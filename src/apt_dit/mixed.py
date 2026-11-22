"""One shared mixed-patch DiT-S for spatial, temporal, and joint schedules.

Implements the proposal's Section 3.1 representation:
- the 32x32 latent is split into sixteen g=8 blocks, each tokenized at its own
  patch size p in {2, 4, 8} (16, 4, or 1 tokens per block);
- every patch size has its own input embedding and output de-embedding; p=2
  reuses the frozen pretrained layers;
- coarse tokens get a learned patch-size embedding and fixed 2D sin-cos
  positions evaluated exactly at their patch centers;
- the transformer is adapted with rank-32 LoRA on attention and MLP linears.
  The LoRA update is applied token-wise to coarse tokens only, so an all-fine
  layout reproduces the frozen teacher exactly (fine-token weights unchanged).
- ``full_finetune=True`` is the pre-registered fallback: a trainable copy of
  all transformer weights (fine path no longer exact).

A layout is an integer tensor [B, 4, 4] of patch sizes. Images in one batch
may use different token counts; shorter sequences are padded and masked.
"""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SIZES = (2, 4, 8)
BLOCKS = 4  # blocks per side; each block is 8x8 latent cells


def sincos_at(dim, coords_w, coords_h):
    from .upstream import imports
    imports()
    from models import get_2d_sincos_pos_embed_from_grid
    grid = np.meshgrid(coords_w, coords_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, len(coords_h), len(coords_w))
    return torch.from_numpy(get_2d_sincos_pos_embed_from_grid(dim, grid)).float()


def size_positions(dim, p):
    """Sin-cos positions at the centers of p-patches, in fine-patch (p=2) units."""
    n = 32 // p
    scale = p // 2
    centers = np.arange(n, dtype=np.float64) * scale + (scale - 1) / 2
    return sincos_at(dim, centers, centers)


def tokens_for(layout):
    """Token count per image for a [B,4,4] patch-size layout."""
    return (64 // layout.long().square()).flatten(1).sum(1)


class TokenLoRA(nn.Module):
    def __init__(self, layer, rank):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, layer.in_features))
        self.b = nn.Parameter(torch.zeros(layer.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x, coarse):
        return F.linear(F.linear(x, self.a), self.b) * coarse


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FineDecoder(nn.Module):
    """Fine-grid correction for coarse regions (cf. EXPERIMENTS.md Sec. 2).

    Input per fine (p=2) cell: the coarse trunk feature covering it, and the
    frozen fine patch embedding of x_t (carries per-pixel noise/detail the
    coarse token cannot). Conditioned on (t, y) via an additive bias.
    Output: one 2x2xC patch per fine cell; zero-initialized.
    """
    def __init__(self, width, cout, hidden=192):
        super().__init__()
        self.inp = nn.Conv2d(2 * width, hidden, 1)
        self.cond = nn.Linear(width, hidden)
        self.mid = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, 4 * cout, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, coarse_feats, fine_emb, c):
        h = self.inp(torch.cat([coarse_feats, fine_emb], 1)) + self.cond(F.silu(c))[:, :, None, None]
        h = self.mid(F.silu(h))
        return self.out(F.silu(h))  # B, 4*C, 16, 16


class MixedPatchDiT(nn.Module):
    def __init__(self, base, rank=32, full_finetune=False):
        super().__init__()
        if base.x_embedder.patch_size != (2, 2) or base.pos_embed.shape[1] != 256:
            raise ValueError("Expects DiT with 32x32 latents and patch size 2")
        self.base = base.requires_grad_(False)
        self.full_finetune = full_finetune
        d = base.pos_embed.shape[-1]
        self.width, self.heads = d, base.blocks[0].attn.num_heads
        cin, cout = base.in_channels, base.out_channels
        self.cout = cout
        self.trunk = copy.deepcopy(base.blocks).requires_grad_(True) if full_finetune else base.blocks

        self.embed = nn.ModuleDict()
        self.deembed = nn.ModuleDict()
        self.size_id = nn.ParameterDict()
        with torch.no_grad():
            w2, b2 = base.x_embedder.proj.weight, base.x_embedder.proj.bias
            head_w, head_b = base.final_layer.linear.weight, base.final_layer.linear.bias
            for p in SIZES[1:]:
                f = p // 2
                conv = nn.Conv2d(cin, d, kernel_size=p, stride=p)
                # Embed(average-pool to 2x2) == W2 applied to the downsampled patch.
                conv.weight.copy_(w2.repeat_interleave(f, 2).repeat_interleave(f, 3) / (f * f))
                conv.bias.copy_(b2)
                self.embed[str(p)] = conv
                lin = nn.Linear(d, p * p * cout)
                # Nearest-upsampled copy of the pretrained 2x2 head.
                ow = head_w.reshape(2, 2, cout, d)
                ob = head_b.reshape(2, 2, cout)
                lin.weight.copy_(ow.repeat_interleave(f, 0).repeat_interleave(f, 1).reshape(-1, d))
                lin.bias.copy_(ob.repeat_interleave(f, 0).repeat_interleave(f, 1).reshape(-1))
                self.deembed[str(p)] = lin
                self.size_id[str(p)] = nn.Parameter(torch.zeros(d))
                self.register_buffer(f"pos{p}", size_positions(d, p)[None])
        self.decoder = FineDecoder(d, cout)
        self.lora = nn.ModuleList()
        if not full_finetune and rank:
            for blk in base.blocks:
                self.lora.append(nn.ModuleDict(dict(
                    qkv=TokenLoRA(blk.attn.qkv, rank), proj=TokenLoRA(blk.attn.proj, rank),
                    fc1=TokenLoRA(blk.mlp.fc1, rank), fc2=TokenLoRA(blk.mlp.fc2, rank))))
        # Noise skip for coarse regions (EXPERIMENTS.md Sec. 5): at high noise
        # eps ~= x_t, which coarse tokens cannot pass back through the trunk.
        from .upstream import imports
        _, create_diffusion = imports()
        self.register_buffer("alphas_cumprod", torch.tensor(create_diffusion("").alphas_cumprod, dtype=torch.float32))
        # Candidate token order: all p2 (256), p4 (64), p8 (16) grid tokens.
        self._offsets = {2: 0, 4: 256, 8: 320}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- layout bookkeeping -------------------------------------------------
    @staticmethod
    def keep_mask(layout):
        """[B,336] bool: which candidate tokens a [B,4,4] layout uses."""
        parts = []
        for p in SIZES:
            per_side = 8 // p  # tokens per block side
            sel = (layout == p).repeat_interleave(per_side, 1).repeat_interleave(per_side, 2)
            parts.append(sel.flatten(1))
        return torch.cat(parts, 1)

    def _linear(self, layer, lora, x, coarse):
        y = F.linear(x, layer.weight, layer.bias)
        return y if lora is None else y + lora(x, coarse)

    def _block(self, i, blk, x, c, coarse, attn_mask):
        lo = dict(self.lora[i]) if len(self.lora) else {}
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaLN_modulation(c).chunk(6, dim=1)
        h = modulate(blk.norm1(x), shift_msa, scale_msa)
        b, n, d = h.shape
        qkv = self._linear(blk.attn.qkv, lo.get("qkv"), h, coarse)
        q, k, v = qkv.reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        a = a.transpose(1, 2).reshape(b, n, d)
        x = x + gate_msa.unsqueeze(1) * self._linear(blk.attn.proj, lo.get("proj"), a, coarse)
        h = modulate(blk.norm2(x), shift_mlp, scale_mlp)
        h = self._linear(blk.mlp.fc1, lo.get("fc1"), h, coarse)
        h = F.gelu(h, approximate="tanh")
        h = self._linear(blk.mlp.fc2, lo.get("fc2"), h, coarse)
        return x + gate_mlp.unsqueeze(1) * h

    def forward(self, x, t, y, layout=None):
        b = self.base
        if layout is None or (not self.full_finetune and bool((layout == 2).all())):
            return b(x, t, y)
        if layout.shape != (len(x), 4, 4):
            raise ValueError("layout must be [B,4,4] patch sizes")
        B = len(x)
        fine_emb = b.x_embedder(x)
        cands = [fine_emb + b.pos_embed]
        size_tag = [torch.zeros(256, device=x.device)]
        for p in SIZES[1:]:
            e = self.embed[str(p)](x).flatten(2).transpose(1, 2)
            cands.append(e + getattr(self, f"pos{p}") + self.size_id[str(p)])
            size_tag.append(torch.ones(e.shape[1], device=x.device))
        cand = torch.cat(cands, 1)  # [B,336,d]
        coarse_tag = torch.cat(size_tag)  # [336]
        keep = self.keep_mask(layout)
        counts = keep.sum(1)
        nmax = int(counts.max())
        order = torch.argsort((~keep).to(torch.uint8), dim=1, stable=True)[:, :nmax]
        tokens = cand.gather(1, order[..., None].expand(-1, -1, cand.shape[-1]))
        coarse = coarse_tag[order][..., None].to(tokens.dtype)
        valid = torch.arange(nmax, device=x.device)[None] < counts[:, None]
        attn_mask = None if bool(valid.all()) else valid[:, None, None, :]
        c = b.t_embedder(t) + b.y_embedder(y, False)
        for i, blk in enumerate(self.trunk):
            tokens = self._block(i, blk, tokens, c, coarse, attn_mask)
        # Scatter back to candidate slots; padded positions are discarded.
        # Padded entries go to an extra dump slot so they cannot overwrite real tokens.
        full = tokens.new_zeros(B, cand.shape[1] + 1, tokens.shape[-1])
        src_idx = torch.where(valid, order, torch.full_like(order, cand.shape[1]))
        full.scatter_(1, src_idx[..., None].expand_as(tokens), tokens)
        full = full[:, :-1]
        fl = b.final_layer
        shift, scale = fl.adaLN_modulation(c).chunk(2, dim=1)
        normed = modulate(fl.norm_final(full), shift, scale)
        out = x.new_zeros(B, self.cout, 32, 32)
        d = normed.shape[-1]
        fmap = normed.new_zeros(B, d, 16, 16)
        for p in SIZES[1:]:
            lo, hi = self._offsets[p], self._offsets[p] + (32 // p) ** 2
            g, f = 32 // p, p // 2
            grid = normed[:, lo:hi].transpose(1, 2).reshape(B, d, g, g)
            grid = grid.repeat_interleave(f, 2).repeat_interleave(f, 3)
            sel = (layout == p).repeat_interleave(4, 1).repeat_interleave(4, 2)[:, None]
            fmap = torch.where(sel, grid, fmap)
        femb = fine_emb.transpose(1, 2).reshape(B, d, 16, 16).to(fmap.dtype)
        corr = self.decoder(fmap, femb, c.to(fmap.dtype))  # B, 4C, 16, 16  (per fine cell 2x2xC)
        corr = corr.reshape(B, 2, 2, self.cout, 16, 16).permute(0, 3, 4, 1, 5, 2).reshape(B, self.cout, 32, 32)
        for p in SIZES:
            lo, hi = self._offsets[p], self._offsets[p] + (32 // p) ** 2
            feats = normed[:, lo:hi]
            head = fl.linear if p == 2 else self.deembed[str(p)]
            pred = head(feats)  # [B, n, p*p*C]
            g = 32 // p
            pred = pred.reshape(B, g, g, p, p, self.cout).permute(0, 5, 1, 3, 2, 4).reshape(B, self.cout, 32, 32)
            region = (layout == p).repeat_interleave(8, 1).repeat_interleave(8, 2)[:, None]
            pred = pred.to(out.dtype)
            if p != 2:
                pred = pred + corr.to(out.dtype)
                ab = self.alphas_cumprod[t].to(out.dtype)[:, None, None, None]
                eps = (1 - ab).sqrt() * x[:, :4].to(out.dtype) + ab.sqrt() * pred[:, :4]
                pred = torch.cat([eps, pred[:, 4:]], 1)
            out = torch.where(region, pred, out)
        return out


def load_mixed(checkpoint_path, base_checkpoint, device="cuda"):
    from .upstream import load_dit
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = MixedPatchDiT(load_dit(base_checkpoint), rank=state["rank"], full_finetune=state["full_finetune"])
    model.load_state_dict(state["model"], strict=True)
    return model.eval().requires_grad_(False).to(device), state
