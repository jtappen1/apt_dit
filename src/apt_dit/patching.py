"""Resize before embedding; repeat features before the unchanged dense head.

The detail branch follows APT's Conv2d -> zero-initialized linear residual.
Blockwise allocation, latent inputs, and feature repetition into DiT's head
are experimental adaptations, not a reproduction of APT classification.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Layout:
    coarse: torch.Tensor  # [batch, 4, 4] boolean; each cell covers 8x8 latent cells
    count: int

    def __post_init__(self):
        if self.coarse.dtype != torch.bool or self.coarse.shape[1:] != (4, 4):
            raise ValueError("Expected boolean [batch,4,4] block mask")
        if not 0 <= self.count <= 16:
            raise ValueError("Coarse count must be in [0,16]")
        if not torch.all(self.coarse.flatten(1).sum(1) == self.count):
            raise ValueError("Every image in a batch must have the declared coarse count")

    @property
    def tokens(self):
        return 256 - 12 * self.count

    def cfg(self):
        return Layout(torch.cat([self.coarse, self.coarse]), self.count)


def uniform_layout(batch, coarse, device):
    return Layout(torch.full((batch, 4, 4), coarse, dtype=torch.bool, device=device), 16 if coarse else 0)


def gather_tokens(tokens, indices):
    return tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))


def packing_indices(layout):
    """Return fine/coarse gather indices and fine-grid -> packed-token mapping."""
    mask = layout.coarse
    fine_keep = ~mask.repeat_interleave(4, 1).repeat_interleave(4, 2).flatten(1)
    coarse_keep = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).flatten(1)
    nf, nc = 256 - 16 * layout.count, 4 * layout.count
    fine = torch.argsort(~fine_keep, dim=1, stable=True)[:, :nf]
    coarse = torch.argsort(~coarse_keep, dim=1, stable=True)[:, :nc]
    fine_rank = fine_keep.long().cumsum(1) - 1
    coarse_rank = (coarse_keep.long().cumsum(1) - 1 + nf).reshape(-1, 8, 8)
    coarse_rank = coarse_rank.repeat_interleave(2, 1).repeat_interleave(2, 2).flatten(1)
    restore = torch.where(fine_keep, fine_rank, coarse_rank)
    return fine, coarse, restore


class DetailBranch(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.aggregate = nn.Conv2d(width, width, kernel_size=2, stride=2)
        self.residual = nn.Linear(width, width)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, fine):
        grid = fine.transpose(1, 2).reshape(fine.shape[0], -1, 16, 16)
        return self.residual(self.aggregate(grid).flatten(2).transpose(1, 2))


class DenseOutputDecoder(nn.Module):
    """Predict a distinct fine-patch correction from coarse transformer features.

    The transformer still runs on the packed sequence. Its final coarse token
    features are expanded to the 16x16 fine grid, combined with the original
    fine input embeddings, then decoded spatially into one 2x2-latent output
    correction per fine token.
    """
    def __init__(self, width, output_dim):
        super().__init__()
        hidden = width // 2
        self.net = nn.Sequential(
            nn.Conv2d(2 * width, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, output_dim, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, transformed, fine_embedding):
        b, n, width = transformed.shape
        if n != 256:
            raise ValueError("Dense output reconstruction expects a restored 16x16 token grid")
        features = torch.cat([transformed, fine_embedding], dim=-1)
        features = features.transpose(1, 2).reshape(b, -1, 16, 16)
        return self.net(features).flatten(2).transpose(1, 2)


class AdaptiveDiT(nn.Module):
    def __init__(self, base, detail_enabled=False):
        super().__init__()
        if base.pos_embed.shape[1] != 256 or base.x_embedder.patch_size != (2, 2):
            raise ValueError("This pilot requires 32x32 latents and base patch size 2")
        self.base = base.requires_grad_(False)
        self.detail = DetailBranch(base.pos_embed.shape[-1])
        self.detail_enabled = detail_enabled
        self.detail.requires_grad_(detail_enabled)
        self.output_decoder = DenseOutputDecoder(
            base.pos_embed.shape[-1], base.final_layer.linear.out_features
        )
        self.reconstruction_enabled = False
        # timm's APT positional resampler defaults to bicubic, antialias=True.
        grid = base.pos_embed.detach().reshape(1, 16, 16, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(grid, size=(8, 8), mode="bicubic", align_corners=False, antialias=True)
        self.register_buffer("coarse_pos", pos.flatten(2).transpose(1, 2))

    def forward(self, x, t, y, layout):
        if x.shape[1:] != (4, 32, 32) or len(x) != len(layout.coarse):
            raise ValueError("Latents and layouts have incompatible shapes")
        b = self.base
        fine = b.x_embedder(x)
        fine_tokens = fine + b.pos_embed
        if layout.count == 0:
            tokens = fine_tokens
            restore = None
        else:
            # Equivalent to individually resizing aligned 4x4 patches to 2x2.
            reduced = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
            coarse = b.x_embedder.proj(reduced).flatten(2).transpose(1, 2)
            if self.detail_enabled:
                coarse = coarse + self.detail(fine)
            fine_idx, coarse_idx, restore = packing_indices(layout)
            tokens = torch.cat([
                gather_tokens(fine_tokens, fine_idx),
                gather_tokens(coarse + self.coarse_pos, coarse_idx),
            ], dim=1)
        # No implicit class dropout: teacher targets already specify each branch.
        c = b.t_embedder(t) + b.y_embedder(y, False)
        for block in b.blocks:
            tokens = block(tokens, c)
        if restore is not None:
            tokens = gather_tokens(tokens, restore)
        output = b.final_layer(tokens, c)
        if self.reconstruction_enabled and layout.count:
            residual = self.output_decoder(tokens, fine_tokens)
            active = layout.coarse.repeat_interleave(4, 1).repeat_interleave(4, 2).flatten(1)
            output = output + residual * active[..., None]
        return b.unpatchify(output)

    def enable_detail_training(self):
        self.detail_enabled = True
        self.detail.requires_grad_(True)
