"""Global patch-size DiT adaptation for a small DDiT-style pilot.

The pretrained DiT path stays intact at patch size 2. Coarse patch sizes use a
separate embedding/de-embedding path, interpolated positions, a patch-size
embedding, FFN LoRA, and a latent-to-output residual projection.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


class FFNLoRA(nn.Module):
    def __init__(self, weight, rank=32):
        super().__init__()
        self.a = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.b = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        self.enabled = False
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, weight):
        if not self.enabled:
            return weight
        return weight + self.b @ self.a


def _positions(pos, size):
    width = pos.shape[-1]
    grid = pos.detach().reshape(1, int(pos.shape[1] ** .5), int(pos.shape[1] ** .5), width)
    grid = grid.permute(0, 3, 1, 2)
    return F.interpolate(grid, size=(size, size), mode="bilinear", align_corners=False)


class DDiT(nn.Module):
    """DiT with patch-2 and a configurable coarse inference path.

    Each active mask entry selects one 8x8 latent region. A selected region
    uses four patch-4 tokens or one patch-8 token.
    """
    def __init__(self, base, rank=32, *, coarse_size=4):
        super().__init__()
        height, width = base.x_embedder.img_size
        if base.x_embedder.patch_size != (2, 2) or height != width or height % 8:
            raise ValueError("DDiT expects square latents divisible by 8 and DiT patch size 2")
        if coarse_size not in (4, 8):
            raise ValueError("Coarse patches must be 4x4 or 8x8")
        self.coarse_size = coarse_size
        self.latent_size = height
        self.layout_size = height // 8
        self.base = base.requires_grad_(False)
        d = base.pos_embed.shape[-1]
        # The patch4 names preserve existing checkpoint keys.
        self.patch4_embed = nn.Conv2d(4, d, kernel_size=coarse_size, stride=coarse_size, bias=True)
        self.patch4_deembed = nn.Linear(d, coarse_size ** 2 * base.out_channels)
        self.patch4_id = nn.Parameter(torch.zeros(1, 1, d))
        self.residual = nn.Conv2d(base.in_channels, base.out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

        # R maps flattened 2x2 pixels to the larger patch via bilinear resize.
        # The minimum-norm coarse embedding satisfying W_coarse R = W2 is W2 pinv(R).
        with torch.no_grad():
            w2 = base.x_embedder.proj.weight
            basis = torch.eye(4, device=w2.device, dtype=torch.float64).reshape(4, 1, 2, 2)
            resize = F.interpolate(basis, size=(coarse_size, coarse_size), mode="bilinear", align_corners=False)
            projection = resize.reshape(4, coarse_size ** 2).T
            coarse_weight = w2.double().flatten(2) @ torch.linalg.pinv(projection)
            self.patch4_embed.weight.copy_(coarse_weight.reshape_as(self.patch4_embed.weight))
            self.patch4_embed.bias.copy_(base.x_embedder.proj.bias)
            # Tile the pretrained 2x2 prediction head over the larger patch.
            old = base.final_layer.linear
            old_w = old.weight.reshape(2, 2, base.out_channels, d)
            old_b = old.bias.reshape(2, 2, base.out_channels)
            new_w = self.patch4_deembed.weight.reshape(coarse_size, coarse_size, base.out_channels, d)
            new_b = self.patch4_deembed.bias.reshape(coarse_size, coarse_size, base.out_channels)
            for y in range(coarse_size):
                for x in range(coarse_size):
                    new_w[y, x].copy_(old_w[y % 2, x % 2])
                    new_b[y, x].copy_(old_b[y % 2, x % 2])
        self.register_buffer("patch4_pos", _positions(base.pos_embed, self.latent_size // coarse_size).flatten(2).transpose(1, 2))
        self.lora = []
        if rank:
            for block in base.blocks:
                for layer in (block.mlp.fc1, block.mlp.fc2):
                    module = FFNLoRA(layer.weight, rank)
                    parametrize.register_parametrization(layer, "weight", module)
                    self.lora.append(module)
        self.rank = rank

    def _set_lora(self, enabled):
        for module in self.lora:
            module.enabled = enabled

    def _condition(self, t, y):
        return self.base.t_embedder(t) + self.base.y_embedder(y, False)

    @staticmethod
    def _gather(tokens, indices):
        return tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))

    def _mixed_indices(self, mask):
        fine_keep = ~mask.repeat_interleave(4, 1).repeat_interleave(4, 2).flatten(1)
        per_side = 8 // self.coarse_size
        coarse_keep = mask.repeat_interleave(per_side, 1).repeat_interleave(per_side, 2).flatten(1)
        coarse_blocks = int(mask[0].sum().item())
        nf = (self.latent_size // 2) ** 2 - 16 * coarse_blocks
        nc = per_side ** 2 * coarse_blocks
        fine_idx = torch.argsort(~fine_keep, dim=1, stable=True)[:, :nf]
        coarse_idx = torch.argsort(~coarse_keep, dim=1, stable=True)[:, :nc]
        return fine_idx, coarse_idx

    def forward(self, x, t, y, patch_size=2, coarse_blocks=None):
        if patch_size == 2 and coarse_blocks is None:
            self._set_lora(False)
            return self.base(x, t, y)
        if patch_size not in (2, self.coarse_size):
            raise ValueError(f"Supported patch sizes are 2 and {self.coarse_size}")
        if coarse_blocks is None:
            coarse_blocks = torch.ones(len(x), self.layout_size, self.layout_size, dtype=torch.bool, device=x.device)
        if coarse_blocks.shape != (len(x), self.layout_size, self.layout_size) or coarse_blocks.dtype != torch.bool:
            raise ValueError("coarse_blocks must have one boolean per 8x8 latent region")
        if not torch.all(coarse_blocks.flatten(1).sum(1) == coarse_blocks[0].sum()):
            raise ValueError("All images in a batch must use the same number of coarse blocks")
        if not coarse_blocks.any():
            return self.forward(x, t, y, patch_size=2)
        if coarse_blocks.all():
            coarse_blocks = torch.ones_like(coarse_blocks)

        self._set_lora(True)
        b = self.base
        fine_tokens = b.x_embedder(x) + b.pos_embed
        coarse_tokens = self.patch4_embed(x).flatten(2).transpose(1, 2) + self.patch4_pos + self.patch4_id
        fine_idx, coarse_idx = self._mixed_indices(coarse_blocks)
        nf, nc = fine_idx.shape[1], coarse_idx.shape[1]
        tokens = torch.cat([self._gather(fine_tokens, fine_idx), self._gather(coarse_tokens, coarse_idx)], dim=1)
        c = self._condition(t, y)
        for block in b.blocks:
            tokens = block(tokens, c)
        fine_features, coarse_features = tokens[:, :nf], tokens[:, nf:]
        fine_out = b.final_layer(fine_features, c).reshape(len(x), nf, 2, 2, b.out_channels)
        shift, scale = b.final_layer.adaLN_modulation(c).chunk(2, dim=1)
        coarse_features = b.final_layer.norm_final(coarse_features)
        coarse_features = coarse_features * (1 + scale[:, None]) + shift[:, None]
        p = self.coarse_size
        coarse_out = self.patch4_deembed(coarse_features).reshape(len(x), nc, p, p, b.out_channels)

        fine_side, coarse_side = self.latent_size // 2, self.latent_size // p
        fine_grid = fine_out.new_zeros(len(x), fine_side ** 2, 2, 2, b.out_channels)
        fine_grid.scatter_(1, fine_idx[..., None, None, None].expand_as(fine_out), fine_out)
        fine_grid = fine_grid.reshape(len(x), fine_side, fine_side, 2, 2, b.out_channels)
        fine_grid = fine_grid.permute(0, 5, 1, 3, 2, 4).reshape(len(x), b.out_channels, self.latent_size, self.latent_size)

        coarse_grid = coarse_out.new_zeros(len(x), coarse_side ** 2, p, p, b.out_channels)
        coarse_grid.scatter_(1, coarse_idx[..., None, None, None].expand_as(coarse_out), coarse_out)
        coarse_grid = coarse_grid.reshape(len(x), coarse_side, coarse_side, p, p, b.out_channels)
        coarse_grid = coarse_grid.permute(0, 5, 1, 3, 2, 4).reshape(len(x), b.out_channels, self.latent_size, self.latent_size)

        active = coarse_blocks.repeat_interleave(8, 1).repeat_interleave(8, 2)[:, None]
        output = torch.where(active, coarse_grid, fine_grid)
        return output + self.residual(x)

    def trainable_parameters(self):
        params = [self.patch4_id, *self.patch4_embed.parameters(), *self.patch4_deembed.parameters(),
                  *self.residual.parameters()]
        for module in self.lora:
            params.extend(module.parameters())
        return params
