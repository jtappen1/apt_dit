"""Global patch-size DiT adaptation for a small DDiT-style pilot.

The pretrained DiT path stays intact at patch size 2. Patch size 4 uses a
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
    """DiT-S with patch-2 and patch-4 inference paths.

    `coarse_blocks` is an optional [B,4,4] mask. Each true block covers an
    8x8 latent region and is represented by four patch-4 tokens. This allows
    the late spatial-selection hypothesis while patch size remains global
    when the mask is entirely true (paper-style DDiT behavior).
    """
    def __init__(self, base, rank=32):
        super().__init__()
        if base.x_embedder.patch_size != (2, 2) or base.pos_embed.shape[1] != 256:
            raise ValueError("DDiT pilot expects a 32x32 latent and DiT patch size 2")
        self.base = base.requires_grad_(False)
        d = base.pos_embed.shape[-1]
        self.patch4_embed = nn.Conv2d(4, d, kernel_size=4, stride=4, bias=True)
        self.patch4_deembed = nn.Linear(d, 4 * 4 * base.out_channels)
        self.patch4_id = nn.Parameter(torch.zeros(1, 1, d))
        self.residual = nn.Conv2d(base.in_channels, base.out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

        # R maps flattened 2x2 pixels to bilinearly interpolated 4x4
        # pixels. W4 R = W2; the minimum-norm solution is W2 pinv(R).
        with torch.no_grad():
            w2 = base.x_embedder.proj.weight
            basis = torch.eye(4, device=w2.device, dtype=torch.float64).reshape(4, 1, 2, 2)
            resize = F.interpolate(basis, size=(4, 4), mode="bilinear", align_corners=False)
            projection = resize.reshape(4, 16).T
            w4 = w2.double().flatten(2) @ torch.linalg.pinv(projection)
            self.patch4_embed.weight.copy_(w4.reshape_as(self.patch4_embed.weight))
            self.patch4_embed.bias.copy_(base.x_embedder.proj.bias)
            # Tile the pretrained 2x2 prediction head over the larger patch.
            old = base.final_layer.linear
            old_w = old.weight.reshape(2, 2, base.out_channels, d)
            old_b = old.bias.reshape(2, 2, base.out_channels)
            new_w = self.patch4_deembed.weight.reshape(4, 4, base.out_channels, d)
            new_b = self.patch4_deembed.bias.reshape(4, 4, base.out_channels)
            for y in range(4):
                for x in range(4):
                    new_w[y, x].copy_(old_w[y % 2, x % 2])
                    new_b[y, x].copy_(old_b[y % 2, x % 2])
        self.register_buffer("patch4_pos", _positions(base.pos_embed, 8).flatten(2).transpose(1, 2))
        self.lora = []
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
        coarse_keep = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).flatten(1)
        nf = 256 - int(mask[0].sum().item()) * 16
        nc = 64 - int((~mask[0]).sum().item()) * 4
        fine_idx = torch.argsort(~fine_keep, dim=1, stable=True)[:, :nf]
        coarse_idx = torch.argsort(~coarse_keep, dim=1, stable=True)[:, :nc]
        return fine_idx, coarse_idx

    def forward(self, x, t, y, patch_size=2, coarse_blocks=None):
        if patch_size == 2 and coarse_blocks is None:
            self._set_lora(False)
            return self.base(x, t, y)
        if patch_size not in (2, 4):
            raise ValueError("Supported patch sizes are 2 and 4")
        if coarse_blocks is None:
            coarse_blocks = torch.ones(len(x), 4, 4, dtype=torch.bool, device=x.device)
        if coarse_blocks.shape != (len(x), 4, 4) or coarse_blocks.dtype != torch.bool:
            raise ValueError("coarse_blocks must be boolean [batch,4,4]")
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
        coarse_out = self.patch4_deembed(coarse_features).reshape(len(x), nc, 4, 4, b.out_channels)

        fine_grid = x.new_zeros(len(x), 256, 2, 2, b.out_channels)
        fine_grid.scatter_(1, fine_idx[..., None, None, None].expand_as(fine_out), fine_out)
        fine_grid = fine_grid.reshape(len(x), 16, 16, 2, 2, b.out_channels)
        fine_grid = fine_grid.permute(0, 5, 1, 3, 2, 4).reshape(len(x), b.out_channels, 32, 32)

        coarse_grid = x.new_zeros(len(x), 64, 4, 4, b.out_channels)
        coarse_grid.scatter_(1, coarse_idx[..., None, None, None].expand_as(coarse_out), coarse_out)
        coarse_grid = coarse_grid.reshape(len(x), 8, 8, 4, 4, b.out_channels)
        coarse_grid = coarse_grid.permute(0, 5, 1, 3, 2, 4).reshape(len(x), b.out_channels, 32, 32)

        active = coarse_blocks.repeat_interleave(8, 1).repeat_interleave(8, 2)[:, None]
        output = torch.where(active, coarse_grid, fine_grid)
        return output + self.residual(x)

    def trainable_parameters(self):
        params = [self.patch4_id, *self.patch4_embed.parameters(), *self.patch4_deembed.parameters(),
                  *self.residual.parameters()]
        for module in self.lora:
            params.extend(module.parameters())
        return params
