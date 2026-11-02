"""Optional attention-only LoRA, disabled in the primary detail-branch run."""
import math
import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankResidual(nn.Module):
    def __init__(self, weight, rank=16):
        super().__init__()
        self.a = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.b = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, weight):
        return weight + self.b @ self.a  # alpha / rank = 1


def add_lora(model, rank=16):
    for block in model.base.blocks:
        for layer in (block.attn.qkv, block.attn.proj):
            parametrize.register_parametrization(layer, "weight", LowRankResidual(layer.weight, rank))


def merge_lora(model):
    for block in model.base.blocks:
        for layer in (block.attn.qkv, block.attn.proj):
            if parametrize.is_parametrized(layer, "weight"):
                parametrize.remove_parametrizations(layer, "weight", leave_parametrized=True)
                layer.weight.requires_grad_(False)
