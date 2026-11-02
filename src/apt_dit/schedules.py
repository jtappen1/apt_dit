"""Causal selectors and fixed-FLOP controls. No model calls in this module."""
import torch
from .patching import Layout, uniform_layout


def block_variation(clean):
    b = clean.reshape(-1, 4, 4, 8, 4, 8).permute(0, 2, 4, 1, 3, 5)
    return (b.diff(dim=-1).abs().mean((3, 4, 5)) + b.diff(dim=-2).abs().mean((3, 4, 5))) / 2


def trajectory_variation(history):
    d = history[-1] - 3 * history[-2] + 3 * history[-3] - history[-4]
    patches = d.unfold(2, 4, 4).unfold(3, 4, 4)
    scores = patches.std(dim=(-2, -1), unbiased=False).mean(1)
    return scores.reshape(-1, 4, 2, 4, 2).permute(0, 1, 3, 2, 4).flatten(3).quantile(.75, dim=-1)


def select_layout(clean, count, *, random=False, generator=None, history=None, signal="clean"):
    if not 0 <= count <= 16:
        raise ValueError("Invalid coarse-block count")
    if random:
        scores = torch.rand((len(clean), 4, 4), device=clean.device, generator=generator)
    elif signal == "clean":
        scores = block_variation(clean)
    elif signal == "trajectory" and history is not None and len(history) >= 4:
        scores = trajectory_variation(history)
    else:
        raise ValueError("Unknown signal or insufficient trajectory history")
    if not torch.isfinite(scores).all():
        return uniform_layout(len(clean), False, clean.device)
    indices = scores.flatten(1).argsort(dim=1, stable=True)[:, :count]
    mask = torch.zeros((len(clean), 16), device=clean.device, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return Layout(mask.reshape(-1, 4, 4), count)


def model_flops(tokens, detail=False, width=384, depth=12):
    """Analytical forward estimate, 2 FLOPs/MAC; excludes elementwise ops.

    Includes full fine embedding, coarse resize projection when used, dense head,
    transformer linear/attention/adaLN work and timestep/class conditioning.
    Scheduler arithmetic is reported separately, not silently counted as zero.
    """
    d, n = width, tokens
    transformer = depth * (24 * n * d * d + 4 * n * n * d + 12 * d * d)
    projections = 2 * 256 * 16 * d + 2 * 256 * d * 32
    conditioning = 2 * (256 * d + d * d) + 4 * d * d
    extra = 0 if n == 256 else 2 * 64 * 16 * d
    if detail and n != 256:
        extra += 2 * 64 * 5 * d * d
    return transformer + projections + conditioning + extra


def schedule_counts(arm, steps=50, count=8, detail=False):
    if arm in ("baseline", "fine"):
        return [0] * steps
    if arm == "coarse":
        return [16] * steps
    active = steps - 4
    if active <= 0:
        raise ValueError("Adaptive schedules require more than four steps")
    target = 4 * model_flops(256, detail) + active * model_flops(256 - 12 * count, detail)
    if arm in ("random", "spatial"):
        return [0] * 4 + [count] * active
    if arm in ("temporal", "joint"):
        if active < 2:
            raise ValueError("A switching schedule needs at least two post-warmup steps")
        early, late = (16, 0) if arm == "temporal" else (12, 4)
        candidates = [[0] * 4 + [early] * k + [late] * (active - k) for k in range(1, active)]
        return min(candidates, key=lambda cs: abs(sum(model_flops(256 - 12 * c, detail) for c in cs) - target))
    raise ValueError(f"Unknown arm: {arm}")
