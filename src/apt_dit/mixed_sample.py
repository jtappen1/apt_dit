"""Batched deterministic DDIM (eta=0) + CFG with per-step patch-size schedulers.

Matches upstream DiT semantics: respaced DDIM, CFG on the first three epsilon
channels, conditional value for the fourth. Both CFG branches share a latent
and a layout. Arms (all on one shared MixedPatchDiT):

  fine            all p=2 (teacher path, exact)
  uniform4/8      all blocks p after warmup
  switch4_k       p=4 for the first k post-warmup steps, then p=2 (DDiT-like fixed switch)
  temporal        DDiT-style: one p per step from the image-wide Delta^3 score vs tau
  spatial         blocks ranked by the first scheduled step's Eq. 3 score, coarsened
                  in rank order to a token budget, then held fixed (causal)
  spatial_random  spatial's per-image size multiset at random block positions (fixed)
  joint           per-block, per-step p from Eq. 3 (proposal's method)
  joint_random    joint's per-step size multiset at random positions (re-drawn each step)

Warmup: Delta^3 needs four latents, so the first three steps run at p=2.
"""
import numpy as np
import torch

from .mixed import tokens_for
from .upstream import imports

WARMUP = 3


def ddim_tables(steps, device):
    _, create_diffusion = imports()
    d = create_diffusion(str(steps))
    ab = torch.tensor(d.alphas_cumprod, dtype=torch.float64)
    abp = torch.tensor(d.alphas_cumprod_prev, dtype=torch.float64)
    tmap = torch.tensor(d.timestep_map, dtype=torch.long)
    return ab, abp, tmap


def window_scores(delta, rho=0.75):
    """Per-block Eq. 3 scores s^(p) for p in (4, 8): [B,4,4] each.

    For each p x p window: spatial std of Delta^3, averaged over channels;
    then the rho-quantile over the block's windows.
    """
    out = {}
    for p in (4, 8):
        w = delta.unfold(2, p, p).unfold(3, p, p)  # B,C,n,n,p,p
        s = w.std(dim=(-2, -1), unbiased=False).mean(1)  # B,n,n
        per = 8 // p
        n = 32 // p
        s = s.reshape(-1, 4, per, 4, per).permute(0, 1, 3, 2, 4).flatten(3)
        out[p] = s.quantile(rho, dim=-1) if per > 1 else s[..., 0]
    return out


def choose(scores, tau, sizes=(2, 4, 8)):
    layout = torch.full_like(scores[4], 2, dtype=torch.long)
    if 4 in sizes:
        layout = torch.where(scores[4] < tau, 4, layout)
    if 8 in sizes:
        layout = torch.where(scores[8] < tau, 8, layout)
    return layout


def budget_layout(score, target_mean, steps):
    """Rank blocks by score (low = coarsen first); p=4 first, then p=8, to hit a
    per-step token count T so that the trajectory mean (incl. 3 fine warmup
    steps) equals target_mean."""
    T = (steps * target_mean - WARMUP * 256) / (steps - WARMUP)
    if T >= 64:
        a, b = 0, int(round((256 - T) / 12))
    else:
        a = int(round((64 - T) / 3))
        b = 16 - a
    a, b = max(0, min(16, a)), max(0, min(16 - a, b))
    order = score.flatten(1).argsort(1)
    flat = torch.full((len(score), 16), 2, dtype=torch.long, device=score.device)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(16, device=score.device).expand_as(order))
    flat = torch.where(rank < a + b, 4, flat)
    flat = torch.where(rank < a, 8, flat)
    return flat.reshape(-1, 4, 4)


def permute_blocks(layout, gen):
    b = len(layout)
    perm = torch.argsort(torch.rand(b, 16, generator=gen, device=layout.device), dim=1)
    return layout.flatten(1).gather(1, perm).reshape(b, 4, 4)


def initial_noise(seeds, device):
    return torch.stack([torch.randn(4, 32, 32, generator=torch.Generator(device=device).manual_seed(int(s)),
                                    device=device) for s in seeds])


@torch.no_grad()
def sample_batch(model, seeds, classes, *, arm="fine", steps=50, tau=None, k=0, cfg=4.,
                 sizes=(2, 4, 8), rho=0.75, base=None, dtype=None, random_seed=0, record=False, target=None):
    """Return final latents [B,4,32,32] and per-step stats."""
    device = next(model.parameters()).device
    x = initial_noise(seeds, device)
    y = torch.as_tensor(classes, device=device)
    B = len(x)
    ab, abp, tmap = ddim_tables(steps, device)
    history = [x]
    gen = torch.Generator(device=device).manual_seed(10_000 + random_seed)
    fixed = None
    prev_layout = None
    stats = dict(tokens=[], turnover=[], frac={2: [], 4: [], 8: []})
    for i, idx in enumerate(range(steps - 1, -1, -1)):
        t = torch.full((2 * B,), int(tmap[idx]), device=device)
        layout = torch.full((B, 4, 4), 2, dtype=torch.long, device=device)
        if arm != "fine" and i >= WARMUP:
            if arm in ("uniform4", "uniform8"):
                layout[:] = int(arm[-1])
            elif arm == "switch4":
                if i - WARMUP < k:
                    layout[:] = 4
            else:
                delta = history[-1] - 3 * history[-2] + 3 * history[-3] - history[-4]
                sc = window_scores(delta.float(), rho)
                if arm == "temporal":
                    glob = {p: s.flatten(1).mean(1)[:, None, None].expand(-1, 4, 4) for p, s in sc.items()}
                    layout = choose(glob, tau, sizes)
                elif arm in ("spatial", "spatial_random"):
                    if fixed is None:
                        # Causal: ranking from the first scheduled step's Eq. 3 score, budget-matched.
                        fixed = budget_layout(sc[8], target, steps) if target is not None else choose(sc, tau, sizes)
                        if arm == "spatial_random":
                            fixed = permute_blocks(fixed, gen)
                    layout = fixed
                elif arm in ("joint", "joint_random"):
                    layout = choose(sc, tau, sizes)
                    if arm == "joint_random":
                        layout = permute_blocks(layout, gen)
                else:
                    raise ValueError(arm)
        xin = torch.cat([x, x])
        yin = torch.cat([y, torch.full_like(y, 1000)])
        lin = torch.cat([layout, layout])
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            if base is not None and arm == "fine":
                out = base(xin, t, yin)
            else:
                out = model(xin, t, yin, lin)
        out = out.float()
        eps_all = out[:, :4]
        cond, uncond = eps_all[:B], eps_all[B:]
        eps = torch.cat([uncond[:, :3] + cfg * (cond[:, :3] - uncond[:, :3]), cond[:, 3:4]], 1)
        a, ap = ab[idx].item(), abp[idx].item()
        x0 = (1 / a) ** .5 * x - (1 / a - 1) ** .5 * eps
        eps2 = ((1 / a) ** .5 * x - x0) / (1 / a - 1) ** .5
        x = x0 * ap ** .5 + (1 - ap) ** .5 * eps2
        history.append(x)
        if len(history) > 4:
            history.pop(0)
        stats["tokens"].append(tokens_for(layout).float().cpu())
        for p in (2, 4, 8):
            stats["frac"][p].append((layout == p).float().mean().item())
        if prev_layout is not None:
            stats["turnover"].append((layout != prev_layout).float().mean().item())
        prev_layout = layout
    stats["tokens"] = torch.stack(stats["tokens"], 1)  # B x steps
    return x, stats
