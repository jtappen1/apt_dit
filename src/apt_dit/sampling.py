"""Use upstream DDIM and CFG semantics, with a causal mixed-layout callback."""
import time
from collections import deque

import torch

from .patching import uniform_layout
from .schedules import model_flops, schedule_counts, select_layout
from .upstream import imports


@torch.no_grad()
def sample(model, *, seed, class_id, arm="baseline", steps=50, count=8,
           cfg_scale=4., allocation_seed=0, signal="clean", budget=None,
           cache=False, device="cuda", bf16=False):
    _, create_diffusion = imports()
    diffusion = create_diffusion(str(steps))
    noise_rng = torch.Generator(device=device).manual_seed(seed)
    allocation_rng = torch.Generator(device=device).manual_seed(seed * 100 + allocation_seed)
    noise = torch.randn(1, 4, 32, 32, generator=noise_rng, device=device)
    z = torch.cat([noise, noise])
    labels = torch.tensor([class_id, 1000], device=device)
    detail = bool(getattr(model, "detail_enabled", False))
    counts = schedule_counts(arm, steps, count, detail)
    clean = None
    history = deque(maxlen=4)
    records, token_counts = [], []
    capture = set(torch.linspace(0, steps - 1, 10).round().long().tolist())
    step_index = 0
    selector_seconds = 0.
    fallbacks = 0

    def predict(x, t, **kwargs):
        nonlocal selector_seconds, fallbacks
        if budget:
            budget.check()
        # Upstream CFG always duplicates the conditional half before inference.
        x = torch.cat([x[:1], x[:1]])
        history.append(x[:1].clone())
        k = counts[step_index]
        start = time.perf_counter()
        if k in (0, 16):
            layout = uniform_layout(1, k == 16, x.device)
        else:
            layout = select_layout(clean, k, random=arm == "random", generator=allocation_rng,
                                   history=list(history), signal=signal)
            fallbacks += int(layout.count != k)
        cfg_layout = layout.cfg()
        # Synchronization is already needed by Layout validation; this is host
        # elapsed selector overhead, not an isolated CUDA kernel measurement.
        selector_seconds += time.perf_counter() - start
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            raw = model(x, t, labels) if arm == "baseline" else model(x, t, labels, cfg_layout)
        raw = raw.float()
        if not torch.isfinite(raw).all():
            raise FloatingPointError(f"Nonfinite prediction at step {step_index}")
        if cache and step_index in capture:
            records.append(dict(x=x.cpu(), t=t.cpu(), y=labels.cpu(), target=raw.cpu(),
                                clean=None if clean is None else clean.cpu(),
                                history=torch.cat(list(history)).cpu(), index=step_index))
        eps, rest = raw[:, :3], raw[:, 3:]
        guided = eps[1:] + cfg_scale * (eps[:1] - eps[1:])
        token_counts.append(layout.tokens)
        return torch.cat([torch.cat([guided, guided]), rest], dim=1)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i, output in enumerate(diffusion.ddim_sample_loop_progressive(
            predict, z.shape, noise=z, clip_denoised=False, device=device, eta=0.)):
        clean = output["pred_xstart"][:1].detach()
        if not torch.isfinite(clean).all():
            raise FloatingPointError(f"Nonfinite clean estimate at step {i}")
        step_index = i + 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latent = output["sample"][:1]
    metrics = dict(seed=seed, class_id=class_id, arm=arm, steps=steps, cfg_scale=cfg_scale,
                   allocation_seed=allocation_seed, signal=signal, tokens=token_counts,
                   denoise_seconds=elapsed, selector_host_seconds=selector_seconds,
                   fallback_steps=fallbacks,
                   estimated_model_gflops=2 * sum(model_flops(n, detail) for n in token_counts) / 1e9)
    return latent, metrics, records
