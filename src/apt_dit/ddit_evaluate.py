"""Paired baseline and schedule-direction evaluation for the DDiT-S pilot."""
import argparse
import json
import time
from collections import deque

import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image

from .ddit import DDiT
from .evaluate import CHECKPOINT, load_vae
from .runtime import GPUBudget, local_path, manifest, sha256, write_json
from .upstream import ROOT, imports, load_dit


def _block_score(history):
    """Third-order latent variation, aggregated within 8x8 latent blocks."""
    d = history[-1] - 3 * history[-2] + 3 * history[-3] - history[-4]
    blocks = d.unfold(2, 8, 8).unfold(3, 8, 8)
    # [B,C,4,4,8,8] -> one variation score per coarse 8x8 region.
    return blocks.std(dim=(1, 4, 5), unbiased=False).flatten(1)


def _mask(strategy, step, steps, history, seed, device):
    half = steps // 2
    if strategy == "all_coarse":
        count, mode = 16, "smooth"
    elif strategy == "coarse_early":
        count = 16 if step < half else 0
        mode = "smooth"
    elif strategy == "coarse_late":
        count = 0 if step < half else 16
        mode = "smooth"
    elif strategy in ("late_signal", "late_random"):
        count = 4 if step < half else 12
        mode = "random" if strategy == "late_random" or step < half else "smooth"
    else:
        raise ValueError(strategy)
    mask = torch.zeros(1, 16, dtype=torch.bool, device=device)
    if count:
        if mode == "smooth" and len(history) == 4 and step >= half:
            scores = _block_score(history)
            chosen = scores.argsort(dim=1, stable=True)[:, :count]
        elif mode == "smooth" and strategy == "coarse_early":
            chosen = torch.arange(count, device=device)[None, :]
        elif mode == "random":
            generator = torch.Generator(device=device).manual_seed(seed * 1009 + step)
            chosen = torch.rand(1, 16, generator=generator, device=device).argsort(dim=1)[:, :count]
        else:
            # Before four latents are available, use a deterministic spatial
            # ordering only for the coarse-early fixed schedule.
            chosen = torch.arange(count, device=device)[None, :]
        mask.scatter_(1, chosen, True)
    return mask.reshape(1, 4, 4)


@torch.no_grad()
def sample(model, *, seed, class_id, strategy, steps=50, cfg_scale=4., budget=None, device="cuda"):
    _, create_diffusion = imports()
    diffusion = create_diffusion(str(steps))
    noise_gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(1, 4, 32, 32, generator=noise_gen, device=device)
    z = torch.cat([noise, noise])
    labels = torch.tensor([class_id, 1000], device=device)
    history = deque(maxlen=4)
    tokens, schedule, step_index = [], [], 0

    def predict(x, t, **kwargs):
        nonlocal step_index
        if budget:
            budget.check()
        current = x[:1].detach()
        history.append(current.clone())
        mask = _mask(strategy, step_index, steps, list(history), seed, device)
        cfg_mask = torch.cat([mask, mask])
        combined = torch.cat([x[:1], x[:1]])
        raw = model(combined, t, labels, coarse_blocks=cfg_mask).float()
        eps = raw[:, :3]
        guided = eps[1:] + cfg_scale * (eps[:1] - eps[1:])
        count = int(mask.sum().item())
        tokens.append(256 - 12 * count)
        schedule.append(count)
        step_index += 1
        return torch.cat([torch.cat([guided, guided]), raw[:, 3:]], dim=1)

    torch.cuda.synchronize()
    started = time.perf_counter()
    output = None
    for output in diffusion.ddim_sample_loop_progressive(
        predict, z.shape, noise=z, clip_denoised=False, device=device, eta=0.
    ):
        pass
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return output["sample"][:1], dict(seed=seed, class_id=class_id, strategy=strategy,
        tokens=tokens, coarse_blocks=schedule, mean_tokens=float(np.mean(tokens)),
        approximate_transformer_gflops=float(sum(
            12 * (24 * n * 384 * 384 + 4 * n * n * 384 + 12 * 384 * 384) for n in tokens
        ) / 1e9), denoise_seconds=elapsed)


def load_model(checkpoint):
    state = torch.load(local_path(checkpoint), map_location="cpu", weights_only=True)
    model_class = DDiT
    if state.get("version") == "ddit_recovery_v1" and state.get("arm") == "C":
        from .ddit_recovery import NoiseSkipDDiT
        model_class = NoiseSkipDDiT
    elif state.get("version") and state["version"] != "ddit_recovery_v1":
        raise ValueError(f"Unsupported checkpoint architecture: {state['version']}")
    model = model_class(load_dit(CHECKPOINT), rank=32)
    model.load_state_dict(state["model"], strict=True)
    return model.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/training/ddit_s_p4/best.pt")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--hours", type=float, default=1.5)
    parser.add_argument("--run-name", default="ddit_schedule_comparison")
    args = parser.parse_args()
    if not 1 <= args.samples <= 64 or args.steps < 8 or args.hours <= 0:
        raise ValueError("Use 1-64 samples, at least 8 denoising steps, and a positive time cap")
    checkpoint = local_path(args.checkpoint)
    folder = local_path(f"artifacts/{args.run_name}")
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(checkpoint_sha256=sha256(checkpoint), steps=args.steps, samples=args.samples,
                  cfg_scale=4., strategies=["baseline", "coarse_early", "coarse_late", "late_signal", "late_random"],
                  note="coarse_early/coarse_late use 25 p4 and 25 p2 steps; late spatial arms use 4 coarse blocks early and 12 late, matching 8000 tokens total")
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run directory already belongs to a different configuration")
    write_json(config_path, config)
    entries = manifest()["final"][:args.samples]
    if any(not (ROOT / f"artifacts/baseline/baseline_0/{e['seed']}.pt").exists() for e in entries):
        raise FileNotFoundError("Missing paired baseline artifacts")
    import lpips
    metric = lpips.LPIPS(net="alex").eval().requires_grad_(False).cuda()
    vae = load_vae().cuda()
    strategies = config["strategies"]
    with GPUBudget("ddit_schedule_evaluation", args.hours) as budget:
        model = load_model(args.checkpoint).cuda()
        for strategy in strategies:
            directory = folder / strategy
            directory.mkdir(exist_ok=True)
            for entry in entries:
                budget.check()
                path = directory / f"{entry['seed']}.pt"
                if path.exists():
                    continue
                if strategy == "baseline":
                    source = torch.load(ROOT / f"artifacts/baseline/baseline_0/{entry['seed']}.pt",
                                        map_location="cpu", weights_only=True)
                    image, latent = source["image"], source["latent"]
                    metrics = source["metrics"].copy()
                else:
                    latent_gpu, metrics = sample(model, **entry, strategy=strategy, steps=args.steps, budget=budget)
                    image = vae.decode(latent_gpu / .18215).sample.clamp(-1, 1).cpu()
                    latent = latent_gpu.cpu()
                    baseline = torch.load(ROOT / f"artifacts/baseline/baseline_0/{entry['seed']}.pt",
                                          map_location="cuda", weights_only=True)
                    metrics["lpips"] = metric(image.cuda(), baseline["image"]).item()
                    metrics["latent_mse"] = (latent_gpu - baseline["latent"]).square().mean().item()
                    del baseline
                torch.save(dict(image=image, latent=latent, metrics=metrics), path)
                print(f"{strategy}: {entry['seed']} LPIPS={metrics.get('lpips',0):.4f} "
                      f"tokens={metrics.get('mean_tokens',256):.1f}", flush=True)
        _summarize(folder, entries, strategies)


def _summarize(folder, entries, strategies):
    rows, image_rows = [], []
    for strategy in strategies:
        metrics, images = [], []
        for entry in entries:
            record = torch.load(folder / strategy / f"{entry['seed']}.pt", map_location="cpu", weights_only=True)
            metrics.append(record["metrics"])
            if len(images) < 8:
                images.append(record["image"])
        if strategy == "baseline":
            rows.append(dict(strategy=strategy, samples=len(metrics), lpips=0., latent_mse=0.,
                             median_denoise_seconds=float(np.median([m["denoise_seconds"] for m in metrics])),
                             mean_tokens=256., mean_gflops=None))
        else:
            rows.append(dict(strategy=strategy, samples=len(metrics),
                             lpips=float(np.mean([m["lpips"] for m in metrics])),
                             latent_mse=float(np.mean([m["latent_mse"] for m in metrics])),
                             median_denoise_seconds=float(np.median([m["denoise_seconds"] for m in metrics])),
                             mean_tokens=float(np.mean([m["mean_tokens"] for m in metrics])),
                             mean_gflops=float(np.mean([m["approximate_transformer_gflops"] for m in metrics]))))
        image_rows.append(torch.cat(images))
    write_json(folder / "summary.json", rows)
    save_image(torch.cat(image_rows), folder / "samples.png", nrow=8, normalize=True, value_range=(-1, 1))
    lines = ["# DDiT-S schedule comparison", "",
        "All methods use the same ImageNet classes and noise seeds. The two global schedules each use 25 patch-4 and 25 patch-2 steps. The late spatial schedules use four coarse blocks during the first half and twelve during the second half; each arm therefore processes 8,000 total tokens over 50 steps. The late-signal arm chooses smooth blocks from the third-order latent difference; late-random is the matched control.", "",
        "| Strategy | Samples | LPIPS to DiT-S | Latent MSE | Median denoise s | Mean tokens | Approx. transformer GFLOPs |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        gflops = "—" if row.get("mean_gflops") is None else f"{row['mean_gflops']:.1f}"
        lines.append(f"| {row['strategy']} | {row['samples']} | {row['lpips']:.4f} | {row['latent_mse']:.4f} | {row['median_denoise_seconds']:.3f} | {row['mean_tokens']:.1f} | {gflops} |")
    lines += ["", "LPIPS measures paired similarity to the original DiT-S output, not distributional quality. The transformer FLOP estimate excludes patch embedding/de-embedding, residual, selector, CFG, and VAE costs; measured latency includes them.", "", f"Samples: `artifacts/{folder.name}/samples.png`."]
    report = ROOT / "results" / f"{folder.name}_report.md"
    report.write_text("\n".join(lines) + "\n")
    print(report.read_text(), flush=True)


if __name__ == "__main__":
    main()
