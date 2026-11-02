"""Paired sampling, immutable run artifacts, and compact comparisons."""
import argparse
import json
import time

import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image

from .patching import AdaptiveDiT, uniform_layout
from .runtime import GPUBudget, local_path, manifest, sha256, write_json
from .sampling import sample
from .upstream import ROOT, load_dit

CHECKPOINT = ROOT / "artifacts/checkpoints/dit_s_256.pt"


def load_vae():
    from huggingface_hub import HfApi
    metadata = ROOT / "artifacts/vae.json"
    if metadata.exists():
        spec = json.loads(metadata.read_text())
    else:
        repo = "stabilityai/sd-vae-ft-ema"
        spec = dict(repo=repo, revision=HfApi().model_info(repo).sha, latent_scale=.18215)
        write_json(metadata, spec)
    return AutoencoderKL.from_pretrained(spec["repo"], revision=spec["revision"],
                                         use_safetensors=True).eval().requires_grad_(False)


def load_student(stage="zero_shot", checkpoint=None):
    torch.manual_seed(6220)
    model = AdaptiveDiT(load_dit(CHECKPOINT))
    if stage == "trained":
        if checkpoint is None:
            raise ValueError("Trained evaluation requires --checkpoint")
        state = torch.load(local_path(checkpoint), map_location="cpu", weights_only=True)
        if state.get("lora_rank", 0):
            from .lora import add_lora, merge_lora
            add_lora(model, state["lora_rank"])
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if unexpected or any(not key.startswith("output_decoder.") for key in missing):
            raise RuntimeError(f"Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")
        if state.get("lora_rank", 0):
            merge_lora(model)
        model.detail_enabled = True
        model.reconstruction_enabled = bool(state.get("reconstruction_enabled", False))
    return model.eval().requires_grad_(False)


@torch.no_grad()
def verify():
    with GPUBudget("verification", .05):
        model = load_student().cuda()
        x = torch.randn(2, 4, 32, 32, device="cuda")
        t = torch.tensor([500, 500], device="cuda")
        y = torch.tensor([1, 1000], device="cuda")
        expected = model.base(x, t, y)
        actual = model(x, t, y, uniform_layout(2, False, "cuda"))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reduced = model.base(x, t, y).float()
        relative = ((reduced - expected).square().mean() / expected.square().mean()).sqrt().item()
        info = dict(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), parameters=sum(p.numel() for p in model.base.parameters()),
                    checkpoint_sha256=sha256(CHECKPOINT), all_fine_max_error=(actual-expected).abs().max().item(),
                    bf16_relative_rmse=relative, sampling_precision="fp32",
                    note="FP32 retained for initial controlled comparison; BF16 difference reported, not assumed harmless.")
        write_json("results/verification.json", info)
        print(json.dumps(info, indent=2), flush=True)


@torch.no_grad()
def evaluate(args):
    stage = args.stage
    folder = local_path(f"artifacts/{args.run_name or stage}")
    folder.mkdir(parents=True, exist_ok=True)
    entries = manifest()["final"][:args.samples]
    if not 1 <= args.samples <= 64:
        raise ValueError("Choose 1 to 64 manifest samples")
    arms = args.arms or (["baseline"] if stage == "baseline" else ["fine", "coarse", "random", "spatial"])
    spec = dict(stage=stage, checkpoint_sha256=sha256(CHECKPOINT if not args.checkpoint else local_path(args.checkpoint)),
                steps=args.steps, cfg_scale=4., signal=args.signal, count=args.count, precision="fp32")
    spec_path = folder / "config.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Run directory already belongs to a different configuration")
    write_json(spec_path, spec)
    vae = load_vae()
    import lpips
    metric = lpips.LPIPS(net="alex").eval().requires_grad_(False)
    with GPUBudget(stage + "_evaluation", args.hours) as budget:
        student = load_student(stage, args.checkpoint) if stage != "baseline" else None
        model = load_dit(CHECKPOINT) if stage == "baseline" else student
        model = model.cuda()
        vae, metric = vae.cuda(), metric.cuda()
        if stage == "zero_shot":
            snapshot = folder / "initial.pt"
            if not snapshot.exists():
                torch.save(dict(model=model.cpu().state_dict(), lora_rank=0), snapshot)
                model.cuda()
        for arm in arms:
            reps = range(3) if arm == "random" else range(1)
            for rep in reps:
                name = f"{arm}_{rep}"
                directory = folder / name
                directory.mkdir(exist_ok=True)
                # Warm up the exact model path; don't include it in per-image timings.
                sample(model, seed=999, class_id=1, arm=arm, count=args.count, signal=args.signal,
                       budget=budget, steps=args.steps)
                for entry in entries:
                    budget.check()
                    path = directory / f"{entry['seed']}.pt"
                    if path.exists():
                        continue
                    baseline = ROOT / f"artifacts/baseline/baseline_0/{entry['seed']}.pt"
                    is_reference = stage == "baseline" and folder.name == "baseline" and args.steps == 50
                    if not is_reference and not baseline.exists():
                        raise FileNotFoundError(f"Generate baseline first: {baseline}")
                    torch.cuda.reset_peak_memory_stats()
                    latent, metrics, _ = sample(model, **entry, arm=arm, count=args.count, signal=args.signal,
                                                allocation_seed=rep, budget=budget, steps=args.steps)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    image = vae.decode(latent / .18215).sample.clamp(-1, 1)
                    torch.cuda.synchronize()
                    metrics["decode_seconds"] = time.perf_counter() - start
                    metrics["end_to_end_seconds"] = metrics["denoise_seconds"] + metrics["decode_seconds"]
                    metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
                    if is_reference:
                        metrics.update(lpips=0., latent_mse=0.)
                    else:
                        reference = torch.load(baseline, map_location="cuda", weights_only=True)
                        metrics["lpips"] = metric(image, reference["image"]).item()
                        metrics["latent_mse"] = (latent-reference["latent"]).square().mean().item()
                    torch.save(dict(image=image.cpu(), latent=latent.cpu(), metrics=metrics), path)
                    print(f"{stage}/{name}: seed={entry['seed']} LPIPS={metrics['lpips']:.4f} time={metrics['denoise_seconds']:.3f}s", flush=True)
        summarize(folder, entries, arms)


def summarize(folder, entries, arms):
    rows, image_rows = [], []
    for arm in arms:
        replicate_metrics = []
        for rep in range(3 if arm == "random" else 1):
            metrics, images = [], []
            for entry in entries:
                path = folder / f"{arm}_{rep}/{entry['seed']}.pt"
                if not path.exists():
                    continue
                record = torch.load(path, map_location="cpu", weights_only=True)
                metrics.append(record["metrics"])
                if len(images) < 8:
                    images.append(record["image"])
            if not metrics:
                continue
            if rep == 0:
                image_rows.append(torch.cat(images))
            replicate_metrics.append(metrics)
        if not replicate_metrics:
            continue
        # Report every completed trajectory; paired analyses use the common prefix.
        flat = [m for ms in replicate_metrics for m in ms]
        rows.append(dict(arm=arm, trajectories=len(replicate_metrics[0]), replicates=len(replicate_metrics),
                         lpips=float(np.mean([m["lpips"] for m in flat])),
                         latent_mse=float(np.mean([m["latent_mse"] for m in flat])),
                         median_denoise_seconds=float(np.median([m["denoise_seconds"] for m in flat])),
                         mean_tokens=float(np.mean([np.mean(m["tokens"]) for m in flat])),
                         estimated_model_gflops=float(np.mean([m["estimated_model_gflops"] for m in flat]))))
    write_json(folder / "summary.json", rows)
    if image_rows:
        save_image(torch.cat(image_rows), folder / "samples.png", nrow=min(8, len(entries)), normalize=True, value_range=(-1, 1))
    lines = [f"# {folder.name}: paired DiT-S evaluation", "", "Rows in samples.png follow the table order; columns share seeds/classes.", "",
             "| Arm | Samples | Repeats | LPIPS to baseline | Latent MSE | Median denoise s | Mean tokens | Model GFLOPs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['trajectories']} | {r['replicates']} | {r['lpips']:.4f} | {r['latent_mse']:.4f} | {r['median_denoise_seconds']:.3f} | {r['mean_tokens']:.1f} | {r['estimated_model_gflops']:.1f} |")
    lines += ["", "LPIPS measures agreement with the original model, not distributional quality. No FID claim.",
              "Model FLOPs are analytical (2 FLOPs/MAC); exclude elementwise, resize, selector, and VAE work. Latency includes selector and resize overhead.",
              "These timings include Python checks; use the dedicated repeated benchmark for a latency conclusion.",
              f"\nArtifacts: `{folder.relative_to(ROOT)}`."]
    target = ROOT / "results" / ("zero_shot_report.md" if folder.name == "zero_shot" else f"{folder.name}_report.md")
    target.parent.mkdir(exist_ok=True)
    target.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["verify", "baseline", "zero_shot", "trained"])
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--hours", type=float, default=1.)
    p.add_argument("--checkpoint")
    p.add_argument("--run-name")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--signal", choices=["clean", "trajectory"], default="clean")
    p.add_argument("--arms", nargs="+", choices=["baseline", "fine", "coarse", "random", "spatial", "temporal", "joint"])
    args = p.parse_args()
    if args.stage == "verify":
        verify()
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
