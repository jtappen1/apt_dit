"""Measure single-image latency and save paired full-compute DiT samples."""
import argparse
import itertools
import json
import math
import random
import statistics
import time

from .runtime import GPUBudget, local_path, manifest, sha256, write_json
from .upstream import imports


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=("xl", "s"), default="xl")
    parser.add_argument("--run-name", required=True, help="New subdirectory in artifacts/")
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 125, 150, 200, 250])
    parser.add_argument("--reference-steps", type=int, default=250)
    parser.add_argument("--target-seconds", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--samples", type=int, default=16, help="Paired quality examples (1–64)")
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--hours", type=float, default=.5)
    parser.add_argument("--resume-quality", action="store_true",
                        help="Finish the image grid using an existing run's saved timings and settings")
    args = parser.parse_args()
    if not args.run_name or args.run_name in (".", "..") or "/" in args.run_name:
        parser.error("--run-name must be a single directory name")
    if any(not 2 <= steps <= 1000 for steps in [*args.steps, args.reference_steps]):
        parser.error("step counts must be between 2 and 1000")
    if args.warmup < 1 or args.repeats < 1 or not 1 <= args.samples <= 64:
        parser.error("use positive warmup and repeat counts and 1–64 samples")
    if any(not math.isfinite(x) or x <= 0 for x in (args.target_seconds, args.hours, args.cfg)):
        parser.error("target, hours and CFG must be finite and positive")
    return args


def summarize(samples, target):
    totals = sorted(sample["total_seconds"] for sample in samples)
    median = statistics.median(totals)
    return dict(median_seconds=median,
                p90_seconds=totals[math.ceil(len(totals) * .9) - 1],
                median_denoise_seconds=statistics.median(s["denoise_seconds"] for s in samples),
                median_decode_seconds=statistics.median(s["decode_seconds"] for s in samples),
                fraction_within_target=sum(t <= target for t in totals) / len(totals),
                within_target=median <= target, trials=samples)


class FullComputeDDIM:
    def __init__(self, model, vae, cfg, precision):
        self.model = model
        self.vae = vae
        self.latent_shape = (model.in_channels, *model.x_embedder.img_size)
        self.cfg = cfg
        self.bf16 = precision == "bf16"

    def predict(self, x, t, y):
        import torch

        # Use the official DiT CFG path, including its first-three-channel convention.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            output = self.model.forward_with_cfg(x, t, y, self.cfg)
        return output.float()

    def generate(self, diffusion, seed, class_id):
        import torch

        torch.cuda.synchronize()
        start = time.perf_counter()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn((1, *self.latent_shape), generator=generator, device="cuda")
        z = torch.cat([noise, noise])
        y = torch.tensor([class_id, 1000], device="cuda")
        latent = diffusion.ddim_sample_loop(
            self.predict, z.shape, noise=z, model_kwargs={"y": y},
            clip_denoised=False, device="cuda", eta=0.,
        )[:1]
        torch.cuda.synchronize()
        decoded_from = time.perf_counter()
        image = self.vae.decode(latent / .18215).sample.clamp(-1, 1)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if not torch.isfinite(image).all().item():
            raise FloatingPointError(f"Nonfinite generation for seed {seed}")
        return (image + 1) / 2, dict(denoise_seconds=decoded_from - start,
                                    decode_seconds=end - decoded_from,
                                    total_seconds=end - start)


def save_comparison(folder, candidates, examples, sampler, schedules, budget):
    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import to_pil_image

    # Same seed and conditioning down each column. Save all images for paired comparisons.
    columns = min(8, len(examples))
    tile, label_height, side = 128, 35, 110
    grid = Image.new("RGB", (side + columns * tile, label_height + len(candidates) * tile), "white")
    draw = ImageDraw.Draw(grid)
    for column, example in enumerate(examples[:columns]):
        draw.text((side + column * tile + 4, 4),
                  f"seed {example['seed']}\nclass {example['class_id']}", fill="black")
    for row, steps in enumerate(candidates):
        destination = folder / f"steps-{steps}"
        destination.mkdir(exist_ok=True)
        print(f"Generating paired samples for {steps} steps", flush=True)
        draw.text((6, label_height + row * tile + 6), f"{steps} steps", fill="black")
        for column, example in enumerate(examples):
            budget.check()
            output = destination / f"seed-{example['seed']}-class-{example['class_id']}.png"
            picture = None
            if output.exists():
                try:
                    with Image.open(output) as saved:
                        picture = saved.copy()
                except OSError:
                    pass
            if picture is None:
                image, _ = sampler.generate(schedules[steps], **example)
                picture = to_pil_image(image[0].cpu())
                temporary = output.with_suffix(".tmp")
                picture.save(temporary, format="PNG")
                temporary.replace(output)
            if column < columns:
                grid.paste(picture.resize((tile, tile), Image.Resampling.LANCZOS),
                           (side + column * tile, label_height + row * tile))
    grid.save(folder / "paired-grid.png")


def main():
    args = parse_args()
    import torch

    from .evaluate import load_vae
    from .upstream import DIT_REVISION, load_dit

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for the latency experiment.")
    checkpoint = local_path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    folder = local_path(f"artifacts/{args.run_name}")
    checkpoint_hash = sha256(checkpoint)
    if args.resume_quality:
        config_path, timings_path = folder / "config.json", folder / "timings.json"
        if not config_path.is_file() or not timings_path.is_file():
            raise SystemExit("Resuming quality requires config.json and completed timings.json.")
        config = json.loads(config_path.read_text())
        timings = json.loads(timings_path.read_text())
        if config["checkpoint_sha256"] != checkpoint_hash:
            raise SystemExit("Checkpoint SHA-256 differs from the original run.")
        if config["dit_revision"] != DIT_REVISION:
            raise SystemExit("DiT revision differs from the original run.")
        args.model, args.cfg, args.precision = config["model"], config["cfg"], config["precision"]
        torch.backends.cuda.matmul.allow_tf32 = config["tf32_matmul"]
        torch.backends.cudnn.allow_tf32 = config["tf32_cudnn"]
        steps_to_load = timings["quality_steps"]
        examples = config["examples"]
    else:
        folder.mkdir(parents=True, exist_ok=False)
        step_counts = sorted(set(args.steps))
        steps_to_load = sorted(set(step_counts + [args.reference_steps]))
        examples = manifest()["final"][:args.samples]
    if not examples:
        raise ValueError("The final manifest contains no examples")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("This GPU does not support BF16.")
    with GPUBudget("fixed_latency_baseline", args.hours) as budget, torch.inference_mode():
        _, create_diffusion = imports()
        if args.model == "xl":
            from models import DiT_models

            model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000)
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(weights.get("ema", weights), strict=True)
            del weights
            model = model.eval().requires_grad_(False).cuda()
        else:
            model = load_dit(checkpoint).cuda()
        vae = load_vae().cuda()
        sampler = FullComputeDDIM(model, vae, args.cfg, args.precision)
        schedules = {steps: create_diffusion(str(steps)) for steps in steps_to_load}
        if args.resume_quality:
            print("Reusing saved latency measurements", flush=True)
            save_comparison(folder, steps_to_load, examples, sampler, schedules, budget)
            print(f"Saved {folder / 'paired-grid.png'}", flush=True)
            return
        write_json(folder / "config.json", dict(vars(args), checkpoint_sha256=checkpoint_hash,
                   dit_revision=DIT_REVISION, gpu=torch.cuda.get_device_name(),
                   torch=torch.__version__, cuda=torch.version.cuda, resolution=256,
                   vae="stabilityai/sd-vae-ft-ema", vae_precision="fp32", batch_size=1,
                   sampler="DDIM", eta=0, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                   tf32_cudnn=torch.backends.cudnn.allow_tf32, examples=examples))
        for steps, repeat in itertools.product(step_counts, range(args.warmup)):
            budget.check()
            sampler.generate(schedules[steps], seed=710000 + repeat, class_id=1)

        rng = random.Random(6220)
        trials = {steps: [] for steps in step_counts}
        for repeat in range(args.repeats):
            seed, class_id = 720000 + repeat, rng.randrange(1000)
            order = step_counts.copy()
            rng.shuffle(order)
            for steps in order:
                budget.check()
                _, timing = sampler.generate(schedules[steps], seed=seed, class_id=class_id)
                trials[steps].append(dict(timing, seed=seed, class_id=class_id))
        rows = [dict(steps=steps, **summarize(trials[steps], args.target_seconds))
                for steps in step_counts]
        eligible = [row["steps"] for row in rows if row["within_target"]]
        over = [row["steps"] for row in rows if not row["within_target"]
                and (not eligible or row["steps"] > eligible[-1])]
        candidates = sorted(set(eligible[-3:] + over[:1] + [args.reference_steps]))
        largest = max(eligible, default=None)
        write_json(folder / "timings.json", dict(target_seconds=args.target_seconds,
                   largest_tested_steps_within_target=largest, quality_steps=candidates, rows=rows))
        for row in rows:
            print(f"{row['steps']:3} steps  median {row['median_seconds']:.3f}s  "
                  f"p90 {row['p90_seconds']:.3f}s  within target {row['fraction_within_target']:.0%}", flush=True)
        print(f"Largest tested step count within target: {largest}", flush=True)
        save_comparison(folder, candidates, examples, sampler, schedules, budget)
        print(f"Saved {folder / 'paired-grid.png'}", flush=True)


if __name__ == "__main__":
    main()
