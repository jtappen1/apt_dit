"""Quality-first DDIM comparison using the native 512px DiT-XL/2 checkpoint."""
import argparse
import shutil
import statistics
from pathlib import Path

from .adaptive_latency import AdaptiveDDIM, AdaptiveExperiment
from .latency_quality import (baseline_settings, blind_scores, comparison_grid, prepare_blind,
                              read_json, score_lpips)
from .fixed_latency import FullComputeDDIM
from .upstream import imports
from .runtime import GPUBudget, local_path, sha256, write_json


IMAGE_SIZE = 512
STEPS = 50
FINE_START = 20
FINE_FINISH = 12
FINE_TOKENS = 1024
REFERENCE = "ddim-50"
CHECKPOINT_NAME = "DiT-XL-2-512x512.pt"
CHECKPOINT_URL = f"https://dl.fbaipublicfiles.com/DiT/models/{CHECKPOINT_NAME}"


def patch_configuration(method):
    if method == "ddim":
        return 2, 0
    shape, _, count = method.removeprefix("adaptive-").rpartition("b")
    size = int(shape.removeprefix("p").rstrip("-")) if shape else 4
    return size, int(count)


class CaptureLatent:
    def __init__(self, key, schedule):
        self.key = key
        self.schedule = schedule
        self.latent = None

    def ddim_sample_loop(self, *args, **kwargs):
        self.latent = self.schedule.ddim_sample_loop(*args, **kwargs)[:1]
        return self.latent


class QualityExperiment(AdaptiveExperiment):
    def __init__(self, config, checkpoint, precision, *, include_patch8=False):
        import torch
        from .evaluate import load_vae
        _, self.diffusion = imports()
        from models import DiT_models
        latent_size = IMAGE_SIZE // 8
        model = DiT_models["DiT-XL/2"](input_size=latent_size, num_classes=1000)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = weights.get("ema", weights)
        original = state["pos_embed"]
        if original.shape != model.pos_embed.shape:
            raise ValueError("Checkpoint positions do not match native 512px DiT-XL/2")
        model.load_state_dict(state, strict=True)
        del weights, state
        model = model.eval().requires_grad_(False).cuda()
        vae = load_vae().cuda()
        self.xl = FullComputeDDIM(model, vae, config["cfg"], precision)
        self.initialize_adaptive()
        self.adaptive8 = self.make_adaptive(8) if include_patch8 else None
        self.latents = {}

    def schedule(self, method, steps):
        diffusion = self.diffusion(str(steps))
        schedule = diffusion
        if method != "ddim":
            size, blocks = patch_configuration(method)
            model = self.adaptive8 if size == 8 else self.adaptive
            schedule = AdaptiveDDIM(diffusion, model, blocks, self.xl.cfg, self.xl.bf16,
                                    self.noise_scale, first_fine=FINE_START, last_fine=FINE_FINISH)
        return CaptureLatent(f"{method}-{steps}", schedule)

    def generate(self, schedule, example, budget):
        image, timing = super().generate(schedule, example, budget)
        self.latents[schedule.key, example["seed"], example["class_id"]] = schedule.latent.detach().float().cpu()
        return image, timing


def choices_for(blocks):
    choices = {REFERENCE: ("ddim", STEPS)}
    choices.update({f"adaptive-b{count}-{STEPS}": (f"adaptive-b{count}", STEPS) for count in blocks})
    return choices


def transformer_gflops(tokens):
    # Match the repository's transformer-only estimate at DiT-XL width and depth.
    width, depth = 1152, 28
    return sum(depth * (24 * count * width**2 + 4 * count**2 * width + 12 * width**2)
               for count in tokens) / 1e9


def summarize_quality(folder, pairs, choices, timings, latents, budget):
    methods = [key for key in choices if key != REFERENCE]
    score_lpips(folder, pairs, methods, budget, reference_method=REFERENCE)
    distances = read_json(folder / "lpips.json")
    result = {}
    for key, (method, steps) in choices.items():
        size, blocks = patch_configuration(method)
        saved_tokens = (16 - (8 // size) ** 2) * blocks
        tokens = ([FINE_TOKENS] * FINE_START + [FINE_TOKENS - saved_tokens] * (steps - FINE_START - FINE_FINISH)
                  + [FINE_TOKENS] * FINE_FINISH)
        records = distances.get(key, [dict(pair, lpips=0.) for pair in pairs])
        for row in records:
            suffix = row["seed"], row["class_id"]
            delta = latents[key, *suffix] - latents[REFERENCE, *suffix]
            row["latent_mse"] = float(delta.square().mean().item())
        result[key] = dict(timings[key], coarse_blocks=blocks, patch_size=size, mean_tokens=statistics.mean(tokens),
                           tokens_by_step=tokens, transformer_gflops=transformer_gflops(tokens),
                           mean_lpips=statistics.mean(row["lpips"] for row in records),
                           mean_latent_mse=statistics.mean(row["latent_mse"] for row in records),
                           paired_quality=records)
        print(f"{key}: LPIPS {result[key]['mean_lpips']:.4f}, latent MSE {result[key]['mean_latent_mse']:.4f}", flush=True)
    write_json(folder / "quality.json", result)
    return result


def publish(folder, pairs, settings, pilot, rows, ratings=None, *, patch_sweep=False):
    output = local_path("results")
    stem = "adaptive_512_patch_sizes" if patch_sweep else "adaptive_512"
    names = {key: "Full DDIM" if key == REFERENCE else f"Adaptive ({row['coarse_blocks']} regions)"
             for key, row in rows.items()}
    if patch_sweep:
        names = {key: "Full DDIM" if key == REFERENCE else f"{row['patch_size']}×{row['patch_size']} adaptive"
                 for key, row in rows.items()}
    comparison_grid(output, pairs[:8], [(names[key], folder, key) for key in rows], f"{stem}_samples")
    reference_grid = f"{stem}_reference_check.png"
    if not patch_sweep:
        shutil.copyfile(folder.parent / "reference" / "native-reference.png", output / reference_grid)
    quality = {}
    if ratings:
        for key, rating in blind_scores(read_json(folder / "blind-key.json"), ratings).items():
            quality[key] = dict(recognizable=statistics.mean(rating["recognizable"]),
                                artifacts=statistics.mean(rating["artifacts"]),
                                wins=rating["wins"], ties=rating["ties"], losses=rating["losses"])
    write_json(output / f"{stem}.json", dict(settings=settings, pilot=pilot, final=rows, blind_ratings=quality))
    lines = ["# DiT-XL adaptive patch quality at 512×512", "",
             f"Both arms use the official native 512px checkpoint, its original positional grid, DDIM with {STEPS} calls, "
             f"CFG {settings['cfg']:g}, {settings['precision'].upper()} denoising, and the same FP32 VAE on {settings['gpu']}.", "",
             f"The adaptive arm uses 4×4 latent patches in the lowest-scoring regions on steps {FINE_START + 1}–{STEPS - FINE_FINISH}. "
             "The other regions and steps use 2×2 patches. The chosen setting was frozen after inspecting the disjoint pilot.", "",
             "The 50-step reference check used four original seed/class pairs. The pilot used 16 disjoint pairs, and the final run used 64 fresh pairs.", "",
             "| Arm | Samples | XL calls | LPIPS to full DDIM | Latent MSE | Median denoise s | Median total s | Mean tokens | Approx. transformer GFLOPs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in rows.items():
        lines.append(f"| {names[key]} | {len(pairs)} | {row['xl_evals']} | {row['mean_lpips']:.4f} | "
                     f"{row['mean_latent_mse']:.4f} | {row['median_denoise_seconds']:.3f} | {row['median_seconds']:.3f} | "
                     f"{row['mean_tokens']:.1f} | {row['transformer_gflops']:.1f} |")
    lines += ["", "LPIPS and latent MSE measure paired agreement with full DDIM.",
              "Total latency includes VAE decoding. The transformer FLOP estimate is per guidance branch and excludes "
              "patch projections, selector, residual, and VAE costs.", "",
              "Columns in the sample grid share starting noise and class. Rows follow the table order.", "",
              f"![Paired 512px samples]({stem}_samples.png)", "",
              f"![Native 512px reference check]({reference_grid})", "",
              f"Raw artifacts are under {folder.relative_to(local_path('.'))}.", "",
              f"[Official 512px weights]({CHECKPOINT_URL})", ""]
    if patch_sweep:
        lines[0] = "# DiT-XL 4×4 vs. 8×8 adaptive patches at 512×512"
        lines[4] = ("Adaptive arms both coarsen the two lowest-scoring 8×8 latent regions on calls 21–38. "
                    "They use the same 4×4-window scoring rule, checkpoint, and seed/class pairs. "
                    "The 4×4 arm emits four coarse tokens per selected region and the 8×8 arm emits one.")
        lines[6] = "All three arms were measured together on the same 64 evaluation pairs as the original 512px quality run."
        lines = [line for line in lines if reference_grid not in line]
    if quality:
        lines += ["", "| Arm | Class recognizable | Obvious artifacts | Blind W/T/L |", "|---|---:|---:|---:|"]
        for key, rating in quality.items():
            lines.append(f"| {names[key]} | {rating['recognizable']:.1%} | {rating['artifacts']:.1%} | "
                         f"{rating['wins']}/{rating['ties']}/{rating['losses']} |")
    else:
        lines += ["", "Human ratings are pending. The blinded images and ballot template are in the raw artifacts' blind folder.", ""]
    (output / f"{stem}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference", "pilot", "final", "report", "patch-sweep"))
    parser.add_argument("--blocks", type=int, choices=(2, 4, 8))
    parser.add_argument("--ratings", type=Path, nargs=2, metavar=("RATER1", "RATER2"))
    args = parser.parse_args()
    if args.stage == "final" and args.blocks is None:
        parser.error("Choose --blocks after inspecting the pilot")
    root = local_path("artifacts/adaptive_512")
    if args.stage == "report":
        if args.ratings and args.ratings[0].resolve() == args.ratings[1].resolve():
            parser.error("Use a separate CSV for each reviewer")
        folder = root / "final"
        rows = read_json(folder / "quality.json")
        examples = [{"seed": row["seed"], "class_id": row["class_id"]} for row in rows[REFERENCE]["paired_quality"]]
        if not (folder / "blind" / "ratings-template.csv").is_file():
            prepare_blind(folder, examples, list(rows), archive=False, link_images=True)
        publish(folder, examples, read_json(root / "settings.json"), read_json(root / "pilot" / "quality.json"),
                rows, args.ratings)
        return
    config, _ = baseline_settings(local_path("artifacts/full-compute-1s"))
    pairs = read_json(local_path("artifacts/latency-quality-1s/pairs.json"))
    checkpoint = local_path(f"artifacts/checkpoints/{CHECKPOINT_NAME}")
    if not checkpoint.is_file():
        raise SystemExit(f"Official checkpoint missing: {checkpoint}")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != config["gpu"]:
        raise SystemExit(f"Use the original GPU type ({config['gpu']})")
    torch.backends.cuda.matmul.allow_tf32 = config["tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = config["tf32_cudnn"]
    root.mkdir(parents=True, exist_ok=True)
    settings_path = root / "settings.json"
    settings = dict(resolution=IMAGE_SIZE, checkpoint=CHECKPOINT_URL, checkpoint_sha256=sha256(checkpoint),
                    dit_revision=config["dit_revision"], cfg=config["cfg"], precision="bf16", vae=config["vae"],
                    vae_precision="fp32", batch_size=1, ddim_eta=0, gpu=config["gpu"], torch=torch.__version__,
                    cuda=torch.version.cuda, tf32_matmul=config["tf32_matmul"], tf32_cudnn=config["tf32_cudnn"],
                    adaptive_score="third-order difference, 0.75-quantile of spatial standard deviation",
                    first_fine_steps=FINE_START, last_fine_steps=FINE_FINISH)
    if settings_path.is_file() and read_json(settings_path) != settings:
        raise SystemExit("The output folder has a different 512px configuration")
    write_json(settings_path, settings)
    folder = root / args.stage
    if args.stage != "reference" and not (root / "reference" / "native-reference.png").is_file():
        raise SystemExit("Run and inspect the native 512px reference first")
    if args.stage == "reference":
        examples, choices = pairs["reference"][:4], choices_for(())
    elif args.stage == "pilot":
        examples, choices = pairs["pilot"], choices_for((2, 4, 8))
    elif args.stage == "patch-sweep":
        examples = pairs["final"]
        choices = choices_for((2,)) | {"adaptive-p8-b2-50": ("adaptive-p8-b2", STEPS)}
        pilot = read_json(root / "pilot" / "quality.json")
    else:
        examples, choices = pairs["final"], choices_for((args.blocks,))
        pilot = read_json(root / "pilot" / "quality.json")
        if not set(choices) <= set(pilot):
            raise SystemExit("The requested setting was not evaluated in the pilot")
        frozen = root / "frozen.json"
        selection = {key: list(choice) for key, choice in choices.items()}
        if frozen.is_file() and read_json(frozen) != selection:
            raise SystemExit("A different final configuration is already frozen")
        write_json(frozen, selection)
    with GPUBudget("adaptive_512_" + args.stage, 1) as budget, torch.inference_mode():
        experiment = QualityExperiment(config, checkpoint, settings["precision"],
                                       include_patch8=args.stage == "patch-sweep")
        timing = experiment.measure(folder, examples, choices, budget, 1., 1)
        if args.stage == "reference":
            comparison_grid(folder, examples, [("50-step DDIM", folder, REFERENCE)], "native-reference")
            return
        result = summarize_quality(folder, examples, choices, timing, experiment.latents, budget)
    if args.stage == "pilot":
        comparison_grid(folder, examples, [(key, folder, key) for key in choices], "quality-pilot")
    else:
        prepare_blind(folder, examples, list(choices), archive=False, link_images=True)
        publish(folder, examples, settings, pilot, result, patch_sweep=args.stage == "patch-sweep")
        print("Published the 512px report, JSON, and sample grid in results/", flush=True)


if __name__ == "__main__":
    main()
