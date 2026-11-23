"""Benchmark adaptive 2x2/4x4 latent patches on the pretrained DiT-XL checkpoint."""
import argparse
import json
from pathlib import Path
import shutil

from .ddit import DDiT
from .latency_quality import (REFERENCE, Experiment, baseline_settings, comparison_grid,
                              prepare_blind, read_json, report, score_lpips)
from .mixed_sample import window_scores
from .runtime import GPUBudget, local_path, sha256, write_json


FINE_START = 8
FINE_FINISH = 5


class AdaptiveDDIM:
    def __init__(self, diffusion, model, coarse_blocks, cfg, bf16, noise_scale, *,
                 first_fine=FINE_START, last_fine=FINE_FINISH):
        self.diffusion = diffusion
        self.model = model
        self.coarse_blocks = coarse_blocks
        self.cfg = cfg
        self.bf16 = bf16
        self.noise_scale = noise_scale
        self.first_fine = first_fine
        self.last_fine = last_fine

    def ddim_sample_loop(self, predict, *args, **kwargs):
        import torch
        from torch.nn import functional as F

        calls = 0
        history = []
        steps = self.diffusion.num_timesteps

        def adaptive(x, t, y):
            nonlocal calls
            index = calls
            calls += 1
            current = x[:1]
            history.append(current)
            if len(history) > 4:
                history.pop(0)
            if index < self.first_fine or index >= steps - self.last_fine:
                return predict(x, t, y)

            latest, previous, third, fourth = reversed(history)
            change = latest - 3 * previous + 3 * third - fourth
            score = window_scores(change.float())[4].flatten(1)
            indices = score.argsort(dim=1, stable=True)[:, :self.coarse_blocks]
            mask = torch.zeros_like(score, dtype=torch.bool).scatter_(1, indices, True)
            mask = mask.reshape(1, current.shape[-2] // 8, current.shape[-1] // 8).expand(len(x), -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                output = self.model(current.expand_as(x), t, y, coarse_blocks=mask)
            epsilon = output[:, :4].float()
            residual = current - F.interpolate(F.avg_pool2d(current, 2), scale_factor=2)
            coarse = F.interpolate(F.avg_pool2d(epsilon, 2), scale_factor=2)
            coarse = coarse + self.noise_scale[t, None, None, None] * residual
            coarse_pixels = mask.repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)
            epsilon = torch.where(coarse_pixels[:, None], coarse, epsilon)
            rest = torch.cat([epsilon[:, 3:], output[:, 4:].float()], dim=1)
            conditional, unconditional = epsilon[:, :3].chunk(2, dim=0)
            guided = unconditional + self.cfg * (conditional - unconditional)
            return torch.cat([torch.cat([guided, guided]), rest], dim=1).float()

        result = self.diffusion.ddim_sample_loop(adaptive, *args, **kwargs)
        if calls != steps:
            raise RuntimeError(f"Expected {steps} DiT-XL calls, got {calls}")
        return result


class AdaptiveExperiment(Experiment):
    def __init__(self, config, checkpoint):
        super().__init__(config, checkpoint, None)
        self.initialize_adaptive()

    def initialize_adaptive(self):
        import torch

        self.adaptive = self.make_adaptive(4)
        original_alphas = self.diffusion("").alphas_cumprod
        self.noise_scale = (1 - torch.as_tensor(original_alphas, device="cuda", dtype=torch.float32)).sqrt()

    def make_adaptive(self, coarse_size):
        import torch
        from torch.nn import functional as F

        adaptive = DDiT(self.xl.model, rank=0, coarse_size=coarse_size).eval().requires_grad_(False).cuda()
        base_head = self.xl.model.final_layer.linear
        coarse_head = adaptive.patch4_deembed
        channels = self.xl.model.out_channels
        width = base_head.in_features
        # Interpolate the pretrained 2x2 readout over the coarse patch.
        basis = torch.eye(4, device=base_head.weight.device).reshape(4, 1, 2, 2)
        resize = F.interpolate(basis, size=(coarse_size, coarse_size), mode="bilinear", align_corners=False).reshape(4, coarse_size ** 2).T
        coarse_head.weight.copy_(torch.einsum("qp,pcd->qcd", resize, base_head.weight.reshape(4, channels, width))
                                 .reshape_as(coarse_head.weight))
        coarse_head.bias.copy_((resize @ base_head.bias.reshape(4, channels)).reshape_as(coarse_head.bias))
        return adaptive

    def schedule(self, method, steps):
        blocks = int(method.removeprefix("adaptive-b"))
        return AdaptiveDDIM(self.diffusion(str(steps)), self.adaptive, blocks,
                            self.xl.cfg, self.xl.bf16, self.noise_scale)


def describe(rows):
    for result in rows.values():
        blocks = int(result["method"].removeprefix("adaptive-b"))
        mixed_steps = result["steps"] - FINE_START - FINE_FINISH
        result.update(coarse_blocks=blocks, mixed_steps=mixed_steps,
                      mean_tokens_per_call=256 - 12 * blocks * mixed_steps / result["steps"])
    return rows


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "final", "report"))
    parser.add_argument("--baseline", default="artifacts/full-compute-1s")
    parser.add_argument("--comparison", default="artifacts/latency-quality-1s")
    parser.add_argument("--configuration", help="Pilot configuration for the final run, e.g. adaptive-b4-20")
    parser.add_argument("--steps", type=int, nargs="+", default=[20, 22])
    parser.add_argument("--blocks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--ratings", type=Path, nargs=2, metavar=("RATER1", "RATER2"))
    args = parser.parse_args()
    if any(not FINE_START + FINE_FINISH < value <= 100 for value in args.steps):
        parser.error("Steps must leave at least one mixed-patch step")
    if any(not 1 <= value <= 15 for value in args.blocks):
        parser.error("Coarsen between 1 and 15 of the 16 blocks")

    baseline = local_path(args.baseline)
    comparison = local_path(args.comparison)
    output = comparison / "final-with-adaptive"
    if args.stage == "report":
        if args.ratings and args.ratings[0].resolve() == args.ratings[1].resolve():
            parser.error("Use a separate ratings CSV for each reviewer")
        report(output, args.ratings)
        return
    settings = read_json(comparison / "settings.json")
    config, _ = baseline_settings(baseline)
    checkpoint = local_path(config["checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != settings["checkpoint_sha256"]:
        raise SystemExit("The original DiT-XL checkpoint is required")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != settings["gpu"]:
        raise SystemExit(f"Use the comparison's GPU type ({settings['gpu']})")
    for setting in ("checkpoint_sha256", "cfg", "precision", "tf32_matmul", "tf32_cudnn", "vae"):
        if config[setting] != settings[setting]:
            raise SystemExit(f"Baseline differs from the comparison in {setting}")
    if (torch.__version__, torch.version.cuda) != (settings["torch"], settings["cuda"]):
        raise SystemExit("Use the original comparison's PyTorch and CUDA versions")
    torch.backends.cuda.matmul.allow_tf32 = config["tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = config["tf32_cudnn"]

    pairs = read_json(comparison / "pairs.json")
    pilot = comparison / "pilot-adaptive-detail"
    original = comparison / "final"
    if args.stage == "pilot":
        choices = {f"adaptive-b{blocks}-{steps}": (f"adaptive-b{blocks}", steps)
                   for blocks in sorted(set(args.blocks)) for steps in sorted(set(args.steps))}
        folder = pilot
    else:
        rows = read_json(pilot / "timings.json")
        if args.configuration not in rows or not rows[args.configuration]["within_target"]:
            raise SystemExit("Choose a measured --configuration with a pilot median at or under one second")
        choice = rows[args.configuration]
        choices = {args.configuration: (choice["method"], choice["steps"])}
        folder = output
        folder.mkdir(parents=True, exist_ok=True)
        frozen = folder / "adaptive-configuration.json"
        selected = dict(configuration=args.configuration, first_fine_steps=FINE_START,
                        last_fine_steps=FINE_FINISH, stage=args.stage)
        if frozen.is_file() and read_json(frozen) != selected:
            raise SystemExit("The final adaptive configuration is already frozen")
        write_json(frozen, selected)

    with GPUBudget("adaptive_latency_" + args.stage, 1) as budget, torch.inference_mode():
        experiment = AdaptiveExperiment(config, checkpoint)
        rows = describe(experiment.measure(folder, pairs[args.stage], choices, budget, 1., 3))
        if args.stage == "pilot":
            write_json(folder / "timings.json", rows)
            dd = f"ddim-{settings['baseline_steps']}"
            comparison_grid(folder, pairs["pilot"], [(dd, comparison / "pilot-xl", dd)] +
                            [(key, folder, key) for key in choices], "adaptive-candidates")
            return
        existing = read_json(original / "timings.json")
        key = args.configuration
        for method in existing:
            shutil.copytree(original / method, folder / method, dirs_exist_ok=True)
        methods = [method for method in existing if method != REFERENCE] + [key]
        write_json(folder / "timings.json", existing | rows)
        prepare_blind(folder, pairs["final"], methods)
        score_lpips(folder, pairs["final"], [key], budget)
        lpips_path = folder / "lpips.json"
        write_json(lpips_path, read_json(original / "lpips.json") | read_json(lpips_path))
        report(folder, None)
        print(json.dumps({field: rows[key][field] for field in
                          ("median_seconds", "mean_tokens_per_call", "fraction_within_target")}))


if __name__ == "__main__":
    main()
