"""Compare DDIM, DPM-Solver++ and DiT-S to DiT-XL stitching at fixed latency."""
import argparse
import csv
import html
from itertools import combinations
import json
import os
import random
import shutil
import statistics
from pathlib import Path

from .fixed_latency import FullComputeDDIM, summarize
from .runtime import GPUBudget, local_path, sha256, write_json
from .upstream import DIT_REVISION, imports, load_dit


REFERENCE = "reference-50"
METHODS = ("ddim", "dpm", "stitch20", "stitch40")


def read_json(path):
    return json.loads(path.read_text())


def tag(method, steps):
    return f"{method}-{steps}"


def stage_pairs(previous, count, first_seed, rng_seed):
    rng = random.Random(rng_seed)
    occupied = {item["seed"] for item in previous}
    seeds = (seed for seed in range(first_seed, first_seed + count + len(occupied))
             if seed not in occupied)
    return [dict(seed=next(seeds), class_id=rng.randrange(1000)) for _ in range(count)]


def small_calls(method, steps):
    ratio = {"stitch20": .2, "stitch40": .4}.get(method, 0)
    return int(steps * ratio + .5) if ratio else 0


class StitchedDDIM:
    def __init__(self, diffusion, small, first_steps):
        self.diffusion = diffusion
        self.small = small
        self.first_steps = first_steps

    def ddim_sample_loop(self, predict, *args, **kwargs):
        calls = 0

        def stitched(x, t, y):
            nonlocal calls
            model = self.small if calls < self.first_steps else predict
            calls += 1
            return model(x, t, y)

        result = self.diffusion.ddim_sample_loop(stitched, *args, **kwargs)
        if calls != self.diffusion.num_timesteps:
            raise RuntimeError(f"Expected {self.diffusion.num_timesteps} DDIM calls, got {calls}")
        return result


class SolverPP:
    def __init__(self, betas, steps):
        from diffusers import DPMSolverMultistepScheduler

        self.steps = steps
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=len(betas), trained_betas=betas,
            algorithm_type="dpmsolver++", prediction_type="epsilon", solver_order=2,
            solver_type="midpoint", lower_order_final=True, timestep_spacing="linspace",
        )

    def reset(self):
        self.scheduler.set_timesteps(self.steps, device="cuda")
        if len(self.scheduler.timesteps) != self.steps:
            raise RuntimeError("DPM-Solver++ did not create the requested number of evaluations")

    def ddim_sample_loop(self, predict, _shape, *, noise, model_kwargs, **kwargs):
        latent = noise[:1]
        for timestep in self.scheduler.timesteps:
            # Keep the original DiT CFG convention and use the conditional epsilon channels.
            epsilon = predict(latent.expand(2, -1, -1, -1), timestep.expand(2), **model_kwargs)
            latent = self.scheduler.step(epsilon[:1, :4], timestep, latent, return_dict=False)[0]
        return latent


def image_path(folder, method, example):
    return folder / method / f"seed-{example['seed']}-class-{example['class_id']}.png"


def comparison_grid(folder, examples, rows, filename, columns=8):
    from PIL import Image, ImageDraw

    tile, side, header = 128, 115, 34
    for offset in range(0, len(examples), columns):
        page = examples[offset:offset + columns]
        grid = Image.new("RGB", (side + len(page) * tile, header + len(rows) * tile), "white")
        draw = ImageDraw.Draw(grid)
        for column, example in enumerate(page):
            draw.text((side + column * tile + 4, 3),
                      f"seed {example['seed']}\nclass {example['class_id']}", fill="black")
        for row, (label, directory, method) in enumerate(rows):
            top = header + row * tile
            draw.text((5, top + 5), label, fill="black")
            for column, example in enumerate(page):
                path = image_path(directory, method, example)
                with Image.open(path) as picture:
                    grid.paste(picture.resize((tile, tile), Image.Resampling.LANCZOS),
                               (side + column * tile, top))
        name = filename if offset == 0 else f"{filename}-{offset // columns + 1}"
        grid.save(folder / f"{name}.png")


class Experiment:
    def __init__(self, config, checkpoint, small_checkpoint):
        import torch
        from .evaluate import load_vae

        _, self.diffusion = imports()
        from models import DiT_models

        model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights.get("ema", weights), strict=True)
        del weights
        model = model.eval().requires_grad_(False).cuda()
        vae = load_vae().cuda()
        self.xl = FullComputeDDIM(model, vae, config["cfg"], config["precision"])
        self.small = None
        if small_checkpoint:
            small = load_dit(small_checkpoint, device="cuda")
            self.small = FullComputeDDIM(small, vae, config["cfg"], config["precision"])

    def schedule(self, method, steps):
        if method == "dpm":
            return SolverPP(self.diffusion("").betas, steps)
        diffusion = self.diffusion(str(steps))
        if method.startswith("stitch"):
            return StitchedDDIM(diffusion, self.small.predict, small_calls(method, steps))
        return diffusion

    def generate(self, schedule, example, budget):
        budget.check()
        if isinstance(schedule, SolverPP):
            schedule.reset()
        return self.xl.generate(schedule, **example)

    def measure(self, folder, examples, candidates, budget, target, warmup):
        from torchvision.transforms.functional import to_pil_image

        schedules = {key: self.schedule(*choice) for key, choice in candidates.items()}
        trials = {key: [] for key in candidates}
        for key, schedule in schedules.items():
            (folder / key).mkdir(parents=True, exist_ok=True)
            for index in range(warmup):
                self.generate(schedule, dict(seed=810000 + index, class_id=1), budget)
        rng = random.Random(6260)
        for index, example in enumerate(examples, 1):
            order = list(candidates)
            rng.shuffle(order)
            for key in order:
                picture, timing = self.generate(schedules[key], example, budget)
                path = image_path(folder, key, example)
                temporary = path.with_suffix(".tmp")
                to_pil_image(picture[0].cpu()).save(temporary, format="PNG")
                temporary.replace(path)
                trials[key].append(dict(timing, **example))
            if index == len(examples) or index % 8 == 0:
                print(f"{folder.name}: {index}/{len(examples)} pairs", flush=True)
        results = {}
        for key, (method, steps) in candidates.items():
            small = small_calls(method, steps)
            results[key] = dict(method=method, steps=steps, s_evals=small,
                                xl_evals=steps - small, **summarize(trials[key], target))
            print(f"{key}: {results[key]['median_seconds']:.3f}s median", flush=True)
        write_json(folder / "timings.json", results)
        return results



def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference", "pilot-xl", "pilot-stitch", "final", "metrics", "report"))
    parser.add_argument("--baseline", default="artifacts/full-compute-1s")
    parser.add_argument("--output", default="artifacts/latency-quality-1s")
    parser.add_argument("--small-checkpoint", default="artifacts/checkpoints/dit_s_256.pt")
    parser.add_argument("--dpm-steps", type=int, nargs="+", help="Pilot DPM-Solver++ candidates")
    parser.add_argument("--stitch-steps", type=int, nargs="+", help="Pilot candidates for both switch points")
    parser.add_argument("--ddim-final", type=int, help="Defaults to the original one-second DDIM step count")
    parser.add_argument("--dpm-final", type=int, help="Pilot-selected DPM-Solver++ step count")
    parser.add_argument("--stitch-final", help="Pilot-selected configuration, e.g. stitch20-24")
    parser.add_argument("--ratings", type=Path, nargs=2, metavar=("RATER1", "RATER2"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--hours", type=float, default=1)
    args = parser.parse_args()
    if args.warmup < 1 or not 0 < args.hours <= 8:
        parser.error("Use positive warmup and hours, up to 8 hours")
    for steps in (args.dpm_steps, args.stitch_steps):
        if steps and any(not 5 <= value <= 100 for value in steps):
            parser.error("Pilot candidates must be between 5 and 100 steps")
    return args


def baseline_settings(folder):
    config = read_json(folder / "config.json")
    timing = read_json(folder / "timings.json")
    expected = dict(model="xl", resolution=256, batch_size=1, sampler="DDIM", eta=0,
                    vae="stabilityai/sd-vae-ft-ema", vae_precision="fp32", dit_revision=DIT_REVISION)
    if any(config.get(key) != value for key, value in expected.items()):
        raise SystemExit("The baseline settings differ from the original 256px DiT-XL DDIM setup")
    if not timing["largest_tested_steps_within_target"] or timing["target_seconds"] != 1.0:
        raise SystemExit("The baseline must have a DDIM configuration measured within one second")
    return config, timing["largest_tested_steps_within_target"]


def pilot_xl(args, experiment, output, pilot, baseline_steps, budget):
    folder = output / "pilot-xl"
    exact = {tag(method, baseline_steps): (method, baseline_steps) for method in ("ddim", "dpm")}
    experiment.measure(folder, pilot, exact, budget, 1.0, args.warmup)
    comparison_grid(folder, pilot, [(key, folder, key) for key in exact], "equal-calls")
    steps = args.dpm_steps or [baseline_steps + delta for delta in (-4, -2, 0, 2, 4, 6)]
    dpm = {tag("dpm", step): ("dpm", step) for step in sorted(set(steps)) if step != baseline_steps and step >= 5}
    if dpm:
        extra = output / "pilot-dpm"
        experiment.measure(extra, pilot, dpm, budget, 1.0, args.warmup)
        comparison_grid(extra, pilot, [(key, folder, key) for key in exact] +
                        [(key, extra, key) for key in dpm], "budget-candidates")


def pilot_stitch(args, experiment, output, pilot, baseline_steps, budget):
    folder = output / "pilot-stitch"
    steps = args.stitch_steps or [baseline_steps + delta for delta in (0, 2, 4, 6, 8, 12)]
    choices = {tag(method, step): (method, step) for method in ("stitch20", "stitch40")
               for step in sorted(set(steps)) if step >= 5}
    experiment.measure(folder, pilot, choices, budget, 1.0, args.warmup)
    for method in ("stitch20", "stitch40"):
        keys = [key for key in choices if key.startswith(method + "-")]
        dd = tag("ddim", baseline_steps)
        comparison_grid(folder, pilot, [(dd, output / "pilot-xl", dd)] +
                        [(key, folder, key) for key in keys], method)


def shortlisted(output, args, baseline_steps):
    if args.dpm_final is None or args.stitch_final is None:
        raise SystemExit("Choose --dpm-final and --stitch-final from the pilot images and timings")
    wanted = (tag("ddim", args.ddim_final or baseline_steps), tag("dpm", args.dpm_final), args.stitch_final)
    rows = {}
    for name in ("pilot-xl", "pilot-dpm", "pilot-stitch"):
        path = output / name / "timings.json"
        if path.is_file():
            rows.update(read_json(path))
    for key in wanted:
        if key not in rows or not rows[key]["within_target"]:
            raise SystemExit(f"{key} must have a pilot median within one second")
    stitch = rows[args.stitch_final]
    if not stitch["method"].startswith("stitch"):
        raise SystemExit("--stitch-final must name a stitch20 or stitch40 configuration")
    return {REFERENCE: ("ddim", 50), **{key: (rows[key]["method"], rows[key]["steps"]) for key in wanted}}


def main():
    args = arguments()
    baseline, output = local_path(args.baseline), local_path(args.output)
    if args.stage == "report":
        if args.ratings and args.ratings[0].resolve() == args.ratings[1].resolve():
            raise SystemExit("Provide a separate ratings CSV for each reviewer")
        report(output / "final", args.ratings)
        return
    if args.stage == "metrics":
        folder = output / "final"
        pairs = read_json(output / "pairs.json")["final"]
        methods = [key for key in read_json(folder / "timings.json") if key != REFERENCE]
        with GPUBudget("latency_quality_metrics", args.hours) as budget:
            score_lpips(folder, pairs, methods, budget)
        report(folder, None)
        return

    import torch

    config, baseline_steps = baseline_settings(baseline)
    checkpoint = local_path(config["checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != config["checkpoint_sha256"]:
        raise SystemExit("The original DiT-XL checkpoint is missing or its hash differs")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != config["gpu"]:
        raise SystemExit(f"Use the original GPU type ({config['gpu']}) for comparable latency")
    torch.backends.cuda.matmul.allow_tf32 = config["tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = config["tf32_cudnn"]
    small_path = None
    if args.stage in ("pilot-stitch", "final"):
        small_path = local_path(args.small_checkpoint)
        if not small_path.is_file():
            raise SystemExit(f"DiT-S checkpoint missing: {small_path}")
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "settings.json"
    settings = dict(baseline=str(baseline), baseline_steps=baseline_steps, **{
        key: config[key] for key in ("checkpoint_sha256", "dit_revision", "cfg", "precision", "tf32_matmul",
                                     "tf32_cudnn", "gpu", "vae", "resolution", "batch_size")})
    settings.update(torch=torch.__version__, cuda=torch.version.cuda)
    if settings_path.exists() and settings != read_json(settings_path):
        raise SystemExit("Output folder contains a different experiment configuration")
    write_json(settings_path, settings)
    if small_path:
        small_hash = sha256(small_path)
        stitch_settings = output / "stitch-settings.json"
        if stitch_settings.exists() and read_json(stitch_settings)["checkpoint_sha256"] != small_hash:
            raise SystemExit("This DiT-S checkpoint differs from the one already used in this experiment")
        write_json(stitch_settings, dict(checkpoint=str(small_path), checkpoint_sha256=small_hash))
    if args.stage != "reference" and not (output / "reference" / "reference-vs-existing.png").is_file():
        raise SystemExit("Run the reference stage first and inspect reference-vs-existing.png")
    if args.stage == "pilot-stitch" and not (output / "pilot-xl" / "timings.json").is_file():
        raise SystemExit("Run pilot-xl before pilot-stitch")
    original = config["examples"]
    pilot = stage_pairs(original, 16, 880000, 2201)
    final = stage_pairs(original + pilot, 64, 990000, 2202)
    write_json(output / "pairs.json", dict(reference=original, pilot=pilot, final=final))
    choices = shortlisted(output, args, baseline_steps) if args.stage == "final" else None
    if choices:
        folder = output / "final"
        folder.mkdir(exist_ok=True)
        frozen = folder / "frozen-configurations.json"
        frozen_choices = {key: list(choice) for key, choice in choices.items()}
        if frozen.exists() and read_json(frozen) != frozen_choices:
            raise SystemExit("This output folder already has frozen final configurations")
        write_json(frozen, frozen_choices)
    with GPUBudget("latency_quality_" + args.stage, args.hours) as budget, torch.inference_mode():
        experiment = Experiment(config, checkpoint, small_path)
        if args.stage == "reference":
            folder = output / "reference"
            experiment.measure(folder, original, {REFERENCE: ("ddim", 50)}, budget, 1.0, args.warmup)
            comparison_grid(folder, original, [(f"Existing {baseline_steps} steps", baseline, f"steps-{baseline_steps}"),
                            ("DDIM 50 steps", folder, REFERENCE)], "reference-vs-existing")
        elif args.stage == "pilot-xl":
            pilot_xl(args, experiment, output, pilot, baseline_steps, budget)
        elif args.stage == "pilot-stitch":
            pilot_stitch(args, experiment, output, pilot, baseline_steps, budget)
        else:
            experiment.measure(folder, final, choices, budget, 1.0, args.warmup)
            prepare_blind(folder, final, list(choices)[1:])
            report(folder, None)
            score_lpips(folder, final, list(choices)[1:], budget)
            report(folder, None)
    print(f"Saved {args.stage} outputs under {output}", flush=True)



def ballot_columns(positions):
    duels = [left + right for left, right in combinations(positions, 2)]
    fields = (["pair"] + [f"{letter}_{item}" for letter in positions
                          for item in ("recognizable", "artifacts")] + duels)
    return duels, fields


def prepare_blind(folder, examples, methods, *, archive=True, link_images=False):
    from torchvision.models._meta import _IMAGENET_CATEGORIES

    blind = folder / "blind"
    pictures = blind / "images"
    pictures.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11804)
    order = examples.copy()
    rng.shuffle(order)
    key = {}
    cards = []
    positions = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(methods)]
    duels, fields = ballot_columns(positions)
    for index, example in enumerate(order, 1):
        pair = f"{index:02d}"
        shuffled = rng.sample(methods, len(methods))
        key[pair] = dict(zip(positions, shuffled))
        class_id = example["class_id"]
        label = html.escape(_IMAGENET_CATEGORIES[class_id])
        tiles = []
        for letter, method in key[pair].items():
            filename = f"{pair}-{letter}.png"
            source = image_path(folder, method, example)
            destination = pictures / filename
            if link_images:
                temporary = destination.with_suffix(".tmp")
                temporary.unlink(missing_ok=True)
                os.link(source, temporary)
                temporary.replace(destination)
            else:
                shutil.copyfile(source, destination)
            tiles.append(f'<figure><img src="images/{filename}" width="256" height="256" '
                         f'alt="Image {letter}" loading="lazy"><figcaption>{letter}</figcaption></figure>')
        cards.append(f'<section><h2>Pair {pair} · {label} ({class_id})</h2>'
                     f'<div class="images">{"".join(tiles)}</div></section>')
    write_json(folder / "blind-key.json", key)
    (blind / "index.html").write_text("""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiT image review</title><style>
body{font:16px system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 16px}
section{margin-top:32px;border-top:1px solid #bbb}.images{display:flex;gap:12px;overflow:auto}
figure{margin:0;text-align:center}img{display:block}figcaption{font-weight:bold;margin-top:5px}
</style><h1>DiT image review</h1>
<p>Each reviewer fills out their own copy of <a href="ratings-template.csv">the ratings sheet</a>.
For each image, score class recognizability and obvious artifacts separately. Use 1 for yes and 0 for no.
For each comparison {duels}, record the preferred letter or enter tie.</p>
""".replace("{duels}", ", ".join(duels)) + "\n".join(cards) + "</html>\n", encoding="utf-8")
    with (blind / "ratings-template.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"pair": pair} for pair in key)
    if archive:
        shutil.make_archive(str(folder / "blind-review"), "zip", root_dir=blind)


def score_lpips(folder, examples, methods, budget, *, reference_method=REFERENCE):
    import lpips
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    metric = lpips.LPIPS(net="alex", verbose=False).eval().cuda()

    def pixels(path):
        with Image.open(path) as image:
            return to_tensor(image.convert("RGB"))[None].cuda() * 2 - 1

    scores = {method: [] for method in methods}
    with torch.inference_mode():
        for example in examples:
            budget.check()
            reference = pixels(image_path(folder, reference_method, example))
            for method in methods:
                generated = pixels(image_path(folder, method, example))
                distance = float(metric(generated, reference).item())
                scores[method].append(dict(**example, lpips=distance))
    write_json(folder / "lpips.json", scores)


def blind_scores(key, ratings):
    first_pair = next(iter(key.values()))
    duels, fields = ballot_columns(sorted(first_pair))
    totals = {method: dict(recognizable=[], artifacts=[], wins=0, ties=0, losses=0)
              for method in first_pair.values()}
    for path in ratings:
        seen = set()
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or ()) != set(fields):
                raise ValueError(f"{path}: use the columns from ratings-template.csv")
            for row in reader:
                pair = row["pair"].strip().zfill(2)
                if pair not in key or pair in seen:
                    raise ValueError(f"{path}: unknown or duplicate pair {pair}")
                seen.add(pair)
                methods = key[pair]
                for letter, method in methods.items():
                    for field in ("recognizable", "artifacts"):
                        value = (row[f"{letter}_{field}"] or "").strip()
                        if value not in ("0", "1"):
                            raise ValueError(f"{path}: pair {pair}, {letter}_{field} must be 0 or 1")
                        totals[method][field].append(int(value))
                for duel in duels:
                    left, right = duel
                    vote = (row[duel] or "").strip().upper()
                    if vote == "TIE":
                        totals[methods[left]]["ties"] += 1
                        totals[methods[right]]["ties"] += 1
                    elif vote in (left, right):
                        loser = right if vote == left else left
                        totals[methods[vote]]["wins"] += 1
                        totals[methods[loser]]["losses"] += 1
                    else:
                        raise ValueError(f"{path}: pair {pair}, {duel} must be {left}, {right}, or tie")
        if seen != set(key):
            raise ValueError(f"{path}: expected ratings for all {len(key)} pairs")
    return totals


def report(folder, ratings):
    timing = read_json(folder / "timings.json")
    lpips_path = folder / "lpips.json"
    distances = read_json(lpips_path) if lpips_path.is_file() else {}
    votes = blind_scores(read_json(folder / "blind-key.json"), ratings) if ratings else {}
    rows = []
    show_tokens = any("mean_tokens_per_call" in row for row in timing.values())
    for key, result in timing.items():
        human = votes.get(key)
        rows.append(dict(configuration=key, median_seconds=round(result["median_seconds"], 4),
                         median_denoise_seconds=round(result["median_denoise_seconds"], 4),
                         median_decode_seconds=round(result["median_decode_seconds"], 4),
                         within_one_second=round(result["fraction_within_target"], 4),
                         s_evals=result["s_evals"], xl_evals=result["xl_evals"],
                         class_recognizability=round(statistics.mean(human["recognizable"]), 4) if human else None,
                         artifact_rate=round(statistics.mean(human["artifacts"]), 4) if human else None,
                         blind_wins=human["wins"] if human else None, blind_ties=human["ties"] if human else None,
                         blind_losses=human["losses"] if human else None,
                         mean_lpips=round(statistics.mean(item["lpips"] for item in distances[key]), 4)
                         if key in distances else None))
        if show_tokens:
            rows[-1]["mean_tokens_per_call"] = round(result.get("mean_tokens_per_call", 256), 1)
    with (folder / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ["| Configuration | Median s | S evals | XL evals | Recognizable | Artifacts | Blind W/T/L | Mean LPIPS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    if show_tokens:
        table[0] += " Mean tokens/call |"
        table[1] += "---:|"
    for row in rows:
        win = row["blind_wins"]
        preference = f"{win}/{row['blind_ties']}/{row['blind_losses']}" if win is not None else "—"
        recognized = row["class_recognizability"]
        artifact_rate = row["artifact_rate"]
        lpips_mean = row["mean_lpips"]
        table.append(f"| {row['configuration']} | {row['median_seconds']:.3f} | {row['s_evals']} | "
                     f"{row['xl_evals']} | {f'{recognized:.1%}' if recognized is not None else '—'} | "
                     f"{f'{artifact_rate:.1%}' if artifact_rate is not None else '—'} | {preference} | "
                     f"{f'{lpips_mean:.3f}' if lpips_mean is not None else '—'} |")
        if show_tokens:
            table[-1] += f" {row['mean_tokens_per_call']:.1f} |"
    notes = ["Median latency includes VAE decoding. Blind preferences are pairwise votes pooled across both reviewers.",
             "The 50-step reference provides the LPIPS target and is excluded from the blind ballot."]
    if show_tokens:
        notes.append("Adaptive sizing uses the original pretrained XL weights with initialized coarse patch layers. "
                     "The other rows reuse the previous run on the same GPU type and seed/class pairs.")
    if not ratings:
        notes.append("Human ratings are pending. Both reviewers can use blind/index.html and their own copy of the ratings template.")
    (folder / "results.md").write_text("\n".join(table) + "\n\n" + "\n\n".join(notes) + "\n")
    print(f"Saved {folder / 'results.csv'} and results.md", flush=True)


if __name__ == "__main__":
    main()

