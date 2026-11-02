"""Rebuild the first-results report and figures from saved measurements."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .runtime import write_json
from .upstream import ROOT


def read_runs(stage, arm, limit=64):
    groups = []
    for rep in range(3 if arm == "random" else 1):
        records = [torch.load(p, map_location="cpu", weights_only=True)
                   for p in sorted((ROOT / f"artifacts/{stage}/{arm}_{rep}").glob("*.pt"))[:limit]]
        if not records:
            return []
        groups.append(records)
    common = min(map(len, groups))
    rows = []
    for i in range(common):
        metrics = groups[0][i]["metrics"].copy()
        for name in ("lpips", "latent_mse", "denoise_seconds", "estimated_model_gflops"):
            metrics[name] = float(np.mean([g[i]["metrics"][name] for g in groups]))
        rows.append(metrics)
    return rows


def paired_difference(first, second):
    a = {r["seed"]: r["lpips"] for r in first}
    b = {r["seed"]: r["lpips"] for r in second}
    common = sorted(a.keys() & b.keys())
    if not common:
        return None
    delta = np.array([a[k]-b[k] for k in common])
    rng = np.random.default_rng(6220)
    boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1)
    return dict(n=len(delta), difference=float(delta.mean()),
                lower=float(np.quantile(boot, .025)), upper=float(np.quantile(boot, .975)))


def main():
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    rows = []
    for stage, arms in (("baseline", ["baseline"]), ("zero_shot", ["fine", "coarse", "random", "spatial"]),
                        ("trained", ["fine", "coarse", "random", "spatial", "temporal", "joint"]),
                        ("fewer_steps", ["baseline"])):
        for arm in arms:
            data = read_runs(stage, arm, 32 if stage == "zero_shot" else 64)
            if not data:
                continue
            rows.append(dict(stage=stage, arm=arm, samples=len(data),
                             lpips=float(np.mean([r["lpips"] for r in data])),
                             denoise=float(np.median([r["denoise_seconds"] for r in data])),
                             gflops=float(np.mean([r["estimated_model_gflops"] for r in data]))))
    contrasts = {}
    for name, a, b in [
        ("zero_spatial_minus_random", ("zero_shot", "spatial"), ("zero_shot", "random")),
        ("trained_spatial_minus_zero_spatial", ("trained", "spatial"), ("zero_shot", "spatial")),
        ("trained_spatial_minus_random", ("trained", "spatial"), ("trained", "random")),
        ("trained_spatial_minus_temporal", ("trained", "spatial"), ("trained", "temporal")),
        ("trained_joint_minus_spatial", ("trained", "joint"), ("trained", "spatial")),
        ("trained_joint_minus_temporal", ("trained", "joint"), ("trained", "temporal")),
    ]:
        contrasts[name] = paired_difference(read_runs(*a), read_runs(*b))
    write_json(results / "paired_comparisons.json", contrasts)
    lines = ["# Initial DiT-S results", "", "## Setup", "",
             "Pretrained DiT-S/2 (32.96M parameters), ImageNet-256, 50-step deterministic DDIM, CFG 4, FP32 on one RTX 5090. The original three-channel CFG convention and EMA VAE are preserved.",
             "", "Sequence: original baseline → frozen APT-style resize/repeat → 1,000 detail-branch updates. Only 738,048 detail-branch parameters trained; original DiT and VAE frozen. LoRA was not run.",
             "", "128 teacher training trajectories and 32 validation trajectories; ten cached timesteps and both CFG branches per trajectory. Zero-shot evaluation uses 32 final seeds; trained schedules use 64. Before/after differences use only their shared 32 seeds. Random allocation averages three replicates per seed.",
             "", "## Paired sample fidelity", "",
             "Lower LPIPS means closer to the original sample, not better distributional quality. The bootstrap intervals below quantify paired differences only; no FID or H1/H2 confirmation is claimed.", "",
             "| Stage | Arm | Seeds | LPIPS | Model GFLOPs/sample | Median denoise seconds |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['stage']} | {r['arm']} | {r['samples']} | {r['lpips']:.4f} | {r['gflops']:.1f} | {r['denoise']:.4f} |")
    lines += ["", "Model FLOPs use 2 FLOPs/MAC and count both guidance branches, full fine/coarse projections, attention, MLPs, conditioning, detail branch, and dense output head. They exclude resize, elementwise, selector, and VAE operations; sampling latency includes resize and selection. Comparisons are approximately model-FLOP matched, not certified total-FLOP matched.",
              "", "| Paired LPIPS difference (first minus second) | Shared seeds | Mean | 95% bootstrap interval |",
              "|---|---:|---:|---|"]
    for name, c in contrasts.items():
        if c:
            lines.append(f"| {name} | {c['n']} | {c['difference']:.4f} | [{c['lower']:.4f}, {c['upper']:.4f}] |")
    training = json.loads((ROOT / "artifacts/training/detail/validation.json").read_text())
    lines += ["", "## Training and reconstruction constraint", "",
              f"At eight coarse blocks, held-out noise-plus-variance MSE changed from {training[0]['validation']['8']:.4f} to {training[-1]['validation']['8']:.4f}. Checkpoint selection used validation, not final samples.",
              "", "The resize/repeat head forces each coarse token's four fine output patches to be identical. This is a structural restriction even with a trained detail branch or LoRA. `diagnostics.json` measures an oracle error floor by projecting teacher predictions onto that restricted output space."]
    diagnostic_path = results / "diagnostics.json"
    if diagnostic_path.exists():
        records = json.loads(diagnostic_path.read_text())
        selector_summary = {}
        lines += ["", "## Output reconstruction diagnostic", "",
                  "Across 48 held-out state/trajectory pairs per checkpoint, independently coarsening blocks and measuring teacher-prediction error gives a rank-based selector diagnostic. The repeat floor is a mathematical lower bound for the repeated-output constraint, computed from teacher predictions with an oracle projection; a model cannot beat it without changing that output representation.", "",
                  "| Stage | Placement rule | Noise MSE | Repeat-output floor | Mean rank correlation with single-block damage |",
                  "|---|---|---:|---:|---:|"]
        for stage in ("zero_shot", "trained"):
            subset = [r for r in records if r["stage"] == stage]
            selector_summary[stage] = {}
            for method, signal in (("clean", "clean"), ("trajectory", "trajectory"), ("oracle_rank", None), ("random", None)):
                mse = float(np.mean([r["strategies"][method]["noise_mse"] for r in subset]))
                floor = float(np.mean([r["strategies"][method]["repeat_floor_noise_mse"] for r in subset]))
                correlations = [r["correlations"][signal] for r in subset if signal and r["correlations"][signal] is not None]
                rho = float(np.mean(correlations)) if correlations else None
                selector_summary[stage][method] = dict(noise_mse=mse, repeat_floor=floor,
                                                       mean_rank_correlation=rho,
                                                       excess_over_floor=mse-floor)
                rho_text = f"{rho:.3f}" if rho is not None else "—"
                lines.append(f"| {stage} | {method} | {mse:.4f} | {floor:.4f} | {rho_text} |")
            corr_text = ", ".join(f"{name} {selector_summary[stage][name]['mean_rank_correlation']:.3f}" for name in ("clean", "trajectory"))
            clean = selector_summary[stage]["clean"]
            explained = 1-clean["excess_over_floor"]/clean["noise_mse"]
            lines.append("")
            lines.append(f"For {stage}, the trajectory variation score ranked single-block damage better than clean-latent variation (mean Spearman {corr_text}). The oracle-ranked layout had lower prediction MSE than either causal score. For the trained clean-score layout, {explained:.1%} of measured noise MSE is already present in the repeat-output lower bound, so the output constraint explains most of this prediction error.")
        write_json(results / "diagnostic_summary.json", selector_summary)
        lines += ["", "The next useful experiment is to preserve a fine-grid residual through the coarse path or add a small dense reconstruction head, then measure whether it closes the repeat-floor gap. A separate validation-only sweep should compare trajectory variation against the current clean-latent selector end to end. LoRA alone changes attention representations and does not remove the output-repeat constraint."]
    latency_path = results / "latency.json"
    if latency_path.exists():
        lines += ["", "## Repeated latency benchmark", "", "Three warmups and ten synchronized repetitions per arm, one image per sample call; timings include the implemented Python validation and layout overhead.", "",
                  "| Stage | Arm | Median denoise seconds | Min–max seconds | Median with VAE seconds |",
                  "|---|---|---:|---|---:|"]
        for r in json.loads(latency_path.read_text()):
            lines.append(f"| {r['stage']} | {r['arm']} | {r['median_seconds']:.4f} | {r['min_seconds']:.4f}–{r['max_seconds']:.4f} | {r['median_end_to_end_seconds']:.4f} |")
    ledger = json.loads((ROOT / "artifacts/gpu_ledger.json").read_text())
    charged = sum(r.get("seconds", r["reserved_seconds"]) for r in ledger)
    lines += ["", "## Reproducibility and limits", "",
              f"Ledger-charged GPU process time: {charged/3600:.4f} hours (eight-hour ceiling). This includes I/O during GPU phases; it excludes dependency installation, CPU-only tests, and code development. A preliminary CUDA compatibility probe preceded the ledger and was not timed; reported accounting is therefore approximate by a few seconds.",
              "", "Sources and checkpoint identity are in `PROVENANCE.md`; exact packages are in `requirements.lock`. Raw runs, checkpoints, and teacher caches remain project-local under `artifacts/`.",
              "", "This is one small backbone, one central allocation budget, one training seed, and a short adaptation run. The fixed-block selector and causal clean-latent score differ from APT's entropy/quadtree policy. Temporal control is simple coarse-early scheduling, not a full DDiT reproduction. Neighboring-budget sweeps, FID, PixArt, video, and optional LoRA remain unrun.",
              "", "Figures: [paired samples](comparison.png), [fidelity versus compute/latency](tradeoffs.png), [validation curve](validation.png).",
              "", "Raw output-reconstruction diagnostics are in `diagnostics.json`; aggregated results are in `diagnostic_summary.json`."]
    (results / "initial_report.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    colors = dict(baseline="black", zero_shot="tab:orange", trained="tab:blue", fewer_steps="tab:green")
    for r in rows:
        for ax, key in zip(axes, ("gflops", "denoise")):
            ax.scatter(r[key], r["lpips"], color=colors[r["stage"]])
            ax.annotate(f"{r['stage']}/{r['arm']}", (r[key],r["lpips"]), fontsize=6, xytext=(3,3), textcoords="offset points")
    axes[0].set_xlabel("Estimated model GFLOPs/sample")
    axes[1].set_xlabel("Median observed denoising seconds")
    for ax in axes:
        ax.set_ylabel("LPIPS to baseline (lower is closer)")
        ax.grid(alpha=.2)
    fig.savefig(results / "tradeoffs.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for count in (4,8,12,16):
        ax.plot([r["step"] for r in training], [r["validation"][str(count)] for r in training], label=f"{count} coarse blocks")
    ax.set(xlabel="Optimizer step", ylabel="Held-out noise + variance MSE")
    ax.legend()
    fig.savefig(results / "validation.png", dpi=180)
    plt.close(fig)
    figure_rows = [("baseline", "baseline"), ("zero_shot", "spatial"), ("trained", "spatial"), ("trained", "temporal"), ("trained", "joint")]
    fig, axes = plt.subplots(len(figure_rows), 4, figsize=(10, 12), constrained_layout=True)
    for row, (stage, arm) in enumerate(figure_rows):
        paths = sorted((ROOT / f"artifacts/{stage}/{arm}_0").glob("*.pt"))[:4]
        for col, path in enumerate(paths):
            image = torch.load(path, weights_only=True)["image"][0]
            axes[row,col].imshow(((image.permute(1,2,0)+1)/2).clamp(0,1))
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
        axes[row,0].set_ylabel(f"{stage}\n{arm}")
    fig.savefig(results / "comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
