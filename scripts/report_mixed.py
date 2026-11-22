"""Build results/mixed/<tag>_report.md and figures from artifacts/mixed_eval/<tag>/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tag = sys.argv[1] if len(sys.argv) > 1 else "lora32"
src = ROOT / "artifacts/mixed_eval" / tag
dst = ROOT / "results/mixed"
dst.mkdir(parents=True, exist_ok=True)

sweep = json.loads((src / "sweep.json").read_text())
flops = {int(k): v for k, v in json.loads((src / "flops.json").read_text()).items()} if (src / "flops.json").exists() else {}

if flops:
    ns = np.array([n for n in flops if n != 256], float)
    fs = np.array([flops[int(n)] for n in ns], float)
    coef = np.linalg.lstsq(np.stack([np.ones_like(ns), ns, ns ** 2], 1), fs, rcond=None)[0]
    resid = np.abs(np.stack([np.ones_like(ns), ns, ns ** 2], 1) @ coef - fs).max() / fs.max()

    def step_flops(mean_n, mean_n2):
        if mean_n >= 255.999:
            return flops[256]
        return coef[0] + coef[1] * mean_n + coef[2] * mean_n2

for name, r in sweep.items():
    if flops:
        r["gflops"] = sum(step_flops(n, n2) for n, n2 in zip(r["tokens_per_step"], r["tokens_sq_per_step"])) / 1e9
    else:
        r["gflops"] = float("nan")

t50 = sweep["teacher50"]
spread = [abs(sweep[k]["fid_imagenet"] - t50["fid_imagenet"]) for k in sweep if k.startswith("teacher50_seedset")]
delta = max(spread) if spread else float("nan")

FAMILIES = [
    ("teacher (fewer DDIM steps)", lambda k: k.startswith("teacher") and "seedset" not in k, "k", "o"),
    ("uniform p (global)", lambda k: k.startswith("uniform"), "tab:gray", "s"),
    ("fixed switch p4->p2", lambda k: k.startswith("switch4"), "tab:brown", "v"),
    ("temporal (DDiT Δ³)", lambda k: k.startswith("temporal"), "tab:blue", "^"),
    ("spatial (fixed mask)", lambda k: k.startswith("spatial_T"), "tab:green", "D"),
    ("spatial random", lambda k: k.startswith("spatial_random"), "tab:olive", "d"),
    ("joint (Eq. 3)", lambda k: k.startswith("joint_T"), "tab:red", "*"),
    ("joint random", lambda k: k.startswith("joint_random"), "tab:pink", "P"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for label, pred, color, marker in FAMILIES:
    rows = sorted([r for k, r in sweep.items() if pred(k)], key=lambda r: r["gflops"])
    if not rows:
        continue
    x = [r["gflops"] for r in rows]
    axes[0].plot(x, [r["fid_imagenet"] for r in rows], marker=marker, color=color, label=label)
    axes[1].plot(x, [r["fid_teacher"] for r in rows], marker=marker, color=color, label=label)
    lp = [(r["gflops"], r["lpips"], r["lpips_ci"]) for r in rows if "lpips" in r]
    if lp:
        axes[2].errorbar([a for a, _, _ in lp], [b for _, b, _ in lp],
                         yerr=np.array([[b - c[0], c[1] - b] for _, b, c in lp]).T,
                         marker=marker, color=color, label=label, capsize=2)
axes[0].axhspan(t50["fid_imagenet"] - delta, t50["fid_imagenet"] + delta, color="k", alpha=.08, label="teacher ± δ")
axes[0].set_ylabel("FID-5K vs ImageNet ref ↓")
axes[1].set_ylabel("FID-5K vs teacher-50 samples ↓")
axes[2].set_ylabel("paired LPIPS vs teacher (1k, 95% CI) ↓")
for ax in axes:
    ax.set_xlabel("executed GFLOPs / image (both CFG branches)")
    ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(dst / f"{tag}_pareto.png", dpi=140)

lines = [f"# Shared mixed-patch DiT-S ({tag}): H1/H2 sweep", "",
         f"One shared model for every adapted arm. {t50['n']} fresh seeds/arm (FID-{t50['n'] // 1000}K), "
         f"first {max(r.get('n_lpips', 0) for r in sweep.values()):,} paired for LPIPS.",
         (f"Pre-registered matching tolerance δ = max |FID(teacher-50, seed set r) − FID(teacher-50)| = **{delta:.2f}** "
         f"(teacher FID {t50['fid_imagenet']:.2f})." if delta == delta else
          f"Teacher FID {t50['fid_imagenet']:.2f}; no seed-set spread measured in this sweep."), "",
         "| Arm | τ | Mean tokens/step | Executed GFLOPs/img | FID-5K (ImageNet) | FID vs teacher | LPIPS [95% CI] | Mask turnover |",
         "|---|---:|---:|---:|---:|---:|---|---:|"]
order = sorted(sweep.items(), key=lambda kv: (kv[0].split("_")[0], -kv[1]["gflops"]))
for name, r in order:
    lp = f"{r['lpips']:.4f} [{r['lpips_ci'][0]:.4f}, {r['lpips_ci'][1]:.4f}]" if "lpips" in r else "—"
    tau = f"{r['tau']:.2e}" if r.get("tau") else "—"
    ok = "✓" if r["fid_imagenet"] <= t50["fid_imagenet"] + delta else ""
    lines.append(f"| {name} | {tau} | {r['mean_tokens']:.1f} | {r['gflops']:.1f} | {r['fid_imagenet']:.2f} | "
                 f"{r['fid_teacher']:.2f} | {lp} | {r['turnover']:.3f} |")
if flops:
    lines += ["", f"Executed FLOPs: `torch.utils.flop_counter` on one CFG step (both branches) per token count, "
              f"including all candidate embeddings/heads, LoRA, attention; quadratic fit max rel. residual {resid:.1e}. "
              "Selector (Δ³ + window std) adds ~0.1 MFLOP/step, negligible."]
lines += ["", "Caution: at CFG 4, FID vs ImageNet is non-monotone in fidelity: several adapted arms score *below* the teacher "
          "(e.g. 18.1 vs 19.6) while being far from it (FID vs teacher ~10, LPIPS ~0.3). δ is therefore not a usable matched-quality "
          "criterion here; use FID vs teacher and paired LPIPS. FID vs teacher for adapted arms uses the same seeds/classes as the "
          "teacher reference, so it is not comparable to the teacher seed-set rows (independent seeds, ~11.4 finite-sample floor).",
          "", f"![pareto]({tag}_pareto.png)", ""]

# Paired differences for the H1/H2 contrasts at matched budgets.
lines += ["## Paired LPIPS contrasts (same seeds; 95% bootstrap CI of the mean difference)", "",
          "| Contrast | Δ LPIPS | 95% CI |", "|---|---:|---|"]
rng = np.random.default_rng(0)


def paired(a, b):
    pa, pb = src / f"lpips_{a}.npy", src / f"lpips_{b}.npy"
    if not (pa.exists() and pb.exists()):
        return None
    d = np.load(pa) - np.load(pb)
    m = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


targets = sorted({int(k.split("_T")[1]) for k in sweep if "_T" in k}, reverse=True)
for T in targets:
    for a, b in ((f"joint_T{T}", f"temporal_T{T}"), (f"spatial_T{T}", f"temporal_T{T}"),
                 (f"joint_T{T}", f"spatial_T{T}"), (f"spatial_T{T}", f"spatial_random_T{T}"),
                 (f"joint_T{T}", f"joint_random_T{T}")):
        res = paired(a, b)
        if res:
            lines.append(f"| {a} − {b} | {res[0]:+.4f} | [{res[1]:+.4f}, {res[2]:+.4f}] |")
lines.append("")
lines.append("Arms at the same target share the calibrated token budget only approximately; compare via the GFLOPs column / Pareto plot.")

gp = src / "gap.json"
if gp.exists():
    g = json.loads(gp.read_text())
    lines += ["", "## Allocation gap (RQ3)", "",
              "Per image, each of 16 blocks is forced to patch size p (others fine) after warmup; damage = LPIPS to the teacher image. "
              "Signals are averaged over the teacher's own scheduled steps. gap = (1 − Spearman)/2; 0 perfect, 0.5 uninformative.", "",
              "| p | Signal | Spearman [95% CI] | Gap |", "|---|---|---|---:|"]
    for p, res in g.items():
        for k in ("fd4", "fd8", "entropy", "variation", "random"):
            r = res[k]
            lines.append(f"| {p} | {k} | {r['spearman']:+.3f} [{r['spearman_ci'][0]:+.3f}, {r['spearman_ci'][1]:+.3f}] | {r['gap']:.3f} |")
        lines.append(f"| {p} | mean single-block damage | {res['mean_damage']:.4f} (within-image sd {res['damage_spread']:.4f}) | |")

ep = src / "efficiency.json"
if ep.exists():
    e = json.loads(ep.read_text())
    lines += ["", "## Measured latency (B200, full 50-step sampling incl. selector; CUDA-synchronized)", "",
              "| Config | dtype | batch | median s | images/s | speedup vs teacher-50 |", "|---|---|---:|---:|---:|---:|"]
    base = {(r["dtype"], r["batch"]): r["median_s"] for r in e["latency"] if r["config"] == "teacher50"}
    fwd = {}
    for r in e["latency"]:
        if r["config"].startswith("forward_"):
            fwd.setdefault((r["dtype"], r["batch"]), {})[r["config"]] = r["median_s"]
            continue
        sp = base[(r["dtype"], r["batch"])] / r["median_s"]
        lines.append(f"| {r['config']} | {r['dtype']} | {r['batch']} | {r['median_s']:.3f} | {r['images_per_s']:.1f} | {sp:.2f}× |")
    lines += ["", "Single CFG forward (trunk upper bound on per-step speedup):", "",
              "| dtype | batch | 256 tok | 160 tok | 64 tok | 16 tok |", "|---|---:|---:|---:|---:|---:|"]
    for (dt, bs), d in fwd.items():
        lines.append(f"| {dt} | {bs} | " + " | ".join(f"{d.get(f'forward_tokens{n}', float('nan')) * 1e3:.2f} ms" for n in (256, 160, 64, 16)) + " |")

(dst / f"{tag}_report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
