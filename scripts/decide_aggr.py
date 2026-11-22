"""Apply the pre-registered Follow-up A decision rule (results/mixed/preregistration_followup.md)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "artifacts/mixed_eval/aggr"
sweep = json.loads((src / "sweep.json").read_text())
rng = np.random.default_rng(0)


def contrast(a, b):
    d = np.load(src / f"lpips_{a}.npy") - np.load(src / f"lpips_{b}.npy")
    m = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(d)


lines = ["# Follow-up A: pre-registered decision", "",
         "Rule at budget T: (i) paired LPIPS(X − Y) 95% CI upper < 0; (ii) FID vs teacher(X) < (Y); (iii) tokens(X) ≤ tokens(Y) + 2.",
         "Narrowed H1 supported iff joint beats temporal at ≥ 3 of {110, 90, 70, 50}.", ""]
verdicts = {}
for X, Y in (("joint", "temporal"), ("joint", "joint_random")):
    lines += [f"## {X} vs {Y}", "",
              f"| T | tokens {X} / {Y} | FID-T {X} / {Y} | ΔLPIPS [95% CI] (n) | (i) | (ii) | (iii) | beats |",
              "|---|---|---|---|:-:|:-:|:-:|:-:|"]
    wins = []
    for T in (130, 110, 90, 70, 50):
        a, b = sweep[f"{X}_T{T}"], sweep[f"{Y}_T{T}"]
        m, lo, hi, n = contrast(f"{X}_T{T}", f"{Y}_T{T}")
        c1, c2, c3 = hi < 0, a["fid_teacher"] < b["fid_teacher"], a["mean_tokens"] <= b["mean_tokens"] + 2
        ok = c1 and c2 and c3
        if T != 130:
            wins.append(ok)
        f = lambda v: "✓" if v else "✗"
        lines.append(f"| {T} | {a['mean_tokens']:.1f} / {b['mean_tokens']:.1f} | {a['fid_teacher']:.2f} / {b['fid_teacher']:.2f} | "
                     f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] ({n}) | {f(c1)} | {f(c2)} | {f(c3)} | {f(ok)} |")
    verdicts[f"{X}_vs_{Y}"] = sum(wins)
    lines += ["", f"Wins at budgets ≤ 110: **{sum(wins)}/4** → "
              + ("**SUPPORTED**" if sum(wins) >= 3 else "**NOT SUPPORTED**") + (" (primary test)" if Y == "temporal" else " (secondary)"), ""]
(ROOT / "results/mixed/aggr_decision.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
