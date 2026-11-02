# Controlled DiT-S recovery: noise skip and longer training

2026-11-02. These are development experiments on the existing held-out validation trajectories, not fresh final-test results. All arms use 50-step deterministic DDIM, CFG 4 with upstream three-channel guidance, and all-patch-4 inference. Pure patch-2 behavior remains exact. Original artifacts are preserved.

## Observed outcome and decision

**None of the three arms passes the image-recognition gate.** Direct inspection of all 16 development images shows corrupted patterns rather than reliably recognizable class/content. C reduces saturation and has more broad color/spatial structure, but is not usable generation.

Longer corrected training materially improves epsilon fit: A falls from 0.6549 to 0.2661, about a 59% reduction. Removing variance loss lowers it further to 0.2334 at matched updates, but paired image LPIPS is not improved (0.8819 versus 0.8739). This loss improvement alone does not repair iterative generation.

The explicit skip package is substantially more promising numerically: epsilon MSE 0.0656, guided MSE 0.0751, and next-state MSE 0.000363. At t=800–999 its epsilon MSE is 0.00000671 versus identity baseline 0.00000776; it preserves the easy high-noise signal. Its largest observed final latent magnitude is 5.09 versus 133.05 for A. Development LPIPS improves to 0.7766, but this remains a failed quality gate.

Errors now concentrate at low/middle timesteps: C epsilon MSE is 0.2051 at t=0–199 and 0.1166 at t=200–399. At the final check, training epsilon MSE is 0.0648 versus development 0.0656. Both fit and development curves are still improving; these trials neither establish convergence nor prove that more training could never help. They do establish that 3,200 more updates are insufficient to recover usable all-patch-4 images.

**Next recommendation:** preserve fine local information in the dense output decoder and begin with conservative count-2/count-4 compression. That is a separate next experiment, not executed here. Keep the explicit skip as a candidate building block. Do not spend further runs on selector sweeps or presume that indefinite continuation will fix the low-noise reconstruction problem.

## Setup

A and B warm-start the same corrected 800-update checkpoint and reset AdamW. Each receives up to 3,200 additional updates with identical timestep-balanced paired sample draws, batch size four conditional/unconditional pairs, learning rate 1e-4, and frozen DiT weights. A minimizes all-eight-channel MSE; B minimizes epsilon MSE. C warm-starts the same trunk but zeros its correction outputs and predicts v with an explicit full-resolution noise skip. C changes initialization, parameterization, and objective together, so it is not a single-factor comparison with B.

`eps = sqrt(1-alpha_bar) * x + sqrt(alpha_bar) * v`; teacher targets use the exact inverse conversion. Learned variance remains an eight-channel interface component but is excluded from B/C training objectives. No mixed-layout or scheduler hypothesis is tested.

The isolated C optimization check reduced loss from 0.237403 to 0.001901 (124.9x) in 800 updates at learning rate 1e-3. Its weights are discarded before the controlled runs.

Checkpoint selection uses each arm’s own minimum development objective. Metrics below use all 320 held-out pairs (640 branches). Image LPIPS uses the same 16 development seeds/classes for each arm and a freshly generated teacher reference. The new 64-pair final-test manifest remains unused.

| Arm | Selected additional updates | Epsilon MSE | Guided epsilon MSE | DDIM next-state MSE | Development LPIPS (16) |
|---|---:|---:|---:|---:|---:|
| A: Longer corrected training | 3200 | 0.266060 | 0.301082 | 0.005910 | 0.8739 |
| B: Longer, epsilon-only | 3200 | 0.233438 | 0.269988 | 0.005246 | 0.8819 |
| C: Explicit noise skip + v correction | 3200 | 0.065617 | 0.075077 | 0.000363 | 0.7766 |

Image rows: teacher, A, B, C. Columns share class and seed. See [paired images](ddit_recovery_samples.png). LPIPS measures same-seed fidelity, not distributional quality.

| Timestep bin | Identity epsilon MSE | A epsilon MSE | B epsilon MSE | C epsilon MSE |
|---|---:|---:|---:|---:|
| 0-199 | 0.86580062 | 0.23149738 | 0.21724063 | 0.20506841 |
| 200-399 | 0.33249393 | 0.29520690 | 0.25876886 | 0.11662790 |
| 400-599 | 0.04555085 | 0.25535834 | 0.22007889 | 0.00615000 |
| 600-799 | 0.00169830 | 0.26355392 | 0.22668201 | 0.00023391 |
| 800-999 | 0.00000776 | 0.28468353 | 0.24441794 | 0.00000671 |

## Does longer training help?

- A: epsilon MSE 0.654887 → 0.266060; guided epsilon MSE 0.665694 → 0.301082.
- B: epsilon MSE 0.654887 → 0.233438; guided epsilon MSE 0.665694 → 0.269988.
- C: epsilon MSE 0.115719 → 0.065617; guided epsilon MSE 0.122017 → 0.075077.

This separates a measured improvement in prediction error from image recognition, which still requires inspection of the saved contact sheet. A/B isolate loss allocation with matched updates. C tests the proposed noise-preservation package. Improvement over initialization is not evidence of convergence or acceptable images.

## Validation and limitations

- CPU suite: 19 tests passed before training, including nonzero final-head conditioning, v conversion, and deterministic-DDIM variance independence.
- Complete pure fine CFG trajectories are checked against pinned upstream before training and for each selected checkpoint. Both raw predictions and final latent are required to be bit-exact.
- Full conditional, unconditional, guided, next-state, spectral, and timestep metrics are retained in `outcomes.json` and each arm’s `validation.json`. The next-state metric is same-state error, not full-rollout error.
- `best.pt` and `last.pt` include optimizer and RNG state, arm/version, source hashes, initialization, and objective. Warm-start histories are explicitly distinguished from exact optimizer resume.
- No cache expansion, final-test sampling, schedule sweeps, or FID run. Existing development data were reused and have been inspected; results are not an unbiased final estimate.
- No speed claim is made. The new architecture retains dense fine embeddings and residual work, and LoRA remains unmerged.

Artifacts: `artifacts/training/ddit_recovery_v1c/`. Reproduce with `source scripts/env.sh`, then `.venv/bin/python -m apt_dit.ddit_recovery --name <new-run> --steps 3200 --tiny-steps 1200 --hours .60`. This command creates a new immutable run directory.

[Learning curves](ddit_recovery_curves.png).

Recovery GPU-process time, including guarded preliminary checks: 200.8 seconds (0.0558 hours), charged in the existing ledger.
