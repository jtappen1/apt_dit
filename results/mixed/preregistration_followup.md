# Pre-registration: follow-up experiments A and B

Written 2026-11-21 ~22:00 EST, **before** any of these runs. Model: the evaluated LoRA-32 checkpoint
`artifacts/training/mixed_lora32/last.pt` (unchanged). Sampler: 50-step DDIM, eta 0, CFG 4, 3 fine warmup steps.
All results are reported regardless of outcome. No budgets, seeds, or metrics are added or dropped after seeing results;
any deviation is listed under "Deviations" at the end with its reason.

## A. Spatial adaptivity at aggressive compression (narrowed H1)

Motivation: in the main sweep, at ~100 tokens joint beat temporal (LPIPS −0.008, CI [−0.010, −0.007]; FID vs teacher
31.5 vs 33.4), and MOSAIK reports gains over DDiT only at aggressive budgets. Mechanism: at small budgets temporal
must coarsen the whole image even late; joint can keep some blocks fine.

- Seeds: 10,000 fresh seeds, offset 800000 (never used before). Classes uniform via `random.Random(800000)`.
- tau calibration: 30 log-spaced thresholds on 256 images at seed offset 510000 (disjoint from evaluation).
- Budgets (target mean tokens/step incl. warmup): 130, 110, 90, 70, 50.
- Arms per budget: temporal, joint, joint_random, spatial (budget-matched, fixed), spatial_random. Plus teacher-50 reference.
- Metrics: paired LPIPS vs teacher-50 on all 10,000 (95% paired bootstrap CI of the mean difference), FID-10K vs teacher-50
  samples, FID-10K vs ImageNet reference (reported, not used for decisions: non-monotone at CFG 4), executed GFLOPs.
- **Decision rule.** At budget T, "joint beats temporal" iff all three hold:
  (i) paired LPIPS(joint − temporal) 95% CI upper bound < 0;
  (ii) FID vs teacher(joint) < FID vs teacher(temporal);
  (iii) joint mean tokens ≤ temporal mean tokens + 2 (joint not given more compute).
  **Narrowed H1 is supported** iff "joint beats temporal" holds at ≥ 3 of the 4 budgets {110, 90, 70, 50}.
  Secondary: whether joint also beats joint_random under the same rule (placement vs per-block variability).

## B. Oracle placement, then a learned damage-predictor selector (RQ3)

- Damage maps: for each image, 16 rollouts, each forcing one block to p=8 after warmup (others fine); damage = LPIPS to the
  teacher-50 image. Training images: 3,000 at seed offset 900000. Evaluation images: 2,000 at seed offset 950000.
- **Stage B1 — oracle (stop/go).** Priority = measured p=8 damage of the evaluation image (oracle, not causal).
  Arms at budgets 100 and 70 tokens on the 2,000 evaluation images: spatial_oracle vs spatial_random, joint_oracle vs
  joint_random (oracle/random arms reuse the joint arm's per-step size multiset; only placement differs).
  **Go** iff, at ≥ 1 budget, an oracle arm beats its random arm with paired LPIPS 95% CI upper bound < −0.01.
  If No-go: stop; report that placement has little headroom on this setup.
- **Stage B2 — learned selector (only if Go).** A small CNN reads causal features at the first scheduled step
  (x_t, predicted x̂0, Delta^3; 12×32×32) plus a projected frozen class embedding, and outputs a 4×4 damage map; trained
  on the 3,000 training images (500 held out for early stopping) with per-image z-scored MSE and horizontal-flip augmentation.
  - Allocation gap on the 2,000 evaluation images vs Delta^3 (fd8), clean variation, random, and oracle.
  - End-to-end on the 2,000 evaluation images at budgets 100 and 70: spatial_learned, joint_learned vs their random arms
    and vs joint (Delta^3).
  **RQ3 positive** iff (i) learned gap < Delta^3 gap with non-overlapping 95% bootstrap CIs, and (ii) joint_learned beats
  joint_random (paired LPIPS CI upper < 0) at both budgets. Partial if (i) holds and (ii) at one budget.

## C. Stage 0: compute headroom at 1024 / 2048 px (added 2026-11-21 ~22:45 EST, before any run)

Question: does token reduction have real wall-clock headroom at resolutions where attention dominates, unlike DiT-S @256?
No training. Real transformer trunks (all blocks) timed in bf16 with CFG (2 sequences/image) at token fractions
1, 0.75, 0.5, 0.25, 1/16, for DiT-S/2 @256 (N=256), PixArt-alpha 1024 (N=4096), PixArt-Sigma 2K (N=16384); batch 1 and 4 images;
executed FLOPs and attention share via `torch.utils.flop_counter`. Upper bound only (excludes embed/head/selector/decoder).
**Gate:** PixArt-alpha 1024, 1 image, trunk speedup at 50% tokens ≥ 1.5× → build Stages 1–2 (mixed-patch PixArt-alpha 1024,
same matched protocol incl. fewer-steps baseline, FID + CLIP score vs teacher). Otherwise Stages 1–2 are not built.

## Not in scope
Latency is not re-measured (hardware may differ from the earlier B200 table). No retraining of the shared model.

## Deviations
1. 2026-11-22 (job 10954676, B200): Stage 0 and A completed as specified. B's damage-map step ran out of GPU memory
   (VAE-decoded 500 images at once while sharing the GPU with the A sweep). The oracle step then had no input and the
   script misreported this as "no-go". **B was not run; it did not fail its gate.** Fix: chunked VAE decoding and LPIPS
   (no change to what is measured) and a corrected failure message. B is re-run unchanged (seeds, budgets, rules)
   as job 11053974 / `runs/followup_B.sh`.
