# Adaptive patching on DiT-S: experiment record

Updated 2026-11-02. This is the consolidated record of completed experiments, conclusions, reproducibility, and possible next work. It replaces the former agent plan, project plans, recovery plans, and experiment log. The original proposal is preserved in [cis_6220_project.pdf](cis_6220_project.pdf); detailed measurements remain in `results/`.

## Current position

The pretrained baseline works and the original fine-patch path remains exact. Spatially mixed models with a fine-information output decoder produce recognizable but distorted images. Global patch-4 models still produce corrupted patterns, even after substantially longer training. **No adapted method has demonstrated useful image quality together with measured acceleration.** Lower teacher-prediction loss has repeatedly failed to translate into usable generated images.

The strongest existing spatial result is the count-4-specific dense decoder: same-seed LPIPS 0.386 on 64 images, with visible artifacts and no measured speedup. The latest global patch-4 noise-skip model reaches development LPIPS 0.7385 on a different 16-image development set. Those numbers are not a matched comparison.

Longer training continues to improve held-out prediction error, with diminishing returns. No sustained held-out-loss deterioration has been observed through 23,200 noise-skip updates. The growing training/development gap and flattening guided-prediction accuracy warn against assuming that unlimited continuation will solve generation.

## Question and fixed setup

The proposal asks whether spatially adaptive patches beat uniform-per-step temporal patching at matched compute, and whether joint space/time adaptation improves the tradeoff. We first need a reliable adapted denoiser; a failed coarse model cannot settle the scheduling hypotheses.

- Backbone: pretrained DiT-S/2, approximately 32.96M parameters, class-conditional ImageNet-256. This is a small-model pilot, not the originally proposed XL-scale evaluation.
- Latents: four channels, 32×32. Original patch size 2 gives 256 tokens; patch size 4 gives 64 tokens.
- Spatial layouts: sixteen 8×8 latent blocks. Coarsening a block replaces sixteen fine tokens with four coarse tokens. Four/eight coarse blocks give 208/160 tokens on an adapted step; reported trajectory averages include fine warmup steps.
- Sampling: official 50-step deterministic DDIM, eta 0, CFG 4 with upstream three-channel guidance; the fourth epsilon channel remains unguided. Both CFG branches receive the same latent and layout. EMA VAE, latent scale 0.18215, FP32, one RTX 5090.
- Original manifests: 128 training, 32 validation, and 64 final trajectories, distinct seeds. Cache: ten timestep states per trajectory and both branches, giving 2,560 training and 640 validation branch records.
- Recovery trials retain paired conditional/unconditional records. Repeatedly inspected original final seeds are now regression examples. Recovery uses the existing validation split as development data and reserves a fresh 64-pair test manifest that has not been sampled.
- Base DiT and VAE remain frozen; adapter/detail/decoder/LoRA components vary by experiment. Source revisions, checkpoint identity, and licenses are in [PROVENANCE.md](PROVENANCE.md).

LPIPS measures fidelity to the same-seed teacher image, not distributional quality. MSE objectives, channel groups, evaluation subsets, and image sets differ across phases; compare only the explicitly matched controls. No FID or distributional-quality claim has been established.

## 1. Baseline, zero-shot patching, and APT detail branch

The initial APT-inspired adaptation downsamples aligned 4×4 patches to 2×2 before the original embedding, interpolates positions, processes the mixed sequence, then repeats coarse features onto the fine output grid before the original head. It is not a reproduction of APT's clean-image entropy/quadtree policy.

| Experiment | Training / evaluation | Result |
|---|---|---|
| Original baseline | 64 seeded images | Recognizable samples; initial median denoising about 0.199 s/image |
| Frozen resize/repeat | 32 paired images, three random-allocation repeats | Fine path exact; coarse/mixed images severely degraded |
| Detail branch | 1,000 updates; 738,048 trainable parameters | Held-out eight-coarse-block noise-plus-variance loss 0.8030 → 0.6506; little image improvement |
| Trained schedules | 64 paired images; random, spatial, temporal, joint controls | Spatial placement modestly beats random/temporal on LPIPS, but all images remain badly degraded |

| Original trained schedule | Paired LPIPS ↓ | Mean tokens | Estimated model GFLOPs/sample | Median denoise seconds |
|---|---:|---:|---:|---:|
| Fine | 0.0000 | 256.0 | 1,211.1 | 0.203 |
| Random spatial | 0.8818 | 167.7 | 777.1 | 0.214 |
| Score-selected spatial | 0.8651 | 167.7 | 777.1 | 0.217 |
| Temporal | 0.8760 | 163.8 | 770.0 | 0.205 |
| Joint | 0.8721 | 167.7 | 781.0 | 0.217 |

Spatial minus random LPIPS was -0.0167 (95% paired bootstrap interval [-0.0192, -0.0142]); spatial minus temporal was -0.0108 ([-0.0122, -0.0094]). Joint was worse than spatial. These are preliminary fidelity differences between degraded models, not confirmation of the proposal's matched-quality hypotheses. FLOPs exclude some overhead; smaller estimates did not yield a measured speedup.

Evidence: [initial report](results/initial_report.md), [zero-shot](results/zero_shot_report.md), [trained schedules](results/trained_report.md), [paired samples](results/comparison.png).

## 2. Output reconstruction, student-state training, and conservative compression

A diagnostic projected teacher outputs onto the repeated-output subspace. For the trained clean-score layout, the best possible repeated output already incurs noise MSE 0.325 versus actual 0.342: about 95% of the error is explained by this restriction. More detail-branch training or LoRA alone cannot remove the repeated-output constraint.

A dense output decoder therefore predicts distinct fine-grid corrections from coarse transformed features and the original fine input embeddings.

| Follow-up | Held-out prediction result | Paired image result |
|---|---|---|
| Dense output decoder, 1,000 updates | Eight-block noise MSE 0.338 → 0.047 on 96 states | LPIPS 0.878 → 0.714 on 32 images |
| Decoder on student-rollout states, 1,000 updates | On-policy noise MSE 0.0369 → 0.0219 | LPIPS 0.714 → 0.675 on the same 32 images |
| Decoder + rank-16 attention LoRA, 1,000 updates | Best on-policy noise MSE 0.0165 | LPIPS 0.680 versus decoder-only 0.675; paired interval includes no improvement |
| Trajectory selector, eight coarse blocks | Better ranking of single-block damage than clean-latent variation | On 64 images, LPIPS 0.6426 versus clean selector 0.6598 |
| Transfer count-8 decoder to four coarse blocks | Less aggressive compression | On 64 images, LPIPS 0.4427 |
| Count-4-specific on-policy decoder, 1,000 updates | Separate count-4 rollout cache | On the same 64 images, LPIPS 0.3860 |

Count-4-specific minus transferred count-8 decoder LPIPS was -0.0567 (95% paired t interval [-0.0722, -0.0412]). Images became more recognizable but remained distorted. Count-4 mean tokens were about 211.8 including warmup, and denoising about 0.223–0.224 s: still slower than baseline. Earlier analytical totals omit the new decoder and must not be treated as complete executed FLOPs.

The student-state follow-ups train on teacher labels at student-generated states, but the training loader draws new random layouts rather than reproducing the inference selector's exact masks. This addresses state-distribution mismatch only partially.

Evidence: [reconstruction](results/reconstruction_report.md), [student-state training](results/onpolicy_report.md), [LoRA](results/lora_report.md), [selector/budget controls](results/selector_budget_report.md), [count-4 fine-tuning](results/k4_finetune_report.md), [count-4 images](results/k4_finetune_samples.png).

## 3. Initial global DDiT-style patch-4 pilot: failed implementation and adaptation

A separate wrapper adds patch-specific embedding/de-embedding, interpolated positions, patch-size embedding, rank-32 FFN LoRA, and a latent-to-output residual. The original pilot used averaged embedding initialization and a tiled output-head prior.

- At 800 updates, combined held-out teacher MSE fell from roughly 5,535 to 17.55. A separate 6,000-update trial lowered it to about 2.14, but images remained unusable checkerboard/noise.
- A 64-image schedule comparison used coarse-early, coarse-late, late spatial signal, and matched random spatial selection. LPIPS was approximately 0.885, 0.869, 0.931, and 0.930 respectively. No useful quality or latency result emerged; signal selection did not beat random.
- Those trials had a missing coarse final normalization/AdaLN modulation and CFG branches that could receive divergent states. Their schedule numbers are records of a failed pilot, not evidence about schedule effectiveness.

Evidence: [historical schedule report](results/ddit_schedule_comparison_report.md), [extended-checkpoint preview report](results/ddit_long_preview_report.md).

## 4. Geometry audit and corrected short trial

The audit verified row/column/channel order, mixed-layout coverage, and upstream output packing. It corrected missing final LayerNorm and timestep/class modulation, fixed CFG latent duplication, and replaced averaged embedding initialization with a bilinear projection pseudo-inverse. The tiled head's spatial order was correct; it was not reordered on visual guesswork.

The corrected 800-update trial on existing data reached full-set epsilon MSE 0.6549 and variance-channel MSE 0.3445. Each timestep bin beat the old target-variance threshold, yet four all-p4 50-step images remained unrecognizable. That threshold was too weak to establish a usable predictor.

Pure patch-2 predictions match the frozen teacher exactly at the same batch size. Against cached batch-two targets, numerical batch-kernel differences give MSE approximately 7.36e-13 and maximum absolute error 1.58e-4. These are distinct comparisons.

The corrected trial was still improving when stopped. The earlier 6,000-update run used broken code and cannot establish convergence or nonviability of the corrected architecture.

Evidence: [audit](results/ddit_adapter_audit.md), [corrected metrics](results/ddit_audit_corrected.json), [failed preview](results/ddit_audit_corrected_p4.png).

## 5. High-noise diagnostic and matched recovery controls

A CPU check of all held-out cache records found that at t=800–999, simply using the input latent as epsilon gives teacher MSE 0.00000776 versus corrected DDiT's 0.7924. The learned residual center diagonal was only about 0.078 rather than an identity coefficient of 1. This implicated poor noise preservation; it did not prove a mathematically unavoidable embedding bottleneck.

Deterministic DDIM's update does not use finite learned-variance predictions, although the original objective gives variance and epsilon equal weight. That observation applies to the current eta-zero sampler, not ancestral sampling.

Three recovery arms received 3,200 updates with matched timestep-balanced paired draws and AdamW at 1e-4:

- A: warm-start corrected 800-update checkpoint; original all-eight-channel loss, new optimizer.
- B: same warm start and draws; epsilon-only loss, new optimizer.
- C: same inherited trunk but zeroed correction output; explicit noise skip and v-target loss: `epsilon = sqrt(1-alpha_bar) * x + sqrt(alpha_bar) * v`. C changes initialization, parameterization, and objective together.

A separate tiny-set optimization check improved loss 125× after 800 updates at 1e-3; those weights were discarded. Full-fine CFG trajectories are bit-exact against upstream.

| Recovery arm | Held-out epsilon MSE | Guided epsilon MSE | DDIM next-state MSE | Development LPIPS, 16 pairs |
|---|---:|---:|---:|---:|
| A: longer corrected training | 0.2661 | 0.3011 | 0.005910 | 0.8739 |
| B: longer, epsilon-only | 0.2334 | 0.2700 | 0.005246 | 0.8819 |
| C: explicit noise skip | 0.0656 | 0.0751 | 0.000363 | 0.7766 |

The skip fixes the easy high-noise signal and reduces extreme latent magnitudes, but all 16 development images per arm remain corrupted. Lowering variance loss alone improves epsilon fit without improving image fidelity. No new schedule or final-test evaluation was run.

Evidence: [diagnostic](results/ddit_recovery_diagnostics.json), [diagnostic plot](results/ddit_recovery_diagnostics.png), [controlled recovery report](results/ddit_recovery_report.md), [paired images](results/ddit_recovery_samples.png).

## 6. Substantially longer noise-skip training

C was continued from 3,200 to 23,200 noise-skip updates with optimizer/data RNG restored, unchanged data/objective/LR, and full training/development checks every 1,000 updates. These correction-update counts exclude the 800 inherited earlier trunk updates. All 1,280 training pairs and 320 development pairs were evaluated, avoiding comparisons of noisy minibatch training loss with full held-out loss.

| Checkpoint | Train epsilon MSE | Held-out epsilon MSE | Held-out v-target MSE | Guided epsilon MSE |
|---|---:|---:|---:|---:|
| Start, 3,200 | 0.06093 | 0.06562 | 0.12252 | 0.07508 |
| Best v-target checkpoint, 22,200 | 0.04695 | 0.05608 | 0.09743 | 0.06858 |
| Last, 23,200 | 0.04650 | 0.05601 | 0.09773 | 0.06908 |

Training epsilon error improved 23.7%, held-out error 14.6%; their absolute gap grew from 0.00469 to 0.00951. No sustained held-out deterioration was observed. The small final v-loss rise is one check, not proof of overfitting. Guided error reached its minimum around 17,200 before rising slightly. The result is diminishing generalization benefit, not evidence that training duration never matters.

The selected 22,200-update checkpoint improves development LPIPS from 0.7766 to 0.7385 on the same 16 pairs. Images still fail reliable content/class recognition. Remaining errors are concentrated at low/middle timesteps: selected epsilon MSE 0.1843 at t=0–199 and 0.0923 at t=200–399, versus 0.00000531 at t=800–999.

Keep both best and last checkpoints. All 20 tests pass, including exact next-update optimizer/data-RNG restoration, nonzero final conditioning, geometry, v conversion, and deterministic-DDIM variance independence. The selected checkpoint's pure-fine full trajectory still matches upstream bit-for-bit.

Evidence: [long-training report](results/ddit_long_training_report.md), [training/generalization curves](results/ddit_long_training_curves.png), [paired images](results/ddit_long_training_samples.png).

## Reproducibility and local checkpoints

Run from the repository root after `source scripts/env.sh`; use the project-local `.venv`. Setup and verification commands are in [README.md](README.md). All artifacts, caches, temporary files, and environments stay in this folder. The serial `GPUBudget` ledger enforces an eight-hour aggregate process-time ceiling; it includes I/O during GPU phases and is an approximate accounting of compute, not a promise of convergence. Do not silently reset or bypass it.

| Local artifact | Role |
|---|---|
| `artifacts/checkpoints/dit_s_256.pt` | Original frozen DiT-S checkpoint |
| `artifacts/cache/{train,validation}/` | Original teacher-state cache |
| `artifacts/training/detail/` | APT detail-only model |
| `artifacts/training/dense_output_onpolicy_k4/best.pt` | Strongest existing count-4 spatial decoder |
| `artifacts/dense_output_onpolicy_k4_adapted/` | Its 64-image evaluation, not the checkpoint directory |
| `artifacts/training/ddit_s_p4{,_long}/` | Historical broken-head pilots; preserve as evidence |
| `artifacts/training/ddit_s_p4_corrected_slice/` | Corrected 800-update initialization |
| `artifacts/training/ddit_recovery_v1c/{A,B,C}/` | Matched 3,200-update recovery controls |
| `artifacts/training/ddit_recovery_long_c/best.pt` | Selected 22,200-update noise-skip model |
| `artifacts/training/ddit_recovery_long_c/last.pt` | 23,200-update model plus optimizer/RNG state |

Recent runs save version/arm, optimizer, RNG, source hashes, and metrics. Evaluation loaders use checkpoint metadata to preserve C's v-to-epsilon conversion. Historical checkpoints do not encode their original forward implementation: use explicit legacy-head audit mode when reproducing historical predictions. Do not silently evaluate old broken-head weights using the corrected forward and label the results historical.

Main entry points:

```bash
# Baseline, original teacher cache, and original spatial detail adaptation
.venv/bin/python -m apt_dit.evaluate verify
.venv/bin/python -m apt_dit.evaluate baseline --samples 64
.venv/bin/python -m apt_dit.evaluate zero_shot --samples 32
.venv/bin/python -m apt_dit.cache
.venv/bin/python -m apt_dit.train --steps 1000 --run-name detail_new

# CPU diagnostic; requires the existing teacher cache and corrected audit
.venv/bin/python scripts/recovery_diagnostics.py

# Matched recovery trials; requires corrected 800-update source checkpoint
.venv/bin/python -m apt_dit.ddit_recovery --name recovery_new --steps 3200 --tiny-steps 1200 --hours .60

# Reproduce the duration experiment from the saved 3,200-update C checkpoint
.venv/bin/python -m apt_dit.ddit_continue --name continuation_new --additional-steps 20000 --eval-every 1000 --hours .60
```

Use new run names; previous artifacts are not overwritten. Later commands depend on local source checkpoints and caches, which are not distributed in Git. Detailed report commands/configuration are retained with their phase's reports and run directories. Reports/selected diagnostic arrays/figures are versioned; model weights, raw rollout tensors, environments, and dependency/model caches are excluded.

## What has not been established

No usable all-p4 predictor, matched-distribution-quality acceleration, FID evaluation, general schedule superiority, larger-model transfer, PixArt, or video result has been demonstrated. Complete FLOPs must include both CFG branches, decoder, fine embedding/skip, residual, LoRA, and selector. Historical DDiT totals exclude CFG; spatial decoder totals omit decoder overhead. Measured latency has not improved. Development results from repeatedly inspected examples are not an unbiased final test.

## Possible next experiments — not yet run

1. **Single-step intervention:** replace one teacher sampling step with the coarse model, then resume teacher sampling. Map which timesteps cause lasting damage and whether teacher steps recover. This is the cheapest localization test.
2. **Conservative dense decoder:** start from the strongest count-4 checkpoint, preserve fine local input features, and coarsen only 1–2 blocks before progressing to 4. Add exact shared CFG layouts and training on inference-matched masks.
3. **Middle-block compression:** preserve dense fine features through early and final transformer blocks; compress only a short middle section and restore fine features with a residual update. Measure all dense-block/restore costs.
4. **Guided/rollout objective alignment:** once basic prediction is competent, compare paired guided epsilon and short student-rollout supervision against the current v objective. Keep raw branch fidelity checks and fixed CFG4 comparisons.

Begin with the intervention diagnostic, then the conservative decoder. Require recognizable development images at a real nonzero compression budget before reopening selector/schedule sweeps. A new all-p4 method must pass its own image gate; a modest spatial method can be assessed at its intended lower compression count. Retain random spatial and fewer-step full-model controls for any eventual quality/compute claim.
