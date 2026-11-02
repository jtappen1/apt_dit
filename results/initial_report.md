# Initial DiT-S results

## Setup

Pretrained DiT-S/2 (32.96M parameters), ImageNet-256, 50-step deterministic DDIM, CFG 4, FP32 on one RTX 5090. The original three-channel CFG convention and EMA VAE are preserved.

Initial sequence: original baseline → frozen APT-style resize/repeat → 1,000 detail-branch updates. Only 738,048 detail-branch parameters trained; original DiT and VAE frozen during that first adaptation. Later follow-ups trained the dense decoder, then tested a separate joint rank-16 LoRA update.

128 teacher training trajectories and 32 validation trajectories; ten cached timesteps and both CFG branches per trajectory. Zero-shot evaluation uses 32 final seeds; the original trained schedule matrix uses 64. Dense-decoder and LoRA adaptation comparisons use 32 shared seeds; selector and budget controls are extended to 64. Random allocation averages three replicates per seed.

## Paired sample fidelity

Lower LPIPS means closer to the original sample, not better distributional quality. The bootstrap intervals below quantify paired differences only; no FID or H1/H2 confirmation is claimed.

| Stage | Arm | Seeds | LPIPS | Model GFLOPs/sample | Median denoise seconds |
|---|---|---:|---:|---:|---:|
| baseline | baseline | 64 | 0.0000 | 1211.1 | 0.1985 |
| zero_shot | fine | 32 | 0.0000 | 1211.1 | 0.2026 |
| zero_shot | coarse | 32 | 0.8905 | 282.6 | 0.2006 |
| zero_shot | random | 32 | 0.8988 | 768.4 | 0.2097 |
| zero_shot | spatial | 32 | 0.8798 | 768.4 | 0.2119 |
| trained | fine | 64 | 0.0000 | 1211.1 | 0.2028 |
| trained | coarse | 64 | 0.8784 | 292.0 | 0.2066 |
| trained | random | 64 | 0.8818 | 777.1 | 0.2144 |
| trained | spatial | 64 | 0.8651 | 777.1 | 0.2173 |
| trained | temporal | 64 | 0.8760 | 770.0 | 0.2048 |
| trained | joint | 64 | 0.8721 | 781.0 | 0.2174 |

Model FLOPs use 2 FLOPs/MAC and count both guidance branches, full fine/coarse projections, attention, MLPs, conditioning, detail branch, and dense output head. They exclude resize, elementwise, selector, and VAE operations; sampling latency includes resize and selection. Comparisons are approximately model-FLOP matched, not certified total-FLOP matched.

| Paired LPIPS difference (first minus second) | Shared seeds | Mean | 95% bootstrap interval |
|---|---:|---:|---|
| zero_spatial_minus_random | 32 | -0.0190 | [-0.0226, -0.0151] |
| trained_spatial_minus_zero_spatial | 32 | -0.0019 | [-0.0051, 0.0012] |
| trained_spatial_minus_random | 64 | -0.0167 | [-0.0192, -0.0142] |
| trained_spatial_minus_temporal | 64 | -0.0108 | [-0.0122, -0.0094] |
| trained_joint_minus_spatial | 64 | 0.0070 | [0.0059, 0.0081] |
| trained_joint_minus_temporal | 64 | -0.0038 | [-0.0044, -0.0033] |

## Training and reconstruction constraint

At eight coarse blocks, held-out noise-plus-variance MSE changed from 0.8030 to 0.6506. Checkpoint selection used validation, not final samples.

The resize/repeat head forces each coarse token's four fine output patches to be identical. This is a structural restriction even with a trained detail branch or LoRA. `diagnostics.json` measures an oracle error floor by projecting teacher predictions onto that restricted output space.

## Output reconstruction diagnostic

Across 48 held-out state/trajectory pairs per checkpoint, independently coarsening blocks and measuring teacher-prediction error gives a rank-based selector diagnostic. The repeat floor is a mathematical lower bound for the repeated-output constraint, computed from teacher predictions with an oracle projection; a model cannot beat it without changing that output representation.

| Stage | Placement rule | Noise MSE | Repeat-output floor | Mean rank correlation with single-block damage |
|---|---|---:|---:|---:|
| zero_shot | clean | 0.5249 | 0.3252 | 0.044 |
| zero_shot | trajectory | 0.5109 | 0.3097 | 0.351 |
| zero_shot | oracle_rank | 0.4840 | 0.2995 | — |
| zero_shot | random | 0.5243 | 0.3236 | — |

For zero_shot, the trajectory variation score ranked single-block damage better than clean-latent variation (mean Spearman clean 0.044, trajectory 0.351). The oracle-ranked layout had lower prediction MSE than either causal score. For the trained clean-score layout, 62.0% of measured noise MSE is already present in the repeat-output lower bound, so the output constraint explains most of this prediction error.
| trained | clean | 0.3420 | 0.3252 | 0.047 |
| trained | trajectory | 0.3283 | 0.3097 | 0.370 |
| trained | oracle_rank | 0.3110 | 0.2939 | — |
| trained | random | 0.3434 | 0.3236 | — |

For trained, the trajectory variation score ranked single-block damage better than clean-latent variation (mean Spearman clean 0.047, trajectory 0.370). The oracle-ranked layout had lower prediction MSE than either causal score. For the trained clean-score layout, 95.1% of measured noise MSE is already present in the repeat-output lower bound, so the output constraint explains most of this prediction error.

The dense reconstruction head follow-up has now been run. On 96 held-out teacher states, it reduced eight-coarse-block noise MSE from 0.338 to 0.047. On 32 same-seed spatial samples it reduced LPIPS to baseline from 0.878 to 0.714 (paired 95% t interval for the reduction: 0.139 to 0.189). The samples are more structured than the repeated-output result, but still visibly distorted. The 777 GFLOPs estimate below and in the earlier schedule table is for the original repeated-output path; the added decoder operations are not included. See [the dense reconstruction report](reconstruction_report.md) and [paired sample montage](reconstruction_samples.png).

An on-policy fine-tune then trained the same decoder on states from its own student rollouts. On a stratified subset of 96 held-out rollout records, MSE fell from 0.0369 to 0.0219. Same-seed LPIPS fell from 0.7136 to 0.6753 on the same 32 final samples (paired 95% t interval for the reduction: 0.0190 to 0.0576). The images remain heavily distorted, so this only partly supports rollout drift as a cause. See [the on-policy report](onpolicy_report.md) and [four-way matched montage](onpolicy_samples.png). The existing 777 GFLOPs estimate omits the separate decoder operations.

Joint rank-16 attention LoRA plus decoder training further reduced held-out MSE from 0.0219 to 0.0165, but its LPIPS was 0.6797 versus 0.6753 for decoder-only on the same seeds (paired 95% interval for the difference: [-0.0136, 0.0223]). This is not a clear fidelity change and denoising was slightly slower. See [the LoRA report](lora_report.md) and [five-way matched montage](lora_samples.png).

The selector and budget controls have now been extended to 64 shared seeds. With the on-policy decoder-only model, trajectory selection beat clean selection at eight coarse blocks by 0.0172 LPIPS (paired 95% interval for the improvement: 0.0109 to 0.0236). Reducing from eight to four coarse blocks improved LPIPS from 0.6598 to 0.4427, with mean tokens increasing from 167.7 to 211.8. Training a count-4-specific decoder on count-4 student rollouts improved LPIPS further to 0.3860 (paired improvement versus the transferred count-8 model at count 4: 0.0412 to 0.0722). The images are more recognizable, but still visibly distorted. See [selector/budget results](selector_budget_report.md) and [count-4 fine-tuning results](k4_finetune_report.md), with their matched sample grids.

The strongest fidelity setting tested is now count-4-specific. The paired comparison covers 64 seeds, but only uses LPIPS, which does not measure distributional quality. A larger distributional evaluation remains to be done if compute allows. The model-FLOP estimate excludes the decoder, and measured denoising latency has not improved.

## Reproducibility and limits

Ledger-charged GPU process time: 0.1512 hours (eight-hour ceiling). This includes I/O during GPU phases; it excludes dependency installation, CPU-only tests, and code development. A preliminary CUDA compatibility probe preceded the ledger and was not timed; reported accounting is therefore approximate by a few seconds.

Sources and checkpoint identity are in `PROVENANCE.md`; exact packages are in `requirements.lock`. Raw runs, checkpoints, and teacher caches remain project-local under `artifacts/`.

This is one small backbone, one central allocation budget, one training seed, and short adaptation runs. The fixed-block selector and causal scores differ from APT's entropy/quadtree policy. Temporal control is simple coarse-early scheduling, not a full DDiT reproduction. Neighboring-budget sweeps, FID, PixArt, and video remain unrun.

Figures: [paired samples](comparison.png), [fidelity versus compute/latency](tradeoffs.png), [validation curve](validation.png).

Raw output-reconstruction diagnostics are in `diagnostics.json`; aggregated results are in `diagnostic_summary.json`. Dense-decoder results are in `reconstruction_report.md` and `reconstruction_comparison.json`; on-policy results are in `onpolicy_report.md` and `onpolicy_comparison.json`; LoRA results are in `lora_report.md` and `lora_comparison.json`; selector/budget and count-4 results are in `selector_budget_report.md` and `k4_finetune_report.md`.
