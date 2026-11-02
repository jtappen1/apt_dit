# On-policy decoder fine-tuning: first results

The current dense decoder generated 128 training and 32 validation spatial-schedule trajectories. We queried the frozen original DiT on ten student-rollout states per trajectory, then trained only the 485,792-parameter output decoder for 1,000 steps. The DiT and prior detail branch remained frozen.

## Held-out student-rollout states

On a stratified subset of 96 held-out records from 32 validation trajectories (ten captured states per trajectory, with both CFG branches represented), the on-policy fine-tune selected step 1,000 with decoded eight-coarse-block noise MSE **0.0219**, versus **0.0369** before fine-tuning on the same records. The repeated-output comparison was **0.3617**.

## Paired sampling

On the same 32 final seeds/classes and same spatial schedule, mean LPIPS to the original samples was **0.7136** before on-policy fine-tuning and **0.6753** after. The paired difference (on-policy minus prior decoder) was **-0.0383** (95% paired t interval **[-0.0576, -0.0190]**). Median denoising time was **0.224s** before and **0.222s** after.

This tests on-policy teacher distillation for the spatial clean-score rollout only. The images still need visual review; LPIPS is same-seed fidelity, not FID or distributional quality. Decoder FLOPs are not included in the existing model-FLOP estimate.

The four-row matched montage is `onpolicy_samples.png`; per-seed metrics are in `onpolicy_comparison.json`.
