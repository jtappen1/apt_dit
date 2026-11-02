# On-policy decoder plus rank-16 attention LoRA

The decoder and rank-16 attention LoRA were trained together for 1,000 steps on the same 128 student-generated training trajectories, with the frozen DiT teacher queried at ten captured states per trajectory. This run has 928,160 trainable parameters.

## Held-out reconstruction

On a stratified subset of 96 records drawn from 32 validation rollouts, decoded eight-coarse-block noise MSE went from **0.0219** at initialization to a best of **0.0165** at step 900. The decoder-only on-policy model started at 0.0219 on the same validation subset, so LoRA lowered prediction error further.

## Paired samples

On the same 32 final seeds/classes and spatial schedule, mean LPIPS to the baseline was **0.6753** for decoder-only on-policy and **0.6797** for decoder plus LoRA. The paired difference (LoRA minus decoder-only) was **0.0044** (95% paired t interval **[-0.0136, 0.0223]**). Median denoising time was **0.222s** versus **0.227s**.

LoRA reduced held-out cached-state prediction error from the decoder-only starting value of 0.0219 to 0.0165, but paired LPIPS was numerically 0.0044 higher than decoder-only. Its 95% paired interval spans zero, so there is no clear sample-fidelity gain. These remain low-fidelity images; no FID was run. The existing model-FLOP estimate omits decoder and LoRA overhead.

See `lora_samples.png` for matched images and `lora_comparison.json` for per-seed results.
