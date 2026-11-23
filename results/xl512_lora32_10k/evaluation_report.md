# DiT-XL/2 512×512 LoRA-32 evaluation

## Summary

- **The small-patch path passed the sanity check.** With 2×2 latent patches everywhere, the update-1,500 model reproduced the full-detail 50-step output pixel for pixel on all eight gentle-run examples.
- **Aggressive coarsening degraded the images.** Uniform 4×4 and 8×8 layouts had mean paired LPIPS of 0.608 and 0.668. Spatial-only, timestep-only, and joint schedules near 600 tokens per step also differed substantially from the reference.
- **Gentle coarsening gave a tradeoff.** Coarsening four regions in the middle of sampling kept mean LPIPS at 0.023 and saved 0.7% of counted denoiser FLOPs on the profiled seed. Coarsening 32 regions saved 13.8% and increased mean LPIPS to 0.173.
- **Full-detail DDIM-25 was the stronger efficiency baseline in these pilots.** It used half the counted denoiser FLOPs of DDIM-50. Its mean LPIPS was lower than all aggressive adaptive arms and lower than the 16- and 32-region gentle schedules on the eight paired examples.

## Setup and how to read the results

Both evaluations used the native pretrained DiT-XL/2 512×512 backbone and the pinned mixed-patch LoRA-32 checkpoint from **update 1,500**. Each strategy used the same starting noise and ImageNet class within an evaluation. DDIM used classifier-free guidance scale 4. The reference used 50 denoising steps with 2×2 latent patches everywhere. The gentle run reused the first eight seeds from the 16-image pilot. Each grid displays the first four seeds, with one strategy per row in the order shown below.

Paired LPIPS measures how much the generated image differs from its same-seed DDIM-50 reference. Lower values indicate closer agreement. The 95% bootstrap intervals are for the mean LPIPS of each arm. Tokens are averaged across images and denoising steps. Counted denoiser FLOPs include both guidance branches for **one profiled seed**. Seconds are median end-to-end times including decoding. Other GPU workloads were active during evaluation, so these timings are preliminary. Compare image quality and compute within each run. Neither run measured ImageNet FID.

## Original pilot · 16 images per strategy

[Paired sample grid](eval_512_pilot/paired_samples.png) · [Raw metrics](eval_512_pilot/metrics.json)

| Row | Strategy | Mean tokens/step | Counted GFLOPs | Median seconds | Paired LPIPS [95% CI] |
|---:|---|---:|---:|---:|---:|
| 1 | Full-detail DDIM-50 | 1,024.0 | 104,908 | 11.407 | 0.000 [0.000, 0.000] |
| 2 | Full-detail DDIM-25 | 1,024.0 | 52,454 | 5.773 | 0.104 [0.063, 0.156] |
| 3 | Uniform 4×4 after warmup | 302.1 | 29,570 | 5.033 | 0.608 [0.557, 0.661] |
| 4 | Uniform 8×8 after warmup | 121.6 | 12,133 | 3.325 | 0.668 [0.639, 0.698] |
| 5 | Switch from uniform 4×4 back to fine | 716.8 | 72,849 | 8.761 | 0.542 [0.483, 0.608] |
| 6 | Spatial-only, target 600 | 595.4 | 59,812 | 7.865 | 0.511 [0.432, 0.590] |
| 7 | Spatial shuffle, target 600 | 595.4 | 59,812 | 7.862 | 0.501 [0.424, 0.579] |
| 8 | Timestep-only, target 600 | 614.8 | 51,983 | 10.580 | 0.509 [0.414, 0.599] |
| 9 | Joint, target 600 | 584.6 | 61,769 | 11.600 | 0.556 [0.514, 0.598] |
| 10 | Joint shuffle, target 600 | 584.8 | 59,771 | 11.484 | 0.559 [0.517, 0.601] |

In the grid, uniform coarsening distorts the pen, and the pink subject nearly disappears. Switching back to small patches recovers some recognizable structure, while color and composition still drift. The similar mean token counts of the spatial, temporal, and joint arms produced different counted FLOPs. The timestep-only and DDIM-25 arms were close in profiled FLOPs, with much lower mean paired LPIPS for DDIM-25.

## Gentle 4×4 pilot · 8 images per strategy

[Paired sample grid](eval_512_gentle_v2/paired_samples.png) · [Raw metrics](eval_512_gentle_v2/metrics.json)

The gentle schedules keep small patches everywhere for the first 20 and last 12 denoising steps. During the middle 18 steps they use 4×4 patches in the selected 4, 16, or 32 smoothest regions out of 64. The selector recomputes the region ranking as sampling progresses.

| Row | Strategy | Mean tokens/step | Counted GFLOPs | Median seconds | Paired LPIPS [95% CI] |
|---:|---|---:|---:|---:|---:|
| 1 | Full-detail DDIM-50 | 1,024.0 | 104,908 | 16.387 | 0.000 [0.000, 0.000] |
| 2 | LoRA model, all small patches, 50 steps | 1,024.0 | 104,908 | 16.488 | 0.000 [0.000, 0.000] |
| 3 | Full-detail DDIM-25 | 1,024.0 | 52,454 | 8.197 | 0.087 [0.030, 0.169] |
| 4 | 4 coarse regions | 1,006.7 | 104,147 | 16.798 | 0.023 [0.014, 0.032] |
| 5 | 16 coarse regions | 954.9 | 98,143 | 14.984 | 0.109 [0.071, 0.150] |
| 6 | 32 coarse regions | 885.8 | 90,438 | 10.362 | 0.173 [0.122, 0.221] |

The four-region schedule preserved most of the reference appearance, although one preview gained a small bison in the background. The 16- and 32-region schedules reduced counted FLOPs by **6.4%** and **13.8%**, respectively. Their mean reference-image difference increased with the number of coarse regions. The four-region timing showed no speed gain, and the measured timing gains for 16 and 32 regions need confirmation on a GPU running evaluation alone.

## Relation to the project questions

- **H1, spatial versus timestep-only.** Spatial-only and timestep-only had nearly identical mean paired LPIPS around 600 tokens. The timestep-only arm had lower counted FLOPs on the profiled seed. These results provide no evidence for a spatial-only advantage in this pilot. A test at matched executed FLOPs would answer H1 more directly.
- **H2, joint versus either axis alone.** Joint had worse mean paired LPIPS than either single-axis arm at the tested target and used more counted FLOPs on the profiled seed. The gentle schedule shows that limiting where and when large patches appear can preserve image agreement, at a small compute saving in the closest-matching configuration. It did not include matched single-axis controls at those gentler budgets.
- **H3, choosing which regions matter.** The spatial and joint selection rules each performed similarly to their respective shuffled controls. These pilot comparisons do not establish a placement advantage. Testing H3 directly would require measuring how much coarsening each region changes the output and checking whether the selector ranks those regions correctly.

This report evaluates the trained 512 LoRA checkpoint at update 1,500. The earlier pretrained 512 patch-size experiment used a separate inference method. The pilot also lacks a comparison against an untrained mixed-patch model under the same schedule, so it does not isolate the contribution of LoRA training itself.
