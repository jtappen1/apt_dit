# DDiT-S schedule comparison

**Failed adapter pilot.** These results used an underfit patch-4 adapter with a missing final normalization/modulation and divergent CFG branch inputs. They do not establish schedule effectiveness. Original artifacts are preserved.

All methods use the same ImageNet classes and noise seeds. The two global schedules each use 25 patch-4 and 25 patch-2 steps. The late spatial schedules use four coarse blocks during the first half and twelve during the second half; each arm therefore processes 8,000 total tokens over 50 steps. The late-signal arm chooses smooth blocks from the third-order latent difference; late-random is the matched control.

| Strategy | Samples | LPIPS to DiT-S | Latent MSE | Median denoise s | Mean tokens | Approx. transformer GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 64 | 0.0000 | 0.0000 | 0.199 | 256.0 | — |
| coarse_early | 64 | 0.8851 | 23439.3121 | 0.223 | 160.0 | 372.9 |
| coarse_late | 64 | 0.8693 | 25.6241 | 0.224 | 160.0 | 372.9 |
| late_signal | 64 | 0.9307 | 332.5218 | 0.246 | 160.0 | 366.5 |
| late_random | 64 | 0.9303 | 330.2787 | 0.245 | 160.0 | 366.5 |

LPIPS measures paired similarity to the original DiT-S output, not distributional quality. The transformer FLOP estimate excludes patch embedding/de-embedding, residual, selector, CFG, and VAE costs; measured latency includes them.

## Paired comparisons and interpretation

- Coarse-late versus coarse-early: mean paired LPIPS change was -0.0158 (95% paired t interval [-0.0276, -0.0039]), favoring coarse-late. The median denoising times were effectively identical (0.224 s vs. 0.223 s), and both were slower than the 0.199 s baseline.
- Late signal versus late random: paired LPIPS change was +0.0003 (95% interval [-0.0033, +0.0039]); there is no measurable benefit from the trajectory-based block selector here.
- Every adaptive arm produced severe visual artifacts. The patch-4 teacher-prediction path had held-out MSE 17.55 after 800 steps. A separate 6,000-step run lowered held-out MSE to 2.13, but its eight-seed preview still showed checkerboard artifacts (see `artifacts/ddit_long_preview/samples.png`).

Thus, the slight coarse-late LPIPS advantage is not evidence for useful late coarsening: the coarse predictor failed the image-quality gate, the images are unusable, and there was no measured speedup. The late spatial idea also did not beat random selection. Do not infer a general schedule conclusion from this underfit DiT-S patch-4 adapter.

Samples: `artifacts/ddit_schedule_comparison/samples.png`.
