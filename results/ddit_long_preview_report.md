# DDiT-S schedule comparison

**Failed adapter pilot.** These results used an underfit patch-4 adapter with a missing final normalization/modulation and divergent CFG branch inputs. They do not establish schedule effectiveness. Original artifacts are preserved.

All methods use the same ImageNet classes and noise seeds. The two global schedules each use 25 patch-4 and 25 patch-2 steps. The late spatial schedules use four coarse blocks during the first half and twelve during the second half; each arm therefore processes 8,000 total tokens over 50 steps. The late-signal arm chooses smooth blocks from the third-order latent difference; late-random is the matched control.

| Strategy | Samples | LPIPS to DiT-S | Latent MSE | Median denoise s | Mean tokens | Approx. transformer GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 0.0000 | 0.0000 | 0.200 | 256.0 | — |
| coarse_early | 8 | 0.9064 | 2291.9981 | 0.222 | 160.0 | 372.9 |
| coarse_late | 8 | 0.9147 | 329.0452 | 0.224 | 160.0 | 372.9 |
| late_signal | 8 | 0.9421 | 595.6916 | 0.247 | 160.0 | 366.5 |
| late_random | 8 | 0.9431 | 451.9909 | 0.246 | 160.0 | 366.5 |

LPIPS measures paired similarity to the original DiT-S output, not distributional quality. The transformer FLOP estimate excludes patch embedding/de-embedding, residual, selector, CFG, and VAE costs; measured latency includes them.

Samples: `artifacts/ddit_long_preview/samples.png`.
