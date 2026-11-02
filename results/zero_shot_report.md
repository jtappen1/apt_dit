# zero_shot: paired DiT-S evaluation

Rows in samples.png follow the table order; columns share seeds/classes.

| Arm | Samples | Repeats | LPIPS to baseline | Latent MSE | Median denoise s | Mean tokens | Model GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine | 32 | 1 | 0.0000 | 0.0000 | 0.203 | 256.0 | 1211.1 |
| coarse | 32 | 1 | 0.8905 | 72846.7510 | 0.201 | 64.0 | 282.6 |
| random | 32 | 3 | 0.8988 | 362.7618 | 0.209 | 167.7 | 768.4 |
| spatial | 32 | 1 | 0.8798 | 216.0674 | 0.212 | 167.7 | 768.4 |

LPIPS measures agreement with the original model, not distributional quality. No FID claim.
Model FLOPs are analytical (2 FLOPs/MAC); exclude elementwise, resize, selector, and VAE work. Latency includes selector and resize overhead.
These timings include Python checks; use the dedicated repeated benchmark for a latency conclusion.

Artifacts: `artifacts/zero_shot`.
