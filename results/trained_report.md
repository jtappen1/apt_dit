# trained: paired DiT-S evaluation

Rows in samples.png follow the table order; columns share seeds/classes.

| Arm | Samples | Repeats | LPIPS to baseline | Latent MSE | Median denoise s | Mean tokens | Model GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine | 64 | 1 | 0.0000 | 0.0000 | 0.203 | 256.0 | 1211.1 |
| coarse | 64 | 1 | 0.8784 | 18577.1801 | 0.207 | 64.0 | 292.0 |
| random | 64 | 3 | 0.8818 | 129.9713 | 0.214 | 167.7 | 777.1 |
| spatial | 64 | 1 | 0.8651 | 87.3579 | 0.217 | 167.7 | 777.1 |
| temporal | 64 | 1 | 0.8760 | 2294.1456 | 0.205 | 163.8 | 770.0 |
| joint | 64 | 1 | 0.8721 | 515.0389 | 0.217 | 167.7 | 781.0 |

LPIPS measures agreement with the original model, not distributional quality. No FID claim.
Model FLOPs are analytical (2 FLOPs/MAC); exclude elementwise, resize, selector, and VAE work. Latency includes selector and resize overhead.
These timings include Python checks; use the dedicated repeated benchmark for a latency conclusion.

Artifacts: `artifacts/trained`.
