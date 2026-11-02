# baseline: paired DiT-S evaluation

Rows in samples.png follow the table order; columns share seeds/classes.

| Arm | Samples | Repeats | LPIPS to baseline | Latent MSE | Median denoise s | Mean tokens | Model GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 64 | 1 | 0.0000 | 0.0000 | 0.199 | 256.0 | 1211.1 |

LPIPS measures agreement with the original model, not distributional quality. No FID claim.
Model FLOPs are analytical (2 FLOPs/MAC); exclude elementwise, resize, selector, and VAE work. Latency includes selector and resize overhead.
These timings include Python checks; use the dedicated repeated benchmark for a latency conclusion.

Artifacts: `artifacts/baseline`.
