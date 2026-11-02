# dense_output_onpolicy: paired DiT-S evaluation

Rows in samples.png follow the table order; columns share seeds/classes.

| Arm | Samples | Repeats | LPIPS to baseline | Latent MSE | Median denoise s | Mean tokens | Model GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial | 64 | 1 | 0.6598 | 0.6467 | 0.222 | 167.7 | 777.1 |

LPIPS measures agreement with the original model, not distributional quality. No FID claim.
Model FLOPs are analytical (2 FLOPs/MAC); exclude elementwise, resize, selector, and VAE work. Latency includes selector and resize overhead.
These timings include Python checks; use the dedicated repeated benchmark for a latency conclusion.

Artifacts: `artifacts/dense_output_onpolicy`.
