# dense_output_onpolicy_k4_adapted: paired DiT-S evaluation

Rows in samples.png follow the table order; columns share seeds/classes.

| Arm | Samples | Repeats | LPIPS to baseline | Latent MSE | Median denoise s | Mean tokens | Model GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial | 64 | 1 | 0.3860 | 0.1962 | 0.223 | 211.8 | 994.6 |

LPIPS measures agreement with the original model, not distributional quality. No FID claim.
Model FLOPs are analytical (2 FLOPs/MAC); exclude elementwise, resize, selector, and VAE work. Latency includes selector and resize overhead.
These timings include Python checks; use the dedicated repeated benchmark for a latency conclusion.

Artifacts: `artifacts/dense_output_onpolicy_k4_adapted`.
