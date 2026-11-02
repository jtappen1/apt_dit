# Selector and compression controls

All arms use the same on-policy decoder or its count-4-specific fine-tune, 64 final seeds/classes, 50-step DDIM, CFG 4, and four fine warmup steps.

| Arm | Mean LPIPS to baseline | Mean tokens/step | Mean estimated GFLOPs/sample* | Mean denoising seconds |
|---|---:|---:|---:|---:|
| Clean score, 8 coarse | 0.6598 | 167.7 | 777.1 | 0.223 |
| Trajectory score, 8 coarse | 0.6426 | 167.7 | 777.1 | 0.228 |
| Clean score, 4 coarse, transferred decoder | 0.4427 | 211.8 | 994.6 | 0.224 |
| Clean score, 4-coarse-specific decoder | 0.3860 | 211.8 | 994.6 | 0.224 |

Paired difference for trajectory-8 minus clean-8 LPIPS: **-0.0172**, 95% t interval **[-0.0236, -0.0109]**. Four-coarse-specific minus transferred count-8 decoder LPIPS: **-0.0567**, 95% t interval **[-0.0722, -0.0412]**.

*The GFLOP estimate omits the dense decoder. LPIPS is same-seed fidelity, not FID. The matched montage is `selector_budget_samples.png`; per-seed values are in `selector_budget_comparison.json`.
