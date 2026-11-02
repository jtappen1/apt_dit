# Count-4 decoder fine-tuning

A separate on-policy cache was collected with four coarse blocks, then the decoder was fine-tuned for 1,000 steps on those rollout states. The DiT and detail branch stayed frozen. Evaluation uses 64 final seeds/classes.

| Model and budget | LPIPS to baseline | Mean tokens/step | Estimated GFLOPs/sample* | Mean denoising s |
|---|---:|---:|---:|---:|
| Count-8 on-policy decoder, evaluated at 8 | 0.6598 | 167.7 | 777.1 | 0.223 |
| Count-8 decoder, transferred to 4 | 0.4427 | 211.8 | 994.6 | 0.224 |
| Count-4-specific on-policy decoder | 0.3860 | 211.8 | 994.6 | 0.224 |

Paired LPIPS change for count-4-specific minus transferred count-8 decoder: **-0.0567** (95% t interval **[-0.0722, -0.0412]**).

*The GFLOP estimate omits the dense decoder. Lower LPIPS means closer to the same-seed baseline; it is not FID. See `k4_finetune_samples.png` and `k4_finetune_comparison.json`.
