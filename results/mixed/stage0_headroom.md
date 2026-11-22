# Stage 0: compute headroom of token reduction (trunk only, bf16, CFG)

GPU: NVIDIA B200

Upper bound for any token-reducing schedule: real transformer blocks at reduced token counts; excludes embedding/head/selector/decoder overhead.

| Model | Images | Tokens (frac) | Trunk TFLOPs | Attention share | Trunk time | Speedup vs full |
|---|---:|---:|---:|---:|---:|---:|
| dit_s_256 | 1 | 256 (1) | 0.024 | 10% | 3.82 ms | 1.00× |
| dit_s_256 | 1 | 192 (0.75) | 0.018 | 8% | 3.89 ms | 0.98× |
| dit_s_256 | 1 | 128 (0.5) | 0.012 | 5% | 3.86 ms | 0.99× |
| dit_s_256 | 1 | 64 (0.25) | 0.006 | 3% | 3.84 ms | 0.99× |
| dit_s_256 | 1 | 16 (0.0625) | 0.001 | 1% | 3.84 ms | 1.00× |
| dit_s_256 | 4 | 256 (1) | 0.097 | 10% | 3.88 ms | 1.00× |
| dit_s_256 | 4 | 192 (0.75) | 0.071 | 8% | 3.86 ms | 1.01× |
| dit_s_256 | 4 | 128 (0.5) | 0.046 | 5% | 3.87 ms | 1.00× |
| dit_s_256 | 4 | 64 (0.25) | 0.023 | 3% | 3.84 ms | 1.01× |
| dit_s_256 | 4 | 16 (0.0625) | 0.006 | 1% | 3.90 ms | 1.00× |
| pixart_alpha_1024 | 1 | 4096 (1) | 13.015 | 34% | 21.12 ms | 1.00× |
| pixart_alpha_1024 | 1 | 3072 (0.75) | 8.959 | 28% | 15.46 ms | 1.37× |
| pixart_alpha_1024 | 1 | 2048 (0.5) | 5.443 | 21% | 13.15 ms | 1.61× |
| pixart_alpha_1024 | 1 | 1024 (0.25) | 2.469 | 12% | 13.45 ms | 1.57× |
| pixart_alpha_1024 | 1 | 256 (0.0625) | 0.593 | 4% | 13.13 ms | 1.61× |
| pixart_alpha_1024 | 4 | 4096 (1) | 52.061 | 34% | 82.56 ms | 1.00× |
| pixart_alpha_1024 | 4 | 3072 (0.75) | 35.834 | 28% | 57.75 ms | 1.43× |
| pixart_alpha_1024 | 4 | 2048 (0.5) | 21.772 | 21% | 35.04 ms | 2.36× |
| pixart_alpha_1024 | 4 | 1024 (0.25) | 9.875 | 12% | 17.36 ms | 4.76× |
| pixart_alpha_1024 | 4 | 256 (0.0625) | 2.373 | 4% | 13.15 ms | 6.28× |
| pixart_sigma_2048 | 1 | 16384 (1) | 104.720 | 67% | 148.64 ms | 1.00× |
| pixart_sigma_2048 | 1 | 12288 (0.75) | 65.574 | 61% | 96.34 ms | 1.54× |
| pixart_sigma_2048 | 1 | 8192 (0.5) | 35.087 | 51% | 52.80 ms | 2.82× |
| pixart_sigma_2048 | 1 | 4096 (0.25) | 13.259 | 35% | 21.73 ms | 6.84× |
| pixart_sigma_2048 | 1 | 1024 (0.0625) | 2.570 | 14% | 13.22 ms | 11.24× |
| pixart_sigma_2048 | 4 | 16384 (1) | 418.881 | 67% | 581.50 ms | 1.00× |
| pixart_sigma_2048 | 4 | 12288 (0.75) | 262.298 | 61% | 372.47 ms | 1.56× |
| pixart_sigma_2048 | 4 | 8192 (0.5) | 140.350 | 51% | 206.33 ms | 2.82× |
| pixart_sigma_2048 | 4 | 4096 (0.25) | 53.036 | 35% | 84.35 ms | 6.89× |
| pixart_sigma_2048 | 4 | 1024 (0.0625) | 10.280 | 14% | 17.59 ms | 33.05× |

**Gate** (PixArt-alpha 1024, bf16, 1 image (CFG batch 2): trunk speedup at 50% tokens >= 1.5x): speedup 1.61× → PASSED: build Stages 1–2.
