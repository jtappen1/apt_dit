# Sources and reproducibility

| Component | Source | Revision / identifier |
|---|---|---|
| DiT model and diffusion code | https://github.com/facebookresearch/DiT | `ed81ce2229091fd4ecc9a223645f95cf379d582b` |
| APT reference, inspected but not imported | https://github.com/rccchoudhury/apt | `707337ee66c7ff62b820538b40b75341d80e32ef` |
| DiT-S weights | https://github.com/NVlabs/T-Stitch/tree/main/dit | Google Drive `1YOSNrFbP36vHlPQPBosqZKbPLokymRuO` |
| Checkpoint SHA-256 | Downloaded DiT-S artifact | `979142b209a6b6b5c9d63e025188abbe6377df7c43097321a31f7b8e5217bd58` |
| VAE | `stabilityai/sd-vae-ft-ema` | Resolved HF revision in `artifacts/vae.json` |

Upstream licenses remain in their downloaded repositories. DiT's upstream code is used directly without edits; consult `vendor/DiT/LICENSE.txt`. T-Stitch lists its licensing in https://github.com/NVlabs/T-Stitch/blob/main/LICENSE.txt. Model artifacts retain their source terms.

APT references: https://arxiv.org/html/2510.18091v2, sections 3.1–3.3 and 4.2. The inspected `patch_tokenizer.py` resizes with bilinear interpolation. `TokenizedZeroConvPatchAttn` in `patch_embed.py` uses a 2×2 convolution followed by a zero-initialized linear residual and timm positional resampling. The pilot omits unused prefix/local-position parameters and supports only scales 2 and 4. Its dense-head repetition follows the paper's dense-prediction construction, applied experimentally to DiT.

The first run uses FP32 after exact all-fine parity was verified. Optional LoRA and the fixed-budget latent scoring rules are project additions, not attributed to the original APT implementation.
