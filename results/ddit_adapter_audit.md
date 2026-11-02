# DiT-S patch-4 audit and stop decision

The corrected DiT-S port fails the image-quality gate. Stop this adaptation under the current setup; do not interpret schedule comparisons or launch further schedule sweeps. Four matched final-set class/seed pairs, sampled for 50 all-patch-4 DDIM steps, produce unrecognizable saturated patterned noise. See [the preview](ddit_audit_corrected_p4.png). Passing a one-step MSE threshold was insufficient for stable iterative generation.

## Verified implementation findings

- The coarse head skipped the frozen DiT final LayerNorm and timestep/class AdaLN modulation. It now applies the same conditioning as the fine head before its patch-specific linear projection.
- The DDiT sampler passed evolving conditional/unconditional states directly to the model. Their fourth epsilon channel is intentionally unguided upstream, so states can diverge. It now duplicates the conditional latent at every step, matching pinned upstream `forward_with_cfg`; labels remain `[class_id, 1000]`. CFG still guides only channels 0–2, preserving upstream behavior; epsilon is channels 0–3 and learned variance is channels 4–7.
- Synthetic row/column/channel-coded predictions verify upstream `[patch_row, patch_col, channel]` packing and both latent assembly paths. Two different mixed masks verify every pixel has exactly the corresponding fine/coarse owner, without overlap, gaps, transposition, or neighboring-token swaps.
- Replaced average-of-four embedding weights with `W4 = W2 @ pinv(R)`, where R is the explicit 2x2-to-4x4 bilinear pixel interpolation matrix (`align_corners=False`). A numerical test verifies embedding equivalence on interpolated patches. This follows the pseudo-inverse initialization described in [DDiT section 4.1](https://arxiv.org/html/2602.16968v1#S4.SS1). The interpolation convention is explicitly our implementation choice; the paper does not specify it.
- The de-embedding initialization retains the pretrained 2x2 head at each of four subpatch offsets. Its repeated pattern is a prior, not a packing bug; no unsupported visual reordering was introduced.
- The learned zero-initialized 3x3 residual maps four latent channels to eight raw prediction channels after spatial assembly. It is neither a latent identity skip nor a pixel-space residual. It affects all pixels on a mixed call, including fine regions. This is a DiT learned-variance adaptation choice: the paper's noise-prediction models do not specify this eight-channel mapping. The pure patch-2 path bypasses it and disables LoRA.
- Only new patch components, patch identifier, rank-32 FFN LoRA, and the residual are trainable. The base parameters remain frozen. Tests verify exact patch-2 preservation even after adapter parameters change.

## Measurements

All held-out measurements use all 640 cached conditional/unconditional records from 32 disjoint validation trajectories. Corrected training used the existing 2,560 training records, AdamW at 1e-4, batch 8, seed 6220, and 800 updates. No cache expansion. Best subset validation MSE decreased from 0.8009 to 0.4989. Full-set results below use that checkpoint; legacy results reproduce the original unnormalized coarse head from the 6,000-step checkpoint, so differing training lengths prevent attributing all differences to the fixes.

| Predictor | Epsilon MSE | Epsilon target variance | Variance-channel MSE | Variance target variance |
|---|---:|---:|---:|---:|
| Legacy 6,000-step | 0.43023 | 0.82808 | 3.94774 | 0.88187 |
| Corrected 800-step validation | 0.65489 | 0.82808 | 0.34445 | 0.88187 |
| Corrected 800-step train | 0.64915 | 0.82133 | 0.29690 | 0.85300 |

| Timestep | Corrected epsilon MSE / target variance | Corrected variance MSE / target variance |
|---|---:|---:|
| 0–199 | 0.28656 / 0.31904 | 1.67266 / 3.65949 |
| 200–399 | 0.66228 / 0.84098 | 0.03597 / 0.07462 |
| 400–599 | 0.74711 / 0.97238 | 0.00820 / 0.01865 |
| 600–799 | 0.78609 / 1.00251 | 0.00288 / 0.01094 |
| 800–999 | 0.79239 / 1.00549 | 0.00256 / 0.01056 |

Corrected epsilon low/high-frequency error power is 0.54574 / 0.68562; variance low/high-frequency error power is 0.53886 / 0.28971. Frequencies are split at max absolute axis frequency 0.25 cycles/pixel using an orthonormal 2D FFT. Each band is normalized by its own number of coefficients. Complete per-timestep spectra and legacy metrics are in the JSON files. Close train/validation errors indicate underfitting/optimization or adaptation limits, rather than demonstrated cache overfitting.

Patch-2 versus the frozen teacher at the same batch size is bit-exact (maximum absolute error 0). Versus cached targets generated at batch size two, MSE is 7.36e-13 and maximum absolute error is 1.58e-4; this is numerical batch-kernel variation, not exact cached equality. No teacher precision was changed.

## Gate and preserved evidence

- Frozen-teacher preservation: pass.
- Both channel groups below target variance overall and in all five timestep bins: pass.
- Recognizable all-patch-4 images without corrupted patterns: **fail**, by direct inspection of the four-image preview.
- Overall: **stop**. No new schedule comparison is justified, and this run does not settle late-coarsening effectiveness.

The four original artifact trees named in the plan were not modified. Historical schedule/preview reports are labeled failed adapter pilots, with their original measurements retained. New training evidence is isolated in `artifacts/training/ddit_s_p4_corrected_slice/`.

Reproduce with `source scripts/env.sh`, then `.venv/bin/python -m pytest -q` (16 passed), `python -m apt_dit.ddit_train --steps 800 --batch-size 8 --hours .25 --run-name <new-name>`, and `python -m apt_dit.ddit_audit --checkpoint artifacts/training/<new-name>/best.pt --name <new-audit-name> --preview`. Use `--split train` for training-set diagnostics. To audit historical weights faithfully use `--legacy-head`; loading historical weights into the corrected forward without this flag changes their behavior.

Diagnostic arrays: `ddit_audit_corrected.npz` and `ddit_audit_legacy.npz` contain the first eight held-out predictions/targets/timesteps. JSON files contain aggregates over every record, not just these eight examples. GPU work is charged to the existing serial project ledger.
