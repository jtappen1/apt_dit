# Longer noise-skip training: progress and overfitting

## Observed result

Longer training continues to reduce held-out prediction error, but with diminishing returns. Over 20,000 extra updates, training epsilon MSE falls from 0.06093 to 0.04650 (23.7%), while held-out epsilon MSE falls from 0.06562 to 0.05601 (14.6%). The absolute generalization gap grows from 0.00469 to 0.00951.

There is **no sustained held-out-loss deterioration** up to 23,200 noise-skip updates. The best held-out v-target objective occurs at 22,200; the final value is only about 0.3% higher, one check rather than a sustained reversal. This is not enough to establish overfitting. Nevertheless, training improves faster than development and guided epsilon error bottoms near 17,200 updates before increasing slightly. These are warnings that further training has diminishing generalization benefit and the selected v-target objective does not perfectly track guided predictions.

The selected 22,200-update model improves development LPIPS from 0.7766 to 0.7385 on the same 16 development pairs. Direct inspection still shows corrupted textures/patterns without reliable class/content recognition; the image-quality gate fails. Greater training duration helps fit and paired fidelity, but has not repaired all-patch-4 generation even after roughly seven times as many v-correction updates. This does not prove that still longer training can never help.

Preserve both `best.pt` (22,200, selected by v-target development MSE) and `last.pt` (23,200, slightly lower raw epsilon MSE). The next efficient investigation is fine-detail reconstruction or alignment with guided/rollout error, rather than assuming unlimited continuation alone will produce clean images. No additional architecture or data experiment was run here.


2026-11-02. Continued the strongest recovery-v1 C checkpoint with its optimizer, Python sampling RNG, and PyTorch RNG states restored. Same frozen teacher, existing cache, timestep-balanced paired batches, v-target objective, and AdamW learning rate 1e-4. No data expansion or architecture change. All-patch-4, 50-step DDIM, CFG 4.

Training continued from 3,200 to 23,200 noise-skip updates; the best development objective was at 22,200. Stop reason: `update_cap`. The source trunk also inherits 800 older epsilon/variance updates before the v-correction output was reset; those are not counted as noise-skip updates.

Training and development metrics evaluate all 1,280 training pairs and all 320 held-out development pairs at every check, rather than comparing a noisy training minibatch with a full validation set. Images use the same 16 development pairs as before. The reserved final-test set remains unused.

| Checkpoint | Train v MSE | Held-out v MSE | Train epsilon MSE | Held-out epsilon MSE | Guided epsilon MSE |
|---|---:|---:|---:|---:|---:|
| Start | 0.112248 | 0.122517 | 0.060928 | 0.065617 | 0.075077 |
| Best held-out checkpoint | 0.078675 | 0.097425 | 0.046950 | 0.056082 | 0.068583 |
| Last check | 0.078095 | 0.097731 | 0.046502 | 0.056010 | 0.069081 |

Paired development LPIPS: **0.7766 → 0.7385** (16 images, lower means closer to the same-seed teacher, not distributional image quality). Largest selected-checkpoint latent magnitude: 5.449.

## Training versus generalization

| Noise-skip updates | Train v MSE | Held-out v MSE | Train epsilon MSE | Held-out epsilon MSE |
|---|---:|---:|---:|---:|
| 3200 | 0.112248 | 0.122517 | 0.060928 | 0.065617 |
| 4200 | 0.105362 | 0.117033 | 0.058351 | 0.063373 |
| 5200 | 0.101991 | 0.114016 | 0.056774 | 0.062029 |
| 6200 | 0.098488 | 0.111079 | 0.055402 | 0.060885 |
| 7200 | 0.095667 | 0.108605 | 0.054268 | 0.060041 |
| 8200 | 0.093332 | 0.106548 | 0.053423 | 0.059320 |
| 9200 | 0.092677 | 0.106688 | 0.052851 | 0.058931 |
| 10200 | 0.089836 | 0.104305 | 0.052111 | 0.058376 |
| 11200 | 0.088948 | 0.103329 | 0.051595 | 0.057935 |
| 12200 | 0.087737 | 0.102591 | 0.051153 | 0.057700 |
| 13200 | 0.086323 | 0.101400 | 0.050625 | 0.057407 |
| 14200 | 0.085610 | 0.101254 | 0.050345 | 0.057491 |
| 15200 | 0.084548 | 0.100224 | 0.049851 | 0.057024 |
| 16200 | 0.084035 | 0.099818 | 0.049462 | 0.056913 |
| 17200 | 0.082934 | 0.099279 | 0.049106 | 0.056744 |
| 18200 | 0.082644 | 0.099323 | 0.048708 | 0.056633 |
| 19200 | 0.081227 | 0.098480 | 0.048283 | 0.056452 |
| 20200 | 0.080689 | 0.098410 | 0.047964 | 0.056456 |
| 21200 | 0.079824 | 0.098031 | 0.047531 | 0.056219 |
| 22200 | 0.078675 | 0.097425 | 0.046950 | 0.056082 |
| 23200 | 0.078095 | 0.097731 | 0.046502 | 0.056010 |

The overfitting stop requires three consecutive checks with development v-target error more than 5% above the best value while training error falls. The plateau stop requires five checks without at least 0.2% improvement in development v-target error. A widening train/development gap alone is a generalization warning, not proof that development quality is deteriorating. Read epsilon metrics and images alongside the v-target checkpoint-selection objective.

## Error by timestep

| Timestep | Start held-out epsilon MSE | Selected held-out epsilon MSE | Selected guided MSE |
|---|---:|---:|---:|
| 0-199 | 0.20506841 | 0.18432298 | 0.20810054 |
| 200-399 | 0.11662790 | 0.09225952 | 0.12304816 |
| 400-599 | 0.00615000 | 0.00368629 | 0.01110755 |
| 600-799 | 0.00023391 | 0.00013354 | 0.00062432 |
| 800-999 | 0.00000671 | 0.00000531 | 0.00003209 |

## Verification and limits

- Original pure-fine full CFG trajectory remains bit-exact at the selected checkpoint; raw prediction and final latent maximum errors are both zero.
- All 20 tests pass, including a bit-exact next-update test after restoring optimizer and data-RNG state. The continuation checks source cache record ordering against the saved manifest and preserves old runs.
- Best and last checkpoints contain model, optimizer, RNG, step, architecture version, source hashes, and full metrics.
- No schedule sweep, fresh test sampling, distributional FID, or speed claim. Development data have been inspected during prior iterations; this is an optimization/generalization diagnosis, not an unbiased final result.

Artifacts: `artifacts/training/ddit_recovery_long_c/`. Reproduce with `source scripts/env.sh`, then `.venv/bin/python -m apt_dit.ddit_continue --name <new-run> --additional-steps 20000 --eval-every 1000 --hours .60`.

[Learning curves](ddit_long_training_curves.png) · [Paired images](ddit_long_training_samples.png).
