# Dense output reconstruction: first results

The head decodes a distinct output correction for each fine-grid token from the packed transformer features and the original fine embeddings. It was trained on the existing cached teacher predictions; the DiT and prior detail branch remained frozen.

## Held-out prediction check

At eight selected coarse blocks, 96 validation states gave repeated-output noise MSE **0.338** and decoded MSE **0.047**. In the selected coarse regions, MSE fell from **0.667** to **0.085**. This is a substantial prediction improvement on cached held-out teacher states.

## Paired sampling check

On the same 32 seeds/classes and the same spatial schedule, LPIPS distance to the original DiT samples averaged **0.878** with repeated outputs and **0.714** with the decoder. The paired change (decoder minus repeated) was **-0.164** (95% paired t interval **[-0.189, -0.139]**). Median denoising time was **0.216s** versus **0.224s**.

The decoder improves similarity substantially, but LPIPS remains high and generated images still contain visible distortion/noise. These are preliminary paired samples, not a distribution-quality result; no FID was run.

See `reconstruction_samples.png` for baseline, previous repeated-output, and dense-decoder samples in matching columns. Detailed paired metrics are in `reconstruction_comparison.json`.
