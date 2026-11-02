# Adaptive patching on DiT-S

Experiments adapting a pretrained ImageNet-256 DiT-S/2 to spatial and temporal patch-size changes.

**Start with [EXPERIMENTS.md](EXPERIMENTS.md)** for the complete experiment history, current conclusions, checkpoint locations, and possible next work. Detailed reports, measurements, and figures are in `results/`. The original proposal is [cis_6220_project.pdf](cis_6220_project.pdf); source revisions and checkpoint identity are in [PROVENANCE.md](PROVENANCE.md).

The baseline works. Fine-information decoders improve spatially compressed images, but artifacts remain. Global patch-4 generation still fails after longer training. No adapted method has demonstrated useful quality together with measured acceleration.

## Setup

Run from this directory. Use a project-local environment; do not change global drivers or configuration.

```bash
source scripts/env.sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
bash scripts/setup_sources.sh
mkdir -p artifacts/checkpoints
.venv/bin/gdown --no-cookies 'https://drive.google.com/uc?id=1YOSNrFbP36vHlPQPBosqZKbPLokymRuO' -O artifacts/checkpoints/dit_s_256.pt
.venv/bin/python -m pytest -q
.venv/bin/python -m apt_dit.evaluate verify
```

The lock records the tested packages; its CUDA build requires a compatible driver. The environment script keeps caches and temporary files inside the project. Pinned upstream implementations are fetched into ignored `vendor/`; checkpoints and raw rollouts stay in ignored `artifacts/`. They are not included in this repository.

## Code

- `src/apt_dit/patching.py`: original APT-inspired mixed-token model, detail branch, and fine-information decoder.
- `src/apt_dit/train.py`, `reconstruct.py`, `onpolicy_cache.py`: original spatial adaptation and student-state teacher distillation.
- `src/apt_dit/ddit.py`: corrected patch-specific DDiT-style adapter and FFN LoRA.
- `src/apt_dit/ddit_recovery.py`, `ddit_continue.py`: paired recovery controls, explicit noise skip, and optimizer/RNG continuation.
- `src/apt_dit/sampling.py`, `ddit_evaluate.py`, `ddit_audit.py`: sampling, checkpoint-aware loading, and diagnostics.
- `scripts/report_ddit_recovery.py`, `report_ddit_long.py`: reproduce reports from saved local run artifacts.
- `tests/`: geometry, fine-path parity, conditioning, sampler semantics, and optimizer restoration checks; 20 tests pass.

See [the experiment record](EXPERIMENTS.md#reproducibility-and-local-checkpoints) before launching dependent runs. Use new run names, preserve old artifacts, and retain the serial eight-hour GPU-process budget enforced by `runtime.py`. No external tracking service is required.
