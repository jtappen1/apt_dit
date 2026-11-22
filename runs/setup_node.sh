#!/usr/bin/env bash
set -euxo pipefail
cd /vast/home/m/maxmerc/apt_dit
source scripts/env.sh
export UV_CACHE_DIR=$PROJECT_ROOT/.cache/uv
~/.local/bin/uv venv --python /usr/bin/python3.12 .venv
~/.local/bin/uv pip install --python .venv/bin/python -r requirements.lock || ~/.local/bin/uv pip install --python .venv/bin/python torch torchvision timm diffusers accelerate safetensors numpy scipy pillow lpips gdown matplotlib pytest
~/.local/bin/uv pip install --python .venv/bin/python --no-deps -e .
~/.local/bin/uv pip install --python .venv/bin/python pytorch-fid
bash scripts/setup_sources.sh
mkdir -p artifacts/checkpoints artifacts/fid
[ -f artifacts/checkpoints/dit_s_256.pt ] || .venv/bin/gdown --no-cookies 'https://drive.google.com/uc?id=1YOSNrFbP36vHlPQPBosqZKbPLokymRuO' -O artifacts/checkpoints/dit_s_256.pt
sha256sum artifacts/checkpoints/dit_s_256.pt
[ -f artifacts/fid/VIRTUAL_imagenet256_labeled.npz ] || curl -sSL -o artifacts/fid/VIRTUAL_imagenet256_labeled.npz https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
ls -la artifacts/fid artifacts/checkpoints
.venv/bin/python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest -q 2>&1 | tail -5
.venv/bin/python -m apt_dit.evaluate verify
echo SETUP_DONE
