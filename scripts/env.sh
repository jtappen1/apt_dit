# Source this file before installing packages or running experiments.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$PROJECT_ROOT/.cache/cuda"
export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/.cache/pycache"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
