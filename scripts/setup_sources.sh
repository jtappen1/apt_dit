#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_ROOT"
mkdir -p vendor
if [[ ! -d vendor/DiT ]]; then
    git clone https://github.com/facebookresearch/DiT.git vendor/DiT
fi
git -C vendor/DiT checkout --detach ed81ce2229091fd4ecc9a223645f95cf379d582b
if [[ ! -d vendor/apt-reference ]]; then
    git clone https://github.com/rccchoudhury/apt.git vendor/apt-reference
fi
git -C vendor/apt-reference checkout --detach 707337ee66c7ff62b820538b40b75341d80e32ef
