#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/apt_dit"
run=runs/xl512_lora32_10k
artifacts=artifacts/xl512_lora32_10k
scripts=scripts/xl512_lora32_10k
export OMP_NUM_THREADS=4 PYTHONPATH=src
python_bin="$HOME/envs/apt-dit/bin/python"
exec 9> "$run/pipeline.lock"
flock -n 9
mkdir -p "$artifacts"
for checkpoint in latest.pt best.pt; do
    if [[ -f "$run/$checkpoint" ]]; then
        if [[ -e "$artifacts/$checkpoint" ]]; then
            printf 'Checkpoint exists in both locations: %s\n' "$checkpoint" >&2
            exit 1
        fi
        mv -- "$run/$checkpoint" "$artifacts/$checkpoint"
    fi
done
printf '%s Starting 10,000-sample pipeline. Follow teacher.log during data generation.\n' "$(date -Is)" >> "$run/train.log"
if [[ ! -f "$artifacts/teacher.complete" ]]; then
    "$python_bin" -u "$scripts/generate_teacher.py" >> "$run/teacher.log" 2>&1
fi
printf '%s Teacher cache ready. Starting or resuming LoRA training.\n' "$(date -Is)" >> "$run/train.log"
exec "$python_bin" -u "$scripts/train.py" >> "$run/train.log" 2>&1
