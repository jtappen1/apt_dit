#!/usr/bin/env bash
set -x
cd /vast/home/m/maxmerc/apt_dit
until grep -q "^done" runs/train_lora32.log; do sleep 15; done
source scripts/env.sh
C=artifacts/training/mixed_lora32/last.pt
P=".venv/bin/python -m apt_dit.mixed_eval"
$P calibrate --checkpoint $C --tag lora32 --points 10 > runs/eval_calibrate.log 2>&1
$P gap --checkpoint $C --tag lora32 --n 256 --batch 256 --sizes 8 4 > runs/eval_gap.log 2>&1 &
$P sweep --checkpoint $C --tag lora32 --n 5000 --n-lpips 1000 --batch 1000 > runs/eval_sweep.log 2>&1
wait
$P efficiency --checkpoint $C --tag lora32 --batch-sizes 1 256 --reps 10 > runs/eval_efficiency.log 2>&1
.venv/bin/python scripts/report_mixed.py lora32 > runs/report.log 2>&1
echo ALL_DONE
