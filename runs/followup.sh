#!/usr/bin/env bash
#SBATCH -J apt_followup
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -o runs/followup_%j.log
set -x
# Repo root: $APT_DIT_ROOT, else this script's parent dir, else the sbatch submit dir.
ROOT="${APT_DIT_ROOT:-}"
if [ -z "$ROOT" ]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
  [ -d "$ROOT/src/apt_dit" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
fi
cd "$ROOT" || exit 1
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
P=".venv/bin/python -m"
C=artifacts/training/mixed_lora32/last.pt
# C / Stage 0: compute headroom at 256/1024/2048 (independent; failure does not block A/B)
[ -f results/mixed/stage0_headroom.json ] || $P apt_dit.pixart_headroom > runs/fu_stage0.log 2>&1 || true
mkdir -p artifacts/mixed_eval/aggr
# A: finer calibration on disjoint seeds
[ -f artifacts/mixed_eval/aggr/calibration.json ] || $P apt_dit.mixed_eval calibrate --checkpoint $C --tag aggr --points 30 --cal-n 256 --cal-offset 510000 > runs/fu_calibrate.log 2>&1
# B damage maps run concurrently with the A sweep
$P apt_dit.mixed_followup damage --batch ${B_BATCH:-500} > runs/fu_damage.log 2>&1 &
DPID=$!
$P apt_dit.mixed_eval sweep --checkpoint $C --tag aggr --n 10000 --n-lpips 10000 --seed-offset 800000 \
   --targets 130 110 90 70 50 --teacher-steps --switch-k --skip-extras --batch ${SWEEP_BATCH:-1000} > runs/fu_sweep.log 2>&1
wait $DPID
$P apt_dit.mixed_followup oracle --batch ${B_BATCH:-500} > runs/fu_oracle.log 2>&1
if [ ! -f artifacts/mixed_eval/followup/oracle.json ]; then
  echo "B1 FAILED TO RUN (no oracle.json) - this is an error, not a no-go; see runs/fu_damage.log and runs/fu_oracle.log"
elif grep -q '"go": true' artifacts/mixed_eval/followup/oracle.json; then
  $P apt_dit.mixed_followup train --steps 3000 > runs/fu_train.log 2>&1
  $P apt_dit.mixed_followup learned --batch ${B_BATCH:-500} > runs/fu_learned.log 2>&1
else
  echo "B1 no-go: learned selector not run (pre-registered stop rule)"
fi
echo FOLLOWUP_DONE
