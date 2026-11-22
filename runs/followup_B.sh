#!/usr/bin/env bash
#SBATCH -J apt_followupB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o runs/followupB_%j.log
# Follow-up B only (A and Stage 0 are complete). Damage maps run alone (no concurrent sweep).
set -x
ROOT="${APT_DIT_ROOT:-}"
if [ -z "$ROOT" ]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
  [ -d "$ROOT/src/apt_dit" ] || ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
fi
cd "$ROOT" || exit 1
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
P=".venv/bin/python -m"
mkdir -p runs/fuB_failed && mv runs/fu_damage.log runs/fu_oracle.log runs/fuB_failed/ 2>/dev/null
$P apt_dit.mixed_followup damage --batch ${B_BATCH:-250} > runs/fu_damage.log 2>&1
$P apt_dit.mixed_followup oracle --batch ${B_BATCH:-250} > runs/fu_oracle.log 2>&1
if [ ! -f artifacts/mixed_eval/followup/oracle.json ]; then
  echo "B1 FAILED TO RUN (no oracle.json) - error, not a no-go"
elif grep -q '"go": true' artifacts/mixed_eval/followup/oracle.json; then
  $P apt_dit.mixed_followup train --steps 3000 > runs/fu_train.log 2>&1
  $P apt_dit.mixed_followup learned --batch ${B_BATCH:-250} > runs/fu_learned.log 2>&1
else
  echo "B1 no-go (pre-registered stop rule): learned selector not run"
fi
echo FOLLOWUP_B_DONE
