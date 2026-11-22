#!/usr/bin/env bash
cd /vast/home/m/maxmerc/apt_dit
while true; do
  S1=$(squeue -h -j 11053974 -o %T 2>/dev/null); S2=$(squeue -h -j 10910187 -o %T 2>/dev/null)
  if [ "$S1" = "RUNNING" ] || [ -z "$S1" ]; then echo "$(date) job 11053974 state='$S1'; watcher exits"; exit 0; fi
  if [ "$S2" = "RUNNING" ]; then
    echo "$(date) hold started first; cancelling 11053974, running B inside hold"
    scancel 11053974
    srun --jobid=10910187 --overlap -N1 -n1 -c 16 runs/followup_B.sh > runs/followupB_hold.log 2>&1
    exit 0
  fi
  sleep 300
done
