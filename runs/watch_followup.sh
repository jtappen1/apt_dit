#!/usr/bin/env bash
# Run the follow-up on whichever GPU job starts first: queued job 10954676 or hold job 10910187.
cd /vast/home/m/maxmerc/apt_dit
while true; do
  S1=$(squeue -h -j 10954676 -o %T 2>/dev/null); S2=$(squeue -h -j 10910187 -o %T 2>/dev/null)
  if [ "$S1" = "RUNNING" ] || [ -z "$S1" ]; then echo "$(date) job 10954676 state='$S1'; watcher exits"; exit 0; fi
  if [ "$S2" = "RUNNING" ]; then
    echo "$(date) hold job started first; cancelling 10954676 and running inside hold"
    scancel 10954676
    srun --jobid=10910187 --overlap -N1 -n1 -c 16 runs/followup.sh > runs/followup_hold.log 2>&1
    exit 0
  fi
  sleep 300
done
