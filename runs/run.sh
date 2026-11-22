#!/usr/bin/env bash
# Usage: runs/run.sh <logname> <python args...>  -- runs on the held B200 node
cd /vast/home/m/maxmerc/apt_dit
source scripts/env.sh
log=runs/$1.log; shift
exec .venv/bin/python "$@"
