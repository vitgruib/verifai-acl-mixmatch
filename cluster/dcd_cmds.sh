#!/bin/bash
# Expand DCD grid configs into one command per line (seeds via --num_trials), using DCD's
# own train_scripts/make_cmd.py. Usage: dcd_cmds.sh <domain-dir> <xvfb:0|1> <config>... > cmds.txt
set -euo pipefail
source "$(dirname "$0")/env.sh"
dom=$1 xvfb=$2; shift 2
cd "$DCD_DIR"
for c in "$@"; do
  python train_scripts/make_cmd.py --dir train_scripts/grid_configs/$dom --json $c \
    --num_trials ${NSEEDS:-10} $([ "$xvfb" = 1 ] && echo --xvfb) \
  | sed -e ':a' -e '/\\$/N; s/ *\\\n */ /; ta' \
  | grep -E '^(xvfb-run|python)' | sed "s|\$| --log_dir=$RUNS/dcd/$dom|"
done
