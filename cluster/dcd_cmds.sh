#!/bin/bash
# Expand DCD grid configs into one command per line (seeds via --num_trials), using DCD's
# own train_scripts/make_cmd.py. Configs in cluster/dcd_configs/<domain-dir> (cells upstream lacks,
# derived from an upstream config by one or two keys) take precedence. Usage: dcd_cmds.sh <domain-dir> <xvfb:0|1> <config>... > cmds.txt
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); source "$here/env.sh"
dom=$1 xvfb=$2; shift 2
cd "$DCD_DIR"
for c in "$@"; do
  d=train_scripts/grid_configs/$dom; [ -f "$here/dcd_configs/$dom/${c%.json}.json" ] && d=$here/dcd_configs/$dom
  ${PYTHON:-python3} train_scripts/make_cmd.py --dir "$d" --json $c \
    --num_trials ${NSEEDS:-5} $([ "$xvfb" = 1 ] && echo --xvfb) \
  | sed -e ':a' -e '/\\$/N; s/ *\\\n */ /; ta' \
  | grep -E '^(xvfb-run|python)' | sed "s|\$| ${DCD_ARGS:-} --log_dir=$RUNS/dcd/$dom|"
done
