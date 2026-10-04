#!/bin/bash
# Run line $SLURM_ARRAY_TASK_ID (0-based) of a DCD command file inside the DCD conda env.
set -euo pipefail
source "$(dirname "$0")/env.sh"
eval "$(conda shell.bash hook)"; conda activate "$DCD_CONDA"
cmd=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "$1")
[ -n "$cmd" ] || { echo "no line ${SLURM_ARRAY_TASK_ID} in $1"; exit 1; }
mkdir -p "$RUNS/dcd"; cd "$DCD_DIR"
echo "host=$(hostname) cmd=$cmd"
eval "$cmd"
