#!/bin/bash
# Usage: bash run/train.sh <first seed> <last seed>   Trains the six methods on Maze for those seeds
# (0-9), one run at a time. Finished runs are skipped, so rerunning resumes.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p runs/slurm
exec > >(tee -a "runs/slurm/run_train_$1_to_$2.log") 2>&1
for seed in $(seq "$1" "$2"); do
  for algo in sfl dr plr rplr accel minimax; do
    ALGOS=$algo SLURM_ARRAY_TASK_ID=$seed bash cluster/1_maze/train.sbatch
  done
done
