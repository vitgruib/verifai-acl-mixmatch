#!/bin/bash
# Trains the 10 reference DR agents (seeds 105-109, then 100-104; 100-104 are copied from the repo
# or fetched from GitHub when they are there, and trained otherwise). Rerunning resumes.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p runs/slurm
exec > >(tee -a runs/slurm/run_ref.log) 2>&1
for i in $(seq 0 9); do ALGOS=ref SLURM_ARRAY_TASK_ID=$i bash cluster/1_maze/train.sbatch; done
