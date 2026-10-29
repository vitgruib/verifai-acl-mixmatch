#!/bin/bash
# After run/train.sh: builds the exam from the reference agents, grades the 60 runs on it and packs
# maze_results.tgz. Uses the first GPU in CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p runs/slurm
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
bash cluster/1_maze/exam.sbatch 2>&1 | tee -a runs/slurm/exam.out
bash cluster/1_maze/grade.sbatch 2>&1 | tee -a runs/slurm/grade.out
bash cluster/collect.sh "$PWD/maze_results.tgz"
