#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/maze.sh
# The whole Maze experiment on the GPUs listed: run/train.sh (all runs), then run/finish.sh (exam,
# grading, maze_results.tgz). Rerunning resumes; finished runs are kept.
set -euo pipefail
cd "$(dirname "$0")/.."
bash run/train.sh
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} bash run/finish.sh
