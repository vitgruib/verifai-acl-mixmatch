#!/bin/bash
# The whole Maze experiment, ending in maze_results.tgz. Rerunning resumes; finished runs are kept.
#   SLURM:  bash run/maze.sh   submits everything (smoke test, 60 runs + reference agents, exam,
#           grading, packing) as dependent jobs; extra flags in SBATCH_ARGS, e.g. "--account=x".
#   Local:  CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/maze.sh   runs it here on those GPUs:
#           run/train.sh, then run/finish.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v sbatch >/dev/null; then exec bash cluster/submit.sh maze; fi
bash run/train.sh
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} bash run/finish.sh
