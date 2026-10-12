#!/bin/bash
# The whole Maze run, from the login node. Arguments are extra sbatch flags, e.g.
#   bash cluster/run_maze.sh -p gpu -A lab
# Checks the flags, sets up (~15 min), then submits.sh maze: smoke test -> full run (only if the
# smoke test succeeds) -> results packed into maze_results.tgz at the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
export SBATCH_ARGS="${SBATCH_ARGS:-} $*"
mkdir -p runs/slurm
sbatch --test-only $SBATCH_ARGS cluster/1_maze/train.sbatch   # fail now, not after setup
bash cluster/setup/clone.sh
bash cluster/setup/setup_jax.sh
bash cluster/setup/setup_sfl.sh
bash cluster/submit.sh maze
echo
echo "Submitted. When the run ends (about a day), send me this file: $PWD/maze_results.tgz"
