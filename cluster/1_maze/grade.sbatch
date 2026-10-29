#!/bin/bash
# Grades every scored Maze run on the frozen exam into scores.csv (atlas.maze.grade). Rerunning
# grades only the missing rows; it exits non-zero if a run is missing or unfinished.
#SBATCH --job-name=atlas-1-maze-grade
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
N=${NUM_UPDATES:-30000}; NSEEDS=${NSEEDS:-10}
root=$(out_root maze_full)   # env.sh: the repo for the full run, else under $RUNS/outputs
EXAM_OUT=${EXAM_OUT:-$root/frozen_sets/suite/maze}; SCORES=${SCORES:-$root/results/suite/maze/scores.csv}
# runs off the full settings (smoke test, tiny test runs) are graded at their own budget
any=""; maze_full || any=--any-budget
source "$JAX_VENV/bin/activate"
python -m atlas.maze.grade --runs "$RUNS" --exam "$EXAM_OUT" --out "$SCORES" --run-name "maze_{algo}_n$N" \
  --seeds $(seq 0 $((NSEEDS - 1))) $any ${GRADE_ARGS:-}
