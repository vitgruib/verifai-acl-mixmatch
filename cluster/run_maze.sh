#!/bin/bash
# Full Maze run: setup, smoke test, full run, results packed into maze_results.tgz.
# On SLURM, arguments are extra sbatch flags. Without SLURM it runs in the background on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
setup() { bash cluster/setup/clone.sh; bash cluster/setup/setup_jax.sh; bash cluster/setup/setup_sfl.sh; }
out=$PWD/maze_results.tgz
command -v "${PYBIN:-python3.11}" >/dev/null || { echo "Needs python3.11 (or PYBIN=/path/to/python3.11)." >&2; exit 1; }

if command -v sbatch >/dev/null; then
  export SBATCH_ARGS="${SBATCH_ARGS:-} $*"
  mkdir -p runs/slurm
  sbatch --test-only $SBATCH_ARGS cluster/1_maze/train.sbatch   # catch bad flags before setup
  setup
  bash cluster/submit.sh maze
  echo
  echo "Submitted. When it finishes (a few hours once the jobs start), send me: $out"
  exit 0
fi

if [ -z "${RUN_MAZE_BG:-}" ]; then
  if ! nvidia-smi -L >/dev/null 2>&1 && [ -z "${ALLOW_CPU:-}" ]; then
    echo "No NVIDIA GPU found (nvidia-smi). On CPU this run takes over a month; ALLOW_CPU=1 runs it anyway." >&2
    exit 1
  fi
  RUN_MAZE_BG=1 nohup bash "$PWD/cluster/run_maze.sh" "$@" > run_maze.log 2>&1 < /dev/null &
  echo "Running in the background. Progress and ETA: tail -f $PWD/run_maze.log"
  echo "When it finishes (about 4 days on one GPU), send me: $out"
  exit 0
fi

echo "start $(date)"
nvidia-smi -L 2>/dev/null || export JAX_EXTRA=""
setup
if LOCAL=1 bash cluster/submit.sh maze; then
  echo "done $(date): send me $out"
else
  echo "FAILED $(date); packing the logs anyway"
  bash cluster/collect.sh "$out"
  echo "send me $out"
  exit 1
fi
