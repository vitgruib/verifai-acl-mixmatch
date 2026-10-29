#!/bin/bash
# Full Maze suite and CartPole bridge: setup, smoke test, full run, results packed into maze_results.tgz.
# On SLURM, arguments are extra sbatch flags. Without SLURM it runs in the background on this machine;
# rerunning it (after a reboot, say) resumes: finished runs, the frozen exam and graded rows are kept.
set -euo pipefail
cd "$(dirname "$0")/.."
setup() { bash cluster/setup/clone.sh; bash cluster/setup/setup_jax.sh; bash cluster/setup/setup_sfl.sh; bash cluster/setup/setup_torch.sh; }
out=$PWD/maze_results.tgz
command -v "${PYBIN:-python3.11}" >/dev/null || { echo "Needs python3.11 (or PYBIN=/path/to/python3.11)." >&2; exit 1; }

if command -v sbatch >/dev/null; then
  export SBATCH_ARGS="${SBATCH_ARGS:-} $*"
  mkdir -p runs/slurm
  sbatch --test-only $SBATCH_ARGS cluster/1_maze/train.sbatch   # catch bad flags before setup
  setup
  bash cluster/submit.sh maze
  echo
  echo "Submitted. When it finishes, send me: $out"
  exit 0
fi

if [ -z "${RUN_MAZE_BG:-}" ]; then
  if ! nvidia-smi -L >/dev/null 2>&1 && [ -z "${ALLOW_CPU:-}" ]; then
    echo "No NVIDIA GPU found (nvidia-smi). On CPU this run takes over a month; ALLOW_CPU=1 runs it anyway." >&2
    exit 1
  fi
  if [ -f run_maze.pid ] && ps -p "$(cat run_maze.pid)" -o command= 2>/dev/null | grep -q run_maze.sh; then
    echo "Already running (pid $(cat run_maze.pid)). Progress: tail -f $PWD/run_maze.log"; exit 0
  fi
  RUN_MAZE_BG=1 nohup bash "$PWD/cluster/run_maze.sh" "$@" >> run_maze.log 2>&1 < /dev/null &
  echo $! > run_maze.pid
  echo "Running in the background. Progress and expected finish time: tail -f $PWD/run_maze.log"
  echo "If the machine restarts, run the same command again; it picks up where it stopped."
  echo "When it finishes, send me: $out"
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
