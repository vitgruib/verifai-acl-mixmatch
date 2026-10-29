#!/bin/bash
# Full Maze suite and CartPole bridge: setup, smoke test, full run, results packed into maze_results.tgz.
# On SLURM, arguments are extra sbatch flags. Without SLURM it runs in the background on this machine;
# rerunning it (after a reboot, say) resumes: finished runs, the frozen exam and graded rows are kept.
# `run_maze.sh stop` stops the background run and every job it started.
set -euo pipefail
# The cluster scripts take these generic names from the environment as overrides; one the caller
# happens to have exported (RUNS, OUT, ...) must not move what this run writes, deletes or packs.
unset ATLAS TP RUNS OUT JAXUED_DIR SFL_DIR KINETIX_DIR JAX_VENV SFL_VENV KINETIX_VENV TORCH_VENV ATLAS_CACHE \
  ALGOS NSEEDS NUM_UPDATES JAXUED_ARGS SFL_STEPS SFL_ARGS SFL_TIME EXAM_ITERS BRIDGE_GRADE BRIDGE_OUT \
  BRIDGE_SEEDS BRIDGE_STEPS CPU_JOBS REF_COPY REF_DIR EXAM_OUT POOL_OUT SCORES GRADE_ARGS GATE KD \
  LOCAL DRY_RUN JAX_EXTRA XLA_PYTHON_CLIENT_MEM_FRACTION
cd "$(dirname "$0")/.."
setup() { bash cluster/setup/clone.sh && bash cluster/setup/setup_jax.sh && bash cluster/setup/setup_sfl.sh && bash cluster/setup/setup_torch.sh; }
out=$PWD/maze_results.tgz
# The background run is a session of its own (setsid), or without setsid a process group of its own,
# with the id in run_maze.pid; GNU timeout moves the jobs it runs to groups of their own, but not out
# of the session. running prints the pgrep/pkill flags that select it, if it is still running. After
# a reboot the id can belong to someone else's session, even this shell's: it counts only if one of
# its processes is this pipeline's, and never if it is the caller's own.
running() { local p; p=$(cat run_maze.pid 2>/dev/null) || return 1
  if command -v setsid >/dev/null; then set -- -s "$p" sid; else set -- -g "$p" pgid; fi
  [ "$(ps -o "$3=" -p $$ | tr -d ' ')" != "$p" ] || return 1
  pgrep "$1" "$2" -f "$PWD/cluster/|cluster/submit\.sh maze|cluster/1_maze/" >/dev/null && echo "$1 $2"; }

if [ "${1:-}" = stop ]; then
  if s=$(running); then
    pkill -TERM $s || true
    for _ in $(seq 60); do pgrep $s >/dev/null || break; sleep 1; done
    pkill -KILL $s 2>/dev/null || true
    echo "Stopped the background run and all its jobs. Run it again to resume."
  else
    echo "Not running."
  fi
  rm -f run_maze.pid; exit 0
fi

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
unset SBATCH_ARGS

if [ -z "${RUN_MAZE_BG:-}" ]; then
  if ! nvidia-smi -L >/dev/null 2>&1 && [ -z "${ALLOW_CPU:-}" ]; then
    echo "No NVIDIA GPU found (nvidia-smi). On CPU this run takes over a month; ALLOW_CPU=1 runs it anyway." >&2
    exit 1
  fi
  if running >/dev/null; then
    echo "Already running. Progress: tail -f $PWD/run_maze.log   Stop: bash $PWD/cluster/run_maze.sh stop"; exit 0
  fi
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv >&2
    echo "Pick the GPUs to use, e.g. CUDA_VISIBLE_DEVICES=0,1 bash $0" >&2; exit 1
  fi
  if command -v setsid >/dev/null; then
    RUN_MAZE_BG=1 setsid nohup bash "$PWD/cluster/run_maze.sh" "$@" >> run_maze.log 2>&1 < /dev/null &
  else
    set -m   # job control: the background job gets a process group of its own
    RUN_MAZE_BG=1 nohup bash "$PWD/cluster/run_maze.sh" "$@" >> run_maze.log 2>&1 < /dev/null &
  fi
  echo $! > run_maze.pid
  echo "Running in the background. Progress and expected finish time: tail -f $PWD/run_maze.log"
  echo "To stop it: bash $PWD/cluster/run_maze.sh stop"
  echo "If the machine restarts, run the same command again; it picks up where it stopped."
  echo "When it finishes, send me: $out"
  exit 0
fi

echo "start $(date); GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi -L 2>/dev/null || export JAX_EXTRA=""
if setup && LOCAL=1 bash cluster/submit.sh maze; then
  echo "done $(date): send me $out"
else
  echo "FAILED $(date); packing the logs anyway"
  bash cluster/collect.sh "$out"
  echo "send me $out"
  exit 1
fi
