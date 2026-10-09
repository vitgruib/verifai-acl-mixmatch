# Shared paths for every cluster script. Source it: `source cluster/env.sh`.
# Override any variable in your shell before sourcing.
export ATLAS=${ATLAS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export TP=${TP:-$ATLAS/third_party}          # upstream clones (gitignored)
export RUNS=${RUNS:-$ATLAS/runs}             # checkpoints, logs, falsifier records (gitignored)
export JAXUED_DIR=${JAXUED_DIR:-$TP/jaxued}
export SFL_DIR=${SFL_DIR:-$TP/sfl}
export KINETIX_DIR=${KINETIX_DIR:-$TP/kinetix}

# Python environments built by cluster/setup/*.sh
export JAX_VENV=${JAX_VENV:-$ATLAS/.venv-jax}   # jaxued + verifai/scenic + atlas (py3.11)
export SFL_VENV=${SFL_VENV:-$ATLAS/.venv-sfl}   # SFL repo (py3.11, 2024-era jax pins; Parts 1 and 3)
export KINETIX_VENV=${KINETIX_VENV:-$ATLAS/.venv-kinetix}   # Kinetix + verifai/scenic (py3.11; Part 2 train and falsify)

# Pinned upstream commits (what our local results used)
export JAXUED_REPO=https://github.com/DramaCow/jaxued.git        JAXUED_COMMIT=0f8f1284677375b889e4f13a32c9617cd009f8c4
export SFL_REPO=https://github.com/amacrutherford/sampling-for-learnability.git SFL_COMMIT=8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b
export KINETIX_REPO=https://github.com/FLAIROx/Kinetix.git     KINETIX_COMMIT=80ee9c292837c825cafaf9889323fdcc212efd1b

export WANDB_MODE=${WANDB_MODE:-offline} WANDB_SILENT=true MPLBACKEND=Agg

# Keep every cache and config write inside the repo (nothing lands in $HOME).
export ATLAS_CACHE=${ATLAS_CACHE:-$ATLAS/.cache}
export XDG_CACHE_HOME=$ATLAS_CACHE PIP_CACHE_DIR=$ATLAS_CACHE/pip MPLCONFIGDIR=$ATLAS_CACHE/matplotlib \
  WANDB_DIR=$RUNS WANDB_CACHE_DIR=$ATLAS_CACHE/wandb/cache WANDB_CONFIG_DIR=$ATLAS_CACHE/wandb/config \
  WANDB_DATA_DIR=$ATLAS_CACHE/wandb/data
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$PIP_CACHE_DIR"

# CPU threads: stay inside the SLURM allocation (BLAS/OpenMP default to every core on the node).
# Falsify jobs run several processes per task; they split this with cap_threads and pin_cpus.
cap_threads() { local n=$1; export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n \
  NUMEXPR_NUM_THREADS=$n VECLIB_MAXIMUM_THREADS=$n; }
cap_threads "${SLURM_CPUS_PER_TASK:-4}"
# pin_cpus <k> <n>: "taskset -c <list>" for the k-th of n equal slices of the CPUs this job may use.
# XLA sizes its CPU thread pool from the affinity mask, so each process stays on its own slice.
# Prints nothing (no pinning) without taskset or Linux affinity, e.g. on a laptop.
pin_cpus() { command -v taskset >/dev/null || return 0; python3 -c '
import os, sys
k, n = map(int, sys.argv[1:]); c = sorted(os.sched_getaffinity(0)); m = len(c) // n
print("taskset -c " + ",".join(map(str, c[k * m:(k + 1) * m] if m else c)))' "$1" "$2" 2>/dev/null || true; }

# GPU jobs: if the scheduler grants a GPU without restricting visibility, see only that GPU.
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ] && export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# Index helper for SLURM arrays: pick <list>[i] from a space-separated list.
pick() { local i=$1; shift; local a=($@); echo "${a[$i]}"; }
