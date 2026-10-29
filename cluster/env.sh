# Shared settings for every cluster script. Any variable can be overridden from the shell.
export ATLAS=${ATLAS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export TP=${TP:-$ATLAS/third_party}
export RUNS=${RUNS:-$ATLAS/runs}
export JAXUED_DIR=${JAXUED_DIR:-$TP/jaxued}
export SFL_DIR=${SFL_DIR:-$TP/sfl}
export KINETIX_DIR=${KINETIX_DIR:-$TP/kinetix}

export JAX_VENV=${JAX_VENV:-$ATLAS/.venv-jax}      # JaxUED, VerifAI
export SFL_VENV=${SFL_VENV:-$ATLAS/.venv-sfl}      # SFL repo (older jax)
export KINETIX_VENV=${KINETIX_VENV:-$ATLAS/.venv-kinetix}
export TORCH_VENV=${TORCH_VENV:-$ATLAS/.venv-torch}  # acl_bench (PyTorch): the CartPole bridge's grader

# Pinned upstream commits
export JAXUED_REPO=https://github.com/DramaCow/jaxued.git        JAXUED_COMMIT=0f8f1284677375b889e4f13a32c9617cd009f8c4
export SFL_REPO=https://github.com/amacrutherford/sampling-for-learnability.git SFL_COMMIT=8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b
export KINETIX_REPO=https://github.com/FLAIROx/Kinetix.git     KINETIX_COMMIT=80ee9c292837c825cafaf9889323fdcc212efd1b
export JAXMARL_REPO=https://github.com/FLAIROx/JaxMARL.git     JAXMARL_COMMIT=976aeb152cb184a5095021968bba94da96eb6394

export WANDB_MODE=${WANDB_MODE:-offline} WANDB_SILENT=true MPLBACKEND=Agg

# Keep caches inside the repo, not in $HOME.
export ATLAS_CACHE=${ATLAS_CACHE:-$ATLAS/.cache}
export XDG_CACHE_HOME=$ATLAS_CACHE PIP_CACHE_DIR=$ATLAS_CACHE/pip MPLCONFIGDIR=$ATLAS_CACHE/matplotlib \
  WANDB_DIR=$RUNS WANDB_CACHE_DIR=$ATLAS_CACHE/wandb/cache WANDB_CONFIG_DIR=$ATLAS_CACHE/wandb/config \
  WANDB_DATA_DIR=$ATLAS_CACHE/wandb/data
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$PIP_CACHE_DIR"

# Stay inside the job's CPU allocation (BLAS/OpenMP otherwise use every core on the node).
cap_threads() { local n=$1; export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n \
  NUMEXPR_NUM_THREADS=$n VECLIB_MAXIMUM_THREADS=$n; }
cap_threads "${SLURM_CPUS_PER_TASK:-4}"
# pin_cpus <k> <n>: a taskset prefix for the k-th of n equal slices of this job's CPUs (empty without taskset)
pin_cpus() { command -v taskset >/dev/null || return 0; python3 -c '
import os, sys
k, n = map(int, sys.argv[1:]); c = sorted(os.sched_getaffinity(0)); m = len(c) // n
print("taskset -c " + ",".join(map(str, c[k * m:(k + 1) * m] if m else c)))' "$1" "$2" 2>/dev/null || true; }

# Only see the GPU the scheduler granted.
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${SLURM_JOB_GPUS:-}" ] && export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# pick <i> <list...>: the i-th item
pick() { local i=$1; shift; local a=($@); echo "${a[$i]}"; }
# maze_tasks <algos...>: one "algo seed" line per Maze training task; ref = the reference DR agents
# (seeds 100-109, never scored), any other algo seeds 0 .. NSEEDS-1
maze_tasks() { local a; for a; do
  if [ "$a" = ref ]; then seq 100 109 | sed 's/^/ref /'; else seq 0 $(( ${NSEEDS:-10} - 1 )) | sed "s/^/$a /"; fi
done; }
# Where the Maze suite and the bridge write scores and the frozen exam: the repo (results/,
# frozen_sets/) for the full run; under $RUNS/outputs, same layout, for a run with any size setting
# off its default (smoke, tiny test runs), so the committed files never change.
maze_full() { [ "${NUM_UPDATES:-30000}" = 30000 ] && [ "${NSEEDS:-10}" = 10 ] && [ -z "${JAXUED_ARGS:-}" ] \
  && [ "${SFL_STEPS:-245760000}" = 245760000 ] && [ -z "${SFL_ARGS:-}" ] && [ "${EXAM_ITERS:-3000}" = 3000 ]; }
bridge_full() { [ "${BRIDGE_STEPS:-614400}" = 614400 ] && [ "${BRIDGE_SEEDS:-20}" = 20 ]; }
exam_frozen() { maze_full && grep -q '"verifai"' "$ATLAS/frozen_sets/suite/maze/manifest.json" 2>/dev/null; }
out_root() { if "$1"; then echo "$ATLAS"; else echo "$RUNS/outputs"; fi; }   # out_root maze_full|bridge_full
