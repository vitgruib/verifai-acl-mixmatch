# Shared paths for every cluster script. Source it: `source cluster/env.sh`.
# Override any variable in your shell (or ~/.bashrc) before sourcing.
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

# Index helper for SLURM arrays: pick <list>[i] from a space-separated list.
pick() { local i=$1; shift; local a=($@); echo "${a[$i]}"; }
