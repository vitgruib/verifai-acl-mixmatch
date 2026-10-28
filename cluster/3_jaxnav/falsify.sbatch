#!/bin/bash
# JaxNav falsification of the trained agents (CPU only). One array task per (algo, seed).
#SBATCH --job-name=atlas-3-jaxnav-falsify
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr minimax plr rplr accel sfl"}; NSEEDS=${NSEEDS:-10}
SPACES=${SPACES:-"dr seg"}; SAMPLERS=${SAMPLERS:-"random ce"}  # ce searches; random is the baseline
BUDGET=${BUDGET:-5000}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
ck=$RUNS/sfl/jaxnav_${algo}/s$seed
echo "task=$i algo=$algo seed=$seed ckpt=$ck host=$(hostname)"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }
[ -n "${DRY_RUN:-}" ] || source "$SFL_VENV/bin/activate"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
# one process per space x sampler, each pinned to its share of the allocated CPUs
np=$(( $(wc -w <<< "$SPACES") * $(wc -w <<< "$SAMPLERS") )); k=0
cap_threads $(( (${SLURM_CPUS_PER_TASK:-4} + np - 1) / np ))
pids=()
for sp in $SPACES; do for sm in $SAMPLERS; do
  run $(pin_cpus $k $np) python -m atlas.jaxnav.falsify --ckpt "$ck" --algo $algo --space $sp --sampler $sm \
    --budget $BUDGET --attempts 10 --tau 0.5 --seed $FSEEDS --out "$RUNS/jaxnav_falsify" &
  pids+=($!); k=$((k + 1))
done; done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
