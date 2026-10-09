#!/bin/bash
# PART 3b - JaxNav: VerifAI falsification of the SFL-repo checkpoints from train.sbatch.
# One array task = one (algo, seed); it runs every space x sampler x falsifier seed into
# $RUNS/jaxnav_falsify/<algo>/s<seed>/<space>_<sampler>_r<fseed>/  (docs/failure_records.md).
#   sbatch --array=0-59 --dependency=afterok:<train_jobid> cluster/3_jaxnav/falsify.sbatch
# DRY_RUN=1 prints the commands instead of running them.
#SBATCH --job-name=atlas-3-jaxnav-falsify
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr minimax plr rplr accel sfl"}; NSEEDS=${NSEEDS:-10}
SPACES=${SPACES:-"dr seg"}; SAMPLERS=${SAMPLERS:-ce}
BUDGET=${BUDGET:-5000}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
ck=$RUNS/sfl/jaxnav_${algo}/s$seed                                 # model.safetensors
echo "task=$i algo=$algo seed=$seed ckpt=$ck host=$(hostname)"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }
[ -n "${DRY_RUN:-}" ] || source "$SFL_VENV/bin/activate"
# CPU only: one process per space x sampler (2 by default) on 4 CPUs; per-level cost is env + oracle, not the network.
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
