#!/bin/bash
# PART 4b - JaxNav: VerifAI falsification of the SFL-repo checkpoints from train.sbatch.
# One array task = one (algo, seed); it runs every space x sampler x falsifier seed into
# $RUNS/jaxnav_falsify/<algo>/s<seed>/<space>_<sampler>_r<fseed>/  (docs/failure_records.md).
#   sbatch --array=0-59 --dependency=afterok:<train_jobid> cluster/4_jaxnav/falsify.sbatch
# DRY_RUN=1 prints the commands instead of running them.
#SBATCH --job-name=atlas-4-jaxnav-falsify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr minimax plr rplr accel sfl"}; NSEEDS=${NSEEDS:-10}
SPACES=${SPACES:-"dr seg"}; SAMPLERS=${SAMPLERS:-"ce mab"}
BUDGET=${BUDGET:-5000}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
ck=$RUNS/sfl/jaxnav_${algo}/s$seed                                 # model.safetensors
echo "task=$i algo=$algo seed=$seed ckpt=$ck host=$(hostname)"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }
[ -n "${DRY_RUN:-}" ] || source "$SFL_VENV/bin/activate"
export MPLBACKEND=Agg
for sp in $SPACES; do
  run python -m atlas.jaxnav.falsify --ckpt "$ck" --algo $algo --space $sp --sampler $SAMPLERS \
    --budget $BUDGET --attempts 10 --tau 0.5 --seed $FSEEDS --out "$RUNS/jaxnav_falsify"
done
