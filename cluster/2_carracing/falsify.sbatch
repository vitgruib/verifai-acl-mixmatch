#!/bin/bash
# PART 2b - CarRacing: VerifAI falsification of the DCD checkpoints from train.sbatch (CPU only).
# One array task = one (algo, trial); it runs every space x sampler x falsifier seed into
# $RUNS/carracing_falsify/<algo>/s<dcd seed>/<space>_<sampler>_r<fseed>/  (DCD seed = 88 + trial).
# The samplers run as parallel processes, one CPU each. Needs $DCDF_VENV (setup/setup_dcd_falsify.sh).
#   sbatch --array=0-29 --dependency=afterok:<train_jobid> cluster/2_carracing/falsify.sbatch
# DRY_RUN=1 prints the commands instead of running them.
#SBATCH --job-name=atlas-2-carracing-falsify
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr plr rplr accel minimax sfl"}; NSEEDS=${NSEEDS:-5}
# ~13 s per level measured on one CPU with an untrained policy (10 attempts; the in-env oracle runs
# on every failure, so trained policies are cheaper): 500 levels x 3 fseeds x 2 spaces ~ 11 h per sampler.
SPACES=${SPACES:-"dr sketch"}; SAMPLERS=${SAMPLERS:-"ce mab"}
BUDGET=${BUDGET:-500}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
ck=$(bash cluster/dcd_ckpt.sh car_racing $algo $seed)            # model.tar + meta.json
echo "task=$i algo=$algo trial=$seed ckpt=$ck host=$(hostname)"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }
[ -n "${DRY_RUN:-}" ] || source "$DCDF_VENV/bin/activate"
export PYTHONPATH="$ATLAS:$DCD_DIR:$TP/baselines" ACL27_SOFT_RENDER=1 MPLBACKEND=Agg OMP_NUM_THREADS=1
pids=()
for sm in $SAMPLERS; do
  ( for sp in $SPACES; do
      run python -m atlas.carracing.falsify --ckpt "$ck" --algo $algo --space $sp --sampler $sm \
        --budget $BUDGET --attempts 10 --tau 0.5 --seed $FSEEDS --out "$RUNS/carracing_falsify"
    done ) & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
