#!/bin/bash
# PART 3b - BipedalWalker: VerifAI falsification of the DCD checkpoints from train.sbatch (CPU only).
# One array task = one (algo, trial); it runs every space x sampler x falsifier seed into
# $RUNS/bipedal_falsify/<algo>/s<dcd seed>/<space>_<sampler>_r<fseed>/  (DCD seed = 88 + trial).
# Solvability witnesses: the scripted walker plus the other algos' checkpoints of the same trial
# (--ref-ckpt; only those whose model.tar exists). Without them, hard counterexamples are limited
# to near-flat levels (atlas/bipedal/falsify.py docstring).
# The samplers run as parallel processes, one CPU each. Needs $DCDF_VENV (setup/setup_dcd_falsify.sh).
#   sbatch --array=0-29 --dependency=afterok:<train_jobid> cluster/3_bipedal/falsify.sbatch
# DRY_RUN=1 prints the commands instead of running them.
#SBATCH --job-name=atlas-3-bipedal-falsify
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr plr rplr accel minimax sfl"}; NSEEDS=${NSEEDS:-5}
# Measured locally: <=5 s per level worst case on one CPU (policy 10x2000 steps ~1.4 s + 5 reference
# witnesses ~3 s); 1000 levels x 3 fseeds <= ~4 h per sampler, samplers in parallel -> 8 h limit.
SPACES=${SPACES:-"dr"}; SAMPLERS=${SAMPLERS:-"ce mab"}
BUDGET=${BUDGET:-1000}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
ck=$(bash cluster/dcd_ckpt.sh bipedal $algo $seed)               # model.tar + meta.json
refs=()
for a in $ALGOS; do
  [ $a = $algo ] && continue
  r=$(bash cluster/dcd_ckpt.sh bipedal $a $seed)
  if [ -n "${DRY_RUN:-}" ] || [ -f "$r/model.tar" ]; then refs+=("$r"); fi
done
echo "task=$i algo=$algo trial=$seed ckpt=$ck refs=${#refs[@]} host=$(hostname)"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }
[ -n "${DRY_RUN:-}" ] || source "$DCDF_VENV/bin/activate"
export PYTHONPATH="$ATLAS:$DCD_DIR:$TP/baselines" ACL27_SOFT_RENDER=1 MPLBACKEND=Agg OMP_NUM_THREADS=1
pids=()
for sm in $SAMPLERS; do
  ( for sp in $SPACES; do
      run python -m atlas.bipedal.falsify --ckpt "$ck" --algo $algo --space $sp --sampler $sm \
        --budget $BUDGET --attempts 10 --tau 0.5 --seed $FSEEDS --out "$RUNS/bipedal_falsify" \
        ${refs[@]+--ref-ckpt "${refs[@]}"}
    done ) & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
