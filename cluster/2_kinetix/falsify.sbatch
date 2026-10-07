#!/bin/bash
# PART 2b - Kinetix: VerifAI falsification of the checkpoints from train.sbatch (atlas.kinetix.falsify).
# One array task = one (algo, seed); it runs every space x sampler x falsifier seed into
# $RUNS/kinetix_falsify/<algo>/s<seed>/<space>_<sampler>_r<fseed>/  (docs/failure_records.md).
#   sbatch --array=0-49 --dependency=afterok:<train_jobid> cluster/2_kinetix/falsify.sbatch
# Spaces: pos (move agent/goal components) and phys (gravity, friction, motor, thrust scales) on the
# 10 S eval levels. Measured on CPU: ~1.2-1.8 s per level incl. the witness oracle, so 1000 levels
# x 3 falsifier seeds ~ 1-1.5 h per process; 2 processes (pos, phys) in parallel.
# DRY_RUN=1 prints the commands instead of running them.
#SBATCH --job-name=atlas-2-kinetix-falsify
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr plr rplr accel sfl"}; NSEEDS=${NSEEDS:-10}
SPACES=${SPACES:-"pos phys"}; SAMPLERS=${SAMPLERS:-ce}
BUDGET=${BUDGET:-1000}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
ck=$RUNS/kinetix/$algo/s$seed                                      # .hydra/config.yaml + ckpt/*/full_model.pbz2
echo "task=$i algo=$algo seed=$seed ckpt=$ck host=$(hostname)"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }
[ -n "${DRY_RUN:-}" ] || source "$KINETIX_VENV/bin/activate"
# CPU only, headless: no renderer is ever called.
export JAX_PLATFORMS=cpu MPLBACKEND=Agg SDL_VIDEODRIVER=dummy
pids=()
for sp in $SPACES; do for sm in $SAMPLERS; do
  run python -m atlas.kinetix.falsify --ckpt "$ck" --algo $algo --space $sp --sampler $sm \
    --budget $BUDGET --attempts 10 --tau 0.5 --seed $FSEEDS --out "$RUNS/kinetix_falsify" &
  pids+=($!)
done; done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
