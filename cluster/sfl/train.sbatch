#!/bin/bash
# Part 4: SFL paper code (MiniGrid maze + JaxNav single-agent), hydra configs, wandb offline.
# One task = one (script, seed). Default: 2 scripts x 10 seeds.
#   sbatch --array=0-19 cluster/sfl/train.sbatch
# XLand lives in $SFL_DIR/xland with its own Dockerfile/jax pin; see cluster/README.md.
#SBATCH --job-name=atlas-sfl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh; source "${SFL_VENV:-$ATLAS/.venv-sfl}/bin/activate"
SCRIPTS=${SCRIPTS:-"minigrid_sfl jaxnav_sfl"}; NSEEDS=${NSEEDS:-10}
i=${SLURM_ARRAY_TASK_ID:-0}
s=$(pick $((i / NSEEDS)) $SCRIPTS); seed=$((i % NSEEDS))
mkdir -p "$RUNS/sfl" && cd "$SFL_DIR"
python sfl/train/$s.py SEED=$seed WANDB_MODE=offline SAVE_PATH="$RUNS/sfl/$s"
