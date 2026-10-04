#!/bin/bash
# PART 4 - JaxNav (SFL paper code, sampling-for-learnability): train. hydra configs, wandb offline.
# DR, PLR, Robust PLR, ACCEL, SFL with the paper's own sweep_configs/jaxnav-sa (single agent).
# One task = one (algo, seed). Default: 5 algos x 10 seeds = 50 tasks.
#   sbatch --array=0-49 cluster/4_jaxnav/train.sbatch
# (The Maze SFL cell also uses this repo but lives in Part 1: cluster/1_maze/train.sbatch.)
# DRY_RUN=1 prints the command instead of running it.
# XLand lives in $SFL_DIR/xland with its own Dockerfile/jax pin; see cluster/README.md.
#SBATCH --job-name=atlas-4-jaxnav
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr plr rplr accel sfl"}; NSEEDS=${NSEEDS:-10}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
JN="env.num_agents=1 env.test_set=single"           # jaxnav-sa sweep settings
case $algo in
  dr)    s=jaxnav_plr; x="$JN ued.PLR_PARAMS.replay_prob=0.0";;
  plr)   s=jaxnav_plr; x="$JN ued.SCORE_FUNCTION=MaxMC";;
  rplr)  s=jaxnav_plr; x="$JN ued.SCORE_FUNCTION=MaxMC ued.EXPLORATORY_GRAD_UPDATES=False";;
  # upstream sweep spells this "use.USE_ACCEL" (a typo hydra rejects); ued.USE_ACCEL is the key
  accel) s=jaxnav_plr; x="$JN ued.SCORE_FUNCTION=MaxMC ued.USE_ACCEL=True";;
  sfl)   s=jaxnav_sfl; x="$JN";;
  *) echo "unknown algo $algo"; exit 2;;
esac
out="$RUNS/sfl/jaxnav_${algo}"
# SFL saves to SAVE_PATH/<wandb run.name>; offline runs have name None unless WANDB_NAME is set.
export WANDB_NAME=s$seed
echo "task=$i algo=$algo seed=$seed script=$s host=$(hostname)"
if [ -n "${DRY_RUN:-}" ]; then echo "+ python sfl/train/$s.py SEED=$seed WANDB_MODE=offline SAVE_PATH=$out $x ${SFL_ARGS:-}"; exit 0; fi
source "$SFL_VENV/bin/activate"; mkdir -p "$out" && cd "$SFL_DIR"
python sfl/train/$s.py SEED=$seed WANDB_MODE=offline SAVE_PATH="$out" $x ${SFL_ARGS:-}
