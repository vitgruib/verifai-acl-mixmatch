#!/bin/bash
# JaxNav training (SFL repo). One array task per (algo, seed): 6 algos x 10 seeds = 60 tasks.
#SBATCH --job-name=atlas-3-jaxnav
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr minimax plr rplr accel sfl"}; NSEEDS=${NSEEDS:-10}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
JN="env.num_agents=1 env.test_set=single"  # single-agent settings from the paper's sweep
# the default eval set is 4-agent and crashes with one agent
EV="EVAL_SAMPLED_SET_PATH=sfl/data/test_sets/sampled_tc_100e_1a.pkl"
case $algo in
  dr)    s=jaxnav_plr; x="$JN $EV ued.PLR_PARAMS.replay_prob=0.0";;
  minimax) s=jaxnav_minimax; x="$JN $EV";;
  plr)   s=jaxnav_plr; x="$JN $EV ued.SCORE_FUNCTION=MaxMC";;
  rplr)  s=jaxnav_plr; x="$JN $EV ued.SCORE_FUNCTION=MaxMC ued.EXPLORATORY_GRAD_UPDATES=False";;
  accel) s=jaxnav_plr; x="$JN $EV ued.SCORE_FUNCTION=MaxMC ued.USE_ACCEL=True";;
  sfl)   s=jaxnav_sfl; x="$JN";;
  *) echo "unknown algo $algo"; exit 2;;
esac
out="$RUNS/sfl/jaxnav_${algo}"
# SFL saves to SAVE_PATH/<wandb run name>, which is None offline unless WANDB_NAME is set
export WANDB_NAME=s$seed
echo "task=$i algo=$algo seed=$seed script=$s host=$(hostname)"
if [ -n "${DRY_RUN:-}" ]; then echo "+ python sfl/train/$s.py SEED=$seed WANDB_MODE=offline SAVE_PATH=$out $x ${SFL_ARGS:-}"; exit 0; fi
source "$SFL_VENV/bin/activate"; mkdir -p "$out" && cd "$SFL_DIR"
python sfl/train/$s.py SEED=$seed WANDB_MODE=offline SAVE_PATH="$out" $x ${SFL_ARGS:-}
