#!/bin/bash
# Part 4: SFL paper code (sampling-for-learnability), hydra configs, wandb offline.
# Covers the JaxNav row of the matrix (DR, PLR, Robust PLR, ACCEL, SFL: the paper's own
# sweep_configs/jaxnav-sa, single agent) and the SFL cell of the Maze row (minigrid_sfl, which
# trains on JaxUED's Maze env, so atlas.maze.falsify loads its model.safetensors directly).
# One task = one (env:algo, seed). Default: 6 x 10 seeds = 60 tasks.
#   sbatch --array=0-59 cluster/sfl/train.sbatch
# XLand lives in $SFL_DIR/xland with its own Dockerfile/jax pin; see cluster/README.md.
#SBATCH --job-name=atlas-sfl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh; source "${SFL_VENV:-$ATLAS/.venv-sfl}/bin/activate"
CELLS=${CELLS:-"maze:sfl jaxnav:dr jaxnav:plr jaxnav:rplr jaxnav:accel jaxnav:sfl"}; NSEEDS=${NSEEDS:-10}
i=${SLURM_ARRAY_TASK_ID:-0}
cell=$(pick $((i / NSEEDS)) $CELLS); seed=$((i % NSEEDS)); env=${cell%%:*}; algo=${cell#*:}
JN="env.num_agents=1 env.test_set=single"           # jaxnav-sa sweep settings
case $cell in
  maze:sfl)     s=minigrid_sfl; x="";;
  jaxnav:dr)    s=jaxnav_plr;   x="$JN ued.PLR_PARAMS.replay_prob=0.0";;
  jaxnav:plr)   s=jaxnav_plr;   x="$JN ued.SCORE_FUNCTION=MaxMC";;
  jaxnav:rplr)  s=jaxnav_plr;   x="$JN ued.SCORE_FUNCTION=MaxMC ued.EXPLORATORY_GRAD_UPDATES=False";;
  # upstream sweep spells this "use.USE_ACCEL" (a typo hydra rejects); ued.USE_ACCEL is the key
  jaxnav:accel) s=jaxnav_plr;   x="$JN ued.SCORE_FUNCTION=MaxMC ued.USE_ACCEL=True";;
  jaxnav:sfl)   s=jaxnav_sfl;   x="$JN";;
  *) echo "unknown cell $cell"; exit 2;;
esac
out="$RUNS/sfl/${env}_${algo}"
# SFL saves to SAVE_PATH/<wandb run.name>; offline runs have name None unless WANDB_NAME is set.
export WANDB_NAME=s$seed
mkdir -p "$out" && cd "$SFL_DIR"
echo "cell=$cell seed=$seed script=$s host=$(hostname)"
python sfl/train/$s.py SEED=$seed WANDB_MODE=offline SAVE_PATH="$out" $x
# Metadata the maze falsifier reads beside model.safetensors (atlas/maze/policy.py).
[ "$env" = maze ] && printf '{"seed": %d, "algo": "%s", "sfl_commit": "%s"}\n' \
  "$seed" "$algo" "$SFL_COMMIT" > "$out/s$seed/config.json"
exit 0
