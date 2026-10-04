#!/bin/bash
# PART 1a - Maze: train every algorithm. One array task = one (algo, seed).
# Default: 6 algos x 10 seeds = 60 tasks.   sbatch --array=0-59 cluster/1_maze/train.sbatch
#   dr plr rplr accel minimax -> JaxUED examples (paper settings, 30k updates)
#   sfl                       -> SFL repo's minigrid_sfl (same JaxUED Maze env and network)
# DRY_RUN=1 prints the command instead of running it. JAXUED_ARGS / SFL_ARGS pass extra flags (smoke tests).
#SBATCH --job-name=atlas-1-maze-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr plr rplr accel minimax sfl"}; NSEEDS=${NSEEDS:-10}; N=${NUM_UPDATES:-30000}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
echo "task=$i algo=$algo seed=$seed host=$(hostname)"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }

if [ "$algo" = sfl ]; then
  # SFL saves to SAVE_PATH/<wandb run name>; offline runs are named None unless WANDB_NAME is set.
  out="$RUNS/sfl/maze_sfl"; export WANDB_NAME=s$seed
  [ -n "${DRY_RUN:-}" ] || { source "$SFL_VENV/bin/activate"; mkdir -p "$out"; cd "$SFL_DIR"; }
  run python sfl/train/minigrid_sfl.py SEED=$seed WANDB_MODE=offline SAVE_PATH="$out" ${SFL_ARGS:-}
  # Metadata atlas/maze/policy.py reads beside model.safetensors.
  [ -n "${DRY_RUN:-}" ] || printf '{"seed": %d, "algo": "sfl", "sfl_commit": "%s"}\n' \
    "$seed" "$SFL_COMMIT" > "$out/s$seed/config.json"
  exit 0
fi

case $algo in
  dr)      f=maze_dr.py;     x="";;
  minimax) f=maze_paired.py; x="--minimax";;               # patches/jaxued_minimax.patch
  rplr)    f=maze_plr.py;    x="";;                         # Robust PLR (no grads on new levels)
  plr)     f=maze_plr.py;    x="--exploratory_grad_updates";;
  accel)   f=maze_plr.py;    x="--use_accel";;
  *) echo "unknown algo $algo"; exit 2;;
esac
r=maze_${algo}_n$N
[ -n "${DRY_RUN:-}" ] || { source "$JAX_VENV/bin/activate"; mkdir -p "$RUNS/maze_train"; cd "$RUNS/maze_train"; }
run python "$JAXUED_DIR/examples/$f" --run_name $r --seed $seed --num_updates $N $x ${JAXUED_ARGS:-}
run python "$JAXUED_DIR/examples/$f" --mode eval --checkpoint_directory checkpoints/$r/$seed
