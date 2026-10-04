#!/bin/bash
# Part 1: JaxUED maze, paper settings (30k updates). One task = one (algo, seed).
# Default array: 5 algos x 10 seeds = 50 tasks.   sbatch --array=0-49 cluster/maze/train.sbatch
#SBATCH --job-name=atlas-maze
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh; source "$JAX_VENV/bin/activate"
ALGOS=${ALGOS:-"dr plr rplr accel paired"}; NSEEDS=${NSEEDS:-10}; N=${NUM_UPDATES:-30000}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
case $algo in
  dr)     f=maze_dr.py;     x="";;
  paired) f=maze_paired.py; x="";;
  rplr)   f=maze_plr.py;    x="";;                          # Robust PLR (no grads on new levels)
  plr)    f=maze_plr.py;    x="--exploratory_grad_updates";;
  accel)  f=maze_plr.py;    x="--use_accel";;
esac
run=maze_${algo}_n$N
mkdir -p "$RUNS/maze_train" && cd "$RUNS/maze_train"        # checkpoints land in ./checkpoints/$run/$seed
echo "algo=$algo seed=$seed updates=$N host=$(hostname)"
python "$JAXUED_DIR/examples/$f" --run_name $run --seed $seed --num_updates $N $x
python "$JAXUED_DIR/examples/$f" --mode eval --checkpoint_directory checkpoints/$run/$seed
