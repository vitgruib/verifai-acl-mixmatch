#!/bin/bash
# PART 1b - Maze: VerifAI falsification of the checkpoints from train.sbatch.
# One array task = one (algo, seed); it runs every space x sampler x falsifier seed into
# $RUNS/maze_falsify/<algo>/s<seed>/<space>_<sampler>_r<fseed>/  (docs/failure_records.md).
#   sbatch --array=0-59 --dependency=afterok:<train_jobid> cluster/1_maze/falsify.sbatch
# DRY_RUN=1 prints the commands instead of running them.
#SBATCH --job-name=atlas-1-maze-falsify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr plr rplr accel minimax sfl"}; NSEEDS=${NSEEDS:-10}; N=${NUM_UPDATES:-30000}
# Samplers from the local prelim (docs/ued_atlas.md, "Falsifier prelim"; results/atlas/prelim_maze.csv):
# ce finds the most failure modes in both spaces; mab is second on counterexample rate. Only these
# two run (ce vs mab is the comparison); random, halton and sa are dropped.
SPACES=${SPACES:-"dr seg"}; SAMPLERS=${SAMPLERS:-"ce mab"}
BUDGET=${BUDGET:-5000}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
if [ "$algo" = sfl ]; then ck=$RUNS/sfl/maze_sfl/s$seed          # model.safetensors
else ck=$RUNS/maze_train/checkpoints/maze_${algo}_n$N/$seed; fi   # orbax checkpoint
echo "task=$i algo=$algo seed=$seed ckpt=$ck host=$(hostname)"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }
[ -n "${DRY_RUN:-}" ] || source "$JAX_VENV/bin/activate"
for sp in $SPACES; do
  run python -m atlas.maze.falsify --ckpt "$ck" --algo $algo --space $sp --sampler $SAMPLERS \
    --budget $BUDGET --attempts 10 --tau 0.5 --seed $FSEEDS --out "$RUNS/maze_falsify"
done
