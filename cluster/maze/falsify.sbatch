#!/bin/bash
# Part 1b: VerifAI falsification of trained maze checkpoints. One task = one (algo, seed);
# it runs every space x sampler x falsifier seed into $RUNS/maze_falsify/<algo>/s<seed>/...
# Run after train.sbatch:   sbatch --array=0-59 --dependency=afterok:<train_jobid> cluster/maze/falsify.sbatch
#SBATCH --job-name=atlas-falsify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh; source "$JAX_VENV/bin/activate"
ALGOS=${ALGOS:-"dr plr rplr accel paired minimax"}; NSEEDS=${NSEEDS:-10}; N=${NUM_UPDATES:-30000}
# Samplers from the local prelim (docs/ued_atlas.md, "Falsifier prelim"; results/atlas/prelim_maze.csv):
# ce finds the most failure modes in both spaces; mab is second on counterexample rate; random
# is the unbiased baseline every ratio is taken against. halton ~ random and sa < random: dropped.
SPACES=${SPACES:-"dr seg"}; SAMPLERS=${SAMPLERS:-"random ce mab"}
BUDGET=${BUDGET:-5000}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
ck=$RUNS/maze_train/checkpoints/maze_${algo}_n$N/$seed
for sp in $SPACES; do
  python -m atlas.maze.falsify --ckpt "$ck" --algo $algo --space $sp --sampler $SAMPLERS \
    --budget $BUDGET --attempts 10 --tau 0.5 --seed $FSEEDS --out "$RUNS/maze_falsify"
done
