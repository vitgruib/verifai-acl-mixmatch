#!/bin/bash
# Part 1b: VerifAI falsification of trained maze checkpoints. One task = one (algo, seed);
# it runs every space x sampler x falsifier seed into $RUNS/maze_falsify/<algo>/s<seed>/...
# Run after train.sbatch:   sbatch --array=0-49 --dependency=afterok:<train_jobid> cluster/maze/falsify.sbatch
#SBATCH --job-name=atlas-falsify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh; source "$JAX_VENV/bin/activate"
ALGOS=${ALGOS:-"dr plr rplr accel paired"}; NSEEDS=${NSEEDS:-10}; N=${NUM_UPDATES:-30000}
SPACES=${SPACES:-"dr seg"}; SAMPLERS=${SAMPLERS:-"random halton ce mab sa"}
BUDGET=${BUDGET:-5000}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
ck=$RUNS/maze_train/checkpoints/maze_${algo}_n$N/$seed
for sp in $SPACES; do
  python -m atlas.maze.falsify --ckpt "$ck" --algo $algo --space $sp --sampler $SAMPLERS \
    --budget $BUDGET --attempts 10 --tau 0.5 --seed $FSEEDS --out "$RUNS/maze_falsify"
done
