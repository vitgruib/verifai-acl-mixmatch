#!/bin/bash
# Maze falsification of the trained agents (CPU only). One array task per (algo, seed).
#SBATCH --job-name=atlas-1-maze-falsify
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr plr rplr accel minimax sfl"}; NSEEDS=${NSEEDS:-10}; N=${NUM_UPDATES:-30000}
# ce searches for failures; random is the unbiased baseline
SPACES=${SPACES:-"dr seg"}; SAMPLERS=${SAMPLERS:-"random ce"}
BUDGET=${BUDGET:-5000}; FSEEDS=${FSEEDS:-"0 1 2"}
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
if [ "$algo" = sfl ]; then ck=$RUNS/sfl/maze_sfl/s$seed
else ck=$RUNS/maze_train/checkpoints/maze_${algo}_n$N/$seed; fi
echo "task=$i algo=$algo seed=$seed ckpt=$ck host=$(hostname)"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }
[ -n "${DRY_RUN:-}" ] || source "$JAX_VENV/bin/activate"
export JAX_PLATFORMS=cpu
# one process per space x sampler, each pinned to its share of the allocated CPUs
np=$(( $(wc -w <<< "$SPACES") * $(wc -w <<< "$SAMPLERS") )); k=0
cap_threads $(( (${SLURM_CPUS_PER_TASK:-4} + np - 1) / np ))
pids=()
for sp in $SPACES; do for sm in $SAMPLERS; do
  run $(pin_cpus $k $np) python -m atlas.maze.falsify --ckpt "$ck" --algo $algo --space $sp --sampler $sm \
    --budget $BUDGET --attempts 10 --tau 0.5 --seed $FSEEDS --out "$RUNS/maze_falsify" &
  pids+=($!); k=$((k + 1))
done; done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; exit $rc
