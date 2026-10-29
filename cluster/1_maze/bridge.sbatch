#!/bin/bash
# CartPole bridge (atlas/bridge, CPU only): JaxUED's gymnax PLR on our CartPole box. Array task i
# trains arm DR, PLR or RPLR (i / 20) with seed i % 20 + 1: 60 tasks; a finished run is skipped.
# BRIDGE_GRADE=1 instead grades every run with our PyTorch grader and compares it with batch 21.
#SBATCH --job-name=atlas-1-bridge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
STEPS=${BRIDGE_STEPS:-614400}; NB=${BRIDGE_SEEDS:-20}; out=$RUNS/bridge/cartpole
root=$(out_root bridge_full)   # env.sh: the repo for the full run, else under $RUNS/outputs
export JAX_PLATFORMS=cpu
if [ -n "${BRIDGE_GRADE:-}" ]; then
  source "$TORCH_VENV/bin/activate"
  python -m atlas.bridge.grade_cartpole --runs "$out" --out "$root/results/suite/cartpole_jaxued/scores.csv"
  mkdir -p "$root/results/suite/analysis"
  python -m atlas.bridge.compare --bridge "$root/results/suite/cartpole_jaxued/scores.csv" --budget "$STEPS" \
    --out "$root/results/suite/analysis/bridge_cartpole.md"
  exit 0
fi
i=${SLURM_ARRAY_TASK_ID:-0}; arm=$(pick $((i / NB)) DR PLR RPLR); seed=$((i % NB + 1)); d=$out/$arm/$seed
echo "task=$i arm=$arm seed=$seed host=$(hostname)"
# config.json is written last, once params.npz is saved
[ -f "$d/config.json" ] && { echo "already trained: $d"; exit 0; }
source "$JAX_VENV/bin/activate"
python -m atlas.bridge.cartpole_jaxued --arm "$arm" --seed "$seed" --steps "$STEPS" --out "$d"
