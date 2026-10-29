#!/bin/bash
# Freezes the Maze exam's VerifAI section from the 10 reference agents (atlas.maze.exam); the
# committed random and custom sections are kept. Rerunning a frozen exam does nothing.
# Smoke test: REF_COPY names one checkpoint, copied in as all 10 reference agents, and EXAM_OUT
# starts from a copy of the committed random and custom sections.
#SBATCH --job-name=atlas-1-maze-exam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
N=${NUM_UPDATES:-30000}
REF_DIR=${REF_DIR:-$RUNS/maze_train/checkpoints/maze_ref_n$N}
EXAM_OUT=${EXAM_OUT:-$ATLAS/frozen_sets/suite/maze}; POOL_OUT=${POOL_OUT:-$ATLAS/frozen_sets/suite_pool/maze}
source "$JAX_VENV/bin/activate"
if [ -n "${REF_COPY:-}" ]; then
  for s in $(seq 100 109); do
    [ -d "$REF_DIR/$s" ] || { mkdir -p "$REF_DIR"; cp -R "$REF_COPY" "$REF_DIR/$s.part"; mv "$REF_DIR/$s.part" "$REF_DIR/$s"; }
  done
  if [ ! -f "$EXAM_OUT/manifest.json" ]; then
    mkdir -p "$EXAM_OUT"; cp "$ATLAS"/frozen_sets/suite/maze/{random,custom}.npz "$EXAM_OUT"
    python -c 'import json, sys; m = json.load(open(sys.argv[1]))
m["sets"] = {k: m["sets"][k] for k in ("random", "custom")}; json.dump(m, open(sys.argv[2], "w"), indent=2, sort_keys=True)' \
      "$ATLAS/frozen_sets/suite/maze/manifest.json" "$EXAM_OUT/manifest.json"
  fi
fi
python -m atlas.maze.exam --ref-dir "$REF_DIR" --out "$EXAM_OUT" --pool-out "$POOL_OUT" --iters "${EXAM_ITERS:-3000}"
