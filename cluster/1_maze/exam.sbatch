#!/bin/bash
# Freezes the Maze exam's VerifAI section from the 10 reference agents (atlas.maze.exam); the
# committed random and custom sections are kept. Rerunning a frozen exam does nothing.
# Outside the repo's frozen_sets (smoke and tiny runs, env.sh out_root) EXAM_OUT starts from a copy
# of the committed random and custom sections. Smoke test: REF_COPY names one checkpoint, copied in
# as all 10 reference agents.
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
root=$(out_root maze_full); frozen=$ATLAS/frozen_sets/suite/maze
EXAM_OUT=${EXAM_OUT:-$root/frozen_sets/suite/maze}; POOL_OUT=${POOL_OUT:-$root/frozen_sets/suite_pool/maze}
source "$JAX_VENV/bin/activate"
if [ -n "${REF_COPY:-}" ]; then
  for s in $(seq 100 109); do
    [ -d "$REF_DIR/$s" ] || { mkdir -p "$REF_DIR"; cp -R "$REF_COPY" "$REF_DIR/$s.part"; mv "$REF_DIR/$s.part" "$REF_DIR/$s"; }
  done
fi
if ! [ "$EXAM_OUT" -ef "$frozen" ] && [ ! -f "$EXAM_OUT/manifest.json" ]; then
  mkdir -p "$EXAM_OUT"; cp "$frozen"/{random,custom}.npz "$EXAM_OUT"
  python -c 'import json, sys; m = json.load(open(sys.argv[1]))
m["sets"] = {k: m["sets"][k] for k in ("random", "custom")}; json.dump(m, open(sys.argv[2], "w"), indent=2, sort_keys=True)' \
    "$frozen/manifest.json" "$EXAM_OUT/manifest.json"
fi
python -m atlas.maze.exam --ref-dir "$REF_DIR" --out "$EXAM_OUT" --pool-out "$POOL_OUT" --iters "${EXAM_ITERS:-3000}"
