#!/bin/bash
# Packs the results into one tarball: the Maze suite's scores and the CartPole bridge's, the frozen
# Maze exam and its full VerifAI pool; from the runs, configs, CSVs, the final checkpoint of each run
# and the logs (earlier atlas runs' falsification results and Kinetix checkpoints too).
# Usage: bash cluster/collect.sh [out.tgz]
set -euo pipefail
source "$(dirname "$0")/env.sh"
out=${1:-$RUNS/atlas_collect_$(date +%Y%m%d_%H%M).tgz}; [[ $out = /* ]] || out=$PWD/$out
tmp=${out%.tgz}.tar; list=$(mktemp); rm -f "$tmp"
cd "$ATLAS"
for p in results/suite/maze results/suite/cartpole_jaxued results/suite/analysis/bridge_cartpole.md \
         frozen_sets/suite/maze frozen_sets/suite_pool/maze run_maze.log; do
  if [ -e "$p" ]; then echo "$p"; fi
done > "$list"
tar cf "$tmp" -T "$list"
# run files, under the runs dir's own name
cd "$(dirname "$RUNS")"; R=$(basename "$RUNS")
find "$R" \( -name config.json -o -name '*.csv' -o -name train.log -o -name timing.txt \
  -o -name model.safetensors -o -name params.npz -o -path '*/hydra/.hydra/*' -o -path "$R/slurm/*" \
  -o -path '*/exam/*' -o -path "$R/outputs/*" -o -path '*_falsify/*' -o -path '*/maze_train/results/*' \
  -o -path "$R/kinetix/*/.hydra/config.yaml" -o -path "$R/kinetix/*/ckpt/*/full_model.pbz2" \) -type f > "$list"
# only the last checkpoint of each JaxUED Maze run
find "$R" -type d -path '*/maze_train/checkpoints/*/*/models' | while read -r d; do
  s=$(ls "$d" | grep -E '^[0-9]+$' | sort -n | tail -1)
  [ -z "$s" ] || find "$d/$s" -type f
done >> "$list"
tar rf "$tmp" -T "$list"
# job logs are always in <repo>/runs/slurm, even when RUNS is elsewhere
[ "$RUNS" -ef "$ATLAS/runs" ] || [ ! -d "$ATLAS/runs/slurm" ] || tar rf "$tmp" -C "$ATLAS" runs/slurm
rm -f "$list"
gzip -c "$tmp" > "$out.part" && mv "$out.part" "$out" && rm -f "$tmp"
echo "$out ($(du -h "$out" | cut -f1))"
