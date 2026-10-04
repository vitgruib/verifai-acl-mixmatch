#!/bin/bash
# Pack what we need back on the laptop: falsifier records, eval CSVs, final checkpoints only.
# Usage: bash cluster/collect.sh  ->  runs/atlas_collect_<date>.tgz
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd "$RUNS"
out=atlas_collect_$(date +%Y%m%d_%H%M).tgz
find . \( -path './maze_falsify/*' -o -path './maze_train/results/*' -o -name 'config.json' \
  -o -name '*.csv' -o -path './slurm/*' \) -type f > .collect_list
# latest model dir of every maze checkpoint (others are intermediate)
for d in maze_train/checkpoints/*/*/models; do [ -d "$d" ] && find "$d/$(ls "$d" | sort -n | tail -1)" -type f; done >> .collect_list
tar czf "$out" -T .collect_list && echo "$RUNS/$out ($(du -h "$out" | cut -f1))"
