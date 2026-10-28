#!/bin/bash
# Packs falsification results, eval CSVs, configs, final checkpoints and logs into one tarball.
# Usage: bash cluster/collect.sh [out.tgz]
set -euo pipefail
source "$(dirname "$0")/env.sh"
out=${1:-$RUNS/atlas_collect_$(date +%Y%m%d_%H%M).tgz}; [[ $out = /* ]] || out=$PWD/$out
cd "$RUNS"
find . \( -path './*_falsify/*' -o -path './maze_train/results/*' -o -name 'config.json' \
  -o -name '*.csv' -o -path './kinetix/*/.hydra/config.yaml' \
  -o -path './kinetix/*/ckpt/*/full_model.pbz2' \) -type f > .collect_list
# only the last Maze checkpoint of each run
for d in maze_train/checkpoints/*/*/models; do [ -d "$d" ] && find "$d/$(ls "$d" | sort -n | tail -1)" -type f; done >> .collect_list
# job logs are always in <repo>/runs/slurm, even when RUNS is elsewhere
L=$ATLAS/runs/slurm; tmp=${out%.tgz}.tar
tar cf "$tmp" -T .collect_list
[ -d "$L" ] && tar rf "$tmp" -C "$ATLAS/runs" slurm
gzip -c "$tmp" > "$out.part" && mv "$out.part" "$out" && rm -f "$tmp" .collect_list
echo "$out ($(du -h "$out" | cut -f1))"
