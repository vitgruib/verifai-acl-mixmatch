#!/bin/bash
# Pack what I need back on the laptop: falsifier records, eval CSVs, final checkpoints only.
# Usage: bash cluster/collect.sh [out.tgz]  ->  runs/atlas_collect_<date>.tgz, or out.tgz
set -euo pipefail
source "$(dirname "$0")/env.sh"
out=${1:-$RUNS/atlas_collect_$(date +%Y%m%d_%H%M).tgz}; [[ $out = /* ]] || out=$PWD/$out
cd "$RUNS"
find . \( -path './*_falsify/*' -o -path './maze_train/results/*' -o -name 'config.json' \
  -o -name '*.csv' -o -path './kinetix/*/.hydra/config.yaml' \
  -o -path './kinetix/*/ckpt/*/full_model.pbz2' \) -type f > .collect_list   # kinetix: one ckpt per run
# latest model dir of every maze checkpoint (others are intermediate)
for d in maze_train/checkpoints/*/*/models; do [ -d "$d" ] && find "$d/$(ls "$d" | sort -n | tail -1)" -type f; done >> .collect_list
# SLURM logs always land in <repo>/runs/slurm (#SBATCH --output), even when RUNS is elsewhere:
# appended from there by path (tar -C), so nothing is linked or copied into RUNS.
L=$ATLAS/runs/slurm; tmp=${out%.tgz}.tar
tar cf "$tmp" -T .collect_list
[ -d "$L" ] && tar rf "$tmp" -C "$ATLAS/runs" slurm
gzip -c "$tmp" > "$out.part" && mv "$out.part" "$out" && rm -f "$tmp" .collect_list
echo "$out ($(du -h "$out" | cut -f1))"
