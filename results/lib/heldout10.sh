#!/bin/zsh
# Held-out no-harm step: sfl_tilt vs DR on PointNav and Maze, seeds 1101-1148.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in pointnav maze; do
  mkdir -p results/$e/lib
  python -m acl_bench.plr.screen --env $e --configs DR sfl_tilt --seeds 1101-1148 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/heldout10.csv
done
