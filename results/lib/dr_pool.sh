#!/bin/zsh
# Shared DR pool for docs/protocol.md (replicates 1-48 per dev environment).
cd /Users/ethancai/Projects/Ongoing/ACL27
for e in cartpole acrobot mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs DR --seeds 1-48 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/runs.csv
done
