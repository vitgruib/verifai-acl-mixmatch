#!/bin/zsh
# Batch 7 stage A (docs/library_log.md): (b) snippets, (c) task-aware PPO with its own DR baseline.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot; do
  python -m acl_bench.plr.screen --env $e --configs var_snip8 var_snip4 DR_po var_low_po --seeds 1-8 \
    --checks 10 --workers 4 --allow-battery --resume --out results/$e/lib/batch7.csv
done
