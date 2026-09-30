#!/bin/zsh
# Batch 7 stage A (docs/library_log.md): (b) snippets, (c) task-aware PPO with its own DR baseline.
# DR_po rows lack lib_counts, so it gets its own file (batch7po.csv).
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot; do
  python -m acl_bench.plr.screen --env $e --configs var_snip8 var_snip4 var_low_po --seeds 1-8 \
    --checks 10 --workers 4 --allow-battery --resume --out results/$e/lib/batch7.csv
  python -m acl_bench.plr.screen --env $e --configs DR_po --seeds 1-8 \
    --checks 10 --workers 4 --allow-battery --resume --out results/$e/lib/batch7po.csv
done
