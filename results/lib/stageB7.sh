#!/bin/zsh
# Batch 7 stage B (docs/library_log.md): var_snip4 vs DR; var_low_po vs DR_po.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole mountaincar pendulum; do
  s=1-8; [[ $e == cartpole ]] && s=9-72
  python -m acl_bench.plr.screen --env $e --configs var_snip4 var_low_po --seeds $s --checks 10 --workers 5 \
    --allow-battery --resume --out results/$e/lib/batch7.csv
  python -m acl_bench.plr.screen --env $e --configs DR_po --seeds $s --checks 10 --workers 5 \
    --allow-battery --resume --out results/$e/lib/batch7po.csv
done
