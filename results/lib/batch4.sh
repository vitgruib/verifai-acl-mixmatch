#!/bin/zsh
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot; do
  python -m acl_bench.plr.screen --env $e --configs var_a2 var_a4 var_a8 var_is --seeds 1-8 \
    --checks 10 --workers 6 --allow-battery --resume --out results/$e/lib/batch4.csv
done
