#!/bin/zsh
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot; do
  python -m acl_bench.plr.screen --env $e --configs var_low var_low16 --seeds 1-8 \
    --checks 10 --workers 6 --allow-battery --resume --out results/$e/lib/batch5.csv
done
