#!/bin/zsh
# Batch 13 stage A (innovation moves on SFL: verify the proxy, amortize scouting): seeds 1-8.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot pendulum; do
  python -m acl_bench.plr.screen --env $e --configs sfl_tilt_verify sfl_tilt_amort --seeds 1-8 --checks 10 --workers 4 \
    --allow-battery --resume --out results/$e/lib/batch13.csv
done
