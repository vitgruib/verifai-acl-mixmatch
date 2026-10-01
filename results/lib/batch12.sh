#!/bin/zsh
# Batch 12 stage A (build on SFL: NCC soft replay, persistent frontier): seeds 1-8.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot pendulum; do
  python -m acl_bench.plr.screen --env $e --configs sfl_tilt_soft sfl_tilt_carry --seeds 1-8 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/batch12.csv
done
