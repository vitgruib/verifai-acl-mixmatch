#!/bin/zsh
# Batch 10 stage C: sfl_tilt on fresh seeds 1101-1200; DR already run.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole mountaincar pendulum acrobot; do
  python -m acl_bench.plr.screen --env $e --configs sfl_tilt --seeds 1101-1200 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/batch10.csv
done
