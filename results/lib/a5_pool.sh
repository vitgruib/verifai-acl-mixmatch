#!/bin/zsh
# Batch 14 (Amendment 5): DR pool seeds 1-48, then stage A (seeds 1-8) on all 4 envs.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs DR --seeds 1-48 --checks 10 --workers 10 \
    --allow-battery --resume --out results/$e/a5/dr.csv
done
for e in cartpole acrobot mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs sfl_halving sfl_ghost sfl_spread sfl_bisect sfl_tilt_sc \
    sfl_tilt sfl_tilt_soft sfl_tilt_carry sfl_tilt_verify sfl_tilt_amort --seeds 1-8 --checks 10 --workers 10 \
    --allow-battery --resume --out results/$e/a5/b14.csv
done
echo finished
