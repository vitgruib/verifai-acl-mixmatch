#!/bin/zsh
# Batch 14 stage B (Amendment 5): CartPole seeds 9-72 for the stage A survivors.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
python -m acl_bench.plr.screen --env cartpole --configs sfl_spread sfl_tilt_carry sfl_tilt_verify sfl_halving \
  sfl_tilt_soft --seeds 9-72 --checks 10 --workers 10 --allow-battery --resume --out results/cartpole/a5/b14B.csv
echo finished
