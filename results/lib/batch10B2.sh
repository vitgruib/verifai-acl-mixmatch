#!/bin/zsh
# Batch 10 stage B, continued: sfl_tilt only (sfl_mut abandoned at 23 seeds, sfl_mut_tilt failed the Pendulum guard).
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
python -m acl_bench.plr.screen --env cartpole --configs sfl_tilt --seeds 9-72 --checks 10 --workers 6 --allow-battery --resume --out results/cartpole/lib/batch10.csv
