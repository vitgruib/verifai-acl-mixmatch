#!/bin/zsh
# Batch 10 stage B: CartPole 9-72, MountainCar 1-8 (Pendulum 1-8 done at stage A). sfl_mut_tilt first.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
python -m acl_bench.plr.screen --env cartpole --configs sfl_mut_tilt --seeds 9-72 --checks 10 --workers 6 --allow-battery --resume --out results/cartpole/lib/batch10.csv
python -m acl_bench.plr.screen --env mountaincar --configs sfl_mut_tilt sfl_mut sfl_tilt --seeds 1-8 --checks 10 --workers 6 --allow-battery --resume --out results/mountaincar/lib/batch10.csv
python -m acl_bench.plr.screen --env cartpole --configs sfl_mut sfl_tilt --seeds 9-72 --checks 10 --workers 6 --allow-battery --resume --out results/cartpole/lib/batch10.csv
