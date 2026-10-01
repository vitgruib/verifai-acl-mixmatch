#!/bin/zsh
# Batch 11: sfl_tilt robustness on CartPole (other learners, half budget), seeds 1201-1248.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
S="--seeds 1201-1248 --checks 10 --workers 6 --allow-battery --resume"
python -m acl_bench.plr.screen --env cartpole --configs DR sfl_tilt --frac 0.5 $S --out results/cartpole/lib/robust11half.csv
python -m acl_bench.plr.screen --env cartpole --configs DR_po sfl_tilt_po $S --out results/cartpole/lib/robust11po.csv
python -m acl_bench.plr.screen --env cartpole --configs DR_lr1e3 sfl_tilt_lr1e3 $S --out results/cartpole/lib/robust11lr.csv
