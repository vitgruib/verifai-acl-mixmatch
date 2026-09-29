#!/bin/zsh
cd /Users/ethancai/Projects/Ongoing/ACL27
while pgrep -f batch3.sh >/dev/null; do sleep 30; done
python -m acl_bench.plr.screen --env cartpole --configs DR --seeds 49-96 --checks 10 --workers 6 \
  --allow-battery --resume --out results/cartpole/lib/runs.csv
python -m acl_bench.plr.screen --env cartpole --configs sir_a4 sir_is --seeds 25-72 --checks 10 --workers 6 \
  --allow-battery --resume --out results/cartpole/lib/batch1.csv
