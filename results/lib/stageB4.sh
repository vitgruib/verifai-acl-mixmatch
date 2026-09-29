#!/bin/zsh
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
A=(var_a4 var_is)
python -m acl_bench.plr.screen --env cartpole --configs $A --seeds 9-72 --checks 10 --workers 6 \
  --allow-battery --resume --out results/cartpole/lib/batch4.csv
for e in mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs $A --seeds 1-8 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/batch4.csv
done
