#!/bin/zsh
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
A=(var_low var_low16)
python -m acl_bench.plr.screen --env cartpole --configs $A --seeds 9-72 --checks 10 --workers 6 \
  --allow-battery --resume --out results/cartpole/lib/batch5.csv
for e in mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs $A --seeds 1-8 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/batch5.csv
done
