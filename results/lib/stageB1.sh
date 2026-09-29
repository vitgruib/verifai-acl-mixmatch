#!/bin/zsh
cd /Users/ethancai/Projects/Ongoing/ACL27
python -m acl_bench.plr.screen --env cartpole --configs fmodel --seeds 9-24 --checks 10 --workers 5 \
  --allow-battery --resume --out results/cartpole/lib/batch1_fmodel.csv
python -m acl_bench.plr.screen --env cartpole --configs sir_a4 sir_is --seeds 9-24 --checks 10 --workers 5 \
  --allow-battery --resume --out results/cartpole/lib/batch1.csv
for e in mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs fmodel --seeds 1-8 --checks 10 --workers 5 \
    --allow-battery --resume --out results/$e/lib/batch1_fmodel.csv
  python -m acl_bench.plr.screen --env $e --configs sir_a4 sir_is --seeds 1-8 --checks 10 --workers 5 \
    --allow-battery --resume --out results/$e/lib/batch1.csv
done
