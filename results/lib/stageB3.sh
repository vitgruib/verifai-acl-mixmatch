#!/bin/zsh
cd /Users/ethancai/Projects/Ongoing/ACL27
while pgrep -f batch1R.sh >/dev/null; do sleep 30; done
A=(sir_is_u50 ens_a4 prog_a4)
python -m acl_bench.plr.screen --env cartpole --configs $A --seeds 9-72 --checks 10 --workers 6 \
  --allow-battery --resume --out results/cartpole/lib/batch3.csv
for e in mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs $A --seeds 1-8 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/batch3.csv
done
