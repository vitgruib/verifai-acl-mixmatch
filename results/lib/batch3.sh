#!/bin/zsh
cd /Users/ethancai/Projects/Ongoing/ACL27
while pgrep -f stageB1.sh >/dev/null; do sleep 30; done
for e in cartpole acrobot; do
  python -m acl_bench.plr.screen --env $e --configs sir_is_a1 sir_is_u50 rw_a4 ens_a4 prog_a4 --seeds 1-8 \
    --checks 10 --workers 5 --allow-battery --resume --out results/$e/lib/batch3.csv
done
