#!/bin/zsh
# Batch 9 stage A (Amendment 4): wider SFL scouting pools on CartPole, Acrobot, Pendulum seeds 1-8.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot pendulum; do
  python -m acl_bench.plr.screen --env $e --configs sfl_n4k sfl_n8k --seeds 1-8 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/batch9.csv
done
