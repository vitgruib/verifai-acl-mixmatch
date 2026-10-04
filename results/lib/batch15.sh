#!/bin/zsh
# Batch 15 stage A (Amendment 5): seeds 1-8 on all 4 dev envs; waits for stage B 14.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
while pgrep -f "stageB14[.]sh" > /dev/null; do sleep 60; done
for e in cartpole acrobot pendulum mountaincar; do
  python -m acl_bench.plr.screen --env $e --configs sfl sfl_spread_carry sfl_spread_verify sfl_spread0 sfl_auto \
    --seeds 1-8 --checks 10 --workers 10 --allow-battery --resume --out results/$e/a5/b15.csv
done
echo finished
