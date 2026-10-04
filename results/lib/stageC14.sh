#!/bin/zsh
# Batch 14 stage C (Amendment 5): fresh seeds 1101-1200, DR + sfl_tilt_carry on all 4 envs. Starts after batch 15.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
while pgrep -f "batch15[.]sh" > /dev/null; do sleep 60; done
for e in cartpole acrobot pendulum mountaincar; do
  python -m acl_bench.plr.screen --env $e --configs DR sfl_tilt_carry sfl_spread_carry --seeds 1101-1200 --checks 10 --workers 10 \
    --allow-battery --resume --out results/$e/a5/c14.csv
done
echo finished
