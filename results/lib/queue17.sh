#!/bin/zsh
# Batch 17 stage A, after stage C 14 finishes.
cd /Users/ethancai/Projects/Ongoing/ACL27 && source .venv/bin/activate
while pgrep -f "queue16[.]sh" >/dev/null || pgrep -f "stageC14[.]sh" >/dev/null; do sleep 300; done
for e in cartpole acrobot pendulum mountaincar; do
  python -m acl_bench.plr.screen --env $e --configs sfl_carry_mem25 sfl_spread2_carry sfl_spread_carry_mem \
    --seeds 1-8 --checks 10 --workers 10 --allow-battery --resume --out results/$e/a5/b17.csv
done
echo finished
