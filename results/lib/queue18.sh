#!/bin/zsh
# Batch 18 (start-state SFL) stage A + exploratory seeds 9-16, after stage C 14.
cd /Users/ethancai/Projects/Ongoing/ACL27 && source .venv/bin/activate
while pgrep -f "queue16[.]sh" >/dev/null || pgrep -f "stageC14[.]sh" >/dev/null; do sleep 300; done
for e in cartpole acrobot pendulum mountaincar; do
  python -m acl_bench.plr.screen --env $e --configs sfl_states --seeds 1-16 \
    --checks 10 --workers 10 --allow-battery --resume --out results/$e/a5/b18.csv
  python -m acl_bench.plr.screen --env $e --configs sfl_tilt --seeds 9-16 \
    --checks 10 --workers 10 --allow-battery --resume --out results/$e/a5/b18.csv
done
echo finished
