#!/bin/zsh
# Stage B 15 (CartPole seeds 9-72), then batch 16 stage A (seeds 1-8 x 4 envs), then stage C 14.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
python -m acl_bench.plr.screen --env cartpole --configs sfl sfl_spread_carry sfl_spread_verify --seeds 9-72 \
  --checks 10 --workers 10 --allow-battery --resume --out results/cartpole/a5/b15B.csv
for e in cartpole acrobot pendulum mountaincar; do
  python -m acl_bench.plr.screen --env $e --configs sfl_carry_mem sfl_carry0 sfl_spread_carry0 --seeds 1-8 \
    --checks 10 --workers 10 --allow-battery --resume --out results/$e/a5/b16.csv
done
echo stage-A-done
zsh results/lib/stageC14.sh
