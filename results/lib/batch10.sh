#!/bin/zsh
# Batch 10 stage A (mix and match, docs/literature.md): CartPole, Acrobot, Pendulum seeds 1-8.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
while pgrep -f "batch9[.]sh" > /dev/null; do sleep 60; done
for e in cartpole acrobot pendulum; do
  python -m acl_bench.plr.screen --env $e --configs sfl_mut sfl_tilt sfl_mut_tilt --seeds 1-8 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/batch10.csv
done
