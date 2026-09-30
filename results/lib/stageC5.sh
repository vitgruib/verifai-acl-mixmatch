#!/bin/zsh
# Stage C for var_low (docs/library_log.md, batch 5): fresh seeds 1001-1100, DR and the arm.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs DR --seeds 1001-1100 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/runs.csv
  python -m acl_bench.plr.screen --env $e --configs var_low --seeds 1001-1100 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/batch5.csv
done
