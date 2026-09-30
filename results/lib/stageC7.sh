#!/bin/zsh
# Re-confirmation of var_low under Amendment 2 (docs/protocol.md): fresh seeds 1101-1200.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs DR --seeds 1101-1200 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/runs.csv
  python -m acl_bench.plr.screen --env $e --configs var_low --seeds 1101-1200 --checks 10 --workers 6 \
    --allow-battery --resume --out results/$e/lib/batch5.csv
done
