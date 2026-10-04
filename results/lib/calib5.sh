#!/bin/zsh
# Amendment 5: independent DR calibration agents (seeds 5001-5010), final snapshots kept.
cd /Users/ethancai/Projects/Ongoing/ACL27
source .venv/bin/activate
for e in cartpole acrobot mountaincar pendulum; do
  python -m acl_bench.plr.screen --env $e --configs DR --seeds 5001-5010 --checks 10 --workers 10 \
    --allow-battery --resume --snapshots --out results/$e/calib/dr_calib.csv
done
echo finished
