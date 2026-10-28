#!/bin/bash
# Batch 21 scored runs, CartPole and FrozenLake (LunarLander on hold, amendment 1).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
for env in cartpole frozenlake; do
  echo "== $env scored $(date)"
  python -m acl_bench.suite.run --env $env --arms DR PLR RPLR ACCEL SFL --seeds 1-20 --workers 8 --allow-battery --resume 2>&1 | grep -v objc
done
echo "== done $(date)"
