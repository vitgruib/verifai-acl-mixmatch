#!/bin/bash
# Batch 21: REF agents, frozen exams, then scored runs, one task at a time. Stops on the first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
for env in cartpole frozenlake lunarlander; do
  echo "== $env REF $(date)"
  python -m acl_bench.suite.run --env $env --arms REF --seeds 1-10 --workers 8 --allow-battery --resume 2>&1 | grep -v objc
  echo "== $env exam $(date)"
  python -m acl_bench.suite.exam --env $env --iters 3000 --seed 20261028 --workers 8 2>&1 | grep -v objc
done
for env in cartpole frozenlake lunarlander; do
  echo "== $env scored $(date)"
  python -m acl_bench.suite.run --env $env --arms DR PLR RPLR ACCEL SFL --seeds 1-20 --workers 8 --allow-battery --resume 2>&1 | grep -v objc
done
echo "== done $(date)"
