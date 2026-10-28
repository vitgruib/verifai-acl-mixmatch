#!/bin/bash
# Batch 21 amendment 2: ACCEL rerun after the fresh-level fix (563f0c1).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
for env in cartpole frozenlake; do
  echo "== $env ACCEL $(date)"
  python -m acl_bench.suite.run --env $env --arms ACCEL --seeds 1-20 --workers 8 --allow-battery --resume 2>&1 | grep -v objc
done
echo "== done $(date)"
