#!/bin/bash
# Batch 21 amendment 3: LunarLander at the doubled budget. REF agents, go/no-go, exam, scored runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
env=lunarlander
echo "== REF $(date)"
python -m acl_bench.suite.run --env $env --arms REF --seeds 1-10 --workers 8 --allow-battery --resume 2>&1 | grep -v objc
python - <<'PY'
import pandas as pd
s = pd.read_csv("results/suite/lunarlander/scores.csv")
r = s[(s.arm == "REF") & (s.section == "random")].sort_values("seed")
print("REF random:", r.pass_rate.round(2).tolist())
if (r.pass_rate < 0.2).sum() >= 3:
    raise SystemExit("no-go: 3+ REF agents below 0.2 on the random section (amendment 3)")
PY
mv results/suite/lunarlander/scores.csv results/suite/lunarlander/scores_ref.csv
echo "== exam $(date)"
python -m acl_bench.suite.exam --env $env --iters 3000 --seed 20261028 --workers 8 2>&1 | grep -v objc
echo "== scored $(date)"
python -m acl_bench.suite.run --env $env --arms DR PLR RPLR ACCEL SFL --seeds 1-20 --workers 8 --allow-battery --resume 2>&1 | grep -v objc
echo "== done $(date)"
