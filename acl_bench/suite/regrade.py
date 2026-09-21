"""Grade saved snapshots against an exam, in the same CSV format as run_arms.

    python -m acl_bench.suite.regrade --snapshots results/snapshots \\
        --sets frozen_sets/cartpole_v2 --out results/regraded.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from acl_bench.suite.sets import evaluate_sets, load_sets
from acl_bench.suite.snapshots import find_runs, load_run


def regrade(snapshot_dir: str, sets_dir: str) -> pd.DataFrame:
    sets = load_sets(sets_dir)
    rows = []
    for arm, replicate, path in find_runs(snapshot_dir):
        for step, agent in load_run(path).items():
            rows.append({"arm": arm, "seed": replicate, "step": step, **evaluate_sets(agent, sets)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--sets", default="frozen_sets/cartpole_v2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    df = regrade(args.snapshots, args.sets)
    df.to_csv(args.out, index=False)
    print(f"graded {df.groupby(['arm', 'seed']).ngroups} runs, {len(df)} snapshots -> {args.out}")


if __name__ == "__main__":
    main()
