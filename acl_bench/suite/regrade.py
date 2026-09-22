"""Grade saved snapshots against an exam, in the same CSV format as run_arms.

    python -m acl_bench.suite.regrade --snapshots results/snapshots \\
        --sets frozen_sets/cartpole_v2 --out results/regraded.csv

`--names` grades only some sections, `--arms` only some arms (e.g. to leave out arms
still training, whose snapshot files may be half-written), `--workers` grades runs
in parallel, and `--resume` skips (arm, seed) pairs already in `--out`, appending the rest.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from acl_bench.suite.sets import evaluate_sets, load_sets
from acl_bench.suite.snapshots import find_runs, load_run


def grade_run(arm: str, replicate: int, path: str, sets: dict) -> list[dict]:
    return [{"arm": arm, "seed": replicate, "step": step, **evaluate_sets(agent, sets)}
            for step, agent in load_run(path).items()]


def regrade(snapshot_dir: str, sets_dir: str, names=None, arms=None) -> pd.DataFrame:
    sets = load_sets(sets_dir, names=names)
    rows = []
    for arm, replicate, path in find_runs(snapshot_dir):
        if arms is None or arm in arms:
            rows += grade_run(arm, replicate, path, sets)
    return pd.DataFrame(rows)


def _grade_job(job):
    import torch
    torch.set_num_threads(1)
    arm, replicate, path, sets_dir, names = job
    return grade_run(arm, replicate, path, load_sets(sets_dir, names=names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--sets", default="frozen_sets/cartpole_v2")
    parser.add_argument("--names", nargs="+", default=None, help="sections to grade (default: all)")
    parser.add_argument("--arms", nargs="+", default=None, help="arms to grade (default: all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="skip (arm, seed) pairs already in --out")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from acl_bench.suite.run_arms import append_rows
    done = set()
    if args.resume and os.path.exists(args.out):
        prior = pd.read_csv(args.out, usecols=["arm", "seed"]).drop_duplicates()
        done = set(zip(prior["arm"], prior["seed"]))
    elif os.path.exists(args.out):
        os.remove(args.out)
    jobs = [(arm, rep, path, args.sets, args.names) for arm, rep, path in find_runs(args.snapshots)
            if (args.arms is None or arm in args.arms) and (arm, rep) not in done]
    print(f"grading {len(jobs)} runs on {args.workers} workers", flush=True)

    if args.workers == 1:
        results = map(_grade_job, jobs)
    else:
        from multiprocessing import get_context
        pool = get_context("spawn").Pool(args.workers)
        results = pool.imap_unordered(_grade_job, jobs)
    for i, rows in enumerate(results, 1):
        append_rows(args.out, rows)
        if i % 50 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} runs graded", flush=True)
    print(f"graded {len(jobs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
