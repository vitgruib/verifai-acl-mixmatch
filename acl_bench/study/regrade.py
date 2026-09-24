"""Grade saved snapshots against the exam: one row per (arm, seed, checkpoint).

    python -m acl_bench.study.regrade --env cartpole --n-checkpoints 10 --workers 8

Reads results/<env>/snapshots, grades on frozen_sets/<env>, writes results/<env>/grades.csv.

`--names` grades only some sections, `--n-checkpoints` only an evenly spaced subset
of check-ins ending at the final model, `--arms` only some arms (e.g. to leave out arms
still training, whose snapshot files may be half-written), `--workers` grades runs
in parallel under the same safeguards as training (acl_bench/study/safety.py), and
`--resume` skips (arm, seed) pairs already in grades.csv, appending the rest.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from acl_bench import envs
from acl_bench.exam.sets import evaluate_sets, load_sets
from acl_bench.snapshots import find_runs, load_run
from acl_bench.study.compare import checkpoint_grid


def grade_run(env, arm: str, replicate: int, path: str, sets: dict, n_checkpoints: int | None = None) -> list[dict]:
    agents = load_run(path)
    steps = checkpoint_grid(agents, n_checkpoints) if n_checkpoints else list(agents)
    return [{"arm": arm, "seed": replicate, "step": step, **evaluate_sets(env, agents[step], sets)} for step in steps]


def _grade_job(job):
    import torch                                   # spawned worker: keep torch to one thread
    torch.set_num_threads(1)
    env_name, arm, replicate, path, names, n_checkpoints = job
    return grade_run(envs.get(env_name), arm, replicate, path, load_sets(envs.exam_dir(env_name), names=names),
                     n_checkpoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=envs.NAMES)
    parser.add_argument("--names", nargs="+", default=None, help="sections to grade (default: all)")
    parser.add_argument("--arms", nargs="+", default=None, help="arms to grade (default: all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-checkpoints", type=int, default=None,
                        help="grade only this many evenly spaced checkpoints ending at the final one")
    parser.add_argument("--resume", action="store_true", help="skip (arm, seed) pairs already graded")
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--min-battery-pct", type=int, default=25)
    parser.add_argument("--stop-file", default="results/STOP_REGRADE")
    args = parser.parse_args()
    out = os.path.join(envs.results_dir(args.env), "grades.csv")
    snapshots = os.path.join(envs.results_dir(args.env), "snapshots")

    from acl_bench.study.run import append_rows
    done = set()
    if args.resume and os.path.exists(out):
        prior = pd.read_csv(out, usecols=["arm", "seed"]).drop_duplicates()
        done = set(zip(prior["arm"], prior["seed"]))
    elif os.path.exists(out):
        os.remove(out)
    jobs = [(args.env, arm, rep, path, args.names, args.n_checkpoints) for arm, rep, path in find_runs(snapshots)
            if (args.arms is None or arm in args.arms) and (arm, rep) not in done]
    print(f"grading {len(jobs)} runs on {args.workers} workers", flush=True)

    from acl_bench.study.safety import Limits, Watchdog, run_jobs
    watchdog = Watchdog(Limits(require_ac=not args.allow_battery, min_battery_pct=args.min_battery_pct),
                        disk_path=os.path.dirname(os.path.abspath(out)))
    state = {"done": 0}

    def on_result(rows):
        append_rows(out, rows)
        state["done"] += 1
        if state["done"] % 50 == 0 or state["done"] == len(jobs):
            print(f"  {state['done']}/{len(jobs)} runs graded", flush=True)

    failed = run_jobs(jobs, _grade_job, args.workers, on_result, watchdog, niceness=args.nice,
                      stop_file=args.stop_file, log=lambda m: print(m, flush=True))
    print(f"graded {state['done']} runs -> {out}, {len(failed)} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
