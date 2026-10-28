"""Train the suite's arms and grade them on the frozen exam.

    python -m acl_bench.suite.run --env cartpole --arms DR PLR RPLR ACCEL SFL --seeds 1-20 \\
        --workers 6 --allow-battery --resume [--steps S] [--checks K]

Every section present in the exam (frozen_sets/suite/<env>) is graded at `--checks` evenly
spaced check-ins; with no exam yet (the REF runs the exam is built from) runs only train and
save. One row per (arm, seed, section, check-in) goes to results/suite/<env>/scores.csv:
`pass_rate` is the mean per-question pass rate, `steps` the training env steps at the
check-in and `scouted_steps` SFL's scouting, which together never exceed the budget. Each
run's agent at every check-in goes to results/suite/<env>/snapshots/<arm>/<seed>.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import pandas as pd

from acl_bench import envs
from acl_bench.suite.arms import ARMS, BUDGET


def run_seed(arm: str, seed: int) -> int:
    return int(hashlib.sha256(f"suite:{arm}:{seed}".encode()).hexdigest()[:8], 16)


def load_exam(directory: str) -> dict:
    from acl_bench.exam.sets import load_sets
    return load_sets(directory) if os.path.exists(os.path.join(directory, "manifest.json")) else {}


def job_fn(job):
    import torch
    torch.set_num_threads(1)
    from acl_bench.plr.fast import FastConfig, train
    from acl_bench.snapshots import save_run
    from acl_bench.suite.exam import STARTS, question_scores
    env_name, arm, seed, steps, n_checks, exam, out = job
    env, sets, levels = envs.get(env_name), load_exam(exam), ARMS[arm]
    snapshots = {}

    def on_check(step, agent):
        if step == 0:
            return {}
        snapshots[step] = {k: v.detach().cpu().numpy().copy() for k, v in agent.state_dict().items()}
        return {"scores": {name: float(question_scores(env, agent, ps).mean()) for name, ps in sets.items()}}

    t0 = time.time()
    _, checks, _ = train(env_name, env, FastConfig(steps=steps, seed=run_seed(arm, seed), levels=levels),
                         n_checks, on_check)
    wall = time.time() - t0
    save_run(os.path.join(out, "snapshots", arm, f"{seed}.npz"), snapshots)
    rows = []
    for c in checks[1:]:
        scouted = c["scouted_steps"]
        trained = c["step"] - (scouted if levels.charge_scouting else 0)
        rows += [{"arm": arm, "seed": seed, "section": name, "n_questions": len(sets[name]) // STARTS,
                  "pass_rate": rate, "steps": trained, "scouted_steps": scouted, "wall_time_sec": wall}
                 for name, rate in c["scores"].items()]
    return rows


def completed(path: Path, sections: list, n_checks: int) -> set:
    """(arm, seed) runs whose rows are all in `path`; rows of incomplete runs are dropped
    from the file so that rerunning them cannot duplicate a row."""
    if not path.exists():
        return set()
    prior = pd.read_csv(path)
    done = {key for key, g in prior.groupby(["arm", "seed"])
            if len(g) == len(sections) * n_checks and set(g.section) == set(sections)}
    keep = [(a, s) in done for a, s in zip(prior.arm, prior.seed)]
    if not all(keep):
        prior[keep].to_csv(path, index=False)
    return done


def main(argv=None):
    from acl_bench.snapshots import find_runs
    from acl_bench.study.run import append_rows, parse_seeds
    from acl_bench.study.safety import Limits, Watchdog, run_jobs
    from acl_bench.suite.exam import exam_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=envs.NAMES)
    ap.add_argument("--arms", nargs="+", required=True, choices=list(ARMS))
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--steps", type=int, help="env steps per run (default: the env's BUDGET)")
    ap.add_argument("--checks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--min-battery-pct", type=int, default=25)
    ap.add_argument("--out", help="default: results/suite/<env>")
    ap.add_argument("--exam", help="default: frozen_sets/suite/<env>")
    args = ap.parse_args(argv)
    steps = args.steps or BUDGET.get(args.env)
    if steps is None:
        raise SystemExit(f"{args.env} has no BUDGET entry yet; pass --steps")
    if steps <= 0 or steps % (1024 * args.checks):
        raise SystemExit(f"--steps must be a positive multiple of 1024 x --checks, got {steps}")
    out = Path(args.out or os.path.join("results", "suite", args.env))
    exam = str(args.exam or exam_dir(args.env))
    sections = list(load_exam(exam))
    scores = out / "scores.csv"
    done = set()
    if args.resume:
        done = (completed(scores, sections, args.checks) if sections else
                {(arm, seed) for arm, seed, _ in find_runs(str(out / "snapshots"))})
    jobs = [(args.env, arm, seed, steps, args.checks, exam, str(out))
            for seed in parse_seeds(args.seeds) for arm in args.arms if (arm, seed) not in done]
    out.mkdir(parents=True, exist_ok=True)
    watchdog = Watchdog(Limits(require_ac=not args.allow_battery, min_battery_pct=args.min_battery_pct),
                        disk_path=str(out))
    print(f"{len(jobs)} runs of {steps} steps on {args.workers} workers, sections: {sections or 'none'}",
          flush=True)

    def on_result(rows):
        if rows:
            append_rows(str(scores), rows)

    failed = run_jobs(jobs, job_fn, args.workers, on_result, watchdog, stop_file="results/STOP",
                      log=lambda m: print(m, flush=True))
    print(f"finished: {len(jobs) - len(failed)} runs, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
