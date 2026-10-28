"""Train the suite's arms and grade them on the frozen exam.

    python -m acl_bench.suite.run --env cartpole --arms DR PLR RPLR ACCEL SFL --seeds 1-20 \\
        --workers 6 --allow-battery --resume [--steps S] [--checks K]

Every section present in the exam (frozen_sets/suite/<env>) is graded at `--checks` evenly
spaced check-ins; with no exam yet (the REF runs the exam is built from) runs only train and
save. One row per (arm, seed, section, check-in) goes to results/suite/<env>/scores.csv:
`check` is the check-in's nominal step count (budget x k / n_checks), `pass_rate` the share
of questions passed (at least half their plays), `play_rate` the share of plays passed,
`steps` the training env steps at the check-in and `scouted_steps` SFL's scouting, which
together never exceed the budget. A scout that spans several check-ins grades the same
agent for each of them. Each run's agent at every graded check-in goes to
results/suite/<env>/snapshots/<arm>/<seed>.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import pandas as pd

from acl_bench import envs
from acl_bench.suite.arms import ARMS, BUDGET, arm_config


def run_seed(arm: str, seed: int) -> int:
    return int(hashlib.sha256(f"suite:{arm}:{seed}".encode()).hexdigest()[:8], 16)


def load_exam(directory: str) -> dict:
    from acl_bench.exam.sets import load_sets
    return load_sets(directory) if os.path.exists(os.path.join(directory, "manifest.json")) else {}


def section_scores(env, agent, pairset) -> tuple[float, float]:
    """(share of questions passed, share of plays passed)."""
    from acl_bench.suite.exam import PASS_RATE, question_scores
    scores = question_scores(env, agent, pairset)
    return float((scores >= PASS_RATE).mean()), float(scores.mean())


def job_fn(job):
    import torch
    torch.set_num_threads(1)
    from acl_bench.plr.fast import FastConfig, train
    from acl_bench.snapshots import save_run
    from acl_bench.suite.exam import STARTS
    env_name, arm, seed, steps, n_checks, exam, out = job
    env, sets, levels = envs.get(env_name), load_exam(exam), arm_config(arm, env_name)
    snapshots = {}

    def on_check(step, agent):
        if step == 0:
            return {}
        snapshots[step] = {k: v.detach().cpu().numpy().copy() for k, v in agent.state_dict().items()}
        return {"scores": {name: section_scores(env, agent, ps) for name, ps in sets.items()}}

    t0 = time.time()
    _, checks, _ = train(env_name, env, FastConfig(steps=steps, seed=run_seed(arm, seed), levels=levels),
                         n_checks, on_check)
    wall = time.time() - t0
    save_run(os.path.join(out, "snapshots", arm, f"{seed}.npz"), snapshots)
    rows, every, k = [], steps // n_checks, 1
    graded = checks[1:]
    for j, c in enumerate(graded):
        scouted = c["scouted_steps"]
        trained = c["step"] - (scouted if levels.charge_scouting else 0)
        # every threshold this check-in crossed; the last one also takes those the run
        # stopped short of (charged scouting can leave less than a rollout unspent)
        while k <= n_checks and (k * every <= c["step"] or j == len(graded) - 1):
            rows += [{"arm": arm, "seed": seed, "section": name, "check": k * every,
                      "n_questions": len(sets[name]) // STARTS, "pass_rate": passed, "play_rate": played,
                      "steps": trained, "scouted_steps": scouted, "budget": steps, "n_checks": n_checks,
                      "wall_time_sec": wall}
                     for name, (passed, played) in c["scores"].items()]
            k += 1
    return rows


def resume(path: Path, wanted: list, sections: list, budget: int, n_checks: int) -> list:
    """The (arm, seed) runs of `wanted` still to do. Stops if `path` was written with other
    settings; drops the rows of the wanted runs that are incomplete, which are rerun."""
    if not path.exists():
        return wanted
    prior = pd.read_csv(path)
    if not {"budget", "n_checks"} <= set(prior.columns):
        raise SystemExit(f"--resume: {path} has no budget / n_checks columns; use a new --out")
    found = (sorted(set(prior.budget)), sorted(set(prior.n_checks)), sorted(set(prior.section)))
    if found != ([budget], [n_checks], sorted(sections)):
        raise SystemExit(f"--resume: {path} has budget {found[0]}, n_checks {found[1]}, sections {found[2]}; "
                         f"this run has budget {budget}, n_checks {n_checks}, sections {sorted(sections)}. "
                         "Use the same settings or a new --out")
    counts = prior.groupby(["arm", "seed"]).size()
    todo = [key for key in wanted if counts.get(key, 0) != len(sections) * n_checks]
    stale = pd.Series(list(zip(prior.arm, prior.seed))).isin(set(todo)).to_numpy()
    if stale.any():
        prior[~stale].to_csv(path, index=False)
    return todo


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
    runs = [(arm, seed) for seed in parse_seeds(args.seeds) for arm in args.arms]
    if args.resume:
        if sections:
            runs = resume(scores, runs, sections, steps, args.checks)
        else:
            saved = {(arm, seed) for arm, seed, _ in find_runs(str(out / "snapshots"))}
            runs = [key for key in runs if key not in saved]
    jobs = [(args.env, arm, seed, steps, args.checks, exam, str(out)) for arm, seed in runs]
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
