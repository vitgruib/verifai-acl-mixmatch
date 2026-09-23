"""Train named arms over independent replicates, grading the exam at every checkpoint
and optionally saving the agent at each, for re-grading later (acl_bench.study.regrade).

    python -m acl_bench.study.run --arms N A S_sa B_sa --seeds 1-100 \\
        --out results/training_new.csv --snapshots results/snapshots

Every (arm, replicate) gets its own independent seed (`arm_seed`), so the runs of
different arms are independent groups and are compared as groups
(acl_bench/study/compare.py); runs are not matched seed by seed. A seed fully
determines a run. One CSV row per (arm, replicate, checkpoint); step 0 is the
untrained policy. `seed` is the replicate number, `run_seed` the seed actually used.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time

import pandas as pd

from acl_bench.study.arms import ARMS, CHECKPOINT_EVERY, DEFAULT_BUDGET, Arm


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return seeds


def arm_seed(arm_name: str, replicate: int) -> int:
    """Independent 32-bit seed for one (arm, replicate). Hashing the arm name means
    replicate 1 of one arm shares nothing with replicate 1 of another, and adding or
    reordering arms never changes an existing run's seed."""
    return int(hashlib.sha256(f"{arm_name}:{replicate}".encode()).hexdigest()[:8], 16)


def train_arm(arm: Arm, seed: int, steps: int, checkpoint_every: int | None = None, on_checkpoint=None):
    """One training run of `arm` with run seed `seed`; returns (RunLog, Agent)."""
    from acl_bench.ppo import PPOConfig, run_training
    from acl_bench.sampling import ScenicTaskSampler
    from acl_bench.scoring import SCORES
    cfg = PPOConfig(total_timesteps=steps, seed=seed, acl=arm.acl)
    return run_training(ScenicTaskSampler.load(arm.sampler), SCORES[arm.score], cfg,
                        checkpoint_every=checkpoint_every, on_checkpoint=on_checkpoint)


def run_job(job: tuple) -> list[dict]:
    arm_name, replicate, steps, checkpoint_every, sets_dir, snapshot_dir = job
    seed = arm_seed(arm_name, replicate)
    import torch
    torch.set_num_threads(1)
    from acl_bench.exam.sets import evaluate_sets, load_sets

    sets = load_sets(sets_dir)
    snapshots: dict[int, dict] = {}

    def grade(step, agent):
        if snapshot_dir:
            snapshots[step] = {k: v.detach().cpu().numpy().copy() for k, v in agent.state_dict().items()}
        return evaluate_sets(agent, sets)

    t0 = time.time()
    log, _ = train_arm(ARMS[arm_name], seed, steps, checkpoint_every, grade)
    wall = time.time() - t0
    if snapshot_dir:
        from acl_bench.snapshots import save_run
        save_run(os.path.join(snapshot_dir, arm_name, f"{replicate}.npz"), snapshots)
    n_new = sum(m == "new" for m in log.episode_modes)
    return [{"arm": arm_name, "seed": replicate, "run_seed": seed, **m, "n_episodes": len(log.episode_modes),
             "n_new": n_new, "wall_time_sec": wall} for m in log.checkpoint_metrics]


def append_rows(path: str, rows: list[dict]) -> None:
    """Append one run's rows to a CSV (header written once). Append-only: rewriting the
    whole file after every run would write gigabytes over a long run for no reason."""
    df = pd.DataFrame(rows)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            header = f.readline().strip().split(",")
        if header != list(df.columns):
            raise ValueError(f"{path} has different columns than this run's rows; use a new --out")
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True, help="names from acl_bench.study.arms.ARMS")
    parser.add_argument("--seeds", default="1-10", help="replicate numbers, e.g. 1-30 or 1,2,5-8; each arm derives its own seeds from them")
    parser.add_argument("--steps", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--sets", default="frozen_sets/cartpole")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 4)),
                        help="parallel runs; default leaves at least 4 cores free")
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshots", default=None, metavar="DIR",
                        help="also save each run's agent at every check-in under DIR/<arm>/<replicate>.npz")
    parser.add_argument("--resume", action="store_true", help="skip (arm, seed) pairs already in --out")
    parser.add_argument("--nice", type=int, default=10, help="lower workers' CPU priority (0-19)")
    parser.add_argument("--allow-battery", action="store_true", help="keep dispatching on battery power")
    parser.add_argument("--min-free-memory-pct", type=int, default=15)
    parser.add_argument("--min-battery-pct", type=int, default=25,
                        help="with --allow-battery, pause dispatching below this charge")
    parser.add_argument("--stop-file", default="results/STOP",
                        help="create this file to stop starting new runs (running ones finish)")
    args = parser.parse_args()

    names = list(args.arms)
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {sorted(ARMS)}")

    done = set()
    if args.resume and os.path.exists(args.out):
        prior = pd.read_csv(args.out, usecols=["arm", "seed"]).drop_duplicates()
        done = set(zip(prior["arm"], prior["seed"]))
    jobs = [(n, s, args.steps, args.checkpoint_every, args.sets, args.snapshots)
            for s in parse_seeds(args.seeds) for n in names if (n, s) not in done]
    if os.path.exists(args.stop_file):
        raise SystemExit(f"{args.stop_file} exists; remove it before starting")

    from acl_bench.study.safety import Limits, Watchdog, free_disk_gb, run_jobs
    n_checkpoints = args.steps // args.checkpoint_every + 1
    need_gb = len(jobs) * n_checkpoints * 40e3 / 1e9 * (1 if args.snapshots else 0) + 5.0
    where = args.snapshots or os.path.dirname(os.path.abspath(args.out))
    os.makedirs(where, exist_ok=True)
    if free_disk_gb(where) < need_gb:
        raise SystemExit(f"need about {need_gb:.1f} GB free at {where}, have {free_disk_gb(where):.1f}")
    watchdog = Watchdog(Limits(min_free_memory_pct=args.min_free_memory_pct, require_ac=not args.allow_battery,
                               min_battery_pct=args.min_battery_pct),
                        disk_path=where)
    ok, reason = watchdog.status()
    print(f"{len(jobs)} runs ({len(names)} arms) on {args.workers} workers at nice {args.nice}; "
          f"health now: {'ok' if ok else reason}; stop with: touch {args.stop_file}", flush=True)

    t0, state = time.time(), {"completed": 0}

    def on_result(rows):
        append_rows(args.out, rows)
        state["completed"] += 1
        c = state["completed"]
        if c % 10 == 0 or c == len(jobs):
            elapsed = time.time() - t0
            print(f"  {c}/{len(jobs)} done, {elapsed / 60:.1f} min elapsed, "
                  f"~{elapsed / c * (len(jobs) - c) / 60:.0f} min left", flush=True)

    failed = run_jobs(jobs, run_job, args.workers, on_result, watchdog, niceness=args.nice,
                      stop_file=args.stop_file, log=lambda m: print(m, flush=True))
    print(f"finished: {state['completed']} runs written to {args.out}, {len(failed)} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
