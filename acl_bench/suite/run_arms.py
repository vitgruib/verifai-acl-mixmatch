"""Run named arms over independent replicates, grading the locked exam at every checkpoint.

    python -m acl_bench.suite.run_arms --arms N A S_sa B_sa --seeds 1-30 \\
        --steps 400000 --workers 9 --out results/arms.csv

Every (arm, replicate) gets its own independent seed (`arm_seed`), so the runs of
different arms are independent groups and are compared as groups
(acl_bench/suite/compare.py); runs are not matched seed by seed. A seed fully
determines a run. One CSV row per (arm, replicate, checkpoint); step 0 is the
untrained policy. `seed` is the replicate number, `run_seed` the seed actually used.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import multiprocessing as mp
import os
import time

import pandas as pd

from acl_bench.suite.ablation import ARMS, CHECKPOINT_EVERY, DEFAULT_BUDGET


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


def run_job(job: tuple) -> list[dict]:
    arm_name, replicate, steps, checkpoint_every, sets_dir, overrides = job
    seed = arm_seed(arm_name, replicate)
    import torch
    torch.set_num_threads(1)
    from acl_bench.envs.registry import ENV_SPECS
    from acl_bench.potential.functions import resolve_potential_fn
    from acl_bench.ppo import PPOConfig, run_training
    from acl_bench.scenic_sampling import ScenicTaskSampler
    from acl_bench.suite.sets import evaluate_sets, load_sets

    arm, spec, sets = ARMS[arm_name], ENV_SPECS["cartpole"], load_sets(sets_dir)
    cfg = PPOConfig(total_timesteps=steps, seed=seed, acl=arm.acl, **{**arm.ppo, **overrides})
    sampler = ScenicTaskSampler.load(spec.scenic_file, arm.sampler, arm.sampler_params)
    t0 = time.time()
    log, _ = run_training(spec, sampler, resolve_potential_fn(arm.potential_fn, spec.success_return), cfg,
                          checkpoint_every=checkpoint_every, on_checkpoint=lambda a: evaluate_sets(a, sets))
    wall = time.time() - t0
    n_new = sum(m == "new" for m in log.episode_modes)
    return [{"arm": arm_name, "seed": replicate, "run_seed": seed, "learning_rate": cfg.learning_rate, **m, "n_episodes": len(log.episode_modes), "n_new": n_new,
             "wall_time_sec": wall} for m in log.checkpoint_metrics]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True, help=f"names from ablation.ARMS, or 'family:<name>'")
    parser.add_argument("--seeds", default="1-10", help="replicate numbers, e.g. 1-30 or 1,2,5-8; each arm derives its own seeds from them")
    parser.add_argument("--steps", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--sets", default="frozen_sets/cartpole_v2")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="PPOConfig overrides applied to every arm, e.g. learning_rate=1e-4")
    parser.add_argument("--resume", action="store_true", help="skip (arm, seed) pairs already in --out")
    args = parser.parse_args()

    names = []
    for a in args.arms:
        names += [x.name for x in ARMS.values() if x.family == a.split(":", 1)[1]] if a.startswith("family:") else [a]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {sorted(ARMS)}")

    from acl_bench.ppo import PPOConfig
    valid = {f.name for f in dataclasses.fields(PPOConfig)}
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in valid:
            raise SystemExit(f"--set {key}: not a PPOConfig field; valid: {sorted(valid)}")
        overrides[key] = ast.literal_eval(value)

    done = set()
    frames = []
    if args.resume and os.path.exists(args.out):
        prior = pd.read_csv(args.out)
        frames.append(prior)
        done = set(zip(prior["arm"], prior["seed"]))
    jobs = [(n, s, args.steps, args.checkpoint_every, args.sets, overrides)
            for s in parse_seeds(args.seeds) for n in names if (n, s) not in done]
    print(f"{len(jobs)} runs ({len(names)} arms x {len(set(j[1] for j in jobs))} seeds) on {args.workers} workers", flush=True)

    t0, completed = time.time(), 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for rows in pool.imap_unordered(run_job, jobs):
            frames.append(pd.DataFrame(rows))
            completed += 1
            pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
            if completed % 10 == 0 or completed == len(jobs):
                print(f"  {completed}/{len(jobs)} done, {time.time() - t0:.0f}s elapsed", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
