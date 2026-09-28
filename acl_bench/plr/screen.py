"""Screen named PLR configurations on the fast harness (acl_bench.plr.fast).

    python -m acl_bench.plr.screen --env acrobot --configs DR plr_paper --seeds 1-8 \\
        --frac 0.5 --out results/acrobot/plr/screen.csv

`--frac` shortens training to that share of the environment's budget. Grades both exam
suites at `--checks` evenly spaced check-ins. One CSV row per (config, seed, check-in).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import time

import pandas as pd

from acl_bench import envs
from acl_bench.plr.levels import LevelConfig

DR = dict(replay_prob=0.0)
SIPACL = dict(replay_prob=0.5, beta=1.0, staleness=0.0, score_ema=0.2, buffer=5000, admit="fifo", min_fill=1)
PAPER = dict(replay_prob=0.5, beta=0.1, staleness=0.1, score_ema=1.0, buffer=1000, admit="min")

# name -> (LevelConfig overrides, FastConfig overrides)
CONFIGS: dict[str, tuple[dict, dict]] = {
    "DR": (DR, {}),
    "sipacl": (SIPACL, {}),
    "paper": (PAPER, {}),
}


def register(name: str, levels: dict, base: dict = PAPER, **fast) -> None:
    CONFIGS[name] = ({**base, **levels}, fast)


# ---- round 1: one direction each, from the PLR/Robust-PLR defaults
register("paper_l1", {"score": "l1"})
register("paper_maxmc", {"score": "maxmc"})
register("paper_robust", {"robust": True})
register("paper_p09", {"replay_prob": 0.9})
register("paper_p02", {"replay_prob": 0.2})
register("paper_beta03", {"beta": 0.3})
register("paper_buf200", {"buffer": 200})
register("paper_negret", {"score": "neg_return"})
register("sipacl_beta01", {"beta": 0.1}, base=SIPACL)
SFL = dict(replay_prob=0.5, sfl=True)
register("sfl", {}, base=SFL)

# ---- round 2: SFL's knobs (round 1 found no PVL-family setting better than DR on Acrobot)
register("sfl_p09", {"replay_prob": 0.9}, base=SFL)
register("sfl_top20", {"sfl_top": 20}, base=SFL)
register("sfl_cheap", {"sfl_n": 500, "sfl_every": 20}, base=SFL)

# ---- round 3: headroom (oracle: half the episodes on the VerifAI suite's own tasks) and
# SFL's MountainCar curve gain with more seeds
register("oracle50", {"replay_prob": 0.5, "oracle": "verifai"}, base=DR)
register("sfl_p09_top20", {"replay_prob": 0.9, "sfl_top": 20}, base=SFL)

# ---- round 6: CartPole, around the leaders (sfl, paper, paper_maxmc)
register("sfl_top50", {"sfl_top": 50}, base=SFL)
register("paper_maxmc_buf200", {"score": "maxmc", "buffer": 200})

# ---- round 7: surprise in either direction (L1), averaged over visits, and negative only
register("l1_avg", {"score": "l1", "score_ema": 0.3})
register("nvl", {"score": "nvl"})

# ---- round 8: make PVL-ranked PLR work. Root cause: a task-blind critic makes PVL track
# easiness. Fixes: a privileged critic (_pc), task-aware networks (_po), exact-start
# replay (_st). Each network change has its own DR baseline.
for _sfx, _fast in (("pc", {"critic_params": True}), ("po", {"obs_params": True})):
    CONFIGS[f"DR_{_sfx}"] = (DR, _fast)
    register(f"paper_{_sfx}", {}, **_fast)
    register(f"sipacl_{_sfx}", {}, base=SIPACL, **_fast)
register("paper_st", {"replay_start": True})
register("paper_pc_st", {"replay_start": True}, critic_params=True)
register("lp", {"score": "lp"})
register("lp_st", {"score": "lp", "replay_start": True})


def run_seed(env_name: str, config: str, replicate: int) -> int:
    return int(hashlib.sha256(f"plr:{config}:{replicate}".encode()).hexdigest()[:8], 16)


def job_fn(job):
    import torch
    torch.set_num_threads(1)
    from acl_bench.exam.sets import load_sets
    from acl_bench.plr.fast import FastConfig, evaluate, train
    env_name, config, replicate, steps, n_checks = job
    env = envs.get(env_name)
    sets = load_sets(envs.exam_dir(env_name))
    lv, fast = CONFIGS[config]
    cfg = FastConfig(steps=steps, seed=run_seed(env_name, config, replicate),
                     levels=dataclasses.replace(LevelConfig(), **lv), **fast)
    t0 = time.time()
    _, checks, stats = train(env_name, env, cfg, n_checks, lambda s, a: evaluate(env, a, sets))
    wall = time.time() - t0
    return [{"config": config, "seed": replicate, **c, **stats, "wall_time_sec": wall} for c in checks]


def main():
    from acl_bench.study.arms import BUDGET
    from acl_bench.study.run import append_rows, parse_seeds
    from acl_bench.study.safety import Limits, Watchdog, run_jobs
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=envs.NAMES)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", default="1-8")
    ap.add_argument("--frac", type=float, default=1.0, help="share of the environment's training budget")
    ap.add_argument("--checks", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--min-battery-pct", type=int, default=25)
    args = ap.parse_args()
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; known: {sorted(CONFIGS)}")
    steps = int(BUDGET[args.env] * args.frac) // (1024 * args.checks) * 1024 * args.checks
    done = set()
    if args.resume and os.path.exists(args.out):
        prior = pd.read_csv(args.out, usecols=["config", "seed", "step"])
        prior = prior[prior.step == prior.groupby(["config", "seed"]).step.transform("max")]
        done = {(c, s) for c, s, st in zip(prior.config, prior.seed, prior.step) if st == steps}
    jobs = [(args.env, c, s, steps, args.checks) for s in parse_seeds(args.seeds) for c in args.configs
            if (c, s) not in done]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    watchdog = Watchdog(Limits(require_ac=not args.allow_battery, min_battery_pct=args.min_battery_pct),
                        disk_path=os.path.dirname(os.path.abspath(args.out)))
    print(f"{len(jobs)} runs of {steps} steps on {args.workers} workers", flush=True)
    t0, n = time.time(), [0]

    def on_result(rows):
        append_rows(args.out, rows)
        n[0] += 1
        if n[0] % 8 == 0 or n[0] == len(jobs):
            el = time.time() - t0
            print(f"  {n[0]}/{len(jobs)} done, {el / 60:.1f} min, ~{el / n[0] * (len(jobs) - n[0]) / 60:.0f} min left",
                  flush=True)

    failed = run_jobs(jobs, job_fn, args.workers, on_result, watchdog, stop_file="results/STOP",
                      log=lambda m: print(m, flush=True))
    print(f"finished: {n[0]} runs, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
