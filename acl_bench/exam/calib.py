"""Amendment 5 suites (docs/protocol.md): hard questions recalibrated on independent DR agents,
and an adversarial (CVaR) level sample.

Calibration agents: DR runs on seeds 5001-5010 (outside every arm's pool), final check-in,
loaded from `results/<env>/snapshots/DR/<seed>.npz` (`results/lib/calib5.sh`).

  - `calib`: every question in the VerifAI search pools (`SEARCH_{ce,mab,sa}`) that the
    calibration agents pass between 10% and 60% of the time (1-6 of 10). At least one agent
    passes it, and the grader is deterministic, so it is proven winnable. Capped at
    `--max-n` by a fixed-seed sample; `ref_fail_frac` = 1 - pass fraction.
  - `adv`: `--levels` uniform tasks x `ADV_STARTS` starts from a fixed seed, kept if every
    start is passed by some calibration agent (proven winnable), first `--keep` levels.
    Rows are grouped by level; `evaluate_sets` reports `adv/cvar`, the mean per-level
    success over the agent's own worst 10% of levels (SFL's CVaR evaluation, 2408.15099).

    python -m acl_bench.exam.calib --env acrobot
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from acl_bench import envs
from acl_bench.exam.grader import grade
from acl_bench.exam.sets import ADV_STARTS, PairSet, load_sets, save_sets
from acl_bench.sampling import ADAPTIVE_SAMPLERS
from acl_bench.snapshots import load_run

CALIB_SEEDS = range(5001, 5011)


def calib_agents(env_name: str) -> list:
    paths = [f"results/{env_name}/snapshots/DR/{s}.npz" for s in CALIB_SEEDS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"calibration snapshots missing: {missing}")
    agents = []
    for p in paths:
        run = load_run(p)
        agents.append(run[max(run)])
    return agents


def pass_matrix(env, agents, params, s0) -> np.ndarray:
    """(n_agents, n_questions) success."""
    return np.stack([grade(env, a, params, s0)[0] for a in agents])


def calib_suite(env, agents, search_dir: str, lo: float, hi: float, max_n: int, seed: int):
    pools = load_sets(search_dir, names=[f"SEARCH_{s}" for s in ADAPTIVE_SAMPLERS])
    params = np.concatenate([p.params for p in pools.values()])
    s0 = np.concatenate([p.s0 for p in pools.values()])
    frac = pass_matrix(env, agents, params, s0).mean(axis=0)
    keep = np.flatnonzero((frac >= lo - 1e-9) & (frac <= hi + 1e-9))
    n_band = len(keep)
    if n_band > max_n:
        keep = np.sort(np.random.default_rng(np.random.SeedSequence(seed)).choice(keep, max_n, replace=False))
    suite = PairSet("calib", params[keep], s0[keep], 1 - frac[keep])
    hist = np.bincount(np.round(frac * len(agents)).astype(int), minlength=len(agents) + 1)
    return suite, {"calib_seeds": list(CALIB_SEEDS), "band": [lo, hi], "n_pool": int(len(params)),
                   "pass_count_hist": hist.tolist(), "n_in_band": int(n_band), "sample_seed": seed,
                   "mean_pass_frac": float(1 - suite.ref_fail_frac.mean()) if len(suite) else float("nan")}


def adv_suite(env, agents, n_levels: int, keep_n: int, seed: int):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    lo, hi = (np.array([env.PARAM_BOUNDS[k][i] for k in env.PARAM_ORDER]) for i in (0, 1))
    levels = rng.uniform(lo, hi, size=(n_levels, len(lo)))
    params = np.repeat(levels, ADV_STARTS, axis=0)
    s0 = np.concatenate([env.sample_starts(p, ADV_STARTS, rng) for p in levels])
    ok = pass_matrix(env, agents, params, s0)
    winnable = ok.any(axis=0).reshape(n_levels, ADV_STARTS).all(axis=1)
    kept = np.flatnonzero(winnable)[:keep_n]
    rows = (kept[:, None] * ADV_STARTS + np.arange(ADV_STARTS)).ravel()
    level_pass = ok.mean(axis=0).reshape(n_levels, ADV_STARTS).mean(axis=1)[kept]
    suite = PairSet("adv", params[rows], s0[rows], 1 - ok.mean(axis=0)[rows])
    return suite, {"calib_seeds": list(CALIB_SEEDS), "n_drawn": n_levels, "n_winnable": int(winnable.sum()),
                   "n_kept": int(len(kept)), "starts": ADV_STARTS, "seed": seed,
                   "calib_mean_level_pass": float(level_pass.mean()) if len(kept) else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=envs.NAMES)
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--max-n", type=int, default=2000)
    ap.add_argument("--levels", type=int, default=1000)
    ap.add_argument("--keep", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--dry", action="store_true", help="print counts, save nothing")
    args = ap.parse_args()
    env, exam = envs.get(args.env), envs.exam_dir(args.env)
    agents = calib_agents(args.env)
    calib, ci = calib_suite(env, agents, envs.search_dir(args.env), args.lo, args.hi, args.max_n, args.seed)
    print(f"calib: {len(calib)} questions {ci}", flush=True)
    adv, ai = adv_suite(env, agents, args.levels, args.keep, args.seed)
    print(f"adv: {len(adv)} questions {ai}", flush=True)
    if not args.dry:
        save_sets({"calib": calib, "adv": adv}, exam, extra={"calib": ci, "adv": ai})
        print(f"saved -> {exam}")


if __name__ == "__main__":
    main()
