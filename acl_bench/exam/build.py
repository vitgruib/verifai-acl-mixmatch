"""Build an environment's two locked exam suites (docs/exam.md).

  - `random`: `--n` random tasks from the task box, one start each, from a fixed seed;
    kept only if proven winnable (acl_bench.exam.certify). Built first: the reference
    agents are chosen by it.
  - `verifai`: every question the VerifAI samplers (ce, mab, sa) discovered
    (acl_bench.exam.search) that at least `--min-fail-frac` of the reference agents fail
    (default 0.6, i.e. 6+ of 10, including questions every agent fails) and that is
    proven winnable: some reference agent passed it (the grader is deterministic, so its
    own play is the proof) or the planner found a replayed certificate.

    python -m acl_bench.exam.build --env acrobot random
    python -m acl_bench.exam.build --env acrobot verifai

CartPole's suites were built by the same rules, with its impossibility proofs
(acl_bench.exam.feasibility) reported alongside; they are locked and not rebuilt.
"""
from __future__ import annotations

import argparse

import numpy as np

from acl_bench import envs
from acl_bench.exam.certify import certify
from acl_bench.exam.sets import PairSet, load_sets, remove_sets, save_sets
from acl_bench.sampling import ADAPTIVE_SAMPLERS


def random_suite(env, n: int, seed: int) -> tuple[PairSet, dict]:
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    lo, hi = (np.array([env.PARAM_BOUNDS[k][i] for k in env.PARAM_ORDER]) for i in (0, 1))
    params = rng.uniform(lo, hi, size=(n, len(lo)))
    s0 = np.concatenate([env.sample_starts(p, 1, rng) for p in params])
    ok = certify(env, params, s0)
    return PairSet("random", params[ok], s0[ok]), {"n_drawn": n, "n_not_certified": int((~ok).sum()), "seed": seed}


def verifai_suite(env, search_dir: str, min_fail_frac: float) -> tuple[PairSet, dict]:
    pools = load_sets(search_dir, names=[f"SEARCH_{s}" for s in ADAPTIVE_SAMPLERS])
    params = np.concatenate([p.params for p in pools.values()])
    s0 = np.concatenate([p.s0 for p in pools.values()])
    fail_frac = np.concatenate([p.ref_fail_frac for p in pools.values()])
    hard = np.flatnonzero(fail_frac >= min_fail_frac - 1e-9)          # fail_frac is k/n_agents
    all_fail = hard[fail_frac[hard] == 1]
    print(f"{len(hard)} questions with fail_frac >= {min_fail_frac}; certifying the {len(all_fail)} "
          "that every reference agent fails...", flush=True)
    winnable = np.zeros(len(params), dtype=bool)
    winnable[hard[fail_frac[hard] < 1]] = True
    winnable[all_fail] = certify(env, params[all_fail], s0[all_fail])
    keep = hard[winnable[hard]]
    suite = PairSet("verifai", params[keep], s0[keep], fail_frac[keep])
    return suite, {"samplers": list(ADAPTIVE_SAMPLERS), "min_fail_frac": min_fail_frac,
                   "n_candidates": int(len(hard)), "n_failed_by_all": int(len(all_fail)),
                   "n_failed_by_all_certified": int(winnable[all_fail].sum()),
                   "mean_fail_frac": float(suite.ref_fail_frac.mean()) if len(suite) else float("nan")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=envs.NAMES)
    parser.add_argument("suite", choices=("random", "verifai"))
    parser.add_argument("--n", type=int, default=300, help="random suite: tasks drawn")
    parser.add_argument("--seed", type=int, default=20260924, help="random suite: seed")
    parser.add_argument("--min-fail-frac", type=float, default=0.6)
    args = parser.parse_args()
    env, exam = envs.get(args.env), envs.exam_dir(args.env)

    if args.suite == "random":
        suite, info = random_suite(env, args.n, args.seed)
    else:
        suite, info = verifai_suite(env, envs.search_dir(args.env), args.min_fail_frac)
        remove_sets([n for n in load_sets(exam) if n != "random"], exam)
    save_sets({suite.name: suite}, exam, extra={suite.name: info})
    print(f"{info}\n{suite.name}: {len(suite)} questions -> {exam}")


if __name__ == "__main__":
    main()
