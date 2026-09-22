"""Find CartPole edge cases by actually falsifying, instead of assuming which
regions are hard (docs/cartpole_suite.md's E1-E5 came from one early probe with
a since-deleted expert controller, then were fixed as parameter quartiles by
hand -- "weak push", "heavy cart", etc.). Each adaptive VerifAI sampler (ce,
mab, sa) already exists for exactly this job (acl_bench/scenic_sampling.py);
this module points it at the reference agents instead of a live training run
and keeps every question it visits, so hard questions accumulate directly
instead of the 0.31% hit rate the passive random POOL got for E6.

A question's `rho` is `success_rate - 0.5` over the reference agents and a few
starts per drawn task: bounded to [-0.5, 0.5] already, so no z-scoring is
needed (contrast acl_bench/curriculum/plr.py's RunningNormalizer, built for
unbounded per-env returns). VerifAI's convention holds: rho < thres (0.0
here) means counterexample, matching E6's fail_frac >= 0.5 threshold exactly.

    python -m acl_bench.suite.find_edge_cases --snapshots results/snapshots \\
        --iters 4000 --starts-per-task 5 --out frozen_sets/cartpole_v3_search
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from acl_bench.scenic_sampling import ADAPTIVE_SAMPLERS, ScenicTaskSampler
from acl_bench.suite.build_pool import pick_reference
from acl_bench.suite.evaluator import PARAM_ORDER, rollout_steps
from acl_bench.suite.sets import MAX_STEPS, PairSet, load_sets, save_sets
from acl_bench.suite.snapshots import find_runs, load_run

SCENIC_FILE = "acl_bench/scenic_scenarios/cartpole.scenic"


def load_reference_agents(snapshot_dir: str, sets_dir: str, arm: str = "REF") -> list:
    e0 = load_sets(sets_dir, names=["E0"])["E0"]
    agents = []
    for a, _replicate, path in find_runs(snapshot_dir):
        if a != arm:
            continue
        _step, agent, _score = pick_reference(load_run(path), e0)
        agents.append(agent)
    if len(agents) < 5:
        raise SystemExit(f"need at least 5 reference runs of arm {arm!r}, found {len(agents)}")
    return agents


def search(sampler_name: str, agents: list, n_iters: int, starts_per_task: int,
           rng: np.random.Generator) -> PairSet:
    """Runs one sampler's falsification loop; returns every `(task, s0)` pair it
    visited as a PairSet, annotated with the fraction of reference agents that
    failed it (`ref_fail_frac`), the same convention `build_pool_sets` uses."""
    sampler = ScenicTaskSampler.load(SCENIC_FILE, sampler_name)
    params_batches, s0_batches, fail_batches = [], [], []
    for _ in range(n_iters):
        task = sampler.draw(PARAM_ORDER)
        task_row = np.array([task[k] for k in PARAM_ORDER])
        init = task_row[PARAM_ORDER.index("init_range")]
        s0 = rng.uniform(-init, init, size=(starts_per_task, 4))
        params = np.tile(task_row, (starts_per_task, 1))
        success = np.array([rollout_steps(a, params, s0, MAX_STEPS) == MAX_STEPS for a in agents])
        success_rate = success.mean()                    # over agents AND starts: one rho per drawn task
        sampler.give_feedback(float(success_rate - 0.5))  # rho < 0 (thres) => counterexample
        params_batches.append(params)
        s0_batches.append(s0)
        fail_batches.append(1.0 - success.mean(axis=0))  # per-start fail_frac over agents
    return PairSet(f"SEARCH_{sampler_name}", np.concatenate(params_batches),
                    np.concatenate(s0_batches), np.concatenate(fail_batches))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--sets", default="frozen_sets/cartpole_v2", help="where E0 lives, for picking reference agents")
    parser.add_argument("--out", default="frozen_sets/cartpole_v3_search")
    parser.add_argument("--iters", type=int, default=4000, help="drawn tasks per sampler")
    parser.add_argument("--starts-per-task", type=int, default=5)
    parser.add_argument("--samplers", nargs="+", default=list(ADAPTIVE_SAMPLERS))
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    agents = load_reference_agents(args.snapshots, args.sets)
    print(f"loaded {len(agents)} reference agents")
    rng = np.random.default_rng(np.random.SeedSequence(args.seed))

    pools = {}
    for name in args.samplers:
        pool = search(name, agents, args.iters, args.starts_per_task, rng)
        n_hard = int(((pool.ref_fail_frac >= 0.5) & (pool.ref_fail_frac < 1.0)).sum())
        n_impossible = int((pool.ref_fail_frac == 1.0).sum())
        print(f"{name}: {len(pool)} questions drawn, {n_hard} hard (winnable, >=50% fail), "
              f"{n_impossible} nobody passes ({100 * n_hard / len(pool):.1f}% hard hit rate)")
        pools[f"SEARCH_{name}"] = pool

    save_sets(pools, args.out, extra={"iters": args.iters, "starts_per_task": args.starts_per_task,
                                       "samplers": args.samplers, "seed": args.seed})
    print(f"saved raw search pools -> {args.out}")


if __name__ == "__main__":
    main()
