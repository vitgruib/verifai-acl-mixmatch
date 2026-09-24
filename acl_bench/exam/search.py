"""Search the task space for questions the reference agents fail: VerifAI falsification.
Each adaptive VerifAI sampler (ce, mab, sa; acl_bench.sampling) is pointed at the
reference agents instead of a live training run and keeps every question it visits.

A drawn task gets `rho = success_rate - 0.5` over the reference agents and a few starts
(bounded in [-0.5, 0.5], so no z-scoring), so VerifAI's convention holds: a task with
rho below the threshold 0 is a counterexample (its falsifier uses rho <= fal_thres = 0;
the ce and mab samplers update on rho < thres = 0). Every visited (task, start) pair is saved with the
share of reference agents that fail it.

    python -m acl_bench.exam.search --env cartpole --iters 3000 --starts-per-task 5

Reference agents are the REF runs under results/<env>/snapshots; pools go to
frozen_sets/<env>/search.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from acl_bench import envs
from acl_bench.exam.grader import grade
from acl_bench.exam.sets import PairSet, load_sets, save_sets
from acl_bench.sampling import ADAPTIVE_SAMPLERS, ScenicTaskSampler
from acl_bench.snapshots import find_runs, load_run

LAST_K = 5


def pick_reference(env, agents_by_step: dict, suite, last_k: int = LAST_K):
    """(step, agent, score): the best of the last `last_k` check-ins on the random suite
    (ties to the later one). Runs sometimes dip after reaching their plateau, and a
    collapsed check-in would misjudge difficulty. Selecting this way is fine because
    reference agents are not among the methods compared."""
    steps = sorted(agents_by_step)[-last_k:]
    scores = {s: float(grade(env, agents_by_step[s], suite.params, suite.s0)[0].mean()) for s in steps}
    best = max(steps, key=lambda s: (scores[s], s))
    return best, agents_by_step[best], scores[best]


def load_reference_agents(env, snapshot_dir: str, exam_dir: str, arm: str = "REF") -> list:
    suite = load_sets(exam_dir, names=["random"])["random"]
    agents = [pick_reference(env, load_run(path), suite)[1] for a, _rep, path in find_runs(snapshot_dir) if a == arm]
    if len(agents) < 5:
        raise SystemExit(f"need at least 5 reference runs of arm {arm!r}, found {len(agents)}")
    return agents


def search(env, sampler_name: str, agents: list, n_iters: int, starts_per_task: int,
           rng: np.random.Generator) -> PairSet:
    """Runs one sampler's falsification loop; returns every `(task, s0)` pair it
    visited, with the fraction of reference agents that failed it (`ref_fail_frac`)."""
    sampler = ScenicTaskSampler.load(sampler_name, env.SCENIC_FILE)
    params_batches, s0_batches, fail_batches = [], [], []
    for _ in range(n_iters):
        task = sampler.draw(env.PARAM_ORDER)
        task_row = np.array([task[k] for k in env.PARAM_ORDER])
        s0 = env.sample_starts(task_row, starts_per_task, rng)
        params = np.tile(task_row, (starts_per_task, 1))
        success = np.array([grade(env, a, params, s0)[0] for a in agents])
        sampler.give_feedback(float(success.mean() - 0.5))   # over agents AND starts: one rho per task
        params_batches.append(params)
        s0_batches.append(s0)
        fail_batches.append(1.0 - success.mean(axis=0))      # per start, over agents
    return PairSet(f"SEARCH_{sampler_name}", np.concatenate(params_batches),
                   np.concatenate(s0_batches), np.concatenate(fail_batches))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=envs.NAMES)
    parser.add_argument("--iters", type=int, default=3000, help="drawn tasks per sampler")
    parser.add_argument("--starts-per-task", type=int, default=5)
    parser.add_argument("--samplers", nargs="+", default=list(ADAPTIVE_SAMPLERS))
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    env, out = envs.get(args.env), envs.search_dir(args.env)

    agents = load_reference_agents(env, os.path.join(envs.results_dir(args.env), "snapshots"), envs.exam_dir(args.env))
    print(f"loaded {len(agents)} reference agents")
    rng = np.random.default_rng(np.random.SeedSequence(args.seed))   # shared, in --samplers order

    manifest = os.path.join(out, "manifest.json")
    provenance = json.load(open(manifest)).get("provenance", {}) if os.path.exists(manifest) else {}
    for i, name in enumerate(args.samplers):
        pool = search(env, name, agents, args.iters, args.starts_per_task, rng)
        hard = int(((pool.ref_fail_frac >= 0.5) & (pool.ref_fail_frac < 1.0)).sum())
        print(f"{name}: {len(pool)} questions, {hard} failed by 5-9 of {len(agents)} agents "
              f"({100 * hard / len(pool):.1f}%), {int((pool.ref_fail_frac == 1.0).sum())} failed by all")
        provenance[pool.name] = {"iters": args.iters, "starts_per_task": args.starts_per_task,
                                 "seed": args.seed, "order_in_run": args.samplers[:i + 1]}
        save_sets({pool.name: pool}, out, extra={"provenance": provenance})
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
