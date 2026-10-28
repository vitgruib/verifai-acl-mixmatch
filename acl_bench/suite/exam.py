"""The suite's `random` and `verifai` exam sections, and the per-question pass rate.

`random`: questions drawn uniformly over the environment's PARAM_BOUNDS (FrozenLake: an
integer layout_seed). `verifai`: the "ce" falsifier is run separately against each reference
agent with the same budget, the visited tasks are pooled, and every pooled task is played by
every reference agent from the same `starts` start states; an agent fails a question when it
passes fewer than half of those plays (4 of 10 fails, 5 of 10 passes), and a task is kept when
at least `min_fail_agents` agents fail it. FrozenLake tasks must also be winnable
(`certify_exact` on every start, `optimal_success` >= 0.5); the continuous tasks are unfiltered.
Each question is `starts` consecutive rows. The whole build is reproducible from `seed`.

    python -m acl_bench.suite.exam --env cartpole --iters 3000 --seed 0

Reference agents are the final snapshots of the REF runs under results/suite/<env>/snapshots;
the exam goes to frozen_sets/suite/<env>.
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np

from acl_bench import envs
from acl_bench.exam.grader import grade
from acl_bench.exam.search import search
from acl_bench.exam.sets import PairSet, save_sets
from acl_bench.snapshots import find_runs, load_run
from acl_bench.suite.custom import MIN_OPTIMAL_SUCCESS, custom_labels, custom_suite

STARTS = 10
N_REF = 10               # reference agents the pre-registered "3 of 10" rule assumes
GRADE_CHUNK = 20_000     # rows per grade call, to bound the memory of FrozenLake's 232-wide states
PASS_RATE = 0.5          # an agent passes a question at this share of its plays or more
WORKERS = max(1, (os.cpu_count() or 1) - 2)


def exam_dir(env_name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "frozen_sets" / "suite" / env_name


def _is_frozenlake(env) -> bool:
    return "layout_seed" in env.PARAM_BOUNDS


def _winnable(env, row: np.ndarray, s0: np.ndarray) -> bool:
    return bool(env.certify_exact(np.tile(row, (len(s0), 1)), s0).all()
                and env.optimal_success(row) >= MIN_OPTIMAL_SUCCESS)


def _pairset(name: str, rows: list, s0s: list, starts: int, ref_fail_frac=None) -> PairSet:
    return PairSet(name, np.repeat(np.stack(rows), starts, axis=0), np.concatenate(s0s), ref_fail_frac)


def question_scores(env, agent, pairset: PairSet, starts: int = STARTS) -> np.ndarray:
    """Per question, the share of its `starts` plays the agent passes."""
    step = GRADE_CHUNK // starts * starts
    success = np.concatenate([grade(env, agent, pairset.params[i:i + step], pairset.s0[i:i + step])[0]
                              for i in range(0, len(pairset), step)])
    return success.reshape(-1, starts).mean(axis=1)


def random_section(env, n_questions: int = 300, starts: int = STARTS, seed: int = 0) -> PairSet:
    rng = np.random.default_rng(seed)
    bounds = [env.PARAM_BOUNDS[k] for k in env.PARAM_ORDER]
    rows, s0s = [], []
    while len(rows) < n_questions:
        row = np.array([rng.integers(lo, hi + 1) if k == "layout_seed" else rng.uniform(lo, hi)
                        for k, (lo, hi) in zip(env.PARAM_ORDER, bounds)], dtype=np.float64)
        s0 = env.sample_starts(row, starts, rng)
        if not _is_frozenlake(env) or _winnable(env, row, s0):
            rows.append(row)
            s0s.append(s0)
    return _pairset("random", rows, s0s, starts)


def _search_one(env, agent, n_iters: int, starts: int, seed: int, i: int, rng) -> PairSet:
    sampler_seed = int(np.random.SeedSequence([seed, i]).generate_state(1)[0])
    random.seed(sampler_seed)                         # Scenic / VerifAI draw from these globals
    np.random.seed(sampler_seed)
    return search(env, "ce", [agent], n_iters, starts, rng)


def _pool_tasks(env, pools: list[PairSet], starts: int, rng):
    """The searches' tasks deduplicated in agent order, each with `starts` fresh start states."""
    tasks: dict[tuple, np.ndarray] = {}
    for pool in pools:
        for row in pool.params[::starts]:
            if _is_frozenlake(env):
                row = row.copy()
                row[env.PARAM_ORDER.index("layout_seed")] = np.floor(row[env.PARAM_ORDER.index("layout_seed")])
            tasks.setdefault(tuple(row), row)
    rows = list(tasks.values())
    s0s = [env.sample_starts(row, starts, rng) for row in rows]
    return rows, s0s, _pairset("verifai", rows, s0s, starts)


def _init_worker() -> None:
    import torch
    torch.set_num_threads(1)                          # spawned worker: keep torch to one thread


def _search_job(job):
    env_name, agent, n_iters, starts, seed, i, state = job
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    pool = _search_one(importlib.import_module(env_name), agent, n_iters, starts, seed, i, rng)
    return pool, rng.bit_generator.state


def _grade_job(job):
    env_name, agent, params, s0 = job
    return grade(importlib.import_module(env_name), agent, params, s0)[0]


def _parallel_searches(env, agents: list, n_iters: int, starts: int, seed: int, rng, pool) -> list[PairSet]:
    """The serial searches share `rng` in agent order. A search draws from it only through
    `sample_starts`, `n_iters` times, and how much each call draws does not depend on the task,
    so each search's start state is found by replaying those calls on a fixed row; every
    search's end state is then checked against the next one's start."""
    row = np.array([env.PARAM_BOUNDS[k][0] for k in env.PARAM_ORDER], dtype=np.float64)
    states = []
    for _ in agents:
        states.append(rng.bit_generator.state)
        for _ in range(n_iters):
            env.sample_starts(row, starts, rng)
    results = pool.map(_search_job, [(env.__name__, agent, n_iters, starts, seed, i, states[i])
                                     for i, agent in enumerate(agents)])
    for i, (_, end) in enumerate(results):
        if end != (states[i + 1] if i + 1 < len(agents) else rng.bit_generator.state):
            raise RuntimeError(f"search {i} drew from rng differently than the replay predicted")
    return [p for p, _ in results]


def _parallel_scores(env, agents: list, pairset: PairSet, starts: int, pool) -> list[np.ndarray]:
    """`question_scores` of every agent, one job per (agent, chunk), the chunks those of
    `question_scores` so every grade call sees the same batch as the serial code."""
    step = GRADE_CHUNK // starts * starts
    chunks = range(0, len(pairset), step)
    flat = pool.map(_grade_job, [(env.__name__, a, pairset.params[i:i + step], pairset.s0[i:i + step])
                                 for a in agents for i in chunks])
    return [np.concatenate(flat[k * len(chunks):(k + 1) * len(chunks)]).reshape(-1, starts).mean(axis=1)
            for k in range(len(agents))]


def verifai_section(env, agents: list, n_iters: int, starts: int = STARTS, min_fail_agents: int = 3,
                    seed: int = 0, workers: int = WORKERS) -> PairSet:
    """`workers` > 1 runs the searches and the regrade in a spawned process pool and gives the
    same result as `workers` == 1, which runs everything in this process."""
    rng = np.random.default_rng(seed)
    if workers > 1:
        for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")            # one thread per worker; inherited by spawned workers
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            pools = _parallel_searches(env, agents, n_iters, starts, seed, rng, pool)
            rows, s0s, pooled = _pool_tasks(env, pools, starts, rng)
            scores = _parallel_scores(env, agents, pooled, starts, pool)
    else:
        pools = [_search_one(env, agent, n_iters, starts, seed, i, rng) for i, agent in enumerate(agents)]
        rows, s0s, pooled = _pool_tasks(env, pools, starts, rng)
        scores = [question_scores(env, a, pooled, starts) for a in agents]
    fails = sum(s < PASS_RATE for s in scores)
    failing = [j for j, n in enumerate(fails) if n >= min_fail_agents]
    keep = [j for j in failing if _winnable(env, rows[j], s0s[j])] if _is_frozenlake(env) else failing
    if not keep:
        raise SystemExit(f"no verifai question survives: {len(rows)} pooled, {len(failing)} failed by "
                         f">= {min_fail_agents} agents, {len(keep)} kept after the winnability filter")
    frac = np.repeat(fails[keep] / len(agents), starts)
    return _pairset("verifai", [rows[j] for j in keep], [s0s[j] for j in keep], starts, frac)


def build_exam(env_name: str, ref_agents: list, n_iters: int, seed: int, min_fail_agents: int = 3,
               workers: int = WORKERS) -> None:
    env = envs.get(env_name)
    sets = {"random": random_section(env, seed=seed),
            "verifai": verifai_section(env, ref_agents, n_iters, min_fail_agents=min_fail_agents, seed=seed,
                                       workers=workers),
            "custom": custom_suite(env, seed=seed)}
    save_sets(sets, str(exam_dir(env_name)), extra={"custom_cases": custom_labels(env)})


def reference_agents(snapshots: Path) -> list:
    """The final snapshot of each REF run; the exam's "3 of 10" rule needs exactly N_REF."""
    runs = [load_run(path) for arm, _rep, path in find_runs(str(snapshots)) if arm == "REF"]
    if len(runs) != N_REF:
        raise SystemExit(f"need exactly {N_REF} REF runs under {snapshots}, found {len(runs)}")
    return [run[max(run)] for run in runs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=envs.NAMES)
    parser.add_argument("--iters", type=int, default=3000, help="drawn tasks per reference agent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=WORKERS, help="processes for the searches and the regrade")
    args = parser.parse_args()
    snapshots = Path("results") / "suite" / args.env / "snapshots"
    build_exam(args.env, reference_agents(snapshots), args.iters, args.seed, workers=args.workers)
    print(f"saved -> {exam_dir(args.env)}")


if __name__ == "__main__":
    main()
