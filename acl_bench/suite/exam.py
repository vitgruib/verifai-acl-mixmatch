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
import hashlib
import importlib
import multiprocessing as mp
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from acl_bench import envs
from acl_bench.exam.grader import grade
from acl_bench.exam.search import search
from acl_bench.exam.sets import PairSet, save_sets
from acl_bench.snapshots import find_runs, load_run
from acl_bench.suite.custom import MIN_OPTIMAL_SUCCESS, custom_labels, custom_suite

STARTS = 10
N_REF = 10               # reference agents the pre-registered "3 of 10" rule assumes
GRADE_CHUNK = 2_000      # rows per grade call: bounds memory (LunarLander keeps ~140 KB per live Box2D env,
                         # so 2,000 rows ~ 0.3 GB per worker; FrozenLake's states are 232 wide)
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


def _search_one(env, agent, n_iters: int, starts: int, seed: int, i: int) -> PairSet:
    """Agent i's "ce" search, with its own sampler seed and its own start-state generator."""
    sampler_seed = int(np.random.SeedSequence([seed, i]).generate_state(1)[0])
    random.seed(sampler_seed)                         # Scenic / VerifAI draw from these globals
    np.random.seed(sampler_seed)
    return search(env, "ce", [agent], n_iters, starts, np.random.default_rng(np.random.SeedSequence([seed, i, 1])))


def _agent_digest(agent) -> str:
    """Names the agent's weights, so a retrained agent at the same index misses the cache."""
    h = hashlib.sha256()
    for tensor in agent.state_dict().values():
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _cache_path(cache_dir, seed: int, i: int, agent, n_iters: int, starts: int) -> Path:
    return Path(cache_dir) / f"seed{seed}_agent{i}_{_agent_digest(agent)}_iters{n_iters}_starts{starts}.npz"


def _save_search(path: Path, pool: PairSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")                # a run killed mid-write leaves no half cache entry
    np.savez(tmp, params=pool.params, s0=pool.s0, ref_fail_frac=pool.ref_fail_frac)
    os.replace(tmp, path)


def _load_search(path: Path) -> PairSet:
    z = np.load(path)
    return PairSet("SEARCH_ce", z["params"], z["s0"], z["ref_fail_frac"])


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
    torch.set_num_threads(1)


def _search_job(job):
    env_name, agent, n_iters, starts, seed, i = job
    return _search_one(importlib.import_module(env_name), agent, n_iters, starts, seed, i)


def _grade_job(job):
    env_name, agent, params, s0 = job
    return grade(importlib.import_module(env_name), agent, params, s0)[0]


def verifai_section(env, agents: list, n_iters: int, starts: int = STARTS, min_fail_agents: int = 3,
                    seed: int = 0, workers: int = WORKERS, cache_dir=None) -> PairSet:
    """`workers` > 1 runs the searches and the regrade in a spawned process pool and gives the
    same result as `workers` == 1, which runs everything in this process. With `cache_dir`, each
    agent's search is saved as it finishes and reused by a rerun."""
    torch.set_num_threads(1)                          # as in the workers, so both paths compute alike
    rng = np.random.default_rng(seed)
    paths = [_cache_path(cache_dir, seed, i, agent, n_iters, starts) if cache_dir else None
             for i, agent in enumerate(agents)]
    pools = [_load_search(p) if p and p.exists() else None for p in paths]
    todo = [i for i, pool in enumerate(pools) if pool is None]

    def finished(i: int, pool: PairSet) -> None:
        pools[i] = pool
        if paths[i]:
            _save_search(paths[i], pool)

    if workers > 1:
        for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")            # one thread per worker; inherited by spawned workers
        # a worker that dies (e.g. killed for memory) raises BrokenProcessPool here instead of hanging
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
            try:
                futures = {ex.submit(_search_job, (env.__name__, agents[i], n_iters, starts, seed, i)): i for i in todo}
                for future in as_completed(futures):
                    finished(futures[future], future.result())
                rows, s0s, pooled = _pool_tasks(env, pools, starts, rng)
                step = GRADE_CHUNK // starts * starts       # question_scores' chunks: the same grade batches
                graded = [[ex.submit(_grade_job, (env.__name__, a, pooled.params[j:j + step], pooled.s0[j:j + step]))
                           for j in range(0, len(pooled), step)] for a in agents]
                scores = [np.concatenate([f.result() for f in fs]).reshape(-1, starts).mean(axis=1) for fs in graded]
            except Exception:                            # surface it now, not after the queued jobs
                ex.shutdown(cancel_futures=True)
                raise
    else:
        for i in todo:
            finished(i, _search_one(env, agents[i], n_iters, starts, seed, i))
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
    cache = exam_dir(env_name) / ".search_cache"
    sets = {"random": random_section(env, seed=seed),
            "verifai": verifai_section(env, ref_agents, n_iters, min_fail_agents=min_fail_agents, seed=seed,
                                       workers=workers, cache_dir=cache),
            "custom": custom_suite(env, seed=seed)}
    save_sets(sets, str(exam_dir(env_name)), extra={"custom_cases": custom_labels(env), "torch_threads": 1})
    shutil.rmtree(cache)


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
