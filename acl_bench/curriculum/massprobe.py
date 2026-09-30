"""Diagnostic (docs/library_log.md, after batch 4): where do the proposers put their mass?

The hard suite's tasks fill a tiny corner of the task box (CartPole: ~0.2% of uniform mass
lies within twice the median nearest-neighbour distance of a dev task). A SIR proposer that
resamples 64 uniform candidates rarely even sees that corner, whatever its score. This trains
plain DR with passive estimators and, at each check-in, draws proposals from several
(estimator, candidates, alpha) SIR configurations and reports
  near_dev: share of proposals near a hard-suite dev task (uniform's share is `uni:near_dev`),
  learn:    mean true p(1-p) of the proposals (greedy agent, random starts) - the quantity a
            frontier sampler is meant to maximize, whatever the environment.

    python -m acl_bench.curriculum.massprobe --env cartpole --seeds 1-2 --checks 4
"""
from __future__ import annotations

import argparse

import numpy as np

CONFIGS = [(m, n, a) for m in ("var", "low", "lowstd") for n in (64, 4096) for a in (4.0, 16.0)]


def sir_draw(score, space, n_cand: int, alpha: float, n: int, rng) -> np.ndarray:
    """`n` independent SIR draws (each from its own `n_cand` uniform candidates)."""
    out = np.empty((n, space.dim))
    per = max(1, 65536 // n_cand)
    for s in range(0, n, per):
        m = min(per, n - s)
        cand = space.uniform(m * n_cand)
        w = (score(cand) + 0.01) ** alpha
        w = w.reshape(m, n_cand)
        w /= w.sum(1, keepdims=True)
        idx = (w.cumsum(1) < rng.uniform(size=(m, 1))).sum(1).clip(max=n_cand - 1)
        out[s:s + m] = cand.reshape(m, n_cand, -1)[np.arange(m), idx]
    return out


def run(env_name: str, seed: int, n_checks: int, n_prop: int, n_starts: int) -> list[dict]:
    import torch
    torch.set_num_threads(1)
    from scipy.spatial import cKDTree
    import acl_bench.curriculum as C
    from acl_bench import envs
    from acl_bench.curriculum.core import TaskSpace
    from acl_bench.curriculum.estimators import PassModel, VarModel
    from acl_bench.exam.grader import grade
    from acl_bench.exam.sets import load_sets
    from acl_bench.plr.fast import FastConfig, LevelConfig, train
    from acl_bench.plr.screen import split_dev_test
    from acl_bench.study.arms import BUDGET

    C.ARMS["_diag_dr"] = {"mix": {"uniform": 1.0}}
    held, orig = {}, C.build

    def build(*a, **kw):
        held["cur"] = orig(*a, **kw)
        return held["cur"]
    C.build = build

    env = envs.get(env_name)
    dev = split_dev_test(load_sets(envs.exam_dir(env_name)))["verifai_dev"]
    bounds = np.array([env.PARAM_BOUNDS[k] for k in env.PARAM_ORDER], dtype=np.float64)
    space = TaskSpace(bounds, np.random.default_rng([seed, 98]))
    rng = np.random.default_rng([seed, 99])
    tree = cKDTree(space.unit(dev.params))
    rad = 2 * np.median(tree.query(space.unit(dev.params), k=2)[0][:, 1])
    models = {"var": VarModel(space, seed), "pass": PassModel(space, seed, window=512, refit_every=128, warmup=256)}
    vm = models["var"]
    # low predicted return (falsification / CVaR direction), alone and times the predicted std
    scores = {"var": vm.score, "pass": models["pass"].learnability,
              "low": lambda x: np.exp(-vm.mean(x)), "lowstd": lambda x: vm.score(x) * np.exp(-vm.mean(x))}
    rows = []

    def learn(agent, params):
        s0 = np.concatenate([env.sample_starts(p, n_starts, rng) for p in params])
        ok, _ = grade(env, agent, np.repeat(params, n_starts, axis=0), s0)
        p = ok.reshape(len(params), n_starts).mean(1)
        return float((p * (1 - p)).mean())

    def on_check(step, agent):
        if not rows:
            held["cur"].estimators.extend(models.values())
        row = dict(seed=seed, step=step)
        props = {"uni": space.uniform(n_prop)}
        for m, n, a in CONFIGS:
            props[f"{m}_n{n}_a{int(a)}"] = sir_draw(scores[m], space, n, a, n_prop, rng)
        for name, x in props.items():
            row[f"{name}:near_dev"] = float((tree.query(space.unit(x))[0] < rad).mean())
            row[f"{name}:learn"] = learn(agent, x[:200])
        rows.append(row)
        return {}

    cfg = FastConfig(steps=BUDGET[env_name], seed=seed, levels=LevelConfig(lib="_diag_dr"))
    train(env_name, env, cfg, n_checks, on_check)
    return rows


def main():
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from acl_bench.study.run import parse_seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cartpole")
    ap.add_argument("--seeds", default="1-2")
    ap.add_argument("--checks", type=int, default=4)
    ap.add_argument("--props", type=int, default=2000)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    with ProcessPoolExecutor(len(seeds)) as ex:
        res = list(ex.map(run, [a.env] * len(seeds), seeds, [a.checks] * len(seeds),
                          [a.props] * len(seeds), [a.starts] * len(seeds)))
    df = pd.DataFrame([r for rs in res for r in rs])
    if a.out:
        df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(df.groupby("step").mean(numeric_only=True).drop(columns="seed").round(3).T)


if __name__ == "__main__":
    main()
