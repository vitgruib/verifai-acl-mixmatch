"""Diagnostic (docs/library_log.md, batch 4): does the pass model, fit to DR's own training
episodes, rate the hard suite's tasks as frontier?

Trains plain DR (every task uniform) with a passive PassModel watching every episode. At
each check-in it compares, on the hard suite's dev tasks and on uniform tasks:
model p, model p(1-p), and the greedy agent's pass rate from random starts (4 per task).

    python -m acl_bench.curriculum.diagnose --env cartpole --seeds 1-3
"""
from __future__ import annotations

import argparse

import numpy as np


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y > 0.5], score[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def run(env_name: str, seed: int, n_checks: int, n_uniform: int, n_starts: int) -> list[dict]:
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    import acl_bench.curriculum as C
    from acl_bench import envs
    from acl_bench.exam.grader import grade
    from acl_bench.exam.sets import load_sets
    from acl_bench.plr.fast import FastConfig, LevelConfig, train
    from acl_bench.plr.screen import split_dev_test
    from acl_bench.study.arms import BUDGET

    C.ARMS["_diag_dr"] = {"mix": {"uniform": 1.0}}
    held = {}
    orig = C.build

    def build(*a, **kw):
        held["cur"] = orig(*a, **kw)
        return held["cur"]
    C.build = build

    env = envs.get(env_name)
    dev = split_dev_test(load_sets(envs.exam_dir(env_name)))["verifai_dev"]
    bounds = np.array([env.PARAM_BOUNDS[k] for k in env.PARAM_ORDER], dtype=np.float64)
    rng = np.random.default_rng([seed, 99])
    uni = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_uniform, len(bounds)))

    def true_p(agent, params):
        s0 = np.concatenate([env.sample_starts(p, n_starts, rng) for p in params])
        ok, _ = grade(env, agent, np.repeat(params, n_starts, axis=0), s0)
        return ok.reshape(len(params), n_starts).mean(1)

    rows = []
    from acl_bench.curriculum.core import TaskSpace
    from acl_bench.curriculum.estimators import PassModel, VarModel
    sp = TaskSpace(bounds, np.random.default_rng(0))
    cands = {"pass_w512": PassModel(sp, seed, window=512, refit_every=128, warmup=256),
             "var_w512": VarModel(sp, seed),
             "var_w2048": VarModel(sp, seed, window=2048),
             "varS_w512": VarModel(sp, seed, outcome="success")}

    def score(m, x):
        return m.learnability(x) if isinstance(m, PassModel) else m.score(x)

    def on_check(step, agent):
        cur = held["cur"]
        if not rows:
            cur.estimators.extend(cands.values())
        model = cur.estimators[0]
        ys = np.array(model.y) if model.y else np.zeros(0)
        pm_d, pm_u = model.p(dev.params), model.p(uni)
        tp_d, tp_u = true_p(agent, dev.params), true_p(agent, uni)
        exam, _ = grade(env, agent, dev.params, dev.s0)
        lu = pm_u * (1 - pm_u)
        ld = pm_d * (1 - pm_d)
        rows.append(dict(
            seed=seed, step=step, seen=model.seen, refits=model.version, window_fail=float(1 - ys.mean()) if len(ys) else np.nan,
            exam_dev=float(exam.mean()),
            true_p_dev=float(tp_d.mean()), model_p_dev=float(pm_d.mean()),
            true_p_uni=float(tp_u.mean()), model_p_uni=float(pm_u.mean()),
            true_learn_dev=float((tp_d * (1 - tp_d)).mean()), true_learn_uni=float((tp_u * (1 - tp_u)).mean()),
            model_learn_dev=float(ld.mean()), model_learn_uni=float(lu.mean()),
            # where the dev tasks sit in the model's frontier ranking of uniform tasks (0.5 = typical)
            dev_pct=float((ld[:, None] > lu[None, :]).mean()),
            auc_uni=auc(pm_u, (tp_u > 0.5).astype(float)),
            auc_dev=auc(pm_d, (tp_d > 0.5).astype(float)),
            # share of the model's top-10% frontier (uniform tasks) that is truly frontier (0 < p < 1)
            top_true_frontier=float(((tp_u > 0) & (tp_u < 1))[lu >= np.quantile(lu, 0.9)].mean()),
            base_true_frontier=float(((tp_u > 0) & (tp_u < 1)).mean()),
            dev_true_frontier=float(((tp_d > 0) & (tp_d < 1)).mean()),
        ))
        tl_u = tp_u * (1 - tp_u)
        for name, m in [("pass", model), *cands.items()]:
            su, sd = score(m, uni), score(m, dev.params)
            rows[-1][f"{name}:dev_pct"] = float((sd[:, None] > su[None, :]).mean())
            rows[-1][f"{name}:top_front"] = float(((tp_u > 0) & (tp_u < 1))[su >= np.quantile(su, 0.9)].mean())
            rows[-1][f"{name}:rho"] = float(pd.Series(su).corr(pd.Series(tl_u), method="spearman"))
        return {}

    cfg = FastConfig(steps=BUDGET[env_name], seed=seed, levels=LevelConfig(lib="_diag_dr"))
    train(env_name, env, cfg, n_checks, on_check)
    return rows


def main():
    import pandas as pd
    from acl_bench.study.run import parse_seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cartpole")
    ap.add_argument("--seeds", default="1-3")
    ap.add_argument("--checks", type=int, default=5)
    ap.add_argument("--uniform", type=int, default=400)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    seeds = parse_seeds(a.seeds)
    with ProcessPoolExecutor(len(seeds)) as ex:
        res = list(ex.map(run, [a.env] * len(seeds), seeds, [a.checks] * len(seeds),
                          [a.uniform] * len(seeds), [a.starts] * len(seeds)))
    df = pd.DataFrame([r for rs in res for r in rs])
    if a.out:
        df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.groupby("step").mean(numeric_only=True).drop(columns="seed").round(3).T)


if __name__ == "__main__":
    main()
