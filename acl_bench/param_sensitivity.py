"""Do an environment's task parameters actually matter? Train one policy under
plain domain randomization (random sampler, ACL off), evaluate it on many
Halton-sampled tasks across the full parameter box, and measure how much of
the per-task performance the parameters explain.

    python -m acl_bench.param_sensitivity --env cartpole --total-timesteps 400000 --seed 1

Reported per environment:
  - noise ceiling: two independent replicate evaluations of every task; their
    correlation r gives the best R^2 any model of the parameters could reach
    (Spearman-Brown, 2r/(1+r) for the mean of both replicates).
  - out-of-sample R^2 of a random forest on the parameters (5-fold CV): the
    share of per-task performance the parameters explain.
  - per-parameter permutation importance (drop in held-out R^2 when that
    parameter is shuffled) with a z-score over folds x repeats, plus the sign
    of its Spearman correlation with performance.
  - the fraction of tasks the policy fails, i.e. how much of the space is
    falsifiable at all.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import stats
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from acl_bench.envs.registry import ENV_SPECS
from acl_bench.potential.functions import resolve_potential_fn
from acl_bench.ppo import PPOConfig, evaluate_agent, run_training
from acl_bench.scenic_sampling import ScenicTaskSampler

N_FOLDS, N_REPEATS = 5, 10


def r2(y, pred):
    return 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=list(ENV_SPECS))
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-tasks", type=int, default=300)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(1)
    spec = ENV_SPECS[args.env]
    names = list(spec.param_bounds)
    lo = np.array([spec.param_bounds[n][0] for n in names])
    hi = np.array([spec.param_bounds[n][1] for n in names])

    cfg = PPOConfig(total_timesteps=args.total_timesteps, seed=args.seed, acl=False)
    _, agent = run_training(spec, ScenicTaskSampler.load(spec.scenic_file, "random"),
                            resolve_potential_fn("neg_return", spec.success_return), cfg)

    unit = qmc.Halton(d=len(names), scramble=True, seed=args.seed).random(args.n_tasks)
    X = lo + unit * (hi - lo)
    tasks = [dict(zip(names, row)) for row in X]
    reps = np.array([[evaluate_agent(agent, spec, [t], episodes_per_task=args.episodes_per_task,
                                     seed=args.seed * 1000 + rep) for t in tasks] for rep in range(2)])
    y = reps.mean(axis=0)

    r = np.corrcoef(reps[0], reps[1])[0, 1]
    ceiling = 2 * r / (1 + r)

    oof = np.zeros_like(y)
    drops = {n: [] for n in names}
    rng = np.random.default_rng(args.seed)
    for train, test in KFold(N_FOLDS, shuffle=True, random_state=args.seed).split(X):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=args.seed, n_jobs=1)
        rf.fit(X[train], y[train])
        oof[test] = rf.predict(X[test])
        base = r2(y[test], oof[test])
        for j, n in enumerate(names):
            for _ in range(N_REPEATS):
                Xp = X[test].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops[n].append(base - r2(y[test], rf.predict(Xp)))

    fail_rate = float(np.mean(y < spec.success_return))
    print(f"\n### {args.env}: {args.total_timesteps:,}-step policy, {args.n_tasks} tasks x "
          f"{2 * args.episodes_per_task} episodes (seed {args.seed})")
    print(f"replicate correlation r={r:.2f} -> noise ceiling R^2={ceiling:.2f}; "
          f"parameters explain out-of-sample R^2={r2(y, oof):.2f}")
    print(f"task return: mean {y.mean():.1f}, 10th-90th pct {np.percentile(y, 10):.1f} to "
          f"{np.percentile(y, 90):.1f}; policy fails on {100 * fail_rate:.0f}% of tasks "
          f"(mean return below {spec.success_return:g})\n")
    print("| parameter | perm. importance (R^2 drop) | z | Spearman with return |")
    print("|---|---|---|---|")
    for j, n in enumerate(names):
        d = np.array(drops[n])
        z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        rho = stats.spearmanr(X[:, j], y).statistic
        print(f"| {n} | {d.mean():.3f} | {z:.0f} | {rho:+.2f} |")


if __name__ == "__main__":
    main()
