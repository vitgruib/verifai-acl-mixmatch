"""Cross-environment report: every arm's gain over DR on every dev env, standardized.

    python -m acl_bench.plr.xenv [--metric fin_vd] [--min-seeds 8]

Reporting only, not a gate (see docs/protocol.md, proposed Amendment 5). Per env and arm:
the gain over DR on `--metric` (seeds 1-99), its p value and seed count, and the random-suite
gain. The gain is divided by DR's seed-to-seed sd on that env (z), and `mean_z` averages z
over the envs where the suite *resolves*: DR passes between 5% and 95% of it with sd >= 0.01.
A floored suite (MountainCar's hard suite: DR 0.000 +- 0.000) cannot show a gain, and one with
near-zero sd would inflate z, so both are left out of `mean_z` and marked `floor`.
"""
from __future__ import annotations

import argparse

import pandas as pd

from acl_bench import envs
from acl_bench.plr.decide import compare, load
from acl_bench.plr.summarize import per_run

DEV_ENVS = ("cartpole", "acrobot", "mountaincar", "pendulum")


def resolves(dr: pd.Series) -> bool:
    return 0.05 <= dr.mean() <= 0.95 and dr.std() >= 0.01


def table(metric: str = "fin_vd", min_seeds: int = 8) -> tuple[pd.DataFrame, dict]:
    rows, info = {}, {}
    for e in DEV_ENVS:
        df = load(e)
        if df is None:
            continue
        runs = per_run(df[df.seed < 100], envs.get(e))
        dr = runs.loc[runs.config == "DR", metric]
        ok = resolves(dr)
        info[e] = f"DR {dr.mean():.3f}+-{dr.std():.3f} (n={len(dr)}){'' if ok else ' floor'}"
        for arm in runs.config.unique():
            if arm == "DR" or (runs.config == arm).sum() < min_seeds:
                continue
            d, p, n = compare(runs, arm, metric)
            r = compare(runs, arm, "fin_r")[0]
            rows.setdefault(arm, {})[e] = f"{d:+.3f}{'*' if p < .05 else ' '}({n}) r{r:+.3f}"
            if ok:
                rows[arm]["z_" + e] = d / dr.std()
    t = pd.DataFrame(rows).T
    z = t[[c for c in t if c.startswith("z_")]].astype(float)
    t["mean_z"], t["n_env"] = z.mean(axis=1), z.notna().sum(axis=1)
    return t.drop(columns=z.columns).sort_values("mean_z", ascending=False), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="fin_vd")
    ap.add_argument("--min-seeds", type=int, default=8)
    args = ap.parse_args()
    t, info = table(args.metric, args.min_seeds)
    for e, s in info.items():
        print(f"{e:12s} {s}")
    print(t.to_string(float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main()
