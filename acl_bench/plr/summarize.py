"""Summarize a screen CSV: per config, final (last three check-ins) and curve (mean over
all check-ins after step 0) scores on each suite, as a difference from a baseline
config with a Welch t-test p-value (uncorrected: screening, not confirmation).

Two views of each suite: the pass rate (`fin_r`, `auc_r`; `_v` for the VerifAI suite)
and a continuous score (`fin_rc`, `auc_rc`, ...): the mean return where the environment
has one (Pendulum), otherwise the mean steps scaled to [0, 1] with higher better (steps
survived / cap for a survive goal, 1 - steps to the goal / cap for a reach goal).

    python -m acl_bench.plr.summarize results/acrobot/plr/screen.csv [--base DR]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from acl_bench import envs

SUITES = {"random": "r", "verifai": "v", "heldout": "h"}


def env_of(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    found = [n for n in envs.NAMES if n in parts]
    if len(found) != 1:
        raise SystemExit(f"cannot tell the environment from {path}; pass --env")
    return found[0]


def continuous(df: pd.DataFrame, suite: str, env) -> pd.Series:
    if f"{suite}/mean_return" in df and df[f"{suite}/mean_return"].notna().all():
        return df[f"{suite}/mean_return"]
    frac = df[f"{suite}/mean_steps"] / env.MAX_EPISODE_STEPS
    return frac if env.GOAL == "survive" else 1 - frac


def per_run(df: pd.DataFrame, env) -> pd.DataFrame:
    df = df[df.step > 0].copy()
    aggs = {}
    for suite, tag in SUITES.items():
        if f"{suite}/success" not in df:
            continue
        df[f"_{tag}c"] = continuous(df, suite, env)
        aggs[tag] = (f"{suite}/success", f"_{tag}c")
    # each run's own last three check-ins (runs charged for scouting end at other steps)
    last3 = df.sort_values("step").groupby(["config", "seed"]).tail(3)
    g_all, g_fin = df.groupby(["config", "seed"]), last3.groupby(["config", "seed"])
    out = {}
    for tag, (succ, cont) in aggs.items():
        out[f"auc_{tag}"], out[f"fin_{tag}"] = g_all[succ].mean(), g_fin[succ].mean()
        out[f"auc_{tag}c"], out[f"fin_{tag}c"] = g_all[cont].mean(), g_fin[cont].mean()
    return pd.DataFrame(out).reset_index()


def summarize(df: pd.DataFrame, env, base: str = "DR") -> pd.DataFrame:
    runs = per_run(df, env)
    cols = [c for c in runs.columns if c not in ("config", "seed")]
    b = runs[runs.config == base]
    rows = []
    for c, g in runs.groupby("config"):
        row = {"config": c, "n": len(g)}
        for m in cols:
            row[m] = g[m].mean()
            if c != base:
                row[f"d_{m}"] = g[m].mean() - b[m].mean()
                p = stats.ttest_ind(g[m], b[m], equal_var=False).pvalue if g[m].std() + b[m].std() > 0 else np.nan
                row[f"p_{m}"] = p
        rows.append(row)
    return pd.DataFrame(rows).set_index("config")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--env", choices=envs.NAMES, help="default: read from the csv's path")
    ap.add_argument("--base", default="DR")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--continuous", action="store_true", help="show the continuous scores instead of pass rates")
    ap.add_argument("--holm", nargs="*", metavar="METRIC",
                    help="confirmatory: Holm-correct these metrics over every non-base config")
    args = ap.parse_args()
    env = envs.get(args.env or env_of(args.csv))
    df = pd.read_csv(args.csv)
    if args.configs:
        df = df[df.config.isin(args.configs + [args.base])]
    s = summarize(df, env, args.base)
    pd.set_option("display.width", 250)
    if args.holm:
        from acl_bench.study.compare import holm
        ps = {f"{c}|{m}": s.loc[c, f"p_{m}"] for c in s.index if c != args.base for m in args.holm}
        adj = holm(ps)
        rows = [{"config": k.split("|")[0], "metric": k.split("|")[1], "diff": s.loc[k.split("|")[0], f"d_{k.split('|')[1]}"],
                 "p": ps[k], "p_holm": adj[k]} for k in ps]
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return
    sfx = "c" if args.continuous else ""
    show = ["n"]
    for tag in SUITES.values():
        show += [f"fin_{tag}{sfx}", f"d_fin_{tag}{sfx}", f"p_fin_{tag}{sfx}", f"d_auc_{tag}{sfx}", f"p_auc_{tag}{sfx}"]
    key = next((k for k in (f"d_auc_v{sfx}", f"d_auc_r{sfx}") if k in s), None)
    s = s[[c for c in show if c in s]]
    print((s.sort_values(key, na_position="first") if key else s).to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
