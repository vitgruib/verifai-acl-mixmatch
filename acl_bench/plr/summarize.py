"""Summarize a screen CSV: per config, final (last three check-ins) and curve (mean over
all check-ins after step 0) success on each suite, as a difference from a baseline
config with a Welch t-test p-value (uncorrected: screening, not confirmation).

    python -m acl_bench.plr.summarize results/acrobot/plr/screen.csv [--base DR]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.step > 0]
    steps = sorted(df.step.unique())
    last3 = df[df.step.isin(steps[-3:])]
    out = df.groupby(["config", "seed"]).agg(
        auc_r=("random/success", "mean"), auc_v=("verifai/success", "mean"),
        auc_rs=("random/mean_steps", "mean"))
    fin = last3.groupby(["config", "seed"]).agg(
        fin_r=("random/success", "mean"), fin_v=("verifai/success", "mean"), fin_rs=("random/mean_steps", "mean"))
    return out.join(fin).reset_index()


def summarize(df: pd.DataFrame, base: str = "DR") -> pd.DataFrame:
    runs = per_run(df)
    cols = ["fin_r", "auc_r", "fin_v", "auc_v", "fin_rs", "auc_rs"]
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
    ap.add_argument("--base", default="DR")
    ap.add_argument("--configs", nargs="*")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    if args.configs:
        df = df[df.config.isin(args.configs + [args.base])]
    s = summarize(df, args.base)
    show = ["n", "fin_r", "d_fin_r", "p_fin_r", "d_auc_r", "p_auc_r", "fin_v", "d_fin_v", "p_fin_v", "d_auc_v", "p_auc_v"]
    pd.set_option("display.width", 250)
    print(s[[c for c in show if c in s]].sort_values("d_auc_v", na_position="first").to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
