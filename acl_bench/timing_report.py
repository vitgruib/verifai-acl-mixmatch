"""Cost vs. information report for a grid_results.csv: where the compute went,
and how much of the between-condition spread is distinguishable from seed noise.

    python -m acl_bench.timing_report [results/grid_results.csv]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from acl_bench.scenic_sampling import NON_ADAPTIVE_SAMPLERS

Z_ALPHA, Z_BETA = 1.96, 0.84  # two-sided alpha=0.05, 80% power


def md(df: pd.DataFrame, floatfmt: str = "{:.1f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def anova_p(groups) -> str:
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return "n/a"
    f, p = stats.f_oneway(*groups)
    return f"{f:.2f} ({p:.2f})"


def main():
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/grid_results.csv")
    total = df["wall_time_sec"].sum()

    print(f"## Compute: {len(df)} runs, {total / 60:.1f} min total\n")
    by = df.pivot_table(index="env", columns="sampler", values="wall_time_sec", aggfunc="sum") / 60
    by["total"] = by.sum(axis=1)
    by.loc["total"] = by.sum()
    print("Total minutes by env x sampler:\n")
    print(md(by), "\n")
    print("Mean seconds per run:\n")
    print(md(df.pivot_table(index="env", columns="sampler", values="wall_time_sec", aggfunc="mean")), "\n")
    print("Mean seconds per run by ACL setting and scoring function:\n")
    print(md(df.pivot_table(index="potential_fn", columns="acl", values="wall_time_sec", aggfunc="mean")), "\n")

    print("## Information: how much is signal vs. seed noise?\n")
    print("(One-way ANOVA F (p) across every cell / across samplers / across ACL / across scoring")
    print("functions, per env. Rule of thumb: detecting a 0.5-SD difference needs ~63 runs per arm.)\n")
    rows = []
    for env, sub in df.groupby("env"):
        cells = [g["eval_return_mean"].values for _, g in sub.groupby(["sampler", "acl", "potential_fn"])]
        rows.append({
            "env": env,
            "within-cell SD": np.sqrt(np.mean([c.var(ddof=1) for c in cells if len(c) > 1])),
            "all cells F (p)": anova_p(cells),
            "sampler F (p)": anova_p([g["eval_return_mean"].values for _, g in sub.groupby("sampler")]),
            "ACL F (p)": anova_p([g["eval_return_mean"].values for _, g in sub.groupby("acl")]),
            "function F (p)": anova_p([g["eval_return_mean"].values for _, g in sub.groupby("potential_fn")]),
        })
    print(md(pd.DataFrame(rows).set_index("env")), "\n")

    print("Minimum detectable effect (80% power, alpha=0.05): 'neither' baseline (non-adaptive sampler,")
    print("ACL off) vs. 'ACL only' (non-adaptive sampler, ACL on), at this grid's actual sample sizes:\n")
    mde_rows = []
    for env, sub in df.groupby("env"):
        sd = np.sqrt(np.mean([g["eval_return_mean"].var(ddof=1)
                               for _, g in sub.groupby(["sampler", "acl", "potential_fn"]) if len(g) > 1]))
        non = sub["sampler"].isin(NON_ADAPTIVE_SAMPLERS)
        n_base = int((non & (sub["acl"] == "off")).sum())
        n_other = int((non & (sub["acl"] == "on")).sum())
        mde = (Z_ALPHA + Z_BETA) * sd * np.sqrt(1 / n_base + 1 / n_other)
        mde_rows.append({"env": env, "n baseline": n_base, "n ACL only": n_other,
                         "MDE (return units)": mde, "MDE (x SD)": mde / sd})
    print(md(pd.DataFrame(mde_rows).set_index("env"), "{:.2f}"))


if __name__ == "__main__":
    main()
