"""Cost vs. information report for a grid_results.csv: where the compute went,
and how much of the between-condition spread is distinguishable from seed noise.

    python -m acl_bench.timing_report [results/grid_results.csv]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

Z_ALPHA, Z_BETA = 1.96, 0.84  # two-sided alpha=0.05, 80% power


def md(df: pd.DataFrame, floatfmt: str = "{:.1f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main():
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/grid_results.csv")
    total = df["wall_time_sec"].sum()

    print(f"## Compute: {len(df)} runs, {total / 60:.1f} min total\n")
    by = df.pivot_table(index="env", columns="sampler", values="wall_time_sec", aggfunc="sum") / 60
    by["total"] = by.sum(axis=1)
    by.loc["total"] = by.sum()
    print("Total minutes by env x sampler:\n")
    print(md(by), "\n")

    per_run = df.pivot_table(index="env", columns="sampler", values="wall_time_sec", aggfunc="mean")
    print("Mean seconds per run:\n")
    print(md(per_run), "\n")

    share = (df.groupby("sampler")["wall_time_sec"].sum() / total * 100).round(1)
    runs = df.groupby("sampler").size()
    print("Share of compute vs share of runs, by sampler:\n")
    print(md(pd.DataFrame({"% of compute": share, "% of runs": runs / len(df) * 100}).rename_axis("sampler")), "\n")

    print("Mean seconds per run by potential-function condition (excluding bo, whose")
    print("cost dwarfs everything; 'none' skips the per-episode scoring entirely):\n")
    fast = df[df["sampler"] != "bo"]
    print(md(fast.groupby("potential_fn")["wall_time_sec"].mean().to_frame("s/run").rename_axis("potential_fn")), "\n")

    print("## Information: how much is signal vs. seed noise?\n")
    print("(One-way ANOVA across all 35 cells / across samplers / across potential fns, per env.")
    print("Rule of thumb: detecting a 0.5-SD difference needs ~63 runs per arm.)\n")
    rows = []
    for env, sub in df.groupby("env"):
        cells = [g["eval_return_mean"].values for _, g in sub.groupby(["sampler", "potential_fn"])]
        f_cells, p_cells = stats.f_oneway(*cells)
        f_s, p_s = stats.f_oneway(*[g["eval_return_mean"].values for _, g in sub.groupby("sampler")])
        f_p, p_p = stats.f_oneway(*[g["eval_return_mean"].values for _, g in sub.groupby("potential_fn")])
        within_sd = np.sqrt(np.mean([c.var(ddof=1) for c in cells]))
        rows.append({
            "env": env, "within-cell SD": within_sd,
            "all cells F (p)": f"{f_cells:.2f} ({p_cells:.2f})",
            "sampler F (p)": f"{f_s:.2f} ({p_s:.2f})",
            "potential_fn F (p)": f"{f_p:.2f} ({p_p:.2f})",
        })
    print(md(pd.DataFrame(rows).set_index("env")), "\n")

    print("Minimum detectable effect (80% power, alpha=0.05) for the 'neither' baseline vs. another")
    print("condition, at this grid's actual sample sizes -- in return units and as a multiple of SD:\n")
    mde_rows = []
    for env, sub in df.groupby("env"):
        sd = np.sqrt(np.mean([g["eval_return_mean"].var(ddof=1) for _, g in sub.groupby(["sampler", "potential_fn"])]))
        n_base = ((sub["sampler"].isin({"random", "halton"})) & (sub["potential_fn"] == "none")).sum()
        n_other = ((sub["sampler"].isin({"random", "halton"})) & (sub["potential_fn"] != "none")).sum()
        mde = (Z_ALPHA + Z_BETA) * sd * np.sqrt(1 / n_base + 1 / n_other)
        mde_rows.append({"env": env, "n baseline": n_base, "n other": n_other,
                         "MDE (return units)": mde, "MDE (x SD)": mde / sd})
    print(md(pd.DataFrame(mde_rows).set_index("env"), "{:.2f}"))


if __name__ == "__main__":
    main()
