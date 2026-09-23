"""The analysis (docs/ablation.md): 13 comparisons (C1-C5) over the eight main arms, on
two suites (`random`, `verifai`), every metric computed on the same N_CHECKPOINTS
evenly spaced check-ins (compare.checkpoint_grid: every 61,440 steps up to the final
model at 614,400).

Per suite, four metrics: success and mean steps survived, each summarized two ways --
`final_*` (the last three check-ins) and `auc_*` (the whole learning curve). All
13 x 8 = 104 tests are Holm-corrected together. Each also reports `detectable`: the
smallest true difference it had 80% power to find at the Bonferroni level.

    python -m acl_bench.study.analyze --grades results/grades.csv --out results/analysis
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from acl_bench.study.arms import MAIN_ARMS
from acl_bench.study.compare import checkpoint_grid, compare_arms, derive_metrics, holm, min_detectable

N_CHECKPOINTS = 10
SUITES = ("random", "verifai")
METRICS = tuple(f"{kind}_{s}" for s in SUITES for kind in ("final", "auc", "final_steps", "auc_steps"))


def comparisons() -> list[tuple[str, str, str]]:
    """(id, a, b): a minus b, in the order docs/ablation.md lists them."""
    samplers = [a[2:] for a in MAIN_ARMS if a.startswith("S_")]
    out = [("C1", "A", "N")]
    out += [(f"C2_{s}", f"S_{s}", "N") for s in samplers]
    out += [(f"C3_{s}", f"B_{s}", "N") for s in samplers]
    out += [(f"C4_{s}", f"B_{s}", f"S_{s}") for s in samplers]
    out += [(f"C5_{s}", f"B_{s}", "A") for s in samplers]
    return out


def on_grid(df: pd.DataFrame, n: int = N_CHECKPOINTS) -> pd.DataFrame:
    grid = checkpoint_grid(df["step"], n)
    df = df[df["step"].isin(grid)]
    short = df.groupby(["arm", "seed"]).size().ne(len(grid))
    if short.any():
        raise SystemExit(f"{short.sum()} runs lack a grid check-in; grid = {grid}")
    return df


def run_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame([{"id": cid, **compare_arms(metrics, a, b, m)}
                          for cid, a, b in comparisons() for m in METRICS])
    table["detectable"] = [min_detectable(r.sd_a, r.sd_b, min(r.n_a, r.n_b), alpha=0.05 / len(table))
                           for r in table.itertuples()]
    keys = [f"{r.id}|{r.metric}" for r in table.itertuples()]
    adjusted = holm(dict(zip(keys, table["p"])))
    table["p_adj"] = [adjusted[k] for k in keys]
    return table


def curves(df: pd.DataFrame) -> dict:
    """Mean success and mean steps per arm per checkpoint, for plotting."""
    cols = [f"{s}/{m}" for s in SUITES for m in ("success", "mean_steps")]
    g = df.groupby(["arm", "step"])[cols].mean().reset_index()
    return {arm: {"step": sub["step"].tolist(), **{c: sub[c].round(4).tolist() for c in cols}}
            for arm, sub in g.groupby("arm")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grades", default="results/grades.csv")
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grades = pd.read_csv(args.grades)
    grades = on_grid(grades[grades["arm"].isin(MAIN_ARMS)])
    metrics = derive_metrics(grades)
    metrics.to_csv(os.path.join(args.out, "per_run_metrics.csv"), index=False)

    tests = run_tests(metrics)
    tests.to_csv(os.path.join(args.out, "tests_holm.csv"), index=False)

    by_arm = metrics.groupby("arm")[list(METRICS)].agg(["mean", "std", "count"])
    by_arm.columns = [f"{m}|{s}" for m, s in by_arm.columns]
    by_arm = by_arm.reindex(list(MAIN_ARMS))
    by_arm.to_csv(os.path.join(args.out, "by_arm.csv"))

    with open(os.path.join(args.out, "report_data.json"), "w") as f:
        json.dump({"by_arm": by_arm.reset_index().to_dict(orient="records"),
                   "tests": tests.to_dict(orient="records"),
                   "checkpoint_steps": checkpoint_grid(grades["step"], N_CHECKPOINTS),
                   "curves": curves(grades)}, f)

    pd.set_option("display.width", 220)
    print(by_arm[[c for c in by_arm.columns if c.endswith("|mean")]].round(3).to_string())
    cols = ["id", "metric", "mean_diff", "ci_lo", "ci_hi", "p", "p_adj", "detectable"]
    print(f"\nTESTS (Holm over {len(tests)}), smallest p:\n",
          tests.sort_values("p")[cols].head(15).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
