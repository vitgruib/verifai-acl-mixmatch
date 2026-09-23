"""The pre-declared analysis (docs/ablation.md): 13 comparisons (C1-C5) over the eight
main arms, every metric computed on the same N_CHECKPOINTS evenly spaced check-ins
(compare.checkpoint_grid: every 61,440 steps up to the final model at 614,400).

  - Primary, Holm-corrected together (26 tests): `AUC_E0` (E0 success averaged over
    the learning curve) and `final_E6` (hard-section success at the end).
  - Secondary, Benjamini-Hochberg-corrected together: easy-section success (E7), the
    learning curve on E6, and every edge section (final and curve) plus their
    macro-averages `S_edge` / `auc_edge`.

Each comparison also reports `detectable`: the smallest true difference it had 80% power
to find at a Bonferroni level for its family (which bounds Holm's first step).

    python -m acl_bench.study.analyze --grades results/grades.csv --out results/analysis
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

from acl_bench.study.arms import MAIN_ARMS
from acl_bench.study.compare import (benjamini_hochberg, checkpoint_grid, compare_arms, derive_metrics, holm,
                                    min_detectable)

N_CHECKPOINTS = 10
PRIMARY = ("AUC_E0", "final_E6")


def edge_sections(df: pd.DataFrame) -> list[str]:
    """Edge sections present as `E<k>v/success` columns, in numeric order."""
    return sorted((c.split("/")[0] for c in df.columns if re.fullmatch(r"E\d+v/success", c)),
                  key=lambda s: int(s[1:-1]))


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


def run_family(metrics: pd.DataFrame, metric_names, correction) -> pd.DataFrame:
    table = pd.DataFrame([{"id": cid, **compare_arms(metrics, a, b, m)}
                          for cid, a, b in comparisons() for m in metric_names])
    table["detectable"] = [min_detectable(r.sd_a, r.sd_b, min(r.n_a, r.n_b), alpha=0.05 / len(table))
                           for r in table.itertuples()]
    keys = [f"{r.id}|{r.metric}" for r in table.itertuples()]
    adjusted = correction(dict(zip(keys, table["p"])))
    table["p_adj"] = [adjusted[k] for k in keys]
    return table


def curves(df: pd.DataFrame, edge: list[str]) -> dict:
    """Mean success per arm per checkpoint, for plotting."""
    df = df.copy()
    df["edge/success"] = df[[f"{s}/success" for s in edge]].mean(axis=1)
    cols = ["E0/success", "E6/success", "E7/success", "edge/success"] + [f"{s}/success" for s in edge]
    g = df.groupby(["arm", "step"])[cols].mean().reset_index()
    return {arm: {"step": sub["step"].tolist(), **{c.split("/")[0]: sub[c].round(4).tolist() for c in cols}}
            for arm, sub in g.groupby("arm")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grades", default="results/grades.csv")
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grades = pd.read_csv(args.grades)
    grades = on_grid(grades[grades["arm"].isin(MAIN_ARMS)])
    edge = edge_sections(grades)
    metrics = derive_metrics(grades)
    metrics["S_edge"] = metrics[[f"final_{s}" for s in edge]].mean(axis=1)
    metrics["auc_edge"] = metrics[[f"auc_{s}" for s in edge]].mean(axis=1)
    metrics.to_csv(os.path.join(args.out, "per_run_metrics.csv"), index=False)

    secondary_names = (["final_E7", "auc_E6", "S_edge", "auc_edge"] + [f"final_{s}" for s in edge]
                       + [f"auc_{s}" for s in edge])
    primary = run_family(metrics, PRIMARY, holm)
    secondary = run_family(metrics, secondary_names, benjamini_hochberg)
    primary.to_csv(os.path.join(args.out, "primary_holm.csv"), index=False)
    secondary.to_csv(os.path.join(args.out, "secondary_bh.csv"), index=False)

    summary = ["AUC_E0", "final_E0", "final_E6", "auc_E6", "final_E7", "S_edge", "auc_edge"] + \
              [f"final_{s}" for s in edge]
    by_arm = metrics.groupby("arm")[summary].agg(["mean", "std", "count"])
    by_arm.columns = [f"{m}|{s}" for m, s in by_arm.columns]
    by_arm = by_arm.reindex(list(MAIN_ARMS))
    by_arm.to_csv(os.path.join(args.out, "by_arm.csv"))

    with open(os.path.join(args.out, "report_data.json"), "w") as f:
        json.dump({"by_arm": by_arm.reset_index().to_dict(orient="records"),
                   "primary": primary.to_dict(orient="records"),
                   "secondary": secondary.to_dict(orient="records"),
                   "edge_sections": edge, "checkpoint_steps": checkpoint_grid(grades["step"], N_CHECKPOINTS),
                   "curves": curves(grades, edge)}, f)

    pd.set_option("display.width", 220)
    print(by_arm[[c for c in by_arm.columns if c.endswith("|mean")]].round(3))
    cols = ["id", "metric", "mean_diff", "ci_lo", "ci_hi", "p", "p_adj", "detectable"]
    print("\nPRIMARY (Holm over 26):\n", primary[cols].round(4).to_string(index=False))
    print(f"\nSECONDARY (BH over {len(secondary)}), smallest p:\n",
          secondary.sort_values("p")[cols].head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
