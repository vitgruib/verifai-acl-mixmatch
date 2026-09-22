"""The full pre-declared analysis (docs/ablation.md, C1-C5) over the eight main arms,
with the v3 edge sections (found by falsification search) as secondary metrics.

Primary: `AUC_E0` and `final_E6` on all 13 comparisons, Holm-corrected together
(26 tests). Secondary: final and AUC success on each v3 edge section plus their
macro-average `S_edge_v3`, on the same 13 comparisons, Benjamini-Hochberg-corrected
together (an exploratory family, like the earlier FDR sweep).

    python -m acl_bench.suite.analyze --main results/main.csv \\
        --edge results/edge_v3.csv --out results/analysis/v3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from acl_bench.scenic_sampling import ADAPTIVE_SAMPLERS
from acl_bench.suite.compare import benjamini_hochberg, compare_arms, derive_metrics, holm

MAIN_ARMS = ["N", "A"] + [f"S_{s}" for s in ADAPTIVE_SAMPLERS] + [f"B_{s}" for s in ADAPTIVE_SAMPLERS]
EDGE_V3 = ("E1v", "E2v", "E3v", "E4v", "E5v")
PRIMARY = ("AUC_E0", "final_E6")
KEEP_SECTIONS = ("E0", "E6", "E7") + EDGE_V3


def comparisons() -> list[tuple[str, str, str]]:
    """(id, a, b): a minus b, in the order docs/ablation.md lists them."""
    out = [("C1", "A", "N")]
    out += [(f"C2_{s}", f"S_{s}", "N") for s in ADAPTIVE_SAMPLERS]
    out += [(f"C3_{s}", f"B_{s}", "N") for s in ADAPTIVE_SAMPLERS]
    out += [(f"C4_{s}", f"B_{s}", f"S_{s}") for s in ADAPTIVE_SAMPLERS]
    out += [(f"C5_{s}", f"B_{s}", "A") for s in ADAPTIVE_SAMPLERS]
    return out


def merge(main: pd.DataFrame, edge: pd.DataFrame) -> pd.DataFrame:
    """One frame with E0/E6/E7 from training-time grading and the v3 edge sections from
    re-grading the saved snapshots, joined on (arm, seed, step)."""
    keep = ["arm", "seed", "step"] + [c for c in main.columns
                                      if c.split("/")[0] in KEEP_SECTIONS and "/" in c]
    df = main[keep].merge(edge, on=["arm", "seed", "step"], how="inner", validate="one_to_one")
    if df.groupby(["arm", "seed"]).ngroups != main.groupby(["arm", "seed"]).ngroups:
        missing = set(map(tuple, main[["arm", "seed"]].drop_duplicates().to_numpy())) - \
                  set(map(tuple, df[["arm", "seed"]].drop_duplicates().to_numpy()))
        raise SystemExit(f"{len(missing)} runs have no v3 edge grades, e.g. {sorted(missing)[:5]}")
    return df


def add_edge_macro(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["S_edge_v3"] = metrics[[f"final_{s}" for s in EDGE_V3]].mean(axis=1)
    metrics["auc_edge_v3"] = metrics[[f"auc_{s}" for s in EDGE_V3]].mean(axis=1)
    return metrics


def run_family(metrics: pd.DataFrame, metric_names, correction) -> pd.DataFrame:
    rows = [{"id": cid, **compare_arms(metrics, a, b, m)}
            for cid, a, b in comparisons() for m in metric_names]
    table = pd.DataFrame(rows)
    keys = [f"{r.id}|{r.metric}" for r in table.itertuples()]
    adjusted = correction(dict(zip(keys, table["p"])))
    table["p_adj"] = [adjusted[k] for k in keys]
    return table


def curves(df: pd.DataFrame) -> dict:
    """Mean success per arm per checkpoint, for plotting."""
    df = df.copy()
    df["edge_v3/success"] = df[[f"{s}/success" for s in EDGE_V3]].mean(axis=1)
    cols = ["E0/success", "E6/success", "E7/success", "edge_v3/success"] + [f"{s}/success" for s in EDGE_V3]
    g = df.groupby(["arm", "step"])[cols].mean().reset_index()
    return {arm: {"step": sub["step"].tolist(), **{c.split("/")[0]: sub[c].round(4).tolist() for c in cols}}
            for arm, sub in g.groupby("arm")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default="results/main.csv")
    parser.add_argument("--edge", default="results/edge_v3.csv")
    parser.add_argument("--out", default="results/analysis/v3")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    main_df = pd.read_csv(args.main)
    main_df = main_df[main_df["arm"].isin(MAIN_ARMS)]
    edge_df = pd.read_csv(args.edge)
    df = merge(main_df, edge_df[edge_df["arm"].isin(MAIN_ARMS)])
    metrics = add_edge_macro(derive_metrics(df))
    metrics.to_csv(os.path.join(args.out, "per_run_metrics.csv"), index=False)

    primary = run_family(metrics, PRIMARY, holm)
    secondary_metrics = ["S_edge_v3", "auc_edge_v3"] + [f"final_{s}" for s in EDGE_V3] + [f"auc_{s}" for s in EDGE_V3]
    secondary = run_family(metrics, secondary_metrics, benjamini_hochberg)
    primary.to_csv(os.path.join(args.out, "primary_holm.csv"), index=False)
    secondary.to_csv(os.path.join(args.out, "edge_v3_fdr.csv"), index=False)

    summary_cols = ["AUC_E0", "final_E0", "final_E6", "final_E7", "S_edge_v3", "auc_edge_v3"] + \
                   [f"final_{s}" for s in EDGE_V3]
    by_arm = metrics.groupby("arm")[summary_cols].agg(["mean", "std", "count"])
    by_arm.columns = [f"{m}|{s}" for m, s in by_arm.columns]
    by_arm = by_arm.reindex(MAIN_ARMS)
    by_arm.to_csv(os.path.join(args.out, "by_arm.csv"))

    with open(os.path.join(args.out, "report_data.json"), "w") as f:
        json.dump({"by_arm": by_arm.reset_index().to_dict(orient="records"),
                   "primary": primary.to_dict(orient="records"),
                   "secondary": secondary.to_dict(orient="records"),
                   "curves": curves(df)}, f)

    pd.set_option("display.width", 200)
    print(by_arm[[c for c in by_arm.columns if c.endswith("|mean")]].round(3))
    cols = ["id", "metric", "mean_diff", "ci_lo", "ci_hi", "p", "p_adj"]
    print("\nPRIMARY (Holm over 26):\n", primary[cols].round(4).to_string(index=False))
    print("\nSECONDARY edge v3 (BH over", len(secondary), "), smallest q:\n",
          secondary.sort_values("p")[cols].head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
