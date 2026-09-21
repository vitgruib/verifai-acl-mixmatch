"""Paired-by-seed analysis of run_arms output.

Two arms trained on the same seed share their random streams, so their outcomes
are (hopefully) positively correlated, and the per-seed *difference* has less
variance than an unpaired comparison would: Var(d) = Var(a) + Var(b) - 2 Cov(a, b).
Pairing only helps to the extent Cov > 0, so `summarize_pair` reports the
measured correlation and the variance actually removed instead of assuming it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EDGE_SETS = ("E1", "E1b", "E2", "E3a", "E3b", "E4", "E5")
FINAL_CHECKPOINTS = 3


def _final(values: np.ndarray) -> float:
    return float(np.mean(values[-FINAL_CHECKPOINTS:]))


def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, seed): AUC_E0 (mean E0 success over the whole curve,
    trapezoid over steps, step 0 included), final_<set> (mean of the last
    checkpoints), S_edge (macro-average final success over the edge sets) and
    steps_to_<pct>_E0 thresholds are left to the caller since they need the
    baseline plateau."""
    rows = []
    for (arm, seed), g in df.groupby(["arm", "seed"]):
        g = g.sort_values("step")
        steps = g["step"].to_numpy(dtype=float)
        row = {"arm": arm, "seed": seed,
               "AUC_E0": float(np.trapezoid(g["E0/success"].to_numpy(), steps) / (steps[-1] - steps[0]))}
        for col in g.columns:
            if col.endswith("/success"):
                row[f"final_{col[:-len('/success')]}"] = _final(g[col].to_numpy())
        edge = [row[f"final_{s}"] for s in EDGE_SETS if f"final_{s}" in row]
        if edge:
            row["S_edge"] = float(np.mean(edge))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_frame(metrics: pd.DataFrame, a: str, b: str, metric: str) -> pd.DataFrame:
    """Rows = seeds that both arms completed; columns a, b, diff (= a - b)."""
    wide = metrics.pivot(index="seed", columns="arm", values=metric)[[a, b]].dropna()
    wide["diff"] = wide[a] - wide[b]
    return wide


def seeds_needed(sd: float, delta: float, alpha: float = 0.0125, power: float = 0.8) -> int:
    """Seeds for a two-sided test to detect a true mean difference `delta` when the
    per-seed *paired difference* has SD `sd` (n = (z_a + z_b)^2 sd^2 / delta^2)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil((z * sd / delta) ** 2))


def summarize_pair(metrics: pd.DataFrame, a: str, b: str, metric: str, n_boot: int = 10_000,
                   rng_seed: int = 0) -> dict:
    w = paired_frame(metrics, a, b, metric)
    n = len(w)
    d = w["diff"].to_numpy()
    out = {"a": a, "b": b, "metric": metric, "n_seeds": n, "mean_a": w[a].mean(), "mean_b": w[b].mean(),
           "mean_diff": d.mean()}
    if n < 3:
        return out
    var_a, var_b, var_d = w[a].var(ddof=1), w[b].var(ddof=1), d.var(ddof=1)
    boot = np.random.default_rng(rng_seed).choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    t = stats.ttest_rel(w[a], w[b])
    out.update({
        "sd_diff_paired": np.sqrt(var_d),
        "sd_diff_unpaired": np.sqrt(var_a + var_b),
        "rho": float(np.corrcoef(w[a], w[b])[0, 1]),
        # share of the unpaired variance of the difference that pairing removed
        "variance_removed": 1.0 - var_d / (var_a + var_b),
        "t": float(t.statistic), "p": float(t.pvalue),
        "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
    })
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, adjusted = len(items), 0.0, {}
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted
