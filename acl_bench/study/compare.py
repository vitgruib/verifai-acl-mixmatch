"""Comparing arms: independent groups of runs, no seed-by-seed pairing.

Each arm is trained on its own seeds (study.run derives an independent seed per
(arm, replicate)), so the runs of two arms are independent groups and are compared
as groups: difference of means, a bootstrap interval that resamples each group
separately, and Welch's t-test.

Seed-by-seed pairing was tried and removed: a pilot (results/calibration/pilot_pairing.csv)
found runs that share a seed are identical at step 0 but uncorrelated by the first
checkpoint (20,480 steps), so pairing removed 0-28% of the variance of a difference
and made some comparisons slightly worse (docs/exam.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FINAL_CHECKPOINTS = 3


def checkpoint_grid(steps, n: int) -> list[int]:
    """`n` evenly spaced checkpoints ending at the last one: every k-th trained
    checkpoint counting back from the final model (k = len // n), step 0 excluded.
    With 30 check-ins at 20,480 steps and n = 10: every 61,440 steps up to 614,400."""
    trained = sorted(s for s in set(steps) if s > 0)
    stride = max(1, len(trained) // n)
    return sorted(trained[::-1][::stride][:n])


def _final(values: np.ndarray) -> float:
    return float(np.mean(values[-FINAL_CHECKPOINTS:]))


def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, seed). For every suite present (columns `<suite>/success` and
    `<suite>/mean_steps`), two summaries of each measure over the run's checkpoints:
    `final_*` (mean of the last FINAL_CHECKPOINTS) and `auc_*` (mean over the whole
    learning curve, trapezoid over steps):
    `final_<suite>`, `auc_<suite>` for success; `final_steps_<suite>`, `auc_steps_<suite>`
    for mean steps survived."""
    rows = []
    for (arm, seed), g in df.groupby(["arm", "seed"]):
        g = g.sort_values("step")
        steps = g["step"].to_numpy(dtype=float)
        span = steps[-1] - steps[0]
        row = {"arm": arm, "seed": seed}
        for col in g.columns:
            for suffix, prefix in (("/success", ""), ("/mean_steps", "steps_")):
                if col.endswith(suffix):
                    name, values = col[:-len(suffix)], g[col].to_numpy()
                    row[f"final_{prefix}{name}"] = _final(values)
                    row[f"auc_{prefix}{name}"] = (float(np.trapezoid(values, steps) / span) if span > 0
                                                  else float(values[-1]))
        rows.append(row)
    return pd.DataFrame(rows)


def group_values(metrics: pd.DataFrame, arm: str, metric: str) -> np.ndarray:
    return metrics.loc[metrics["arm"] == arm, metric].dropna().to_numpy(dtype=float)


def z_total(alpha: float = 0.0125, power: float = 0.8) -> float:
    return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))


def min_detectable(sd_a: float, sd_b: float, n: int, alpha: float = 0.0125, power: float = 0.8) -> float:
    """Smallest true difference detectable with `n` runs per arm."""
    return z_total(alpha, power) * float(np.sqrt((sd_a ** 2 + sd_b ** 2) / n))


def compare_arms(metrics: pd.DataFrame, a: str, b: str, metric: str, n_boot: int = 10_000,
                 rng_seed: int = 0) -> dict:
    """`a` minus `b` on `metric`, treating the arms as independent groups."""
    x, y = group_values(metrics, a, metric), group_values(metrics, b, metric)
    out = {"a": a, "b": b, "metric": metric, "n_a": len(x), "n_b": len(y),
           "mean_a": float(x.mean()), "mean_b": float(y.mean()), "mean_diff": float(x.mean() - y.mean())}
    if len(x) < 2 or len(y) < 2:
        return out
    rng = np.random.default_rng(rng_seed)
    boot = (rng.choice(x, size=(n_boot, len(x))).mean(axis=1)
            - rng.choice(y, size=(n_boot, len(y))).mean(axis=1))          # each group resampled separately
    t = stats.ttest_ind(x, y, equal_var=False)
    out.update({"sd_a": float(x.std(ddof=1)), "sd_b": float(y.std(ddof=1)),
                "sd_diff": float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))),
                "t": float(t.statistic), "p": float(t.pvalue),
                "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))})
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values: controls the family-wise error rate (chance
    of *any* false positive among the tests). Conservative; use for a small,
    pre-declared set of confirmatory comparisons. A NaN p (no variation in either arm,
    e.g. every run scores 0) still counts toward m but ranks last and stays NaN;
    otherwise it would sort unpredictably and min(1.0, nan) would set every later
    test to 1."""
    items = sorted(pvalues.items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    m, running, adjusted = len(items), 0.0, {}
    for i, (key, p) in enumerate(items):
        if np.isnan(p):
            adjusted[key] = np.nan
            continue
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted
