"""Apply docs/protocol.md's stage rules to screen results.

    python -m acl_bench.plr.decide --stage A --arms sir_p sir_is

Reads every `results/<env>/lib/*.csv` for each dev environment (the shared DR pool,
replicates 1-48, in `runs.csv`; arms in their own files so parallel screens never share one) and prints, per arm, the primary gain (CartPole hard
suite, dev half), the guards (random suite on every environment) and the verdict.
Screens read only `*_dev` columns; `--confirm` switches to the test halves and seeds 1001+.

`--a5` applies Amendment 5 (docs/protocol.md): it reads `results/<env>/a5/*.csv` (runs graded
on the recalibrated `calib` suite), every stage uses all four dev envs, and the primary is
`a5_primary`: the mean over envs where DR's calib suite resolves of the arm's gain over DR on
`fin_cd` divided by DR's sd, with a normal-approximation p value. Guards are unchanged.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from acl_bench import envs
from acl_bench.plr.summarize import per_run

DISCOVERY, NO_HARM = ("cartpole",), ("acrobot", "mountaincar", "pendulum")
NO_HARM_ALL = DISCOVERY + NO_HARM
STAGE_ENVS = {"A": ("cartpole", "acrobot", "pendulum"), "B": ("cartpole", "acrobot", "mountaincar", "pendulum"),
              "C": ("cartpole", "acrobot", "mountaincar", "pendulum")}


def load(env: str, sub: str = "lib") -> pd.DataFrame | None:
    import glob
    files = sorted(glob.glob(os.path.join(envs.results_dir(env), sub, "*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def compare(runs: pd.DataFrame, arm: str, metric: str, base: str = "DR") -> tuple[float, float, int]:
    """(arm mean - base mean, Welch p, n arm seeds)."""
    a, b = runs.loc[runs.config == arm, metric], runs.loc[runs.config == base, metric]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, len(a)
    d = a.mean() - b.mean()
    p = stats.ttest_ind(a, b, equal_var=False).pvalue if a.std() + b.std() > 0 else np.nan
    return d, p, len(a)


def resolves(dr: pd.Series) -> bool:
    """Amendment 5: a suite resolves a gain when DR passes 5-95% of it with seed sd >= 0.01."""
    return len(dr) >= 2 and 0.05 <= dr.mean() <= 0.95 and dr.std() >= 0.01


def a5_primary(tables: dict, arm: str, metric: str, base: str = "DR") -> tuple[float, float, dict]:
    """(Z, p, per-env z): Z = mean over resolving envs of (arm - base mean) / base sd, and its
    two-sided p from the summed per-env Welch variances (envs are independent runs)."""
    zs, var = {}, []
    for e, runs in tables.items():
        a, b = runs.loc[runs.config == arm, metric].dropna(), runs.loc[runs.config == base, metric].dropna()
        if len(a) < 2 or not resolves(b):
            continue
        sd = b.std()
        zs[e] = (a.mean() - b.mean()) / sd
        var.append((a.var() / len(a) + b.var() / len(b)) / sd ** 2)
    if not zs:
        return np.nan, np.nan, zs
    z = float(np.mean(list(zs.values())))
    se = np.sqrt(np.sum(var)) / len(var)
    return z, float(2 * stats.norm.sf(abs(z) / se)) if se > 0 else np.nan, zs


def lower_bound(runs: pd.DataFrame, arm: str, metric: str, base: str = "DR", level: float = 0.95) -> float:
    """One-sided `level` lower confidence bound on (arm mean - base mean), Welch."""
    a, b = runs.loc[runs.config == arm, metric], runs.loc[runs.config == base, metric]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var() / len(a), b.var() / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return a.mean() - b.mean()
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return a.mean() - b.mean() - stats.t.ppf(level, df) * se


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    order, m, adj, run = np.argsort(ps), len(ps), [np.nan] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def verdict(stage: str, res: dict, envs_=None) -> str:
    """Amendment 2 (docs/protocol.md): a guard fails when the random-suite loss exceeds tol[e],
    20% of DR's own random-suite pass rate on that env (stage C: the one-sided 95% bound; stage
    B: the point estimate; stage A: the point estimate with 0.07 slack, as before)."""
    prim_d, prim_p, _ = res[("cartpole", "primary")]
    envs_ = envs_ or STAGE_ENVS[stage]
    guards = {e: res[(e, "guard")] for e in envs_ if (e, "guard") in res}
    tol = res["tol"]
    if np.isnan(prim_d):
        return "no data"
    if stage == "C":
        if prim_d <= 0 or not res["holm_p"] < 0.05:
            return f"FAIL (primary {prim_d:+.3f}, Holm p={res['holm_p']:.3f})"
        bad = [e for e in envs_ if not res.get((e, "lb"), np.nan) >= -tol.get(e, 0.0)]
        return f"FAIL (guard bound below -20% of DR: {bad})" if bad else "CONFIRMED"
    if stage == "A":
        if prim_d < 0:
            return "ABANDON (primary < 0)"
        bad = [e for e, (d, p, n) in guards.items() if d < -(tol[e] + 0.07)]
        return f"ABANDON (guard: {bad})" if bad else "advance to B"
    if prim_d <= 0 or not prim_p < 0.10:
        return f"ABANDON (primary {prim_d:+.3f}, p={prim_p:.2f})"
    bad = [e for e, (d, p, n) in guards.items() if d < -tol[e]]
    return f"ABANDON (guard: {bad})" if bad else "ADVANCE to C"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("A", "B", "C"), required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--confirm", action="store_true", help="read the test halves and seeds 1001+")
    ap.add_argument("--base", default="DR", help="baseline config (e.g. DR_po for a task-aware learner)")
    ap.add_argument("--seeds", default=None, help="confirmation seed range, e.g. 1101-1200 (default 1001-1100)")
    ap.add_argument("--a5", action="store_true", help="Amendment 5: calib suites, multi-env primary")
    args = ap.parse_args()
    args.confirm = args.confirm or args.stage == "C"
    hard = "fin_vt" if args.confirm else "fin_vd"
    if args.a5:
        hard = "fin_ct" if args.confirm else "fin_cd"
    tables = {}
    for e in (NO_HARM_ALL if args.a5 else STAGE_ENVS[args.stage]):
        df = load(e, "a5" if args.a5 else "lib")
        if df is None:
            continue
        if args.confirm:
            lo, hi = map(int, (args.seeds or "1001-1100").split("-"))
            df = df[(df.seed >= lo) & (df.seed <= hi)]
        else:
            df = df[df.seed < 1000]
        tables[e] = per_run(df, envs.get(e))
    rows = []
    for arm in args.arms:
        res, row = {}, {"arm": arm}
        res["tol"] = {e: 0.2 * float(t.loc[t.config == args.base, "fin_r"].mean()) for e, t in tables.items()}
        if args.a5:
            z, p, zs = a5_primary(tables, arm, hard, args.base)
            res[("cartpole", "primary")] = (z, p, 0)
            row["n"] = {e[:4]: int((t.config == arm).sum()) for e, t in tables.items()}
            row["Z"], row["Z_p"] = z, p
            row["z"] = " ".join(f"{e[:4]}{v:+.2f}" for e, v in zs.items())
            if all("fin_cvar" in t for t in tables.values()):
                row["cvar_d"] = np.nanmean([compare(t, arm, "fin_cvar", args.base)[0] for t in tables.values()])
        for e, runs in tables.items():
            if e in DISCOVERY and not args.a5:
                res[(e, "primary")] = compare(runs, arm, hard, args.base)
                row["n"] = res[(e, "primary")][2]
                row["hard_d"], row["hard_p"] = res[(e, "primary")][:2]
            res[(e, "guard")] = compare(runs, arm, "fin_r", args.base)
            row[f"{e[:4]}_r_d"] = res[(e, "guard")][0]
            if args.stage == "C":
                res[(e, "lb")] = row[f"{e[:4]}_r_lb"] = lower_bound(runs, arm, "fin_r", args.base)
        rows.append((row, res))
    if args.stage == "C":
        for (row, res), q in zip(rows, holm([res[("cartpole", "primary")][1] for row, res in rows])):
            res["holm_p"] = row["holm_p"] = q
    for row, res in rows:
        row["verdict"] = verdict(args.stage, res, NO_HARM_ALL if args.a5 else None)
    rows = [row for row, res in rows]
    base_n = {e: int((t.config == args.base).sum()) for e, t in tables.items()}
    print(f"{args.base} pool seeds: {base_n}")
    if args.a5:
        for e, t in tables.items():
            b = t.loc[t.config == args.base, hard].dropna() if hard in t else pd.Series(dtype=float)
            print(f"  {e:12s} DR {hard} {b.mean():.3f}+-{b.std():.3f}{'' if resolves(b) else '  (does not resolve)'}")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
