"""Held-out no-harm verdict (Maze, PointNav) for an arm confirmed at stage C.

    python -m acl_bench.plr.heldout --arm sfl_tilt --csv heldout10.csv

Per environment: final random-suite success (`fin_r`), arm minus DR, and its one-sided 95%
lower bound; pass if the bound >= -0.20 x DR's own rate (Amendment 2's rule, as registered
in docs/library_log.md). Maze's named held-out mazes (`fin_h`) are reported, not tested.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from acl_bench import envs
from acl_bench.plr.decide import compare, lower_bound
from acl_bench.plr.summarize import per_run

HELD_OUT = ("pointnav", "maze")
TOL = 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--csv", default="heldout10.csv")
    args = ap.parse_args()
    rows, ok = [], True
    for e in HELD_OUT:
        path = os.path.join(envs.results_dir(e), "lib", args.csv)
        if not os.path.exists(path):
            rows.append({"env": e, "verdict": "missing"})
            ok = False
            continue
        runs = per_run(pd.read_csv(path), envs.get(e))
        d, p, n = compare(runs, args.arm, "fin_r")
        lb = lower_bound(runs, args.arm, "fin_r")
        base = runs.loc[runs.config == "DR", "fin_r"].mean()
        passed = lb >= -TOL * base
        ok &= bool(passed)
        row = {"env": e, "n": n, "n_DR": int((runs.config == "DR").sum()), "DR_r": base, "r_d": d,
               "r_lb": lb, "tol": -TOL * base, "p": p, "verdict": "pass" if passed else "FAIL"}
        if "fin_h" in runs:
            row["h_d"], row["h_p"] = compare(runs, args.arm, "fin_h")[:2]
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("HELD-OUT NO-HARM:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
