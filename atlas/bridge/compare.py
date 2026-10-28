"""The CartPole bridge's pre-registered reading (docs/superpowers/plans/2026-10-28-maze-gpu-suite.md):
PASS if every arm's mean pass rate under JaxUED training is within 0.10 of batch 21's on VerifAI
and on random, and DR > PLR and DR > RPLR on VerifAI under JaxUED training; else FAIL.

    python -m atlas.bridge.compare [--ours results/suite/cartpole/scores.csv] \\
        [--bridge results/suite/cartpole_jaxued/scores.csv] [--budget 614400]

writes results/suite/analysis/bridge_cartpole.md. Only rows at the final check-in of runs at
`--budget` steps count.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ARMS = ("DR", "PLR", "RPLR")
RULE_SECTIONS = ("verifai", "random")
TOLERANCE = 0.10


def means(scores: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Mean pass rate and seed count per (arm, section) at the final check-in."""
    d = scores[(scores.check == budget) & (scores.budget == budget) & scores.arm.isin(ARMS)]
    m = d.groupby(["arm", "section"]).pass_rate.agg(["mean", "count"])
    missing = [(a, s) for a in ARMS for s in RULE_SECTIONS if (a, s) not in m.index]
    if missing:
        raise ValueError(f"no rows at budget {budget} for {missing}")
    return m


def reading(ours: pd.DataFrame, bridge: pd.DataFrame, budget: int) -> dict:
    o, b = means(ours, budget), means(bridge, budget)
    rules = []
    for section in RULE_SECTIONS:
        for arm in ARMS:
            diff = b.loc[(arm, section), "mean"] - o.loc[(arm, section), "mean"]
            rules.append({"name": f"{arm} {section} within {TOLERANCE:.2f}",
                          "pass": bool(round(abs(diff), 9) <= TOLERANCE), "detail": f"difference {diff:+.3f}"})
    for arm in ARMS[1:]:
        dr, other = b.loc[("DR", "verifai"), "mean"], b.loc[(arm, "verifai"), "mean"]
        rules.append({"name": f"DR > {arm} on verifai", "pass": bool(dr > other),
                      "detail": f"DR {dr:.3f}, {arm} {other:.3f}"})
    table = o.join(b, lsuffix="_ours", rsuffix="_bridge", how="outer")
    return {"verdict": "PASS" if all(r["pass"] for r in rules) else "FAIL", "rules": rules, "means": table}


def markdown(result: dict, budget: int, ours_path: str, bridge_path: str) -> str:
    lines = ["# CartPole bridge: JaxUED PLR training vs our PyTorch training", "",
             f"Both graded by our grader on frozen_sets/suite/cartpole, final check-in at {budget} steps.",
             f"Ours (PyTorch): `{ours_path}`. JaxUED: `{bridge_path}`.", "",
             f"**Verdict: {result['verdict']}**", "", "| rule | result | detail |", "|---|---|---|"]
    lines += [f"| {r['name']} | {'PASS' if r['pass'] else 'FAIL'} | {r['detail']} |" for r in result["rules"]]
    lines += ["", "Mean pass rate per arm and section (n = seeds):", "",
              "| arm | section | ours | n | JaxUED | n | difference |", "|---|---|---|---|---|---|---|"]
    for (arm, section), r in result["means"].iterrows():
        lines.append(f"| {arm} | {section} | {r.mean_ours:.3f} | {r.count_ours:.0f} | {r.mean_bridge:.3f} | "
                     f"{r.count_bridge:.0f} | {r.mean_bridge - r.mean_ours:+.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="results/suite/cartpole/scores.csv")
    ap.add_argument("--bridge", default="results/suite/cartpole_jaxued/scores.csv")
    ap.add_argument("--budget", type=int, default=614_400)
    ap.add_argument("--out", default="results/suite/analysis/bridge_cartpole.md")
    args = ap.parse_args(argv)
    result = reading(pd.read_csv(args.ours), pd.read_csv(args.bridge), args.budget)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(markdown(result, args.budget, args.ours, args.bridge))
    print(f"{result['verdict']}: wrote {args.out}")


if __name__ == "__main__":
    main()
