"""Batch 21 analysis: the pre-registered screen, per-task tables, and the custom-case breakdown.

    python -m acl_bench.suite.analyze [--envs cartpole frozenlake]

Reads results/suite/<env>/scores.csv (one row per arm, seed and section; the final check is
analysed) and writes screen.csv, table_<env>.csv, custom_<env>.csv and summary.md to
results/suite/analysis/. The screen is one-sided Mann-Whitney U (method's VerifAI pass rate >
DR's) per method and task, Holm-corrected across the tests run (docs/library_log.md, batch 21).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from acl_bench import envs
from acl_bench.exam.sets import load_sets
from acl_bench.snapshots import find_runs, load_run
from acl_bench.study.compare import holm
from acl_bench.suite.exam import PASS_RATE, exam_dir, question_scores

RESULTS = Path(__file__).resolve().parents[2] / "results" / "suite"
ALPHA = 0.05
N_SEEDS = 20
TASK_SEEDS = {"maze": 10}
BASELINE = "DR"
METHODS = ("PLR", "RPLR", "ACCEL", "SFL", "MINIMAX")
TASK_ONLY = {"MINIMAX": "maze"}   # an arm scored on one task only
SECTIONS = ("random", "verifai", "custom")


def load_scores(results: Path = RESULTS, env_names=None) -> pd.DataFrame:
    """Every env's final-check rows, with a `task` column."""
    frames = []
    for path in sorted(results.glob("*/scores.csv")):
        if env_names and path.parent.name not in env_names:
            continue
        df = pd.read_csv(path)
        df = df[df["arm"].isin((BASELINE, *METHODS))]
        df = df[df["check"] == df.groupby(["arm", "seed", "section"])["check"].transform("max")]
        frames.append(df.assign(task=path.parent.name))
    return pd.concat(frames, ignore_index=True)


def require_complete(scores: pd.DataFrame) -> None:
    """Stop unless each task's arms have TASK_SEEDS (default N_SEEDS) seeds per section at full budget."""
    done = scores[scores["check"] == scores["budget"]]
    counts = done.groupby(["task", "arm", "section"])["seed"].nunique()
    for task in scores["task"].unique():
        n_seeds = TASK_SEEDS.get(task, N_SEEDS)
        for arm in (BASELINE, *METHODS):
            if TASK_ONLY.get(arm, task) != task:
                continue
            for section in SECTIONS:
                n = counts.get((task, arm, section), 0)
                if n != n_seeds:
                    raise SystemExit(f"{task}: {arm} {section} has {n} seeds at full budget, need {n_seeds} "
                                     "(--allow-partial to preview)")


def screen(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, sub in scores[scores["section"] == "verifai"].groupby("task"):
        dr = sub[sub["arm"] == BASELINE]["pass_rate"]
        for method in METHODS:
            x = sub[sub["arm"] == method]["pass_rate"]
            if len(x) and len(dr):
                u, p = mannwhitneyu(x, dr, alternative="greater")
                rows.append(dict(method=method, task=task, U=u, p=p, diff=x.mean() - dr.mean(),
                                 n_dr=len(dr), n_method=len(x)))
    out = pd.DataFrame(rows)
    adjusted = holm({f"{r.method}|{r.task}": r.p for r in out.itertuples()})
    out["holm_p"] = [adjusted[f"{r.method}|{r.task}"] for r in out.itertuples()]
    out["improved"] = out["holm_p"] < ALPHA
    return out


def table(scores: pd.DataFrame) -> pd.DataFrame:
    grouped = scores.groupby(["task", "arm", "section"])["pass_rate"]
    return grouped.agg(mean="mean", std="std", n_seeds="count").reset_index()


def custom_breakdown(env_name: str, snapshots: Path | None = None, exam: Path | None = None,
                     score=question_scores) -> pd.DataFrame:
    """Per custom case, each method's mean question score over seeds (final agent)."""
    exam = exam or exam_dir(env_name)
    snapshots = snapshots or RESULTS / env_name / "snapshots"
    env = envs.get(env_name)
    custom = load_sets(str(exam), names=["custom"])["custom"]
    with open(exam / "manifest.json") as f:
        cases = json.load(f)["custom_cases"]
    per_arm: dict[str, list] = {}
    for arm, _, path in find_runs(str(snapshots)):
        if arm in (BASELINE, *METHODS):
            agents = load_run(path)
            per_arm.setdefault(arm, []).append(score(env, agents[max(agents)], custom))
    out = pd.DataFrame({"case": cases, **{arm: np.mean(v, axis=0) for arm, v in per_arm.items()}})
    arms = list(per_arm)
    top = out[arms].max(axis=1)
    out["best"] = ["=".join(arm for arm in arms if np.isclose(row[arm], t))
                   for (_, row), t in zip(out.iterrows(), top)]
    # "No agent passes": no individual agent reaches PASS_RATE on the case.
    best_single = np.max([np.max(v, axis=0) for v in per_arm.values()], axis=0)
    out["no_agent_passes"] = best_single < PASS_RATE
    return out


def summary(screened: pd.DataFrame, tables: pd.DataFrame, customs: dict) -> str:
    lines = ["# Batch 21 analysis", "", "Per-case custom scores are the mean share of plays passed, averaged over seeds.", "",
             f"## Screen (one-sided Mann-Whitney, Holm over {len(screened)} tests, alpha {ALPHA})", "",
             "| method | task | diff | p | Holm p | improved |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.method} | {r.task} | {r.diff:+.3f} | {r.p:.4f} | {r.holm_p:.4f} | {r.improved} |"
              for r in screened.itertuples()]
    for task, sub in tables.groupby("task"):
        wide = sub.assign(cell=sub.apply(lambda r: f"{r['mean']:.3f} +- {r['std']:.3f}", axis=1))
        wide = wide.pivot(index="arm", columns="section", values="cell")[list(SECTIONS)]
        lines += ["", f"## {task}: pass rate, mean +- std over {sub['n_seeds'].max()} seeds", "",
                  "| arm | " + " | ".join(SECTIONS) + " |", "|---|" + "---|" * len(SECTIONS)]
        lines += [f"| {arm} | " + " | ".join(row) + " |" for arm, row in wide.iterrows()]
    for task, df in customs.items():
        flagged = df[df["no_agent_passes"]]["case"].tolist()
        lines += ["", f"## {task}: custom cases no agent passes", "", ", ".join(flagged) if flagged else "none"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--allow-partial", action="store_true", help="preview with incomplete scores")
    args = parser.parse_args()
    scores = load_scores(RESULTS, args.envs)
    if not args.allow_partial:
        require_complete(scores)
    out = RESULTS / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    screened, tables = screen(scores), table(scores)
    screened.to_csv(out / "screen.csv", index=False)
    customs = {}
    for task in sorted(scores["task"].unique()):
        tables[tables["task"] == task].to_csv(out / f"table_{task}.csv", index=False)
        customs[task] = custom_breakdown(task)
        customs[task].to_csv(out / f"custom_{task}.csv", index=False)
    text = summary(screened, tables, customs)
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
