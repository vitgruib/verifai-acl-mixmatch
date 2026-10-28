"""Grade the CartPole bridge runs with our PyTorch grader on our frozen exam. Runs in .venv:

    python -m atlas.bridge.grade_cartpole --runs DIR --out results/suite/cartpole_jaxued/scores.csv

Every run under DIR (a params.npz beside a config.json, as atlas/bridge/cartpole_jaxued.py writes
them) is loaded into acl_bench.ppo.Agent and graded on each section of frozen_sets/suite/cartpole
with acl_bench.suite.run.section_scores, the grading batch 21 used. Rows have the suite's columns
(acl_bench/suite/run.py), one check-in at the run's final step. Re-running grades only the runs
not yet in the CSV; a run with rows for only some sections is dropped and graded again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acl_bench import envs
from acl_bench.ppo import Agent

KEY = ["arm", "seed", "budget"]


def load_agent(path) -> Agent:
    env = envs.get("cartpole")
    agent = Agent(env.OBS_DIM, env.ACTION_DIM)
    with np.load(path) as z:
        agent.load_state_dict({k: torch.as_tensor(z[k]) for k in z.files})
    return agent.eval()


def grade_run(run: Path, env, sets: dict) -> list[dict]:
    from acl_bench.suite.exam import STARTS
    from acl_bench.suite.run import section_scores
    cfg = json.loads((run / "config.json").read_text())
    agent = load_agent(run / "params.npz")
    rows = []
    for name, pairset in sets.items():
        passed, played = section_scores(env, agent, pairset)
        rows.append({"arm": cfg["arm"], "seed": cfg["seed"], "section": name, "check": cfg["steps"],
                     "n_questions": len(pairset) // STARTS, "pass_rate": passed, "play_rate": played,
                     "steps": cfg["steps"], "scouted_steps": 0, "budget": cfg["steps"], "n_checks": 1,
                     "wall_time_sec": cfg["wall_time_sec"]})
    return rows


def todo(out: Path, runs: list[Path], sections: list[str]) -> list[Path]:
    """The runs without a complete set of rows in `out`; drops the rows of incomplete ones."""
    if not out.exists():
        return runs
    prior = pd.read_csv(out)
    counts = prior.groupby(KEY).size()
    done = {key for key, n in counts.items() if n == len(sections)}
    stale = ~pd.Series(list(zip(*(prior[k] for k in KEY)))).isin(done).to_numpy()
    if stale.any():
        prior[~stale].to_csv(out, index=False)
    keys = [(c["arm"], c["seed"], c["steps"]) for c in (json.loads((r / "config.json").read_text()) for r in runs)]
    return [r for r, key in zip(runs, keys) if key not in done]


def main(argv=None):
    from acl_bench.study.run import append_rows
    from acl_bench.suite.exam import exam_dir
    from acl_bench.suite.run import load_exam
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="results/suite/cartpole_jaxued/scores.csv")
    ap.add_argument("--exam", default=None, help="default: frozen_sets/suite/cartpole")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    env = envs.get("cartpole")
    sets = load_exam(str(args.exam or exam_dir("cartpole")))
    if not sets:
        raise SystemExit("no exam found")
    runs = sorted(p.parent for p in Path(args.runs).rglob("config.json") if (p.parent / "params.npz").exists())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    left = todo(out, runs, list(sets))
    print(f"{len(runs)} runs found, {len(left)} to grade on {list(sets)}", flush=True)
    for run in left:
        rows = grade_run(run, env, sets)
        append_rows(str(out), rows)
        print(run, {r["section"]: r["pass_rate"] for r in rows}, flush=True)


if __name__ == "__main__":
    main()
