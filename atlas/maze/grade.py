"""Grade every scored Maze run's final agent on the frozen exam, in the classic suite's columns.

  python -m atlas.maze.grade --runs runs --exam frozen_sets/suite/maze --out results/suite/maze/scores.csv

Runs are where cluster/1_maze/train.sbatch puts them under --runs: JaxUED's
maze_train/checkpoints/maze_<algo>_n30000/<seed> (orbax, graded at the final step its
config.json implies, as atlas.maze.exam does) and SFL's sfl/maze_sfl/s<seed> (model.safetensors
plus a config.json that must give `total_timesteps`). One row per (arm, seed, section):
`check` = `steps` = `budget` = BUDGET, `n_checks` 1, `pass_rate` the share of levels solved in
at least half of 10 plays, `play_rate` the mean solve rate, `scouted_steps` 0 except SFL's
uncharged scouting (docs/maze_gpu_audit.md section 4-6), `wall_time_sec` the grading time.

Rows already in --out are kept and only missing ones are graded. The section digests graded
are recorded beside --out (<name>.exam.json); a later exam whose section differs is refused.
A run that is missing, trained for another budget or lacks its final checkpoint is listed on
stderr, and the command exits non-zero after grading the rest.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from atlas.maze import exam, suite

BUDGET = 245_760_000
ARMS = {"DR": "dr", "PLR": "plr", "RPLR": "rplr", "ACCEL": "accel", "MINIMAX": "minimax", "SFL": "sfl"}
SEEDS = range(10)
COLUMNS = ["arm", "seed", "section", "check", "n_questions", "pass_rate", "play_rate", "steps",
           "scouted_steps", "budget", "n_checks", "wall_time_sec"]
# SFL repo defaults (minigrid-sfl.yaml, ippo-minigrid.yaml), Recipe A of the audit; config.json
# entries of the same (lower-cased Hydra) names override them.
SFL_RECIPE = {"num_steps": 256, "num_envs": 256, "eval_freq": 100, "num_batches": 5, "batch_size": 5000,
              "rollout_steps": 2000}


def _load(run_dir: str, step: int):
    from atlas.maze.policy import Policy
    return Policy(run_dir, step)


def _jaxued_run(runs: Path, arm: str, seed: int):
    """-> (run dir, step to grade, scouted steps), or a problem string."""
    d = runs / "maze_train" / "checkpoints" / f"maze_{ARMS[arm]}_n30000" / str(seed)
    if not (d / "config.json").exists():
        return "missing"
    with open(d / "config.json") as f:
        cfg = json.load(f)
    steps = cfg["num_updates"] * cfg["num_train_envs"] * cfg["student_num_steps" if arm == "MINIMAX" else "num_steps"]
    if steps != BUDGET:
        return f"trains {steps} env steps, not the budget {BUDGET}"
    step = exam._final_step(cfg)
    if not (d / "models" / str(step)).is_dir():
        return f"no checkpoint at its final step {step} (training unfinished)"
    return str(d), step, 0


def _sfl_run(runs: Path, seed: int):
    d = runs / "sfl" / "maze_sfl" / f"s{seed}"
    if not (d / "config.json").exists():
        return "missing"
    if not (d / "model.safetensors").exists():
        return "no model.safetensors (training unfinished)"
    with open(d / "config.json") as f:
        cfg = {**SFL_RECIPE, **json.load(f)}
    if "total_timesteps" not in cfg:
        return "config.json has no total_timesteps"
    if cfg["total_timesteps"] != BUDGET:
        return f"trains {cfg['total_timesteps']} env steps, not the budget {BUDGET}"
    # scouting runs once per round, before each round's eval_freq updates
    rounds = cfg["total_timesteps"] // cfg["num_steps"] // cfg["num_envs"] // cfg["eval_freq"]
    return str(d), -1, rounds * cfg["num_batches"] * cfg["batch_size"] * cfg["rollout_steps"]


def _check_exam(sets: dict, record: Path) -> None:
    """Refuse if a section graded into --out has changed; record the current digests."""
    digests = {name: ms.digest() for name, ms in sets.items()}
    stored = json.loads(record.read_text()) if record.exists() else {}
    changed = sorted(name for name in stored.keys() & digests.keys() if stored[name] != digests[name])
    if changed:
        sys.exit(f"exam sections {changed} differ from the ones graded into {record.parent}; "
                 "refusing to mix them (grade into a new --out)")
    record.write_text(json.dumps({**stored, **digests}, indent=2, sort_keys=True))


def _graded(out: Path) -> set:
    if not out.exists():
        return set()
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            sys.exit(f"{out} has columns {reader.fieldnames}, not {COLUMNS}")
        return {(r["arm"], int(r["seed"]), r["section"]) for r in reader}


def _append(out: Path, row: dict) -> None:
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="the runs root train.sbatch writes under ($RUNS)")
    ap.add_argument("--exam", default="frozen_sets/suite/maze")
    ap.add_argument("--out", default="results/suite/maze/scores.csv")
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    a = ap.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    sets = suite.load_sets(a.exam)
    os.makedirs(out.parent, exist_ok=True)
    _check_exam(sets, out.with_suffix(".exam.json"))
    done, problems = _graded(out), []
    for arm in a.arms:
        for seed in a.seeds:
            todo = [name for name in sets if (arm, seed, name) not in done]
            if not todo:
                continue
            run = _sfl_run(runs, seed) if arm == "SFL" else _jaxued_run(runs, arm, seed)
            if isinstance(run, str):
                problems.append(f"{arm} {seed}: {run}")
                continue
            run_dir, step, scouted = run
            policy = _load(run_dir, step)
            for name in todo:
                t0 = time.time()
                rates = exam.solve_rates(policy, sets[name], label=f"{arm} {seed} {name}")
                _append(out, {"arm": arm, "seed": seed, "section": name, "check": BUDGET,
                              "n_questions": len(rates), "pass_rate": float((rates >= exam.TAU).mean()),
                              "play_rate": float(rates.mean()), "steps": BUDGET, "scouted_steps": scouted,
                              "budget": BUDGET, "n_checks": 1, "wall_time_sec": time.time() - t0})
            print(f"graded {arm} {seed}", flush=True)
    if problems:
        print(f"{len(problems)} runs not graded:\n  " + "\n  ".join(problems), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
