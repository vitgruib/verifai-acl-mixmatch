"""Maze grader: suite-format rows per (arm, seed, section), resume, and incomplete runs reported."""
import csv
import json

import numpy as np
import pytest

from atlas.maze import grade, suite

COLUMNS = ["arm", "seed", "section", "check", "n_questions", "pass_rate", "play_rate", "steps",
           "scouted_steps", "budget", "n_checks", "wall_time_sec"]
BUDGET = 245_760_000


class WallLimitAgent:
    """Solves every play of a level with fewer than `limit` walls and 3 of 10 plays of the rest."""

    def __init__(self, limit):
        self.limit = limit

    def run_batch(self, walls, agent, agent_dir, goal, attempts, seeds):
        solved = np.where(np.asarray(walls).sum(axis=(1, 2)) < self.limit, attempts, 3)
        return (np.arange(attempts)[None] < solved[:, None]).astype(float)


def section(n_levels=4):
    """Levels with 0, 1, ..., n_levels - 1 walls, labelled case0, case1, ... (as custom's are)."""
    levels = []
    for k in range(n_levels):
        walls = np.zeros((13, 13), bool)
        walls[0, 1:1 + k] = True
        levels.append((walls, (6, 6), 0, (7, 6)))
    return suite._from_levels(levels, [f"case{k}" for k in range(n_levels)])


def jaxued_run(runs, algo, seed, num_updates=30000, steps=(116, 118)):
    """With eval_freq 250 and interval 2 a 30,000-update run's last checkpoint is step 118."""
    d = runs / "maze_train" / "checkpoints" / f"maze_{algo}_n30000" / str(seed)
    for step in steps:
        (d / "models" / str(step)).mkdir(parents=True)
    key = "student_num_steps" if algo == "minimax" else "num_steps"
    (d / "config.json").write_text(json.dumps({"seed": seed, "num_updates": num_updates, "eval_freq": 250,
                                               "checkpoint_save_interval": 2, key: 256, "num_train_envs": 32}))
    return d


def sfl_run(runs, seed, total_timesteps=BUDGET, weights=True):
    d = runs / "sfl" / "maze_sfl" / f"s{seed}"
    d.mkdir(parents=True)
    if weights:
        (d / "model.safetensors").write_bytes(b"")
    (d / "config.json").write_text(json.dumps({"seed": seed, "algo": "sfl", "total_timesteps": total_timesteps}))
    return d


@pytest.fixture
def exam_dir(tmp_path):
    d = tmp_path / "exam"
    for name in ("random", "custom", "verifai"):
        suite.save_set(str(d), name, section())
    return d


@pytest.fixture
def loads(monkeypatch):
    """Every policy load, as (run dir, step); each agent solves levels with fewer than 2 walls."""
    seen = []

    def load(run_dir, step):
        seen.append((run_dir, step))
        return WallLimitAgent(2)
    monkeypatch.setattr(grade, "_load", load)
    return seen


def grade_cli(runs, exam, out, *extra):
    grade.main(["--runs", str(runs), "--exam", str(exam), "--out", str(out), *extra])


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_rows_match_suite_columns(tmp_path, exam_dir, loads):
    runs, out = tmp_path / "runs", tmp_path / "scores.csv"
    dr = jaxued_run(runs, "dr", 0)
    sfl = sfl_run(runs, 0)
    grade_cli(runs, exam_dir, out, "--arms", "DR", "SFL", "--seeds", "0")
    with open(out) as f:
        assert next(csv.reader(f)) == COLUMNS
    rows = read(out)
    assert sorted((r["arm"], r["section"]) for r in rows) == sorted(
        (arm, s) for arm in ("DR", "SFL") for s in ("random", "custom", "verifai"))
    for r in rows:
        assert int(r["seed"]) == 0
        assert int(r["check"]) == int(r["budget"]) == BUDGET
        assert int(r["n_checks"]) == 1 and int(r["n_questions"]) == 4
        # rates per level 1, 1, 0.3, 0.3
        assert float(r["pass_rate"]) == pytest.approx(0.5) and float(r["play_rate"]) == pytest.approx(0.65)
        # SFL Recipe A: 3750 // 100 = 37 rounds x 5 x 5000 levels x 2000 steps (docs/maze_gpu_audit.md)
        assert int(r["scouted_steps"]) == (1_850_000_000 if r["arm"] == "SFL" else 0)
        # SFL trains 37 rounds x 100 updates x 256 steps x 256 envs; JaxUED the whole budget
        assert int(r["steps"]) == (242_483_200 if r["arm"] == "SFL" else BUDGET)
    cases = read(tmp_path / "custom_cases.csv")
    assert [(c["arm"], c["seed"], c["case"], float(c["solve_rate"])) for c in cases if c["arm"] == "DR"] == [
        ("DR", "0", "case0", 1.0), ("DR", "0", "case1", 1.0), ("DR", "0", "case2", 0.3), ("DR", "0", "case3", 0.3)]
    assert {c["arm"] for c in cases} == {"DR", "SFL"} and len(cases) == 8
    assert sorted(loads) == sorted([(str(dr), 118), (str(sfl), -1)])


def test_resume_appends_only_missing(tmp_path, loads):
    runs, exam, out = tmp_path / "runs", tmp_path / "exam", tmp_path / "scores.csv"
    jaxued_run(runs, "minimax", 3)
    for name in ("random", "custom"):
        suite.save_set(str(exam), name, section())
    grade_cli(runs, exam, out, "--arms", "MINIMAX", "--seeds", "3")
    first = read(out)
    assert len(first) == 2
    suite.save_set(str(exam), "verifai", section(6))
    grade_cli(runs, exam, out, "--arms", "MINIMAX", "--seeds", "3")
    rows = read(out)
    assert rows[:2] == first
    assert [(r["section"], r["n_questions"]) for r in rows[2:]] == [("verifai", "6")]
    loads.clear()
    grade_cli(runs, exam, out, "--arms", "MINIMAX", "--seeds", "3")
    assert read(out) == rows and loads == []
    assert [c["case"] for c in read(tmp_path / "custom_cases.csv")] == ["case0", "case1", "case2", "case3"]


def test_missing_checkpoint_reported_nonzero_exit(tmp_path, exam_dir, loads, capsys):
    runs, out = tmp_path / "runs", tmp_path / "scores.csv"
    jaxued_run(runs, "plr", 0)
    jaxued_run(runs, "plr", 1, steps=(114, 116))             # killed before its final checkpoint
    jaxued_run(runs, "plr", 2, num_updates=50, steps=(0,))     # a smoke-sized run
    sfl_run(runs, 0, weights=False)                            # SFL killed before saving
    sfl_run(runs, 1, total_timesteps=1_000_000)
    with pytest.raises(SystemExit) as e:
        grade_cli(runs, exam_dir, out, "--arms", "PLR", "SFL", "--seeds", "0", "1", "2")
    assert e.value.code not in (0, None)
    err = capsys.readouterr().err
    for problem in ("PLR 1: no checkpoint at its final step 118", "PLR 2: trains 409600 env steps",
                    "SFL 0: no model.safetensors", "SFL 1: trains 1000000 env steps", "SFL 2: missing"):
        assert problem in err
    assert {(r["arm"], r["seed"]) for r in read(out)} == {("PLR", "0")}


def test_changed_exam_refuses_to_regrade(tmp_path, exam_dir, loads):
    runs, out = tmp_path / "runs", tmp_path / "scores.csv"
    jaxued_run(runs, "dr", 0)
    grade_cli(runs, exam_dir, out, "--arms", "DR", "--seeds", "0")
    before = out.read_bytes()
    suite.save_set(str(exam_dir), "custom", section(5))
    with pytest.raises(SystemExit, match="custom"):
        grade_cli(runs, exam_dir, out, "--arms", "DR", "--seeds", "0")
    assert out.read_bytes() == before
