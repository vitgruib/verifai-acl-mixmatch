"""The suite runner: arms, per-section score rows, charged SFL scouting, resume, snapshots."""
import dataclasses
import shutil

import numpy as np
import pandas as pd
import pytest

from acl_bench import envs
from acl_bench.envs import cartpole, frozenlake
from acl_bench.exam.sets import load_sets, save_sets
from acl_bench.plr.fast import FastConfig, train
from acl_bench.plr.levels import LevelConfig, LevelSampler
from acl_bench.snapshots import load_run
from acl_bench.suite import run
from acl_bench.suite.arms import ARMS
from acl_bench.suite.exam import question_scores, random_section

SECTIONS = ("random", "verifai", "custom")
BUDGET = 4096


def _exam(directory):
    env = envs.get("cartpole")
    sets = {name: dataclasses.replace(random_section(env, n_questions=2, seed=i), name=name)
            for i, name in enumerate(SECTIONS)}
    save_sets(sets, str(directory))
    return directory


def _run(out, exam, seeds, *extra):
    run.main(["--env", "cartpole", "--arms", "DR", "SFL", "--seeds", seeds, "--steps", str(BUDGET),
              "--workers", "2", "--allow-battery", "--min-battery-pct", "0",
              "--out", str(out), "--exam", str(exam), *extra])
    return pd.read_csv(out / "scores.csv")


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    exam = _exam(root / "exam")
    _run(root / "out", exam, "1-2")
    return root


def test_smoke_run_writes_one_row_per_section(first_run):
    rows = pd.read_csv(first_run / "out" / "scores.csv")
    assert list(rows.columns) == ["arm", "seed", "section", "n_questions", "pass_rate", "steps",
                                  "scouted_steps", "wall_time_sec"]
    assert sorted(zip(rows.arm, rows.seed, rows.section)) == sorted(
        (a, s, sec) for a in ("DR", "SFL") for s in (1, 2) for sec in SECTIONS)
    assert (rows.n_questions == 2).all()


def test_sfl_scouting_counts_against_the_budget(first_run):
    rows = pd.read_csv(first_run / "out" / "scores.csv")
    dr, sfl = rows[rows.arm == "DR"], rows[rows.arm == "SFL"]
    assert (dr.steps == BUDGET).all() and (dr.scouted_steps == 0).all()
    assert (sfl.steps + sfl.scouted_steps <= BUDGET).all()


def test_resume_neither_duplicates_nor_drops_rows(first_run, tmp_path):
    shutil.copytree(first_run / "out", tmp_path / "out")
    rows = _run(tmp_path / "out", first_run / "exam", "1-3", "--resume")
    keys = list(zip(rows.arm, rows.seed, rows.section))
    assert len(keys) == len(set(keys)) == 3 * 2 * 3
    before = pd.read_csv(first_run / "out" / "scores.csv")
    pd.testing.assert_frame_equal(rows.iloc[:len(before)], before)     # earlier rows kept as they were


def test_saved_agent_grades_the_same_after_reload(first_run):
    rows = pd.read_csv(first_run / "out" / "scores.csv")
    env, sets = envs.get("cartpole"), load_sets(str(first_run / "exam"))
    snaps = load_run(str(first_run / "out" / "snapshots" / "DR" / "1.npz"))
    assert list(snaps) == [BUDGET]
    for name in SECTIONS:
        row = rows[(rows.arm == "DR") & (rows.seed == 1) & (rows.section == name)].iloc[0]
        assert question_scores(env, snaps[BUDGET], sets[name]).mean() == pytest.approx(row.pass_rate)


def test_without_an_exam_it_only_trains_and_saves(tmp_path):
    run.main(["--env", "cartpole", "--arms", "REF", "--seeds", "1", "--steps", "2048", "--workers", "1",
              "--allow-battery", "--min-battery-pct", "0", "--out", str(tmp_path / "out"),
              "--exam", str(tmp_path / "no_exam")])
    assert not (tmp_path / "out" / "scores.csv").exists()
    assert list(load_run(str(tmp_path / "out" / "snapshots" / "REF" / "1.npz"))) == [2048]


def test_ref_seeds_differ_from_scored_seeds():
    assert {run.run_seed("REF", s) for s in range(1, 11)}.isdisjoint(
        {run.run_seed(a, s) for a in ARMS if a != "REF" for s in range(1, 21)})


class _CountingEnv:
    """cartpole, counting the training rollout's env steps (batches of `n_envs` rows)."""

    def __init__(self, n_envs):
        self.n_envs, self.train_steps = n_envs, 0

    def __getattr__(self, name):
        return getattr(cartpole, name)

    def step(self, state, action, params):
        if len(state) == self.n_envs:
            self.train_steps += len(state)
        return cartpole.step(state, action, params)


def test_charged_sfl_never_spends_more_than_the_budget():
    lv = dataclasses.replace(ARMS["SFL"], sfl_n=50, sfl_k=4, sfl_top=10, sfl_every=1)
    env = _CountingEnv(FastConfig().n_envs)
    _, _, stats = train("cartpole", env, FastConfig(steps=BUDGET, seed=3, levels=lv), 1, None)
    assert stats["scouted_steps"] > 0
    assert env.train_steps + stats["scouted_steps"] <= BUDGET


def test_frozenlake_mutation_keeps_the_layout_seed_and_size():
    bounds = np.array([frozenlake.PARAM_BOUNDS[k] for k in frozenlake.PARAM_ORDER], dtype=np.float64)
    fixed = [frozenlake.PARAM_ORDER.index(k) for k in frozenlake.MUTATE_FIXED]
    sampler = LevelSampler(ARMS["ACCEL"], bounds, np.random.default_rng(0), fixed=fixed)
    parent = np.array([8.0, 0.2, 0.3, 12345.0])
    children = np.stack([sampler._mutate(parent) for _ in range(20)])
    assert (children[:, fixed] == parent[fixed]).all()
    assert (children[:, frozenlake.PARAM_ORDER.index("hole_density")] != 0.2).all()


def test_other_envs_mutate_every_parameter():
    assert not hasattr(cartpole, "MUTATE_FIXED")
    bounds = np.array([cartpole.PARAM_BOUNDS[k] for k in cartpole.PARAM_ORDER], dtype=np.float64)
    sampler = LevelSampler(ARMS["ACCEL"], bounds, np.random.default_rng(0))
    parent = bounds.mean(axis=1)
    assert (sampler._mutate(parent) != parent).all()


def test_arms_are_configured_as_preregistered():
    plr = {"replay_prob": 0.8, "score": "maxmc", "beta": 0.3, "staleness": 0.3, "buffer": 1000}
    assert {k: getattr(ARMS["PLR"], k) for k in plr} == plr and not ARMS["PLR"].robust
    assert ARMS["RPLR"] == dataclasses.replace(ARMS["PLR"], robust=True)
    assert ARMS["ACCEL"] == dataclasses.replace(ARMS["RPLR"], accel=True, mut_sigma=LevelConfig().mut_sigma)
    assert ARMS["SFL"].sfl and ARMS["SFL"].charge_scouting and ARMS["SFL"].replay_prob == 0.5
    assert ARMS["DR"] == ARMS["REF"] == LevelConfig(replay_prob=0.0)


def test_resume_reruns_a_run_with_missing_rows_without_duplicating(first_run, tmp_path):
    shutil.copytree(first_run / "out", tmp_path / "out")
    prior = pd.read_csv(tmp_path / "out" / "scores.csv")
    partial = (prior.arm == "DR") & (prior.seed == 2) & (prior.section != "random")
    prior[~partial].to_csv(tmp_path / "out" / "scores.csv", index=False)
    rows = _run(tmp_path / "out", first_run / "exam", "1-2", "--resume")
    keys = list(zip(rows.arm, rows.seed, rows.section))
    assert len(keys) == len(set(keys)) == 2 * 2 * 3
