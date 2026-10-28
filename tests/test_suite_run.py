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
from acl_bench.suite.arms import ARMS, SFL_SCOUT, arm_config
from acl_bench.suite.exam import question_scores, random_section

SECTIONS = ("random", "verifai", "custom")
BUDGET, CHECKS = 65536, 4          # SFL's first scout spans several check-ins at this budget
THRESHOLDS = [BUDGET // CHECKS * k for k in range(1, CHECKS + 1)]


def _exam(directory):
    env = envs.get("cartpole")
    sets = {name: dataclasses.replace(random_section(env, n_questions=2, seed=i), name=name)
            for i, name in enumerate(SECTIONS)}
    save_sets(sets, str(directory))
    return directory


def _run(out, exam, seeds, *extra, arms=("DR", "SFL"), steps=BUDGET, checks=CHECKS):
    run.main(["--env", "cartpole", "--arms", *arms, "--seeds", seeds, "--steps", str(steps),
              "--checks", str(checks), "--workers", "2", "--allow-battery", "--min-battery-pct", "0",
              "--out", str(out), "--exam", str(exam), *extra])
    return pd.read_csv(out / "scores.csv")


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    exam = _exam(root / "exam")
    _run(root / "out", exam, "1-2")
    return root


def _keys(rows):
    return list(zip(rows.arm, rows.seed, rows.section, rows.check))


def test_smoke_run_writes_one_row_per_section_and_check_in(first_run):
    rows = pd.read_csv(first_run / "out" / "scores.csv")
    assert list(rows.columns) == ["arm", "seed", "section", "check", "n_questions", "pass_rate", "play_rate",
                                  "steps", "scouted_steps", "budget", "n_checks", "wall_time_sec"]
    assert sorted(_keys(rows)) == sorted(
        (a, s, sec, c) for a in ("DR", "SFL") for s in (1, 2) for sec in SECTIONS for c in THRESHOLDS)
    assert (rows.n_questions == 2).all() and (rows.budget == BUDGET).all() and (rows.n_checks == CHECKS).all()


def test_sfl_scouting_counts_against_the_budget(first_run):
    rows = pd.read_csv(first_run / "out" / "scores.csv")
    dr, sfl = rows[rows.arm == "DR"], rows[rows.arm == "SFL"]
    assert (dr.steps == dr.check).all() and (dr.scouted_steps == 0).all()
    assert (sfl.scouted_steps > 0).all()
    assert (sfl.steps + sfl.scouted_steps <= BUDGET).all()


def test_resume_neither_duplicates_nor_drops_rows(first_run, tmp_path):
    shutil.copytree(first_run / "out", tmp_path / "out")
    rows = _run(tmp_path / "out", first_run / "exam", "1-3", "--resume")
    assert len(_keys(rows)) == len(set(_keys(rows))) == 3 * 2 * 3 * CHECKS
    before = pd.read_csv(first_run / "out" / "scores.csv")
    pd.testing.assert_frame_equal(rows.iloc[:len(before)], before)     # earlier rows kept as they were


@pytest.mark.parametrize("settings", [{"steps": BUDGET // 2}, {"checks": 2, "arms": ("DR",)}])
def test_resume_with_other_settings_stops_and_keeps_the_scores(first_run, tmp_path, settings):
    shutil.copytree(first_run / "out", tmp_path / "out")
    with pytest.raises(SystemExit, match="--resume"):
        _run(tmp_path / "out", first_run / "exam", "1-2", "--resume", **settings)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "out" / "scores.csv"),
                                  pd.read_csv(first_run / "out" / "scores.csv"))


def test_resume_reruns_only_the_incomplete_runs_it_was_asked_for(first_run, tmp_path):
    shutil.copytree(first_run / "out", tmp_path / "out")
    prior = pd.read_csv(tmp_path / "out" / "scores.csv")
    cut = (prior.seed == 2) & (prior.arm == "DR") | (prior.seed == 1) & (prior.arm == "SFL")
    prior[~(cut & (prior.check > THRESHOLDS[0]))].to_csv(tmp_path / "out" / "scores.csv", index=False)
    rows = _run(tmp_path / "out", first_run / "exam", "2", "--resume", arms=("DR",))
    assert len(_keys(rows)) == len(set(_keys(rows)))
    assert len(rows[(rows.arm == "DR") & (rows.seed == 2)]) == 3 * CHECKS
    assert len(rows[(rows.arm == "SFL") & (rows.seed == 1)]) == 3        # not asked for: left as it was


def test_saved_agent_grades_the_same_after_reload(first_run):
    rows = pd.read_csv(first_run / "out" / "scores.csv")
    env, sets = envs.get("cartpole"), load_sets(str(first_run / "exam"))
    snaps = load_run(str(first_run / "out" / "snapshots" / "DR" / "1.npz"))
    assert list(snaps) == THRESHOLDS
    for name in SECTIONS:
        row = rows[(rows.arm == "DR") & (rows.seed == 1) & (rows.section == name) & (rows.check == BUDGET)].iloc[0]
        scores = question_scores(env, snaps[BUDGET], sets[name])
        assert (scores >= 0.5).mean() == pytest.approx(row.pass_rate)
        assert scores.mean() == pytest.approx(row.play_rate)


def test_pass_rate_counts_questions_passed_and_play_rate_counts_plays(monkeypatch):
    from acl_bench.exam.sets import PairSet
    from acl_bench.suite import exam
    passes = {1.0: 4, 2.0: 5, 3.0: 10}           # plays passed of 10, per question

    def fake_grade(env, agent, params, s0, extras=None):
        return np.arange(len(params)) % 10 < np.array([passes[x] for x in params[:, 0]]), np.zeros(len(params))
    monkeypatch.setattr(exam, "grade", fake_grade)
    ps = PairSet("q", np.repeat([1.0, 2.0, 3.0], 10)[:, None], np.zeros((30, 1)))
    assert run.section_scores(None, None, ps) == pytest.approx((2 / 3, 19 / 30))


def test_sfl_on_an_env_without_scouting_size_stops_before_any_run(tmp_path):
    with pytest.raises(SystemExit, match="SFL has no SFL_SCOUT entry for lunarlander"):
        run.main(["--env", "lunarlander", "--arms", "DR", "SFL", "--seeds", "1", "--steps", "4096",
                  "--allow-battery", "--out", str(tmp_path / "out"), "--exam", str(tmp_path / "exam")])
    assert not (tmp_path / "out").exists()


def test_without_an_exam_it_only_trains_and_saves(tmp_path):
    run.main(["--env", "cartpole", "--arms", "REF", "--seeds", "1", "--steps", "2048", "--workers", "1",
              "--allow-battery", "--min-battery-pct", "0", "--out", str(tmp_path / "out"),
              "--exam", str(tmp_path / "no_exam")])
    assert not (tmp_path / "out" / "scores.csv").exists()
    assert list(load_run(str(tmp_path / "out" / "snapshots" / "REF" / "1.npz"))) == [2048]


def test_charged_sfl_keeps_no_buffer_from_a_scout_that_cannot_fit():
    _, _, stats = train("cartpole", envs.get("cartpole"), FastConfig(steps=4096, seed=3, levels=ARMS["SFL"]), 1)
    assert stats["scouted_steps"] == 0 and stats["replays"] == 0


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
    _, _, stats = train("cartpole", env, FastConfig(steps=4096, seed=3, levels=lv), 1, None)
    assert stats["scouted_steps"] > 0
    assert env.train_steps + stats["scouted_steps"] <= 4096


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


def test_sfl_scouting_is_sized_per_env_and_other_arms_are_not():
    scout = {k: getattr(arm_config("SFL", "cartpole"), k) for k in ("sfl_n", "sfl_k", "sfl_top", "sfl_every")}
    assert scout == {"sfl_k": 8, **SFL_SCOUT["cartpole"]}
    assert arm_config("SFL", "cartpole") == dataclasses.replace(ARMS["SFL"], **SFL_SCOUT["cartpole"])
    for env in ("lunarlander", "frozenlake"):
        with pytest.raises(KeyError, match=f"SFL has no SFL_SCOUT entry for {env}"):
            arm_config("SFL", env)
    assert all(arm_config(arm, "cartpole") == ARMS[arm] for arm in ARMS if arm != "SFL")


