from types import SimpleNamespace

import numpy as np

from acl_bench.envs import cartpole, frozenlake
from acl_bench.exam.sets import PairSet, load_sets, save_sets
from acl_bench.suite import exam
from acl_bench.suite.exam import (
    build_exam,
    exam_dir,
    question_scores,
    random_section,
    verifai_section,
)

STARTS = 10


def fake_env():
    return SimpleNamespace(
        PARAM_ORDER=("x",), PARAM_BOUNDS={"x": (0, 3)},
        sample_starts=lambda row, n, rng: np.zeros((n, 1)))


def fake_world(monkeypatch, passes_by_task, width=1):
    """Tasks are told apart by param x; an agent (an int) passes the first
    `passes_by_task[x][agent]` starts of a task's 10 and fails the rest."""
    tasks = np.array(sorted(passes_by_task))

    def fake_search(env, name, agents, n_iters, starts_per_task, rng):
        params = np.pad(np.repeat(tasks, starts_per_task)[:, None], ((0, 0), (0, width - 1)))
        return PairSet("SEARCH", params, np.zeros((len(params), 1)), np.zeros(len(params)))

    def fake_grade(env, agent, params, s0, extras=None):
        n_passed = np.array([passes_by_task[x][agent] for x in params[:, 0]])
        return np.arange(len(params)) % STARTS < n_passed, np.zeros(len(params))

    monkeypatch.setattr(exam, "search", fake_search)
    monkeypatch.setattr(exam, "grade", fake_grade)


def test_question_kept_only_when_three_of_ten_agents_fail(monkeypatch):
    fail, ok = 0, 10
    fake_world(monkeypatch, {1.0: [fail] * 3 + [ok] * 7, 2.0: [fail] * 2 + [ok] * 8})
    s = verifai_section(fake_env(), list(range(10)), n_iters=2, starts=STARTS, min_fail_agents=3, seed=0)
    assert set(s.params[:, 0]) == {1.0}
    assert len(s) == STARTS
    np.testing.assert_allclose(s.ref_fail_frac, 0.3)


def test_agent_fails_a_question_below_five_of_ten_plays(monkeypatch):
    # three agents pass 4 of 10 on task 1 (fail), but 5 of 10 on task 2 (pass)
    fake_world(monkeypatch, {1.0: [4] * 3 + [10] * 7, 2.0: [5] * 3 + [10] * 7})
    s = verifai_section(fake_env(), list(range(10)), n_iters=2, starts=STARTS, min_fail_agents=3, seed=0)
    assert set(s.params[:, 0]) == {1.0}
    scores = question_scores(fake_env(), 0, PairSet("x", np.repeat([1.0, 2.0], STARTS)[:, None], np.zeros((20, 1))))
    np.testing.assert_allclose(scores, [0.4, 0.5])


def test_random_section_is_unfiltered_on_continuous_tasks():
    s = random_section(cartpole, n_questions=50, starts=10, seed=0)
    assert len(s.params) == 50 * 10
    lo, hi = np.array([cartpole.PARAM_BOUNDS[k] for k in cartpole.PARAM_ORDER]).T
    assert ((s.params >= lo) & (s.params <= hi)).all()
    assert (s.params == np.repeat(s.params[::10], 10, axis=0)).all()


def test_random_section_is_reproducible_from_its_seed():
    a, b = random_section(cartpole, 5, seed=3), random_section(cartpole, 5, seed=3)
    assert a.digest() == b.digest()
    assert a.digest() != random_section(cartpole, 5, seed=4).digest()


def test_frozenlake_random_section_keeps_only_winnable_maps():
    s = random_section(frozenlake, n_questions=6, starts=10, seed=0)
    assert len(s) == 60
    assert frozenlake.certify_exact(s.params, s.s0).all()
    assert all(frozenlake.optimal_success(p) >= 0.5 for p in np.unique(s.params, axis=0))
    assert (s.params[:, -1] == s.params[:, -1].round()).all()


def test_exam_dir_is_inside_the_repo_whatever_the_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert exam_dir("cartpole").parts[-3:] == ("frozen_sets", "suite", "cartpole")
    assert exam_dir("cartpole").parent.parent.parent.joinpath("acl_bench").is_dir()


def test_exam_round_trips_with_checksums(tmp_path):
    sets = {"random": random_section(cartpole, 4, seed=1)}
    sets["verifai"] = PairSet("verifai", sets["random"].params, sets["random"].s0, np.full(40, 0.3))
    save_sets(sets, str(tmp_path))
    loaded = load_sets(str(tmp_path))
    assert {k: v.digest() for k, v in loaded.items()} == {k: v.digest() for k, v in sets.items()}


def test_build_exam_writes_all_three_sections(monkeypatch, tmp_path):
    fake_world(monkeypatch, {1.0: [0] * 3}, width=5)
    monkeypatch.setattr(exam, "exam_dir", lambda name: tmp_path)
    monkeypatch.setattr(cartpole, "sample_starts", lambda row, n, rng: np.zeros((n, 4)))
    build_exam("cartpole", list(range(3)), n_iters=2, seed=0, min_fail_agents=3)
    assert set(load_sets(str(tmp_path))) == {"random", "verifai", "custom"}
