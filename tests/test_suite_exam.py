from types import SimpleNamespace

import numpy as np
import pytest

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
    s = verifai_section(fake_env(), list(range(10)), n_iters=2, starts=STARTS, min_fail_agents=3, seed=0, workers=1)
    assert set(s.params[:, 0]) == {1.0}
    assert len(s) == STARTS
    np.testing.assert_allclose(s.ref_fail_frac, 0.3)


def test_agent_fails_a_question_below_five_of_ten_plays(monkeypatch):
    # three agents pass 4 of 10 on task 1 (fail), but 5 of 10 on task 2 (pass)
    fake_world(monkeypatch, {1.0: [4] * 3 + [10] * 7, 2.0: [5] * 3 + [10] * 7})
    s = verifai_section(fake_env(), list(range(10)), n_iters=2, starts=STARTS, min_fail_agents=3, seed=0, workers=1)
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
    monkeypatch.setattr(exam, "_agent_digest", str)
    monkeypatch.setattr(exam, "exam_dir", lambda name: tmp_path)
    monkeypatch.setattr(exam, "pool_dir", lambda name: tmp_path / "pool")
    monkeypatch.setattr(cartpole, "sample_starts", lambda row, n, rng: np.zeros((n, 4)))
    build_exam("cartpole", list(range(3)), n_iters=2, seed=0, min_fail_agents=3, workers=1)
    assert set(load_sets(str(tmp_path))) == {"random", "verifai", "custom"}
    assert set(load_sets(str(tmp_path / "pool"))) == {"verifai"}
    assert not (tmp_path / ".search_cache").exists()


def test_verifai_section_stops_with_counts_when_no_task_survives(monkeypatch):
    fake_world(monkeypatch, {1.0: [10] * 10})
    with pytest.raises(SystemExit, match="1 pooled"):
        verifai_section(fake_env(), list(range(10)), n_iters=2, seed=0, workers=1)


def test_reference_agents_require_exactly_ten_ref_runs(tmp_path):
    with pytest.raises(SystemExit, match="exactly 10"):
        exam.reference_agents(tmp_path)


def test_regrade_is_chunked(monkeypatch):
    sizes = []

    def fake_grade(env, agent, params, s0, extras=None):
        sizes.append(len(params))
        return np.ones(len(params), dtype=bool), np.zeros(len(params))

    monkeypatch.setattr(exam, "grade", fake_grade)
    monkeypatch.setattr(exam, "GRADE_CHUNK", 30)
    pairset = PairSet("x", np.zeros((100, 1)), np.zeros((100, 1)))
    np.testing.assert_allclose(question_scores(fake_env(), 0, pairset), 1.0)
    assert max(sizes) <= 30


def test_frozenlake_verifai_tasks_are_whole_maps_and_must_be_winnable(monkeypatch):
    env = SimpleNamespace(
        PARAM_ORDER=("x", "layout_seed"), PARAM_BOUNDS={"x": (0, 3), "layout_seed": (0, 9)},
        sample_starts=lambda row, n, rng: np.zeros((n, 1)),
        certify_exact=lambda params, s0: np.ones(len(s0), dtype=bool),
        optimal_success=lambda row: 0.9 if row[0] == 1.0 else 0.2)

    def fake_search(env, name, agents, n_iters, starts_per_task, rng):
        params = np.repeat([[1.0, 7.6], [2.0, 3.2]], starts_per_task, axis=0)
        return PairSet("SEARCH", params, np.zeros((len(params), 1)), np.zeros(len(params)))

    monkeypatch.setattr(exam, "search", fake_search)
    monkeypatch.setattr(exam, "grade", lambda e, a, p, s, extras=None: (np.zeros(len(p), dtype=bool), np.zeros(len(p))))
    s = verifai_section(env, list(range(10)), n_iters=1, seed=0, workers=1)
    assert set(map(tuple, s.params)) == {(1.0, 7.0)}


def test_verifai_section_is_the_same_with_two_workers_as_with_one():
    import torch

    from acl_bench.ppo import Agent
    torch.manual_seed(0)
    agents = [Agent(cartpole.OBS_DIM, cartpole.ACTION_DIM) for _ in range(3)]
    serial = verifai_section(cartpole, agents, n_iters=4, min_fail_agents=1, seed=5, workers=1)
    parallel = verifai_section(cartpole, agents, n_iters=4, min_fail_agents=1, seed=5, workers=2)
    assert len(serial) > 0
    np.testing.assert_array_equal(parallel.params, serial.params)
    np.testing.assert_array_equal(parallel.s0, serial.s0)
    np.testing.assert_array_equal(parallel.ref_fail_frac, serial.ref_fail_frac)


def test_rerun_with_a_search_cache_skips_the_searches(monkeypatch, tmp_path):
    fake_world(monkeypatch, {1.0: [0] * 3 + [10] * 7, 2.0: [0] * 5 + [10] * 5})
    monkeypatch.setattr(exam, "_agent_digest", str)          # the fake agents are ints
    first = verifai_section(fake_env(), list(range(10)), n_iters=2, seed=0, workers=1, cache_dir=tmp_path)

    def no_search(*args):
        raise AssertionError("search ran despite the cache")

    monkeypatch.setattr(exam, "search", no_search)
    again = verifai_section(fake_env(), list(range(10)), n_iters=2, seed=0, workers=1, cache_dir=tmp_path)
    assert again.digest() == first.digest()


def test_cached_search_is_not_reused_for_an_agent_with_different_weights(monkeypatch, tmp_path):
    import torch

    from acl_bench.ppo import Agent
    searched = []

    def fake_search(env, name, agents, n_iters, starts_per_task, rng):
        searched.append(agents[0])
        return PairSet("SEARCH", np.ones((starts_per_task, 1)), np.zeros((starts_per_task, 1)), np.zeros(starts_per_task))

    monkeypatch.setattr(exam, "search", fake_search)
    monkeypatch.setattr(exam, "grade", lambda e, a, p, s, extras=None: (np.zeros(len(p), dtype=bool), np.zeros(len(p))))
    torch.manual_seed(0)
    old, retrained = Agent(cartpole.OBS_DIM, cartpole.ACTION_DIM), Agent(cartpole.OBS_DIM, cartpole.ACTION_DIM)
    verifai_section(fake_env(), [old], n_iters=1, min_fail_agents=1, seed=0, workers=1, cache_dir=tmp_path)
    verifai_section(fake_env(), [retrained], n_iters=1, min_fail_agents=1, seed=0, workers=1, cache_dir=tmp_path)
    assert searched == [old, retrained]


def test_cap_questions_keeps_whole_questions_and_their_fail_fractions():
    params = np.repeat(np.arange(50.0)[:, None], 10, axis=0)
    s0 = np.arange(500.0)[:, None]
    frac = np.repeat(np.arange(50) / 100, 10)
    capped = exam.cap_questions(PairSet("verifai", params, s0, frac), max_questions=20, seed=7)
    q = capped.params[::10, 0].astype(int)
    assert len(capped) == 200 and len(set(q)) == 20
    assert (capped.params[:, 0] == np.repeat(q, 10)).all()
    assert (capped.s0[:, 0] == (np.repeat(q, 10) * 10 + np.tile(np.arange(10), 20))).all()
    assert np.allclose(capped.ref_fail_frac, np.repeat(q / 100, 10))
    again = exam.cap_questions(PairSet("verifai", params, s0, frac), max_questions=20, seed=7)
    assert again.digest() == capped.digest()


def test_cap_questions_leaves_a_small_section_alone():
    ps = PairSet("verifai", np.zeros((30, 2)), np.zeros((30, 4)), np.full(30, 0.3))
    assert exam.cap_questions(ps, max_questions=20, seed=7) is ps
