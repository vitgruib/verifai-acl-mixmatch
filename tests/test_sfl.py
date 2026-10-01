"""SFL (and `sfl_tilt`, the confirmed arm): the scouting score ranks levels by
p(1-p)(1-p)^tilt, the sampler replays the scouted buffer at `replay_prob`, scouting is
reported but never charged, and nothing in training reads the exam."""
import dataclasses
import types

import numpy as np
import pytest

from acl_bench import envs
from acl_bench.plr.fast import FastConfig, make_agent, sfl_select, train
from acl_bench.plr.levels import NEW, REPLAY, LevelConfig, LevelSampler
from acl_bench.plr.screen import CONFIGS


def _coin_env():
    """One-step 'reach' env whose level x in [0, 1] passes with probability x, whatever the
    policy does: scouting's p estimates x directly."""
    env = types.ModuleType("coin")
    env.PARAM_BOUNDS, env.PARAM_ORDER = {"x": (0.0, 1.0)}, ["x"]
    env.OBS_DIM, env.ACTION_DIM, env.MAX_EPISODE_STEPS, env.GOAL = 1, 2, 1, "reach"
    rng = np.random.default_rng(0)
    env.sample_starts = lambda p, n, r: np.zeros((n, 1))
    env.observe = lambda s: s
    env.step = lambda s, a, p: rng.uniform(size=(len(s), 1))
    env.terminated = lambda s: s[:, 0] < env._params[:, 0]
    return env


def _select(tilt, n=400, k=400, top=20):
    env = _coin_env()
    bounds = np.array([[0.0, 1.0]])
    lc = LevelConfig(sfl=True, sfl_n=n, sfl_k=k, sfl_top=top, sfl_tilt=tilt)
    levels = LevelSampler(lc, bounds, np.random.default_rng(1))
    agent = make_agent(env, bounds, FastConfig())
    step = env.step

    def step_with_params(s, a, p):              # terminated() needs this step's levels
        env._params = p
        return step(s, a, p)
    env.step = step_with_params
    return sfl_select(env, agent, levels, lc, np.random.default_rng(2)), lc


@pytest.mark.parametrize("tilt,peak", [(0.0, 0.5), (1.0, 1 / 3)])
def test_score_keeps_levels_near_its_peak(tilt, peak):
    (top, sim), lc = _select(tilt)
    assert top.shape == (lc.sfl_top, 1)
    assert np.all((top >= 0) & (top <= 1))
    assert abs(np.median(top) - peak) < 0.06     # tilt=0 is plain SFL; tilt=1 moves to p=1/3
    assert sim == lc.sfl_n * lc.sfl_k            # one step per rollout in this env


def test_pick_replays_buffer_at_replay_prob_only_after_scouting():
    lc = LevelConfig(sfl=True, replay_prob=0.5)
    s = LevelSampler(lc, np.array([[0.0, 1.0], [5.0, 6.0]]), np.random.default_rng(0))
    assert all(s.pick()[2] == NEW for _ in range(500))      # pure DR before the first scout
    buf = np.array([[0.123, 5.5], [0.456, 5.9]])
    s.set_sfl(buf)
    picks = [s.pick() for _ in range(4000)]
    rep = [p for p, _, kind in picks if kind == REPLAY]
    assert abs(len(rep) / len(picks) - 0.5) < 0.03
    assert all(any(np.array_equal(p, b) for b in buf) for p in rep)
    new = np.array([p for p, _, kind in picks if kind == NEW])
    assert np.all((new[:, 0] >= 0) & (new[:, 0] <= 1) & (new[:, 1] >= 5) & (new[:, 1] <= 6))


def _short(config, steps=4096, seed=7):
    lv, fast = CONFIGS[config]
    cfg = FastConfig(steps=steps, seed=seed, levels=dataclasses.replace(LevelConfig(), **lv), **fast)
    lv_small = dict(sfl_n=50, sfl_k=4, sfl_top=10, sfl_every=1) if cfg.levels.sfl else {}
    cfg = dataclasses.replace(cfg, levels=dataclasses.replace(cfg.levels, **lv_small))
    steps_seen = []
    _, checks, stats = train("cartpole", envs.get("cartpole"), cfg, 4,
                             lambda step, agent: steps_seen.append(step) or {})
    return checks, stats, steps_seen


def test_sfl_tilt_is_configured_as_registered():
    lv, fast = CONFIGS["sfl_tilt"]
    cfg = dataclasses.replace(LevelConfig(), **lv)
    assert cfg.sfl and cfg.sfl_tilt == 1.0 and cfg.replay_prob == 0.5
    assert not cfg.charge_scouting and not cfg.oracle and fast == {}


def test_scouting_is_uncharged_and_reported():
    dr_checks, dr_stats, dr_steps = _short("DR")
    sf_checks, sf_stats, sf_steps = _short("sfl_tilt")
    assert dr_steps == sf_steps                       # check-ins at the same training steps
    assert dr_stats["scouted_steps"] == 0
    assert sf_stats["scouted_steps"] >= 4 * 50 * 4    # 4 scouts x n x k rollouts, >= 1 step each
    assert sf_stats["replays"] > 0 and dr_stats["replays"] == 0


def test_same_seed_same_run():
    a, b = _short("sfl_tilt")[1], _short("sfl_tilt")[1]
    assert a == b


def test_training_never_reads_the_exam(monkeypatch):
    import acl_bench.exam.sets as sets
    import acl_bench.plr.fast as fast

    def boom(*a, **k):
        raise AssertionError("training touched the exam")
    monkeypatch.setattr(sets, "load_sets", boom)
    monkeypatch.setattr(sets, "evaluate_sets", boom)
    monkeypatch.setattr(fast, "oracle_tasks", boom)
    _short("sfl_tilt")
