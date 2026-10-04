"""SFL (and `sfl_tilt`, the confirmed arm): the scouting score ranks levels by
p(1-p)(1-p)^tilt, the sampler replays the scouted buffer at `replay_prob`, scouting is
reported but never charged, and nothing in training reads the exam."""
import dataclasses
import types

import numpy as np
import pytest

from acl_bench import envs
from acl_bench.plr.fast import FastConfig, _archive, make_agent, sfl_select, train
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


def _select(tilt, n=400, k=400, top=20, **kw):
    env = _coin_env()
    bounds = np.array([[0.0, 1.0]])
    lc = LevelConfig(sfl=True, sfl_n=n, sfl_k=k, sfl_top=top, sfl_tilt=tilt, **kw)
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
    (top, sim, w), lc = _select(tilt)
    assert w is None
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


def test_soft_keeps_every_learnable_level_weighted_by_score():
    (lv, sim, w), _ = _select(1.0, n=200, k=50, sfl_soft=True)
    assert len(lv) == len(w) and np.all(w > 0)
    x = lv[:, 0]
    p_hat = np.clip(x, 0, 1)                          # pass rate of level x is x
    near, far = w[np.abs(p_hat - 1 / 3) < 0.1].mean(), w[p_hat > 0.9].mean()
    assert near > 3 * far                              # weight follows p(1-p)^2


def test_soft_pick_follows_weights():
    s = LevelSampler(LevelConfig(sfl=True, replay_prob=1.0), np.array([[0.0, 1.0]]),
                     np.random.default_rng(0))
    s.set_sfl(np.array([[0.1], [0.2]]), np.array([3.0, 1.0]))
    reps = np.array([s.pick()[0][0] for _ in range(4000)])
    assert abs((reps == 0.1).mean() - 0.75) < 0.03


def test_carry_rescouts_the_last_buffer():
    env = _coin_env()
    bounds = np.array([[0.0, 1.0]])
    lc = LevelConfig(sfl=True, sfl_n=100, sfl_k=200, sfl_top=10, sfl_tilt=1.0, sfl_carry=True)
    levels = LevelSampler(lc, bounds, np.random.default_rng(1))
    levels.set_sfl(np.full((10, 1), 1 / 3))           # a buffer of ideal levels
    step = env.step

    def step_with_params(s, a, p):
        env._params = p
        return step(s, a, p)
    env.step = step_with_params
    top, _, _ = sfl_select(env, make_agent(env, bounds, FastConfig()), levels, lc, np.random.default_rng(2))
    assert np.sum(np.isclose(top[:, 0], 1 / 3)) >= 5   # most of the old frontier survives


def test_verify_shrinks_the_winners_curse():
    (plain, _, _), _ = _select(1.0, n=1000, k=8, top=100)
    (ver, sim, _), lc = _select(1.0, n=1000, k=8, top=100, sfl_verify=200)
    err = lambda lv: np.abs(lv[:, 0] - 1 / 3).mean()
    assert err(ver) < 0.8 * err(plain)                 # more rollouts pick truer 1/3 levels
    assert sim == (1000 + 200 * 3) * 8                    # the re-rolls are reported as scouting


def test_prescreen_predicts_the_frontier():
    from acl_bench.plr.fast import _prescreen
    lc = LevelConfig(sfl=True, sfl_n=500, sfl_tilt=1.0, sfl_amort=10)
    s = LevelSampler(lc, np.array([[0.0, 1.0]]), np.random.default_rng(0))
    x = s.rng.uniform(size=(500, 1))
    out = _prescreen(s, [(x, x[:, 0])], lc, 100)       # measured p of level x is x
    assert out.shape == (100, 1) and abs(np.median(out) - 1 / 3) < 0.05


@pytest.mark.parametrize("arm", ["sfl_tilt_soft", "sfl_tilt_carry", "sfl_tilt_verify", "sfl_tilt_amort"])
def test_batch12_arms_train(arm):
    checks, stats, _ = _short(arm)
    assert stats["scouted_steps"] > 0
    if arm != "sfl_tilt_soft":
        assert stats["replays"] > 0     # soft keeps no buffer (pure DR) while every score is 0


def test_halving_spends_rollouts_on_the_frontier():
    (plain, _, _), _ = _select(1.0, n=1000, k=8, top=100)
    (hv, sim, _), lc = _select(1.0, n=4000, k=2, top=100, sfl_halving=2)
    err = lambda lv: np.abs(lv[:, 0] - 1 / 3).mean()
    assert hv.shape == (100, 1) and err(hv) < 0.8 * err(plain)
    assert sim == 4000 * 2 + 1000 * 6 + 250 * 24


def test_bisect_adds_boundary_levels_near_the_peak():
    from acl_bench.plr.fast import _bisect
    env = _coin_env()
    step = env.step

    def step_with_params(s, a, p):
        env._params = p
        return step(s, a, p)
    env.step = step_with_params
    lc = LevelConfig(sfl=True, sfl_k=200, sfl_tilt=1.0, sfl_bisect=50)
    s = LevelSampler(lc, np.array([[0.0, 1.0]]), np.random.default_rng(0))
    cand = np.array([[0.02], [0.98], [0.05], [0.95]])
    mid, pm, sim = _bisect(env, make_agent(env, s.bounds, FastConfig()), s, cand, cand[:, 0], lc,
                           np.random.default_rng(1))
    assert mid.shape == (50, 1) and sim == 3 * 50 * 200
    assert abs(np.median(mid) - 1 / 3) < 0.1           # 3 halvings of [0.02, 0.95]: within ~0.12


def test_spread_picks_distinct_levels():
    from acl_bench.plr.fast import _farthest
    s = LevelSampler(LevelConfig(sfl=True), np.array([[0.0, 1.0]]), np.random.default_rng(0))
    x = np.array([[0.5], [0.51], [0.0], [1.0], [0.49]])
    assert sorted(_farthest(s, x, 3)) == [0, 2, 3]


def test_ghost_rescores_with_the_last_policy():
    env = _coin_env()
    step = env.step

    def step_with_params(s, a, p):
        env._params = p
        return step(s, a, p)
    env.step = step_with_params
    bounds = np.array([[0.0, 1.0]])
    lc = LevelConfig(sfl=True, sfl_n=100, sfl_k=8, sfl_top=10, sfl_tilt=1.0, sfl_ghost=0.5)
    levels = LevelSampler(lc, bounds, np.random.default_rng(1))
    agent = make_agent(env, bounds, FastConfig())
    _, sim1, _ = sfl_select(env, agent, levels, lc, np.random.default_rng(2))
    assert levels.sfl_ghost_agent is not agent
    _, sim2, _ = sfl_select(env, agent, levels, lc, np.random.default_rng(3))
    assert sim1 == 800 and sim2 == 1600                 # the old policy re-rolls every candidate


@pytest.mark.parametrize("arm", ["sfl_tilt_sc", "sfl_halving", "sfl_ghost", "sfl_spread", "sfl_bisect"])
def test_batch14_arms_train(arm):
    checks, stats, _ = _short(arm)
    assert stats["scouted_steps"] > 0


def test_auto_tries_every_arm_then_sets_replay_prob_and_charges_probe():
    from acl_bench.plr.fast import AUTO_ARMS
    env = _coin_env()
    step = env.step

    def step_with_params(s, a, p):
        env._params = p
        return step(s, a, p)
    env.step = step_with_params
    bounds = np.array([[0.0, 1.0]])
    lc = LevelConfig(sfl=True, sfl_n=50, sfl_k=4, sfl_top=5, sfl_auto=20)
    levels = LevelSampler(lc, bounds, np.random.default_rng(1))
    agent = make_agent(env, bounds, FastConfig())
    for i in range(len(AUTO_ARMS) + 2):
        _, sim, _ = sfl_select(env, agent, levels, lc, np.random.default_rng(i))
        assert sim == (50 + 20) * 4                     # scouting + robustness probe
        assert levels.sfl_rp == AUTO_ARMS[levels.auto["arm"]]
    assert levels.auto["trace"][:len(AUTO_ARMS)] == list(range(len(AUTO_ARMS)))
    assert (levels.auto["n"] >= 1).all()


def test_auto_replay_prob_drives_pick():
    lc = LevelConfig(sfl=True, replay_prob=0.5)
    s = LevelSampler(lc, np.array([[0.0, 1.0]]), np.random.default_rng(0))
    s.set_sfl(np.array([[0.3]]))
    for rp, want in [(0.0, 0.0), (0.75, 0.75)]:
        s.sfl_rp = rp
        kinds = [s.pick()[2] for _ in range(4000)]
        assert abs(np.mean([k == REPLAY for k in kinds]) - want) < 0.03


@pytest.mark.parametrize("arm", ["sfl_spread_carry", "sfl_spread_verify", "sfl_spread0", "sfl_auto"])
def test_batch15_arms_train(arm):
    checks, stats, _ = _short(arm)
    assert stats["scouted_steps"] > 0


def test_memory_pools_a_carried_levels_earlier_rollouts():
    env = _coin_env()
    bounds = np.array([[0.0, 1.0]])
    step = env.step

    def step_with_params(s, a, p):
        env._params = p
        return step(s, a, p)
    env.step = step_with_params
    kept = {}
    for memory in (0.0, 1.0):
        lc = LevelConfig(sfl=True, sfl_n=100, sfl_k=4, sfl_top=10, sfl_tilt=1.0, sfl_carry=True,
                         sfl_memory=memory)
        levels = LevelSampler(lc, bounds, np.random.default_rng(1))
        levels.set_sfl(np.full((10, 1), 1 / 3))       # ideal levels, known from 1000 earlier rollouts
        levels.sfl_counts = (np.full(10, 1000 / 3), np.full(10, 1000.0))
        top, _, _ = sfl_select(env, make_agent(env, bounds, FastConfig()), levels, lc, np.random.default_rng(2))
        kept[memory] = np.sum(np.isclose(top[:, 0], 1 / 3))
    assert kept[1.0] == 10 > kept[0.0]                 # the evidence outweighs 4 noisy rollouts
    assert np.allclose(levels.sfl_counts[1], 1004.0)   # and it accumulates


@pytest.mark.parametrize("arm", ["sfl_carry_mem", "sfl_carry0", "sfl_spread_carry0", "sfl_carry_mem25", "sfl_spread2_carry", "sfl_spread_carry_mem"])
def test_batch16_arms_train(arm):
    _, stats, _ = _short(arm)
    assert stats["scouted_steps"] > 0


def test_start_state_sfl_selects_and_replays_archived_states():
    """Level-blind one-step env: an episode from state (q, .) passes with probability q, and
    the level's own starts have q = 0 (never pass, unlearnable). Start-state SFL must keep
    archived states, near the tilted optimum q = 1/3, and replay from them."""
    env = types.ModuleType("states")
    env.PARAM_BOUNDS, env.PARAM_ORDER = {"x": (0.0, 1.0)}, ["x"]
    env.OBS_DIM, env.ACTION_DIM, env.MAX_EPISODE_STEPS, env.GOAL = 2, 2, 1, "reach"
    rng = np.random.default_rng(0)
    env.sample_starts = lambda p, n, r: np.zeros((n, 2))
    env.observe = lambda s: s
    env.step = lambda s, a, p: np.column_stack([s[:, 0], rng.uniform(size=len(s))])
    env.terminated = lambda s: s[:, 1] < s[:, 0]
    bounds = np.array([[0.0, 1.0]])
    lc = LevelConfig(sfl=True, sfl_n=400, sfl_k=200, sfl_top=20, sfl_tilt=1.0, sfl_states=0.5)
    levels = LevelSampler(lc, bounds, np.random.default_rng(1))
    q = np.linspace(0, 1, 50)
    _archive(levels, lc, np.full((50, 1), 0.5), np.column_stack([q, q]), np.ones(50, bool))
    top, _, _ = sfl_select(env, make_agent(env, bounds, FastConfig()), levels, lc, np.random.default_rng(2))
    st, has = levels.sfl_top_starts
    assert has.all() and abs(st[:, 0].mean() - 1 / 3) < 0.1
    levels.set_sfl(top, None, levels.sfl_top_starts)
    levels.sfl_rp = 1.0
    _, _, m = levels.pick()
    assert m == REPLAY and levels.next_start is not None and levels.next_start[0] in st[:, 0]


@pytest.mark.parametrize("arm", ["sfl_states", "sfl_states_carry"])
def test_start_state_arms_train(arm):
    _, stats, _ = _short(arm)
    assert stats["scouted_steps"] > 0
