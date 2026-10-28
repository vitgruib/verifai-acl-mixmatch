"""LunarLander: a pool of Gymnasium Box2D envs behind the batched env contract."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from acl_bench.envs import lunarlander
from acl_bench.exam.grader import grade
from acl_bench.ppo import Agent


def play_gym(row, seed, actions):
    """Initial obs, then (obs, reward, terminated) per step of a fresh Gymnasium env."""
    env = lunarlander.make_env(dict(zip(lunarlander.PARAM_ORDER, row)))
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for a in actions:
        obs, r, term, _, _ = env.step(int(a))
        out.append((obs, r, term))
        if term:
            break
    return out


def test_params_reach_the_simulator():
    env = lunarlander.make_env({"gravity": -3.0, "wind_power": 12.0, "turbulence_power": 1.0})
    u = env.unwrapped
    assert u.gravity == -3.0 and u.enable_wind and u.wind_power == 12.0


def test_no_wind_when_both_wind_params_are_zero():
    env = lunarlander.make_env({"gravity": -10.0, "wind_power": 0.0, "turbulence_power": 0.0})
    assert not env.unwrapped.enable_wind


def test_gravity_at_the_open_end_of_the_box_is_made_valid():
    env = lunarlander.make_env({"gravity": -12.0, "wind_power": 0.0, "turbulence_power": 0.0})
    assert lunarlander.valid({"gravity": env.unwrapped.gravity, "wind_power": 0.0, "turbulence_power": 0.0})


@pytest.mark.parametrize("params, ok", [
    ({"gravity": -10.0, "wind_power": 0.0, "turbulence_power": 0.0}, True),
    ({"gravity": -12.0, "wind_power": 0.0, "turbulence_power": 0.0}, False),
    ({"gravity": 0.0, "wind_power": 0.0, "turbulence_power": 0.0}, False),
    ({"gravity": -5.0, "wind_power": -1.0, "turbulence_power": 0.0}, False),
    ({"gravity": -5.0, "wind_power": 0.0, "turbulence_power": -0.1}, False),
])
def test_valid(params, ok):
    assert lunarlander.valid(params) is ok


def test_batched_step_and_reward_match_a_fresh_gym_env():
    rows = np.array([[-10.0, 0.0, 0.0], [-4.0, 15.0, 1.5], [-11.0, 3.0, 0.5]])
    state = np.concatenate([lunarlander.sample_starts(r, 1, np.random.default_rng(i)) for i, r in enumerate(rows)])
    plans = np.random.default_rng(0).integers(0, 4, size=(len(rows), 40))
    gym = [play_gym(r, int(sd), p) for r, sd, p in zip(rows, state[:, 1], plans)]
    for t in range(40):
        live = [i for i in range(len(rows)) if len(gym[i]) > t + 1]
        for i in live:
            expect = gym[i][0] if t == 0 else gym[i][t][0]
            np.testing.assert_allclose(lunarlander.observe(state[i:i + 1])[0], expect, atol=1e-5)
        r = lunarlander.reward(state, plans[:, t], rows)
        state = lunarlander.step(state, plans[:, t], rows)
        for i in live:
            _, r_gym, term_gym = gym[i][t + 1]
            assert r[i] == pytest.approx(r_gym)
            assert bool(lunarlander.terminated(state[i:i + 1])[0]) == term_gym
        if not live:
            break
    assert (state[:, 2] > 0).all()


def graded(s0, p):
    torch.manual_seed(0)
    agent = Agent(lunarlander.OBS_DIM, lunarlander.ACTION_DIM)
    extras = {}
    success, steps = grade(lunarlander, agent, p, s0, extras=extras)
    return success.tolist(), steps.tolist(), extras["return"].tolist()


def test_play_is_reproducible_from_its_start():
    p = np.array([[-10.0, 15.0, 1.5], [-3.0, 0.0, 0.0]])
    s0 = np.concatenate([lunarlander.sample_starts(r, 1, np.random.default_rng(3)) for r in p])
    first = graded(s0, p)
    assert graded(s0, p) == first
    assert min(first[2]) < 0                      # returns are real numbers, not all zero


def test_play_is_reproducible_in_another_process(tmp_path):
    p = np.array([[-10.0, 15.0, 1.5], [-3.0, 0.0, 0.0]])
    s0 = np.concatenate([lunarlander.sample_starts(r, 1, np.random.default_rng(3)) for r in p])
    np.savez(tmp_path / "q.npz", s0=s0, p=p)
    script = (
        "import json, sys, numpy as np\n"
        "from tests.test_lunarlander import graded\n"
        "d = np.load(sys.argv[1]); print(json.dumps(graded(d['s0'], d['p'])))\n")
    out = subprocess.run([sys.executable, "-c", script, str(tmp_path / "q.npz")], check=True,
                         capture_output=True, text=True, cwd=Path(__file__).parent.parent).stdout
    assert json.loads(out.splitlines()[-1]) == list(graded(s0, p))


def test_two_rows_replaying_one_start_do_not_share_an_env():
    row = np.array([[-8.0, 5.0, 0.5]])
    s0 = lunarlander.sample_starts(row[0], 1, np.random.default_rng(1))
    both, params = np.concatenate([s0, s0]), np.concatenate([row, row])
    state = both
    for _ in range(5):
        state = lunarlander.step(state, np.array([2, 0]), params)
    solo = lunarlander.sample_starts(row[0], 1, np.random.default_rng(1))
    solo = np.concatenate([solo, solo])
    for _ in range(5):
        solo = lunarlander.step(solo, np.array([2, 2]), params)
    assert state[0, 3] == solo[0, 3] and state[0, 3] != state[1, 3]


def starts(n, seed=0):
    row = np.array([-10.0, 0.0, 0.0])
    return lunarlander.sample_starts(row, n, np.random.default_rng(seed)), np.tile(row, (n, 1))


def test_a_running_batch_is_never_swept_as_idle(monkeypatch):
    monkeypatch.setattr(lunarlander, "IDLE_STEPS", 2)
    monkeypatch.setattr(lunarlander, "SWEEP_EVERY", 1)
    state, params = starts(6)
    for _ in range(3):
        state = lunarlander.step(state, np.full(6, 2), params)
    assert state[:, 2].tolist() == [3] * 6


def test_envs_that_hit_the_time_limit_are_released(monkeypatch):
    monkeypatch.setattr(lunarlander, "MAX_EPISODE_STEPS", 3)
    before = len(lunarlander._LIVE)
    state, params = starts(4)
    for _ in range(3):
        state = lunarlander.step(state, np.full(4, 2), params)
    assert len(lunarlander._LIVE) == before


def test_envs_nobody_steps_any_more_are_freed(monkeypatch):
    monkeypatch.setattr(lunarlander, "IDLE_STEPS", 5)
    monkeypatch.setattr(lunarlander, "SWEEP_EVERY", 1)
    abandoned, params = starts(3)
    abandoned = lunarlander.step(abandoned, np.full(3, 2), params)
    running, params = starts(2, seed=1)
    for _ in range(8):
        running = lunarlander.step(running, np.full(2, 2), params)
    assert not set(abandoned[:, 0].astype(int)) & set(lunarlander._LIVE)
    assert set(running[:, 0].astype(int)) <= set(lunarlander._LIVE)


def test_stepping_an_old_state_again_is_rejected():
    state, params = starts(1)
    one = lunarlander.step(state, np.array([2]), params)
    lunarlander.step(one, np.array([2]), params)
    with pytest.raises(RuntimeError, match="restored"):
        lunarlander.step(one, np.array([2]), params)


def test_observing_an_unknown_live_row_is_a_clear_error():
    state, _ = starts(1)
    state[0, 0], state[0, 2] = 10**9, 4
    with pytest.raises(RuntimeError, match="restored"):
        lunarlander.observe(state)


def test_rows_paused_for_several_full_passes_are_still_there():
    state, params = starts(2)
    for _ in range(3):
        state = lunarlander.step(state, np.full(2, 2), params)
    other = np.zeros((1, 8))
    other[0, 0] = -1                                  # finished: stepping it only advances the clock
    for _ in range(5 * lunarlander.MAX_EPISODE_STEPS):
        lunarlander.step(other, np.zeros(1, dtype=int), params[:1])
    state = lunarlander.step(state, np.full(2, 2), params)
    assert state[:, 2].tolist() == [4, 4]
