"""LunarLander: a pool of Gymnasium Box2D envs behind the batched env contract."""
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


def test_play_is_reproducible_from_its_start():
    torch.manual_seed(0)
    agent = Agent(lunarlander.OBS_DIM, lunarlander.ACTION_DIM)
    p = np.array([[-10.0, 15.0, 1.5]])
    s0 = lunarlander.sample_starts(p[0], 1, np.random.default_rng(3))
    a = grade(lunarlander, agent, p, s0, extras={})
    b = grade(lunarlander, agent, p, s0, extras={})
    assert a[0].tolist() == b[0].tolist() and a[1].tolist() == b[1].tolist()
