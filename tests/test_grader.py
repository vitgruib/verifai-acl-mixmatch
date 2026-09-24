"""Each environment's batched exam physics must agree with its real gymnasium env."""
import numpy as np
import pytest
import torch

from acl_bench import envs
from acl_bench.exam.grader import grade, rollout
from acl_bench.ppo import Agent
from acl_bench.study.arms import ARMS
from acl_bench.study.run import train_arm


def random_pairs(env, n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = (np.array([env.PARAM_BOUNDS[k][i] for k in env.PARAM_ORDER]) for i in (0, 1))
    params = rng.uniform(lo, hi, size=(n, len(lo)))
    return params, np.concatenate([env.sample_starts(p, 1, rng) for p in params])


def real_env(env, row, s0):
    e = env.make_env(dict(zip(env.PARAM_ORDER, row)))
    e.reset(seed=0)
    e.state = tuple(s0) if env is envs.get("mountaincar") else s0.copy()
    return e


@pytest.fixture(scope="module")
def cartpole_agent():
    return train_arm(envs.get("cartpole"), ARMS["N"], seed=1, steps=60_000)[1]


@pytest.mark.parametrize("name", envs.NAMES)
def test_physics_matches_gym_step_by_step(name):
    env = envs.get(name)
    params, s0 = random_pairs(env, 80, seed=0)
    rng = np.random.default_rng(1)
    real = [real_env(env, p, s) for p, s in zip(params, s0)]
    state = s0.copy()
    done = np.zeros(len(params), dtype=bool)
    compared = 0
    for _ in range(60):
        action = rng.integers(0, env.ACTION_DIM, len(params))
        state = env.step(state, action, params)
        for i, e in enumerate(real):
            if done[i]:
                continue
            _, _, term, _, _ = e.step(int(action[i]))
            np.testing.assert_allclose(state[i], np.asarray(e.state, dtype=np.float64), rtol=0, atol=1e-9)
            compared += 1
            assert bool(env.terminated(state[i:i + 1])[0]) == term
            done[i] = term
    assert compared > 500


def reference_steps(env, agent, row, s0):
    """(steps, ended) playing the real env one question at a time."""
    e = real_env(env, row, s0)
    obs = env.observe(np.asarray(s0, dtype=np.float64)[None])[0]
    for step in range(1, env.MAX_EPISODE_STEPS + 1):
        with torch.no_grad():
            action = int(agent.actor(torch.as_tensor(np.asarray(obs, dtype=np.float32))[None]).argmax(dim=1).item())
        obs, _, term, _, _ = e.step(action)
        if term:
            return step, True
    return env.MAX_EPISODE_STEPS, False


def test_cartpole_rollout_matches_per_question_reference_loop(cartpole_agent):
    env = envs.get("cartpole")
    params, s0 = random_pairs(env, 120, seed=2)
    steps, ended = rollout(env, cartpole_agent, params, s0)
    reference = [reference_steps(env, cartpole_agent, p, s) for p, s in zip(params, s0)]
    np.testing.assert_array_equal(steps, [r[0] for r in reference])
    np.testing.assert_array_equal(ended, [r[1] for r in reference])
    assert 1 < np.median(steps) and steps.max() > 100, "test needs non-trivial trajectories"


@pytest.mark.parametrize("name", ["acrobot", "mountaincar"])
def test_reach_rollout_matches_per_question_reference_loop(name):
    """An untrained policy on a reach task. The physics agrees to rounding error (above),
    and a chaotic system can amplify that over hundreds of steps, so almost all, not all."""
    env = envs.get(name)
    torch.manual_seed(0)
    agent = Agent(env.OBS_DIM, env.ACTION_DIM)
    params, s0 = random_pairs(env, 60, seed=2)
    steps, ended = rollout(env, agent, params, s0)
    reference = np.array([reference_steps(env, agent, p, s) for p, s in zip(params, s0)])
    assert (steps == reference[:, 0]).mean() >= 0.95
    assert (ended == reference[:, 1].astype(bool)).mean() >= 0.95


@pytest.mark.parametrize("name", envs.NAMES)
def test_pass_rule_and_bounds(name):
    env = envs.get(name)
    torch.manual_seed(0)
    agent = Agent(env.OBS_DIM, env.ACTION_DIM)
    params, s0 = random_pairs(env, 50, seed=3)
    steps, ended = rollout(env, agent, params, s0, max_steps=40)
    assert steps.min() >= 1 and steps.max() <= 40
    assert (steps[~ended] == 40).all()
    success, _ = grade(env, agent, params, s0)
    full_steps, full_ended = rollout(env, agent, params, s0)
    expected = full_ended if env.GOAL == "reach" else full_steps == env.MAX_EPISODE_STEPS
    np.testing.assert_array_equal(success, expected)
