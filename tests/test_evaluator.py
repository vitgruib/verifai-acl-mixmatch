"""The batched evaluator must agree with the real ParamCartPoleEnv."""
import numpy as np
import pytest
import torch

from acl_bench.envs.param_cartpole import PARAM_BOUNDS, make_env
from acl_bench.envs.registry import ENV_SPECS
from acl_bench.potential.functions import resolve_potential_fn
from acl_bench.ppo import PPOConfig, run_training
from acl_bench.scenic_sampling import ScenicTaskSampler
from acl_bench.suite.evaluator import PARAM_ORDER, rollout_steps, step_physics, terminated

SPEC = ENV_SPECS["cartpole"]


def random_pairs(n, seed):
    rng = np.random.default_rng(seed)
    params = np.array([[rng.uniform(*PARAM_BOUNDS[k]) for k in PARAM_ORDER] for _ in range(n)])
    s0 = np.array([rng.uniform(-p[4], p[4], 4) for p in params])
    return params, s0


def as_dict(row):
    return dict(zip(PARAM_ORDER, row))


@pytest.fixture(scope="module")
def agent():
    cfg = PPOConfig(total_timesteps=60_000, seed=1, acl=False)
    _, trained = run_training(SPEC, ScenicTaskSampler.load(SPEC.scenic_file, "random"),
                              resolve_potential_fn("neg_return", SPEC.success_return), cfg)
    return trained


def test_physics_matches_gym_step_by_step():
    params, s0 = random_pairs(80, seed=0)
    rng = np.random.default_rng(1)
    envs = []
    for p, s in zip(params, s0):
        env = make_env(as_dict(p)); env.reset(seed=0); env.state = s.copy(); envs.append(env)
    state = s0.copy()
    done = np.zeros(len(params), dtype=bool)
    compared = 0
    for _ in range(60):
        action = rng.integers(0, 2, len(params))
        force = np.where(action == 1, params[:, 3], -params[:, 3])
        state = step_physics(state, force, params)
        for i, env in enumerate(envs):
            if done[i]:
                continue
            _, _, term, _, _ = env.step(int(action[i]))
            np.testing.assert_allclose(state[i], env.state, rtol=0, atol=1e-9)
            compared += 1
            assert bool(terminated(state[i:i + 1])[0]) == term
            done[i] = term
    assert compared > 500


def reference_steps(agent, row, s0, max_steps=500):
    env = make_env(as_dict(row)); env.reset(seed=0); env.state = s0.copy()
    for step in range(1, max_steps + 1):
        obs = torch.as_tensor(np.array(env.state, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            action = int(agent.actor(obs).argmax(dim=1).item())
        _, _, term, _, _ = env.step(action)
        if term:
            return step
    return max_steps


def test_rollout_matches_per_pair_reference_loop(agent):
    params, s0 = random_pairs(120, seed=2)
    batched = rollout_steps(agent, params, s0)
    reference = np.array([reference_steps(agent, p, s) for p, s in zip(params, s0)])
    np.testing.assert_array_equal(batched, reference)
    assert 1 < np.median(batched) and batched.max() > 100, "test needs non-trivial trajectories"


def test_finished_pairs_do_not_change_and_all_pairs_are_bounded(agent):
    params, s0 = random_pairs(50, seed=3)
    steps = rollout_steps(agent, params, s0, max_steps=40)
    assert steps.min() >= 1 and steps.max() <= 40
