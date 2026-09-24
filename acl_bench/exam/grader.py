"""Batched evaluator: steps every (task, start) question in lockstep with one batched
policy forward pass, instead of one Python env step per question.

The physics is each environment's batched transcription of its gymnasium env
(acl_bench.envs), checked against the real env in tests/test_grader.py. The policy is
deterministic (argmax action). A question's `steps` is the step at which the episode
terminated, or `max_steps` if it never did. Passing depends on the goal: a "survive"
question passes if `steps == max_steps` (CartPole: the pole is still up at the cap), a
"reach" question if it terminated at all (Acrobot, MountainCar: the goal was reached).
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def rollout(env, agent, params: np.ndarray, s0: np.ndarray, max_steps: int | None = None):
    """(steps, ended) per question under the deterministic (argmax) policy."""
    max_steps = max_steps or env.MAX_EPISODE_STEPS
    n = len(params)
    state = np.array(s0, dtype=np.float64)
    steps = np.full(n, max_steps, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    for t in range(1, max_steps + 1):
        obs = torch.as_tensor(env.observe(state).astype(np.float32))
        action = agent.actor(obs).argmax(dim=1).numpy()
        state = np.where(alive[:, None], env.step(state, action, params), state)
        ended = alive & env.terminated(state)
        steps[ended] = t
        alive &= ~ended
        if not alive.any():
            break
    return steps, ~alive


def passed(env, steps: np.ndarray, ended: np.ndarray, max_steps: int | None = None) -> np.ndarray:
    return ended if env.GOAL == "reach" else steps == (max_steps or env.MAX_EPISODE_STEPS)


def grade(env, agent, params: np.ndarray, s0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(success, steps) per question."""
    steps, ended = rollout(env, agent, params, s0)
    return passed(env, steps, ended), steps
