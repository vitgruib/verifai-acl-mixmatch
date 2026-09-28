"""Batched evaluator: steps every (task, start) question in lockstep with one batched
policy forward pass, instead of one Python env step per question.

The physics is each environment's batched transcription of its gymnasium env
(acl_bench.envs), checked against the real env in tests/test_grader.py. The policy is
deterministic (argmax action). A question's `steps` is the step at which the episode
terminated, or `max_steps` if it never did. Passing depends on the goal: a "survive"
question passes if `steps == max_steps` (CartPole: the pole is still up at the cap), a
"reach" question if it terminated at all (Acrobot, MountainCar: the goal was reached), a
"balance" question if the state is upright for each of the last `env.HOLD_STEPS` steps
(Pendulum, which never terminates). An environment with a batched `reward` also gets
each question's return (Pendulum's continuous score).
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def rollout(env, agent, params: np.ndarray, s0: np.ndarray, max_steps: int | None = None,
            extras: dict | None = None):
    """(steps, ended) per question under the deterministic (argmax) policy. If `extras`
    is a dict it receives `return` (for an env with `reward`) and `streak` (consecutive
    upright steps at the end, for a "balance" goal)."""
    max_steps = max_steps or env.MAX_EPISODE_STEPS
    n = len(params)
    state = np.array(s0, dtype=np.float64)
    steps = np.full(n, max_steps, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    has_reward, balance = hasattr(env, "reward"), env.GOAL == "balance"
    ret, streak = np.zeros(n), np.zeros(n, dtype=np.int64)
    for t in range(1, max_steps + 1):
        obs = torch.as_tensor(env.observe(state).astype(np.float32))
        action = agent.actor(obs).argmax(dim=1).numpy()
        if has_reward:
            ret += np.where(alive, env.reward(state, action, params), 0.0)
        state = np.where(alive[:, None], env.step(state, action, params), state)
        if balance:
            streak = np.where(alive, np.where(env.upright(state), streak + 1, 0), streak)
        ended = alive & env.terminated(state)
        steps[ended] = t
        alive &= ~ended
        if not alive.any():
            break
    if extras is not None:
        if has_reward:
            extras["return"] = ret
        if balance:
            extras["streak"] = streak
    return steps, ~alive


def passed(env, steps: np.ndarray, ended: np.ndarray, max_steps: int | None = None,
           streak: np.ndarray | None = None) -> np.ndarray:
    if env.GOAL == "balance":
        return streak >= env.HOLD_STEPS
    return ended if env.GOAL == "reach" else steps == (max_steps or env.MAX_EPISODE_STEPS)


def grade(env, agent, params: np.ndarray, s0: np.ndarray, extras: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(success, steps) per question; `extras` as in `rollout`."""
    extras = {} if extras is None else extras
    steps, ended = rollout(env, agent, params, s0, extras=extras)
    return passed(env, steps, ended, streak=extras.get("streak")), steps
