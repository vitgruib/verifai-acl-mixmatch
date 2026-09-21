"""Batched CartPole evaluator: steps every (task, start) pair in lockstep with
one batched policy forward pass, instead of one Python env step per pair.

Physics is transcribed from gymnasium's `CartPoleEnv.step` (Euler integrator,
same termination test) with per-pair parameters, and is checked against
`ParamCartPoleEnv` in tests/test_evaluator.py. The policy is deterministic
(argmax action). A pair's `steps` is the step index at which the episode
terminated, or `max_steps` if it never did; success is `steps == max_steps`.
"""
from __future__ import annotations

import numpy as np
import torch

# CartPole's physics constants, as in gymnasium's CartPoleEnv.
G = 9.8
TAU = 0.02
THETA_THRESHOLD = 12 * 2 * np.pi / 360   # failure angle (0.2095 rad)
X_THRESHOLD = 2.4

PARAM_ORDER = ("length", "masspole", "masscart", "force_mag", "init_range")
_LENGTH, _MASSPOLE, _MASSCART, _FORCE = 0, 1, 2, 3


def step_physics(state: np.ndarray, force: np.ndarray, params: np.ndarray) -> np.ndarray:
    """One Euler step for N pairs. `state` is (N, 4) float64, `force` (N,) signed,
    `params` (N, >=4) in PARAM_ORDER. Same arithmetic as CartPoleEnv.step."""
    length, masspole = params[:, _LENGTH], params[:, _MASSPOLE]
    total_mass = masspole + params[:, _MASSCART]
    polemass_length = masspole * length
    x, x_dot, theta, theta_dot = state.T
    costheta, sintheta = np.cos(theta), np.sin(theta)
    temp = (force + polemass_length * np.square(theta_dot) * sintheta) / total_mass
    thetaacc = (G * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * np.square(costheta) / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    return np.stack([x + TAU * x_dot, x_dot + TAU * xacc,
                     theta + TAU * theta_dot, theta_dot + TAU * thetaacc], axis=1)


def terminated(state: np.ndarray) -> np.ndarray:
    return (state[:, 0] < -X_THRESHOLD) | (state[:, 0] > X_THRESHOLD) | \
           (state[:, 2] < -THETA_THRESHOLD) | (state[:, 2] > THETA_THRESHOLD)


@torch.no_grad()
def rollout_steps(agent, params: np.ndarray, s0: np.ndarray, max_steps: int = 500) -> np.ndarray:
    """Steps each pair survives under the deterministic (argmax) policy."""
    assert agent.action_type == "discrete", "batched evaluator supports discrete-action agents"
    n = len(params)
    state = np.array(s0, dtype=np.float64)
    steps = np.full(n, max_steps, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    for t in range(1, max_steps + 1):
        obs = torch.as_tensor(state.astype(np.float32))
        force = np.where(agent.actor(obs).argmax(dim=1).numpy() == 1,
                         params[:, _FORCE], -params[:, _FORCE])
        state = np.where(alive[:, None], step_physics(state, force, params), state)
        died = alive & terminated(state)
        steps[died] = t
        alive &= ~died
        if not alive.any():
            break
    return steps
