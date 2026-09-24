"""MountainCar with its physical parameters exposed as a task that Scenic/VerifAI samples.

A task is three scalars (`PARAM_ORDER`), declared again as `VerifaiRange`s in
`mountaincar.scenic`. The car starts at rest at a position drawn from
-0.5 +- init_range (gymnasium: -0.5 +- 0.1). The goal is to reach: a question is passed
by reaching the flag (position >= 0.5) within `MAX_EPISODE_STEPS` (gymnasium's
registered limit).

The batched physics (`step`, `terminated`) is transcribed from gymnasium's
`MountainCarEnv.step`; tests/test_grader.py checks it against `ParamMountainCarEnv`.
"""
from __future__ import annotations

import os

import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

PARAM_BOUNDS = {
    "force": (0.001, 0.004),      # engine force per step (gymnasium: 0.001)
    "gravity": (0.0015, 0.0035),  # gravity per step (gymnasium: 0.0025)
    "init_range": (0.05, 0.3),    # start position ~ U(-0.5 - init_range, -0.5 + init_range)
}
PARAM_ORDER = tuple(PARAM_BOUNDS)
MAX_EPISODE_STEPS = 200
OBS_DIM, ACTION_DIM = 2, 3
GOAL = "reach"
SCENIC_FILE = os.path.join(os.path.dirname(__file__), "mountaincar.scenic")

MIN_POSITION, MAX_POSITION = -1.2, 0.6
MAX_SPEED = 0.07
GOAL_POSITION, GOAL_VELOCITY = 0.5, 0.0
_FORCE, _GRAVITY, _INIT = range(3)


class ParamMountainCarEnv(MountainCarEnv):
    def set_task(self, params: dict) -> None:
        self.force = float(params["force"])
        self.gravity = float(params["gravity"])
        self._init_range = float(params["init_range"])

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("low", -0.5 - self._init_range)
        options.setdefault("high", -0.5 + self._init_range)
        return super().reset(seed=seed, options=options)


def make_env(params: dict) -> ParamMountainCarEnv:
    env = ParamMountainCarEnv()
    env.set_task(params)
    return env


# ---------------------------------------------------------------- batched physics
def sample_starts(task: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    position = rng.uniform(-0.5 - task[_INIT], -0.5 + task[_INIT], size=n)
    return np.stack([position, np.zeros(n)], axis=1)


def step(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Action 0 / 1 / 2 pushes left / not at all / right. Same arithmetic as MountainCarEnv.step."""
    position, velocity = state[:, 0], state[:, 1]
    velocity = velocity + (np.asarray(action) - 1) * params[:, _FORCE] + np.cos(3 * position) * (-params[:, _GRAVITY])
    velocity = np.clip(velocity, -MAX_SPEED, MAX_SPEED)
    position = np.clip(position + velocity, MIN_POSITION, MAX_POSITION)
    velocity = np.where((position == MIN_POSITION) & (velocity < 0), 0.0, velocity)
    return np.stack([position, velocity], axis=1)


def observe(state: np.ndarray) -> np.ndarray:
    return state


def terminated(state: np.ndarray) -> np.ndarray:
    return (state[:, 0] >= GOAL_POSITION) & (state[:, 1] >= GOAL_VELOCITY)


# ---------------------------------------------------------------- planning heuristic
def planning_cost(params: np.ndarray):
    """Returns cost(states (N, K, 2)) -> (N, K), lower = better: minus the energy
    v^2 / 2 + (gravity / 3) sin(3 x), which the valley conserves when the engine is off."""
    gravity = params[:, _GRAVITY][:, None]
    return lambda s: -(0.5 * s[..., 1] ** 2 + gravity / 3 * np.sin(3 * s[..., 0]))
