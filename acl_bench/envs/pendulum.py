"""Pendulum with its physical parameters exposed as a task that Scenic/VerifAI samples:
the continuous-reward environment of the four.

A task is five scalars (`PARAM_ORDER`), declared again as `VerifaiRange`s in
`pendulum.scenic`. The pendulum starts at an angle drawn from +-init_range (0 = upright;
gymnasium: +-pi, a full swing-up) and an angular speed from +-1. Every step pays
gymnasium's cost, theta^2 + 0.1 thetadot^2 + 0.001 u^2, as a negative reward, and the
episode never terminates: it runs `MAX_EPISODE_STEPS` (gymnasium's registered limit).
Actions are `ACTION_DIM` evenly spaced torques in [-max_torque, max_torque], so the
discrete-action PPO applies unchanged.

The goal is to balance: a question is passed when the pendulum is within `UPRIGHT` rad of
upright for each of the last `HOLD_STEPS` steps. The pass rate is the thresholded view;
the return (`mean_return`) is the continuous one.

The batched physics (`step`, `reward`) is transcribed from gymnasium's
`PendulumEnv.step`; tests/test_grader.py checks it against `ParamPendulumEnv`.
"""
from __future__ import annotations

import os

import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv

PARAM_BOUNDS = {
    "mass": (0.5, 2.0),            # kg (gymnasium: 1)
    "length": (0.5, 1.5),          # m (gymnasium: 1)
    "gravity": (5.0, 15.0),        # m/s^2 (gymnasium: 10)
    "max_torque": (0.5, 3.0),      # N m (gymnasium: 2)
    "init_range": (0.3, np.pi),    # |initial angle| ~ U(-init_range, init_range) (gymnasium: pi)
}
PARAM_ORDER = tuple(PARAM_BOUNDS)
MAX_EPISODE_STEPS = 200
OBS_DIM, ACTION_DIM = 3, 5
GOAL = "balance"
UPRIGHT, HOLD_STEPS = 0.3, 50
SCENIC_FILE = os.path.join(os.path.dirname(__file__), "pendulum.scenic")

DT, MAX_SPEED = 0.05, 8.0                     # as in gymnasium's PendulumEnv
_M, _L, _G, _TORQUE, _INIT = range(5)


def torque(action, max_torque):
    """Action i of ACTION_DIM -> torque, evenly spaced in [-max_torque, max_torque]."""
    return (np.asarray(action) / (ACTION_DIM - 1) * 2 - 1) * max_torque


class ParamPendulumEnv(PendulumEnv):
    def set_task(self, params: dict) -> None:
        self.m = float(params["mass"])
        self.l = float(params["length"])
        self.g = float(params["gravity"])
        self.max_torque = float(params["max_torque"])
        self._init_range = float(params["init_range"])

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("x_init", self._init_range)
        options.setdefault("y_init", 1.0)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        return super().step(np.array([torque(int(action), self.max_torque)], dtype=np.float64))


def make_env(params: dict) -> ParamPendulumEnv:
    env = ParamPendulumEnv()
    env.set_task(params)
    return env


# ---------------------------------------------------------------- batched physics
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def sample_starts(task: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform([-task[_INIT], -1.0], [task[_INIT], 1.0], size=(n, 2))


def step(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    th, thdot = state[:, 0], state[:, 1]
    m, l, g = params[:, _M], params[:, _L], params[:, _G]
    u = torque(action, params[:, _TORQUE])
    newthdot = np.clip(thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * DT, -MAX_SPEED, MAX_SPEED)
    return np.stack([th + newthdot * DT, newthdot], axis=1)


def reward(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """The reward of taking `action` in `state` (gymnasium charges the pre-step state)."""
    u = torque(action, params[:, _TORQUE])
    return -(angle_normalize(state[:, 0]) ** 2 + 0.1 * state[:, 1] ** 2 + 0.001 * u**2)


def observe(state: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(state[:, 0]), np.sin(state[:, 0]), state[:, 1]], axis=1)


def terminated(state: np.ndarray) -> np.ndarray:
    return np.zeros(len(state), dtype=bool)


def upright(state: np.ndarray) -> np.ndarray:
    return np.abs(angle_normalize(state[:, 0])) < UPRIGHT


# ---------------------------------------------------------------- planning heuristic
def planning_cost(params: np.ndarray):
    """Returns cost(states (N, K, 2)) -> (N, K), lower = better: energy shaping for the
    swing-up (Astrom and Furuta, 2000). The squared gap between the pendulum's energy and
    the energy of resting upright (a uniform rod: inertia m l^2 / 3, centre of mass at l/2),
    relative to that energy, plus a small angle term that settles it at the top."""
    m, l, g = (params[:, i][:, None] for i in (_M, _L, _G))
    e_top = m * g * l / 2

    def cost(s):
        energy = m * l**2 / 6 * s[..., 1] ** 2 + e_top * np.cos(s[..., 0])
        return ((energy - e_top) / e_top) ** 2 + 0.1 * angle_normalize(s[..., 0]) ** 2
    return cost
