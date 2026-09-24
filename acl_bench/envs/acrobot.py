"""Acrobot with its physical parameters exposed as a task that Scenic/VerifAI samples.

A task is five scalars (`PARAM_ORDER`), declared again as `VerifaiRange`s in
`acrobot.scenic`. The first link is a uniform rod (its centre of mass at half its
length); the second link and both moments of inertia keep gymnasium's values, and so
does the goal line (tip height -cos(t1) - cos(t1 + t2) > 1, in gymnasium's unit link
lengths). The goal is to reach: a question is passed by reaching the line within
`MAX_EPISODE_STEPS` (gymnasium's registered limit).

The batched physics (`step`, `terminated`) is transcribed from gymnasium's
`AcrobotEnv.step` ("book" dynamics, one RK4 step of 0.2 s, angle wrapping, velocity
bounds); tests/test_grader.py checks it against `ParamAcrobotEnv`.
"""
from __future__ import annotations

import os

import numpy as np
from gymnasium.envs.classic_control.acrobot import AcrobotEnv

PARAM_BOUNDS = {
    "link_length_1": (0.5, 1.5),  # first link length (m); its centre of mass is at half of it
    "link_mass_1": (0.5, 3.0),    # first link mass (kg)
    "link_mass_2": (0.5, 3.0),    # second link mass (kg)
    "torque": (0.25, 2.0),        # magnitude of the joint torque (N m)
    "init_range": (0.05, 0.3),    # |initial state| ~ U(-init_range, init_range)
}
PARAM_ORDER = tuple(PARAM_BOUNDS)
MAX_EPISODE_STEPS = 500
OBS_DIM, ACTION_DIM = 6, 3
GOAL = "reach"
SCENIC_FILE = os.path.join(os.path.dirname(__file__), "acrobot.scenic")

# Constants, as in gymnasium's AcrobotEnv.
G = 9.8
DT = 0.2
LC2 = AcrobotEnv.LINK_COM_POS_2
MOI = AcrobotEnv.LINK_MOI
MAX_VEL_1, MAX_VEL_2 = AcrobotEnv.MAX_VEL_1, AcrobotEnv.MAX_VEL_2
_L1, _M1, _M2, _TORQUE, _INIT = range(5)


class ParamAcrobotEnv(AcrobotEnv):
    def set_task(self, params: dict) -> None:
        self.LINK_LENGTH_1 = float(params["link_length_1"])
        self.LINK_COM_POS_1 = self.LINK_LENGTH_1 / 2
        self.LINK_MASS_1 = float(params["link_mass_1"])
        self.LINK_MASS_2 = float(params["link_mass_2"])
        torque = float(params["torque"])
        self.AVAIL_TORQUE = [-torque, 0.0, torque]
        self._init_range = float(params["init_range"])

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("low", -self._init_range)
        options.setdefault("high", self._init_range)
        return super().reset(seed=seed, options=options)


def make_env(params: dict) -> ParamAcrobotEnv:
    env = ParamAcrobotEnv()
    env.set_task(params)
    return env


# ---------------------------------------------------------------- batched physics
def sample_starts(task: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-task[_INIT], task[_INIT], size=(n, 4))


def _dsdt(s: np.ndarray, a: np.ndarray, params: np.ndarray) -> np.ndarray:
    """AcrobotEnv._dsdt ("book"), term by term, for N states."""
    m1, m2, l1 = params[:, _M1], params[:, _M2], params[:, _L1]
    lc1, lc2, I1, I2 = l1 / 2, LC2, MOI, MOI
    theta1, theta2, dtheta1, dtheta2 = s.T
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
    phi2 = m2 * lc2 * G * np.cos(theta1 + theta2 - np.pi / 2.0)
    phi1 = (-m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
            + (m1 * lc1 + m2 * l1) * G * np.cos(theta1 - np.pi / 2) + phi2)
    ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2) - phi2) / (
        m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return np.stack([dtheta1, dtheta2, ddtheta1, ddtheta2], axis=1)


def _wrap(x: np.ndarray, m: float, M: float) -> np.ndarray:
    diff = M - m
    for _ in range(2):                       # one RK4 step turns less than 2 * pi * 2
        x = np.where(x > M, x - diff, x)
        x = np.where(x < m, x + diff, x)
    return x


def step(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Action 0 / 1 / 2 applies -torque / 0 / +torque. One RK4 step as in gymnasium's rk4."""
    a = (np.asarray(action) - 1) * params[:, _TORQUE]
    dt2 = DT / 2.0
    k1 = _dsdt(state, a, params)
    k2 = _dsdt(state + dt2 * k1, a, params)
    k3 = _dsdt(state + dt2 * k2, a, params)
    k4 = _dsdt(state + DT * k3, a, params)
    ns = state + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.stack([_wrap(ns[:, 0], -np.pi, np.pi), _wrap(ns[:, 1], -np.pi, np.pi),
                     np.clip(ns[:, 2], -MAX_VEL_1, MAX_VEL_1), np.clip(ns[:, 3], -MAX_VEL_2, MAX_VEL_2)], axis=1)


def observe(state: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(state[:, 0]), np.sin(state[:, 0]), np.cos(state[:, 1]), np.sin(state[:, 1]),
                     state[:, 2], state[:, 3]], axis=1)


def terminated(state: np.ndarray) -> np.ndarray:
    return -np.cos(state[:, 0]) - np.cos(state[:, 1] + state[:, 0]) > 1.0


# ---------------------------------------------------------------- planning heuristic
def planning_cost(params: np.ndarray):
    """Returns cost(states (N, K, 4)) -> (N, K), lower = better: minus the total
    mechanical energy, the classic heuristic for pumping an acrobot up (Spong, 1995)."""
    m1, m2, l1 = (params[:, i][:, None] for i in (_M1, _M2, _L1))
    lc1 = l1 / 2

    def cost(s):
        t1, t2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        d11 = m1 * lc1**2 + m2 * (l1**2 + LC2**2 + 2 * l1 * LC2 * np.cos(t2)) + 2 * MOI
        d12 = m2 * (LC2**2 + l1 * LC2 * np.cos(t2)) + MOI
        d22 = m2 * LC2**2 + MOI
        kinetic = 0.5 * (d11 * w1**2 + 2 * d12 * w1 * w2 + d22 * w2**2)
        potential = -G * ((m1 * lc1 + m2 * l1) * np.cos(t1) + m2 * LC2 * np.cos(t1 + t2))
        return -(kinetic + potential)
    return cost
