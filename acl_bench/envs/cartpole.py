"""CartPole with its physical parameters exposed as a task that Scenic/VerifAI samples.

A task is five scalars (`PARAM_ORDER`); `cartpole.scenic` declares the same box as
`VerifaiRange` parameters so the samplers in `acl_bench.sampling` can draw from it.
Episodes are capped at `MAX_EPISODE_STEPS` (the raw gymnasium class has no cap).
The goal is to survive: a question is passed by not terminating (docs/exam.md).

The batched physics (`step`, `terminated`) is transcribed from gymnasium's
`CartPoleEnv.step` (Euler integrator, same termination test) with per-question
parameters; tests/test_grader.py checks it against `ParamCartPoleEnv`.
"""
from __future__ import annotations

import os

import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from scipy.linalg import solve_discrete_are

PARAM_BOUNDS = {
    "length": (0.25, 1.5),       # half-length of the pole (m)
    "masspole": (0.05, 0.5),     # pole mass (kg)
    "masscart": (0.5, 2.0),      # cart mass (kg)
    "force_mag": (4.0, 16.0),    # magnitude of the applied push (N)
    "init_range": (0.05, 0.3),   # |initial state| ~ U(-init_range, init_range)
}
PARAM_ORDER = tuple(PARAM_BOUNDS)
MAX_EPISODE_STEPS = 500
OBS_DIM, ACTION_DIM = 4, 2
GOAL = "survive"
SCENIC_FILE = os.path.join(os.path.dirname(__file__), "cartpole.scenic")

# Physics constants, as in gymnasium's CartPoleEnv.
G = 9.8
TAU = 0.02
THETA_THRESHOLD = 12 * 2 * np.pi / 360   # failure angle (0.2095 rad)
X_THRESHOLD = 2.4
_LENGTH, _MASSPOLE, _MASSCART, _FORCE, _INIT = range(5)


class ParamCartPoleEnv(CartPoleEnv):
    def set_task(self, params: dict) -> None:
        self.length = float(params["length"])
        self.masspole = float(params["masspole"])
        self.masscart = float(params["masscart"])
        self.force_mag = float(params["force_mag"])
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self._init_range = float(params["init_range"])

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("low", -self._init_range)
        options.setdefault("high", self._init_range)
        return super().reset(seed=seed, options=options)


def make_env(params: dict) -> ParamCartPoleEnv:
    env = ParamCartPoleEnv()
    env.set_task(params)
    return env


# ---------------------------------------------------------------- batched physics
def sample_starts(task: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` starting states for one task (a row in PARAM_ORDER), drawn as reset() draws them."""
    return rng.uniform(-task[_INIT], task[_INIT], size=(n, 4))


def step_physics(state: np.ndarray, force: np.ndarray, params: np.ndarray) -> np.ndarray:
    """One Euler step for N questions. `state` is (N, 4) float64, `force` (N,) signed,
    `params` (N, 5) in PARAM_ORDER. Same arithmetic as CartPoleEnv.step."""
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


def step(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Action 1 pushes right, 0 pushes left."""
    return step_physics(state, np.where(action == 1, params[:, _FORCE], -params[:, _FORCE]), params)


def observe(state: np.ndarray) -> np.ndarray:
    return state


def terminated(state: np.ndarray) -> np.ndarray:
    return (state[:, 0] < -X_THRESHOLD) | (state[:, 0] > X_THRESHOLD) | \
           (state[:, 2] < -THETA_THRESHOLD) | (state[:, 2] > THETA_THRESHOLD)


# ---------------------------------------------------------------- planning heuristic
def lqr_cost_matrices(params: np.ndarray) -> np.ndarray:
    """(N, 4, 4) discrete-time LQR cost-to-go P for each question's linearization about
    upright, with the Euler step the grader uses. Only a ranking heuristic."""
    Q = np.diag([1 / X_THRESHOLD ** 2, 0.1, 1 / THETA_THRESHOLD ** 2, 0.1])
    out = np.empty((len(params), 4, 4))
    for i, p in enumerate(params):
        length, mp, mc, f = p[_LENGTH], p[_MASSPOLE], p[_MASSCART], p[_FORCE]
        M = mp + mc
        D = length * (4.0 / 3.0 - mp / M)
        Ac = np.array([[0, 1, 0, 0], [0, 0, -mp * length * G / (M * D), 0], [0, 0, 0, 1], [0, 0, G / D, 0]])
        Bc = np.array([[0], [1 / M + mp * length / (M * M * D)], [0], [-1 / (M * D)]])
        out[i] = solve_discrete_are(np.eye(4) + TAU * Ac, TAU * Bc, Q, np.array([[1 / f ** 2]]))
    return out


def planning_cost(params: np.ndarray):
    """Returns cost(states (N, K, 4)) -> (N, K), lower = better: the LQR cost-to-go
    (the standard value-function heuristic for underactuated balancing, cf. LQR trees)."""
    P = lqr_cost_matrices(params)
    return lambda states: np.einsum("nci,nij,ncj->nc", states, P, states)
