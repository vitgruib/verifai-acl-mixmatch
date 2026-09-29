"""Point-mass navigation through a gap: the continuous-reward navigation task (in the
spirit of SFL's JaxNav, reduced to a batched 2D point mass).

The arena is [-1, 1]^2. A wall along x = 0 spans the arena except for a gap. The agent
starts on the left (x = -0.8, height drawn per episode within +-0.8) at rest; the goal is
on the right at (0.8, goal_y). A task is five scalars (`PARAM_ORDER`), declared again in
`pointnav.scenic`: where the gap is and how wide, a constant crosswind along y, the
engine's force, and the goal's height. The agent sees its position and velocity, the
gap and the goal; the wind and its engine force are hidden, like the other
environments' physical parameters.

Actions are 9 discrete thrusts: none or one of 8 unit directions, times `force`.
Dynamics per step (dt = 0.1, unit mass): v += (thrust + wind - 0.5 v) dt, speed capped at
1; x += v dt, clipped to the arena. Crossing x = 0 outside the gap is blocked: the agent
stays on its side and its x-velocity is zeroed.

Reward is shaped and continuous: each step pays 10 x the progress made along the
shortest route to the goal (through the gap while on the left); arriving (within
`GOAL_RADIUS`) pays 10 more and ends the episode. The goal is to reach: a question is
passed by arriving within `MAX_EPISODE_STEPS`; the return is the continuous score.

The geometry lives in the state, so observation and termination need no parameters: a
state row is [x, y, vx, vy, gap_y, gap_width, goal_y].
"""
from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

PARAM_BOUNDS = {
    "gap_y": (-0.8, 0.8),        # centre of the gap in the wall at x = 0
    "gap_width": (0.08, 0.8),    # full width of the gap
    "wind": (-0.6, 0.6),         # constant force along y (hidden)
    "force": (0.4, 1.5),         # engine thrust (hidden)
    "goal_y": (-0.8, 0.8),       # the goal is at (0.8, goal_y)
}
PARAM_ORDER = tuple(PARAM_BOUNDS)
MAX_EPISODE_STEPS = 200
OBS_DIM, ACTION_DIM = 8, 9
GOAL = "reach"
GOAL_RADIUS = 0.1
SCENIC_FILE = os.path.join(os.path.dirname(__file__), "pointnav.scenic")

DT, DRAG, VMAX, START_X, GOAL_X = 0.1, 0.5, 1.0, -0.8, 0.8
_GAP_Y, _GAP_W, _WIND, _FORCE, _GOAL_Y = range(5)
_X, _Y, _VX, _VY, _SGAP, _SWID, _SGOAL = range(7)
_DIRS = np.vstack([[0.0, 0.0], np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1)])


def sample_starts(task: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    geo = np.array([task[_GAP_Y], task[_GAP_W], task[_GOAL_Y]])
    return np.column_stack([np.full(n, START_X), rng.uniform(-0.8, 0.8, n), np.zeros((n, 2)), np.tile(geo, (n, 1))])


def step(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    s = state.copy()
    x, y, vx, vy = s[:, _X], s[:, _Y], s[:, _VX], s[:, _VY]
    thrust = _DIRS[np.asarray(action)] * params[:, _FORCE][:, None]
    vx = vx + (thrust[:, 0] - DRAG * vx) * DT
    vy = vy + (thrust[:, 1] + params[:, _WIND] - DRAG * vy) * DT
    speed = np.hypot(vx, vy)
    scale = np.where(speed > VMAX, VMAX / np.maximum(speed, 1e-12), 1.0)
    vx, vy = vx * scale, vy * scale
    nx = np.clip(x + vx * DT, -1.0, 1.0)
    ny = np.clip(y + vy * DT, -1.0, 1.0)
    crosses = (x < 0) != (nx < 0)
    frac = np.divide(-x, nx - x, out=np.zeros_like(x), where=nx != x)
    y_at_wall = y + (ny - y) * frac
    in_gap = np.abs(y_at_wall - s[:, _SGAP]) <= s[:, _SWID] / 2
    blocked = crosses & ~in_gap
    nx = np.where(blocked, np.where(x < 0, -1e-6, 1e-6), nx)
    vx = np.where(blocked, 0.0, vx)
    s[:, _X], s[:, _Y], s[:, _VX], s[:, _VY] = nx, ny, vx, vy
    return s


def route_length(state: np.ndarray) -> np.ndarray:
    """Shortest distance to the goal: straight once on the right, else via the nearest
    point of the gap opening."""
    x, y = state[..., _X], state[..., _Y]
    gy, hw, goal_y = state[..., _SGAP], state[..., _SWID] / 2, state[..., _SGOAL]
    py = np.clip(y, gy - hw, gy + hw)
    via = np.hypot(x, py - y) + np.hypot(GOAL_X, goal_y - py)
    return np.where(x >= 0, np.hypot(GOAL_X - x, goal_y - y), via)


def terminated(state: np.ndarray) -> np.ndarray:
    return np.hypot(state[:, _X] - GOAL_X, state[:, _Y] - state[:, _SGOAL]) < GOAL_RADIUS


def reward(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    nxt = step(state, action, params)
    arrived = terminated(nxt) & ~terminated(state)
    return 10 * (route_length(state) - route_length(nxt)) + 10 * arrived


def observe(state: np.ndarray) -> np.ndarray:
    return np.column_stack([state[:, _X], state[:, _Y], state[:, _VX], state[:, _VY],
                            state[:, _SGAP], state[:, _SWID],
                            GOAL_X - state[:, _X], state[:, _SGOAL] - state[:, _Y]])


def planning_cost(params: np.ndarray):
    """For the exam's beam search: the remaining route length."""
    return route_length


class PointNavEnv(gym.Env):
    """One task as a gymnasium env, stepping the same batched functions."""

    def __init__(self, params: dict):
        self.task = np.array([params[k] for k in PARAM_ORDER], dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = sample_starts(self.task, 1, self.np_random)[0]
        return observe(self.state[None])[0].astype(np.float32), {}

    def step(self, action):
        s, a, p = self.state[None], np.array([int(action)]), self.task[None]
        r = float(reward(s, a, p)[0])
        self.state = step(s, a, p)[0]
        return observe(self.state[None])[0].astype(np.float32), r, bool(terminated(self.state[None])[0]), False, {}


def make_env(params: dict) -> PointNavEnv:
    return PointNavEnv(params)
