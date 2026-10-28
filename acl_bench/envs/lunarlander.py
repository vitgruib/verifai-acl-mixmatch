"""LunarLander (discrete) with gravity, wind and turbulence exposed as a task that Scenic/VerifAI samples.

A task is three scalars (`PARAM_ORDER`); `lunarlander.scenic` declares the same box. The goal
is a return: a question is passed by an episode return of at least `RETURN_THRESHOLD`, the
level Gymnasium calls solved.

Box2D cannot be batched in NumPy, so the batched contract (`step`, `observe`, `terminated`,
`reward`) runs on a pool of Gymnasium envs. A state row is `[pool_id, reset_seed, t, ret]`:
`sample_starts` registers a start (task and seed) and the env is built, and reset with that
seed, when first needed, so a play is reproducible from `(params, reset_seed)` (wind and
turbulence are drawn from the env's own seeded generator). A row whose episode terminated
holds `-pool_id - 1` and its env is freed. There is no `planning_cost`: continuous tasks are
not certified. A row cannot be restored to a mid-episode state (`t > 0`) once stepped.
"""
from __future__ import annotations

import itertools
import os

import gymnasium as gym
import numpy as np

PARAM_BOUNDS = {
    "gravity": (-12.0, -2.0),            # m/s^2; Gymnasium requires -12 < gravity < 0
    "wind_power": (0.0, 20.0),           # linear wind force
    "turbulence_power": (0.0, 2.0),      # rotational wind torque
}
PARAM_ORDER = tuple(PARAM_BOUNDS)
DEFAULTS = {"gravity": -10.0, "wind_power": 0.0, "turbulence_power": 0.0}
MAX_EPISODE_STEPS = 1000
OBS_DIM, ACTION_DIM = 8, 4
GOAL = "return"
RETURN_THRESHOLD = 200.0
SCENIC_FILE = os.path.join(os.path.dirname(__file__), "lunarlander.scenic")

_LOWEST_GRAVITY = -11.999


def valid(params: dict) -> bool:
    return -12.0 < params["gravity"] < 0.0 and params["wind_power"] >= 0.0 and params["turbulence_power"] >= 0.0


def make_env(params: dict) -> gym.Env:
    # VerifAI can return the open end of the box exactly; Gymnasium rejects -12.0.
    gravity = max(float(params["gravity"]), _LOWEST_GRAVITY)
    wind, turbulence = float(params["wind_power"]), float(params["turbulence_power"])
    return gym.make("LunarLander-v3", gravity=gravity, enable_wind=wind > 0 or turbulence > 0,
                    wind_power=wind, turbulence_power=turbulence)


# ---------------------------------------------------------------- the pool
class _Start:
    """One registered start. `env` is None until needed and after the episode ends."""

    def __init__(self, task: np.ndarray, seed: int):
        self.task, self.seed = task, seed
        self.env, self.obs, self.t = None, None, 0

    def reset(self) -> None:
        self.env = make_env(dict(zip(PARAM_ORDER, self.task))).unwrapped
        self.obs, _ = self.env.reset(seed=self.seed)
        self.t = 0

    def at(self, t: int) -> _Start:
        """This start's env, positioned at step `t` of its episode (only 0 can be re-made)."""
        if t == 0 and (self.t != 0 or self.env is None):
            self.reset()
        elif self.t != t:
            raise RuntimeError(f"LunarLander state at t={t} cannot be restored (pool is at t={self.t})")
        return self


_POOL: dict[int, _Start] = {}
_IDS = itertools.count()
_last_reward: tuple | None = None        # (state, action, next state, reward) from the latest `reward` call


def _start_of(row: np.ndarray) -> _Start:
    pid = int(row[0])
    return _POOL[pid if pid >= 0 else -pid - 1]


def sample_starts(task: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` starting states for one task (a row in PARAM_ORDER): fresh reset seeds."""
    state = np.zeros((n, 4))
    for k, seed in enumerate(rng.integers(0, 2**31 - 1, size=n)):
        pid = next(_IDS)
        _POOL[pid] = _Start(np.array(task, dtype=np.float64), int(seed))
        state[k, :2] = pid, seed
    return state


def _advance(state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nxt, reward = state.copy(), np.zeros(len(state))
    for i, row in enumerate(state):
        if row[0] < 0:
            continue
        start = _start_of(row).at(int(row[2]))
        start.obs, reward[i], term, _, _ = start.env.step(int(action[i]))
        start.t += 1
        nxt[i, 2:] = start.t, row[3] + reward[i]
        if term:
            nxt[i, 0] = -row[0] - 1
        if term or start.t >= MAX_EPISODE_STEPS:
            start.env.close()
            start.env = None
    return nxt, reward


def reward(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """The reward of taking `action` in `state`. Box2D yields it only by stepping, so this
    steps the pool and keeps the result for the `step` call on the same arguments."""
    global _last_reward
    nxt, r = _advance(state, action)
    _last_reward = (state.copy(), np.array(action), nxt, r)
    return r


def step(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Actions are Gymnasium's: 0 noop, 1 left engine, 2 main engine, 3 right engine."""
    global _last_reward
    last, _last_reward = _last_reward, None
    if last is not None and np.array_equal(last[0], state) and np.array_equal(last[1], action):
        return last[2]
    return _advance(state, action)[0]


def observe(state: np.ndarray) -> np.ndarray:
    return np.stack([_start_of(row).at(int(row[2]) if row[0] >= 0 else _start_of(row).t).obs for row in state])


def terminated(state: np.ndarray) -> np.ndarray:
    return state[:, 0] < 0
