"""LunarLander (discrete) with gravity, wind and turbulence exposed as a task that Scenic/VerifAI samples.

A task is three scalars (`PARAM_ORDER`); `lunarlander.scenic` declares the same box. The goal
is a return: a question is passed by an episode return of at least `RETURN_THRESHOLD`, the
level Gymnasium calls solved.

Box2D cannot be batched in NumPy, so the batched contract (`step`, `observe`, `terminated`,
`reward`) runs on a pool of Gymnasium envs. A state row is
`[handle, reset_seed, t, ret, *task]`. A start (`sample_starts`) has handle 0: it names no env,
so it is self-describing and a play is reproducible from `(task, reset_seed)` alone, in any
process (wind and turbulence are drawn from the env's own seeded generator). The first `step`
builds the env and gives the row a process-local handle; a row whose episode terminated holds
`-handle` and its env is freed. At most `MAX_LIVE` envs stay open: the oldest are closed, and
stepping such a row raises. There is no `planning_cost`: continuous tasks are not certified.
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
MAX_LIVE = 8192                          # open Box2D worlds; the oldest are closed beyond this
_LIVE: dict[int, list] = {}              # handle -> [env or None, last obs], oldest first
_HANDLES = itertools.count(1)
_last_reward: tuple | None = None        # (state, action, next state, reward) from the latest `reward` call


def _new_env(task, seed: int):
    env = make_env(dict(zip(PARAM_ORDER, task))).unwrapped
    return env, env.reset(seed=int(seed))[0]


def _close(handle: int) -> None:
    env = _LIVE.pop(handle)[0]
    if env is not None:
        env.close()


def sample_starts(task: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` starting states for one task (a row in PARAM_ORDER): fresh reset seeds."""
    state = np.zeros((n, 4 + len(PARAM_ORDER)))
    state[:, 1] = rng.integers(0, 2**31 - 1, size=n)
    state[:, 4:] = task
    return state


def _advance(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nxt, reward = state.copy(), np.zeros(len(state))
    for i, row in enumerate(state):
        if row[0] < 0:
            continue
        if row[2] == 0:
            env, obs = _new_env(params[i], row[1])
            nxt[i, 0] = next(_HANDLES)
            _LIVE[int(nxt[i, 0])] = [env, obs]
            while len(_LIVE) > MAX_LIVE:
                _close(next(iter(_LIVE)))
        handle = int(nxt[i, 0])
        if handle not in _LIVE or _LIVE[handle][0] is None:
            raise RuntimeError("LunarLander state cannot be restored mid-episode (only starts can be replayed)")
        entry = _LIVE[handle]
        entry[1], reward[i], term, _, _ = entry[0].step(int(action[i]))
        nxt[i, 2:4] = row[2] + 1, row[3] + reward[i]
        if term:
            nxt[i, 0] = -handle
            _close(handle)
        elif nxt[i, 2] >= MAX_EPISODE_STEPS:
            entry[0].close()
            entry[0] = None
    return nxt, reward


def reward(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """The reward of taking `action` in `state`. Box2D yields it only by stepping, so this
    steps the pool and keeps the result for the `step` call on the same arguments."""
    global _last_reward
    nxt, r = _advance(state, action, params)
    _last_reward = (state.copy(), np.array(action), nxt, r)
    return r


def step(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Actions are Gymnasium's: 0 noop, 1 left engine, 2 main engine, 3 right engine."""
    global _last_reward
    last, _last_reward = _last_reward, None
    if last is not None and np.array_equal(last[0], state) and np.array_equal(last[1], action):
        return last[2]
    return _advance(state, action, params)[0]


def observe(state: np.ndarray) -> np.ndarray:
    obs = np.zeros((len(state), OBS_DIM))
    for i, row in enumerate(state):
        if row[2] == 0:
            env, obs[i] = _new_env(row[4:], row[1])    # a start has no env yet; reset is deterministic
            env.close()
        elif row[0] > 0:
            obs[i] = _LIVE[int(row[0])][1]
    return obs


def terminated(state: np.ndarray) -> np.ndarray:
    return state[:, 0] < 0
