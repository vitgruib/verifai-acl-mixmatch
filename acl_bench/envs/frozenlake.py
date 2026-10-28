"""FrozenLake on random maps: Gymnasium's FrozenLake-v1 dynamics, generalized.

A training task is four scalars (`PARAM_ORDER`), declared again as `VerifaiRange`s in
`frozenlake.scenic`: a `size` x `size` map (rounded to an int) whose cells are holes with
probability `hole_density`, drawn from `layout_seed`; the start (top-left) and the goal
(bottom-right) are always cleared, so a map can still be unsolvable. Actions are
Gymnasium's (0 left, 1 down, 2 right, 3 up). With probability `slip` the agent moves in
one of the two perpendicular directions instead (equal odds; Gymnasium's slippery ice is
slip 2/3); a move off the map leaves it in place. The agent sees the 5x5 window centred
on itself, as hole, goal and off-map channels. Falling in a hole ends the play with no
reward; reaching the goal ends it with reward 1 - 0.9 t / MAX_EPISODE_STEPS.

The goal is "return" with a threshold any arrival clears: a hole also terminates, so the
"reach" rule (pass = terminated) would pass a play that fell in.

Plays reproduce from their start state alone: a state row is
[x, y, t, size, play_seed, goal_x, goal_y, MAX_SIZE^2 hole cells], and each step's slip
is a hash of (play_seed, t). Negative layout seeds name fixed maps (`held_out_questions`),
so every question is a parameter row. Two exact checks: `certify_exact` (a hole-free
path exists; since the slip draw is fixed per step and the agent can aim any direction,
that is exactly winnability of the seeded play) and `optimal_success` (the best policy's
success probability over the slip, by value iteration).
"""
from __future__ import annotations

import os
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import MAPS

PARAM_BOUNDS = {
    "size": (4, 10),
    "hole_density": (0.1, 0.3),
    "slip": (0.0, 0.67),
    "layout_seed": (0, 1_000_000),
}
PARAM_ORDER = tuple(PARAM_BOUNDS)
# ACCEL edits keep the map's seed and size: holes are seeded uniforms < hole_density, so a
# nudged density adds or removes a few holes, while a new seed would draw a whole new map.
MUTATE_FIXED = ("layout_seed", "size")
DEFAULTS = {"size": 8, "hole_density": 0.2, "slip": 0.67}
MAX_SIZE, VIEW = 15, 5
MAX_EPISODE_STEPS = 100
N_ACTIONS = 4
OBS_DIM, ACTION_DIM = 3 * VIEW * VIEW, N_ACTIONS
GOAL = "return"
RETURN_THRESHOLD = 0.05              # an arrival earns at least 1 - 0.9 = 0.1, a hole 0
SCENIC_FILE = os.path.join(os.path.dirname(__file__), "frozenlake.scenic")

_X, _Y, _T, _SIZE, _SEED, _GX, _GY = range(7)
_H0 = 7                              # holes start here, MAX_SIZE^2 cells, row-major [y][x]
STATE_DIM = _H0 + MAX_SIZE * MAX_SIZE
_P_SIZE, _P_DENSITY, _P_SLIP, _P_SEED = range(4)
_MOVE = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])           # Gymnasium's action order
_VIEW_DY, _VIEW_DX = np.meshgrid(np.arange(VIEW) - VIEW // 2, np.arange(VIEW) - VIEW // 2, indexing="ij")

# Only one hole-free route, a snake from the start to the goal; on any slip a perfect
# policy passes it less than half the time (0.40 at slip 0.05), so it is played without.
_SINGLE_PATH = ["SFFFFFFF", "HHHHHHHF", "FFFFFFFF", "FHHHHHHH", "FFFFFFFF", "HHHHHHHF", "HHHHHHHF", "HHHHHHHG"]
_NAMED = {-1: ("4x4", MAPS["4x4"], 2 / 3), -2: ("8x8", MAPS["8x8"], 2 / 3),
          -3: ("single_path_8x8", _SINGLE_PATH, 0.0)}


def valid(params: dict) -> bool:
    return (2 <= round(params["size"]) <= MAX_SIZE and 0 <= params["hole_density"] < 1
            and 0 <= params["slip"] <= 1)


def layout(size: float, hole_density: float, layout_seed: float):
    """(holes (size, size) bool, start (x, y), goal (x, y)); a negative seed names a fixed map."""
    seed = int(layout_seed)
    if seed < 0:
        holes = np.array([[c == "H" for c in row] for row in _NAMED[seed][1]])
    else:
        size = round(size)
        holes = np.random.default_rng(seed).random((size, size)) < hole_density
        holes[0, 0] = holes[-1, -1] = False
    n = len(holes)
    return holes, (0, 0), (n - 1, n - 1)


def make_state(holes: np.ndarray, play_seed: int) -> np.ndarray:
    n = len(holes)
    s = np.zeros(STATE_DIM)
    s[_SIZE], s[_SEED], s[_GX], s[_GY] = n, play_seed, n - 1, n - 1
    padded = np.zeros((MAX_SIZE, MAX_SIZE))
    padded[:n, :n] = holes
    s[_H0:] = padded.reshape(-1)
    return s


def sample_starts(task: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` plays of one task (a row in PARAM_ORDER): the same map, distinct slip seeds."""
    holes, _, _ = layout(task[_P_SIZE], task[_P_DENSITY], task[_P_SEED])
    return np.stack([make_state(holes, int(k)) for k in rng.choice(2**31 - 1, size=n, replace=False)])


def _uniform(seed: np.ndarray, t: np.ndarray) -> np.ndarray:
    """A uniform [0, 1) per (seed, t), the splitmix64 finalizer of seed * 2^32 + t."""
    z = (seed.astype(np.uint64) << np.uint64(32)) + t.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return (z >> np.uint64(11)).astype(np.float64) / 2.0**53


def _on_map(state: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = state[:, _SIZE].reshape((-1,) + (1,) * (x.ndim - 1))
    return (x >= 0) & (x < n) & (y >= 0) & (y < n)


def _hole_at(state: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Holes at on-map cells (x, y), shaped (N, ...) like x."""
    idx = _H0 + np.clip(y, 0, MAX_SIZE - 1) * MAX_SIZE + np.clip(x, 0, MAX_SIZE - 1)
    rows = np.arange(len(state)).reshape((-1,) + (1,) * (x.ndim - 1))
    return _on_map(state, x, y) & (state[rows, idx] > 0.5)


def step(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """The map is in the state; `params` gives the slip."""
    s = state.copy()
    a = np.asarray(action).astype(int)
    u, slip = _uniform(s[:, _SEED], s[:, _T]), np.asarray(params)[:, _P_SLIP]
    d = np.where(u < slip / 2, (a - 1) % 4, np.where(u < slip, (a + 1) % 4, a))
    x, y = s[:, _X].astype(int), s[:, _Y].astype(int)
    nx, ny = x + _MOVE[d, 0], y + _MOVE[d, 1]
    on = _on_map(s, nx, ny)
    s[:, _X], s[:, _Y] = np.where(on, nx, x), np.where(on, ny, y)
    s[:, _T] += 1
    return s


def _at_goal(state: np.ndarray) -> np.ndarray:
    return (state[:, _X] == state[:, _GX]) & (state[:, _Y] == state[:, _GY])


def terminated(state: np.ndarray) -> np.ndarray:
    return _at_goal(state) | _hole_at(state, state[:, _X].astype(int), state[:, _Y].astype(int))


def reward(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """The reward of taking `action` in `state`: 1 - 0.9 t / cap on arriving at the goal."""
    nxt = step(state, action, params)
    return np.where(_at_goal(nxt) & ~terminated(state), 1 - 0.9 * nxt[:, _T] / MAX_EPISODE_STEPS, 0.0)


def observe(state: np.ndarray) -> np.ndarray:
    n = len(state)
    vx = state[:, _X].astype(int)[:, None, None] + _VIEW_DX[None]
    vy = state[:, _Y].astype(int)[:, None, None] + _VIEW_DY[None]
    hole = _hole_at(state, vx, vy).reshape(n, -1)
    goal = ((vx == state[:, _GX].astype(int)[:, None, None]) & (vy == state[:, _GY].astype(int)[:, None, None])).reshape(n, -1)
    off = ~_on_map(state, vx, vy).reshape(n, -1)
    return np.concatenate([hole, goal, off], axis=1).astype(np.float64)


# ---------------------------------------------------------------- exact checks
def shortest_steps(holes: np.ndarray, start, goal) -> int:
    """Fewest moves from start to goal over hole-free cells (BFS); a large number if none."""
    n = len(holes)
    seen = {tuple(start)}
    q = deque([(start[0], start[1], 0)])
    while q:
        x, y, t = q.popleft()
        if (x, y) == tuple(goal):
            return t
        for dx, dy in _MOVE:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and not holes[ny, nx] and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny, t + 1))
    return 1 << 30


def certify_exact(params: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """True where a hole-free path reaches the goal within MAX_EPISODE_STEPS (exact, by BFS)."""
    out = np.zeros(len(s0), dtype=bool)
    for i, s in enumerate(s0):
        n = int(s[_SIZE])
        holes = s[_H0:].reshape(MAX_SIZE, MAX_SIZE)[:n, :n] > 0.5
        t = shortest_steps(holes, (int(s[_X]), int(s[_Y])), (int(s[_GX]), int(s[_GY])))
        out[i] = t <= MAX_EPISODE_STEPS
    return out


def optimal_success(params: np.ndarray) -> float:
    """The best policy's probability of reaching the goal within MAX_EPISODE_STEPS from the
    start, over the slip (finite-horizon value iteration)."""
    holes, start, goal = layout(params[_P_SIZE], params[_P_DENSITY], params[_P_SEED])
    slip, n = float(params[_P_SLIP]), len(holes)
    y, x = np.divmod(np.arange(n * n), n)
    nx, ny = x[None] + _MOVE[:, :1], y[None] + _MOVE[:, 1:]
    on = (nx >= 0) & (nx < n) & (ny >= 0) & (ny < n)
    nxt = np.where(on, ny * n + nx, y * n + x)                     # (direction, cell) -> cell
    is_goal = np.arange(n * n) == goal[1] * n + goal[0]
    v = np.zeros(n * n)
    for _ in range(MAX_EPISODE_STEPS):
        landed = np.where(is_goal, 1.0, np.where(holes.reshape(-1), 0.0, v))[nxt]
        v = ((1 - slip) * landed + slip / 2 * (np.roll(landed, 1, axis=0) + np.roll(landed, -1, axis=0))).max(axis=0)
    return float(v[start[1] * n + start[0]])


def held_out_questions() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Gymnasium's classic 4x4 and 8x8 maps on its slippery ice, and a single-safe-path 8x8
    map without slip: name -> (params row, s0 rows, one per play)."""
    out = {}
    for seed, (name, rows, slip) in _NAMED.items():
        holes, _, _ = layout(0, 0, seed)
        params = np.array([len(holes), holes.mean(), slip, seed], dtype=np.float64)
        out[name] = params, sample_starts(params, 10, np.random.default_rng(-seed))
    return out


# ---------------------------------------------------------------- gymnasium wrapper
class FrozenLakeEnv(gym.Env):
    """One map as a gymnasium env, stepping the same batched functions (for the study
    pipeline, which trains through gymnasium envs)."""

    def __init__(self, params: dict):
        self.task = np.array([params[k] for k in PARAM_ORDER], dtype=np.float64)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = sample_starts(self.task, 1, self.np_random)[0]
        return observe(self.state[None])[0].astype(np.float32), {}

    def step(self, action):
        s, a = self.state[None], np.array([int(action)])
        r = float(reward(s, a, self.task[None])[0])
        self.state = step(s, a, self.task[None])[0]
        term = bool(terminated(self.state[None])[0])
        return observe(self.state[None])[0].astype(np.float32), r, term, False, {}


def make_env(params: dict) -> FrozenLakeEnv:
    return FrozenLakeEnv(params)


def planning_cost(params: np.ndarray):
    """For the generic beam search, which FrozenLake does not need (certify_exact is exact):
    Manhattan distance to the goal."""
    return lambda s: np.abs(s[..., _X] - s[..., _GX]) + np.abs(s[..., _Y] - s[..., _GY])
