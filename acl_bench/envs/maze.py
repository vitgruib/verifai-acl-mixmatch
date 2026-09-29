"""Grid maze navigation: the standard testbed of the UED literature (PAIRED, PLR, Robust
PLR, ACCEL, SFL), in the conventions of JaxUED and minimax.

The grid is 15x15 including an outer wall, so a 13x13 interior. A training task is two
scalars (`PARAM_ORDER`), declared again as `VerifaiRange`s in `maze.scenic`: `n_walls`
interior cells become walls, and `layout_seed` fixes which cells, where the goal is and
where the agent starts (all distinct, uniformly among the interior cells; a level can be
unsolvable, as in the literature's domain randomization). The starting direction is drawn
per episode. Actions: turn left, turn right, move forward (blocked by walls). The agent
sees a 5x5 window ahead of it (4 cells forward, 2 to each side; walls do not block sight,
as in DCD's mazes), as wall and goal channels, plus its direction one-hot. Reaching the
goal ends the episode with reward 1 - 0.9 t / MAX_EPISODE_STEPS (t = the step it arrived);
every other step gives 0. The goal is to reach: a question is passed by reaching the goal
within `MAX_EPISODE_STEPS`.

The layout lives in the state, not the parameters: a state row is
[x, y, dir, t, goal_x, goal_y, 225 wall cells], so the named held-out mazes of the
literature (`maze_layouts.HELD_OUT`) are questions like any other. Winnability is exact:
a breadth-first search over (cell, direction) (`certify_exact`).
"""
from __future__ import annotations

import os
from collections import deque

import gymnasium as gym
import numpy as np

PARAM_BOUNDS = {
    "n_walls": (0.0, 60.0),          # interior wall cells (JaxUED / minimax: at most 60)
    "layout_seed": (0.0, 1e6),       # which cells, and the goal and start
}
PARAM_ORDER = tuple(PARAM_BOUNDS)
SIZE, VIEW = 15, 5
MAX_EPISODE_STEPS = 250
N_ACTIONS = 3                        # 0 turn left, 1 turn right, 2 forward
OBS_DIM, ACTION_DIM = 2 * VIEW * VIEW + 4, N_ACTIONS
GOAL = "reach"
SCENIC_FILE = os.path.join(os.path.dirname(__file__), "maze.scenic")

_X, _Y, _DIR, _T, _GX, _GY = range(6)
_W0 = 6                              # walls start here, SIZE * SIZE cells, row-major [y][x]
STATE_DIM = _W0 + SIZE * SIZE
_FWD = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])          # dir 0 right, 1 down, 2 left, 3 up
_RIGHT = np.stack([-_FWD[:, 1], _FWD[:, 0]], axis=1)          # the agent's right-hand side
_VIEW_F, _VIEW_R = np.meshgrid(np.arange(VIEW - 1, -1, -1), np.arange(VIEW) - VIEW // 2, indexing="ij")


def layout(n_walls: float, layout_seed: float):
    """(walls (SIZE, SIZE) bool, start (x, y), goal (x, y)) for a training task."""
    rng = np.random.default_rng(int(layout_seed))
    cells = rng.permutation(13 * 13)
    k = int(round(n_walls))
    walls = np.ones((SIZE, SIZE), dtype=bool)
    walls[1:-1, 1:-1] = False
    inner = walls[1:-1, 1:-1]
    inner.flat[cells[:k]] = True
    goal, start = cells[k], cells[k + 1]
    return walls, (start % 13 + 1, start // 13 + 1), (goal % 13 + 1, goal // 13 + 1)


def make_state(walls: np.ndarray, start, goal, direction: int) -> np.ndarray:
    s = np.zeros(STATE_DIM)
    s[_X], s[_Y], s[_DIR] = start[0], start[1], direction
    s[_GX], s[_GY] = goal
    s[_W0:] = walls.reshape(-1)
    return s


def sample_starts(task: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    walls, start, goal = layout(task[0], task[1])
    return np.stack([make_state(walls, start, goal, int(d)) for d in rng.integers(0, 4, size=n)])


def _wall_at(state: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    inside = (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE)
    idx = _W0 + np.clip(y, 0, SIZE - 1) * SIZE + np.clip(x, 0, SIZE - 1)
    return ~inside | (state[np.arange(len(state))[:, None], idx.reshape(len(state), -1)].reshape(x.shape) > 0.5)


def step(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """The layout is in the state; `params` is unused (kept for the common contract)."""
    s = state.copy()
    a = np.asarray(action)
    d = s[:, _DIR].astype(int)
    d = np.where(a == 0, (d - 1) % 4, np.where(a == 1, (d + 1) % 4, d))
    x, y = s[:, _X].astype(int), s[:, _Y].astype(int)
    nx, ny = x + _FWD[d, 0], y + _FWD[d, 1]
    move = (a == 2) & ~_wall_at(s, nx[:, None], ny[:, None])[:, 0]
    s[:, _X], s[:, _Y], s[:, _DIR] = np.where(move, nx, x), np.where(move, ny, y), d
    s[:, _T] += 1
    return s


def terminated(state: np.ndarray) -> np.ndarray:
    return (state[:, _X] == state[:, _GX]) & (state[:, _Y] == state[:, _GY])


def reward(state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
    """The reward of taking `action` in `state`: 1 - 0.9 t / cap on arriving at the goal."""
    nxt = step(state, action, params)
    return np.where(terminated(nxt) & ~terminated(state), 1 - 0.9 * nxt[:, _T] / MAX_EPISODE_STEPS, 0.0)


def observe(state: np.ndarray) -> np.ndarray:
    n = len(state)
    d = state[:, _DIR].astype(int)
    x, y = state[:, _X].astype(int), state[:, _Y].astype(int)
    vx = x[:, None, None] + _VIEW_F[None] * _FWD[d, 0][:, None, None] + _VIEW_R[None] * _RIGHT[d, 0][:, None, None]
    vy = y[:, None, None] + _VIEW_F[None] * _FWD[d, 1][:, None, None] + _VIEW_R[None] * _RIGHT[d, 1][:, None, None]
    wall = _wall_at(state, vx, vy).reshape(n, -1)
    goal = ((vx == state[:, _GX].astype(int)[:, None, None]) & (vy == state[:, _GY].astype(int)[:, None, None])).reshape(n, -1)
    return np.concatenate([wall, goal, np.eye(4)[d]], axis=1).astype(np.float64)


# ---------------------------------------------------------------- exact winnability
def shortest_steps(walls: np.ndarray, start, goal, direction: int) -> int:
    """Fewest actions from (start, direction) to the goal (BFS over cell x direction);
    a large number if unreachable."""
    seen = {(start[0], start[1], direction)}
    q = deque([(start[0], start[1], direction, 0)])
    while q:
        x, y, d, t = q.popleft()
        if (x, y) == tuple(goal):
            return t
        for nd, (nx, ny) in (((d - 1) % 4, (x, y)), ((d + 1) % 4, (x, y)), (d, (x + _FWD[d, 0], y + _FWD[d, 1]))):
            if walls[ny, nx] or (nx, ny, nd) in seen:
                continue
            seen.add((nx, ny, nd))
            q.append((nx, ny, nd, t + 1))
    return 1 << 30


def certify_exact(params: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """True where the goal is reachable within MAX_EPISODE_STEPS (exact, by BFS)."""
    out = np.zeros(len(s0), dtype=bool)
    for i, s in enumerate(s0):
        walls = s[_W0:].reshape(SIZE, SIZE) > 0.5
        t = shortest_steps(walls, (int(s[_X]), int(s[_Y])), (int(s[_GX]), int(s[_GY])), int(s[_DIR]))
        out[i] = t <= MAX_EPISODE_STEPS
    return out


def held_out_questions() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The literature's named held-out mazes, one question per starting direction:
    (params placeholder, s0, labels)."""
    from acl_bench.envs.maze_layouts import HELD_OUT
    states, labels = [], []
    for name, (*rows, start, goal) in HELD_OUT.items():
        walls = np.ones((SIZE, SIZE), dtype=bool)
        walls[1:-1, 1:-1] = np.array([[c == "1" for c in r] for r in rows])
        for d in range(4):
            states.append(make_state(walls, start, goal, d))
            labels.append(f"{name}/dir{d}")
    s0 = np.stack(states)
    return np.full((len(s0), len(PARAM_ORDER)), -1.0), s0, labels


# ---------------------------------------------------------------- gymnasium wrapper
class MazeEnv(gym.Env):
    """One maze as a gymnasium env, stepping the same batched functions (for the study
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


def make_env(params: dict) -> MazeEnv:
    return MazeEnv(params)


def planning_cost(params: np.ndarray):
    """For the generic beam search, which the maze does not need (certify_exact is exact):
    Manhattan distance to the goal."""
    return lambda s: np.abs(s[..., _X] - s[..., _GX]) + np.abs(s[..., _Y] - s[..., _GY])
