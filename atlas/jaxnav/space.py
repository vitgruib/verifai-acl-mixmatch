"""Falsification spaces over single-agent JaxNav levels (the SFL repo's Grid-Rand-Poly maps:
11x11 grid, border walls, 9x9 interior, cell size 1 m). Pure numpy, so it runs without JAX.

A space maps a point of a VerifAI box (`bounds`, a dict name -> (lo, hi)) to a level:
(grid (11, 11) int indexed [y, x], start cell (x, y), theta, goal cell (x, y)). Start and goal
sit at cell centres (world position = cell + 0.5), as in the env's grid_sample_test_case.

  dr   the training generator's own knobs: wall count (training draws 0..47), a wall seed,
       start, goal and heading. Wall positions come from the seed, so that dimension is not
       smooth.
  seg  structured walls: up to 8 straight segments (x, y, orientation, length) plus start,
       goal and heading. Smooth in every dimension, and reaches corridor and room layouts
       that uniform random walls almost never produce.

Walls under the start or goal are removed and start != goal. Reachability is left to the
oracle (atlas.jaxnav.oracle), which also proves solvability under the robot's dynamics.
"""
from __future__ import annotations

from collections import deque

import numpy as np

N = 11            # full grid incl. border
M = N - 2         # interior
N_FILL = 48       # training: n_walls ~ randint(0, floor(81 * 0.6) = 48)
N_SEG = 8
_POS = {"start_x": (0, M), "start_y": (0, M), "goal_x": (0, M), "goal_y": (0, M),
        "theta": (-np.pi, np.pi)}
SPACES = {
    "dr": {"n_walls": (0, N_FILL), "wall_seed": (0, 1e6), **_POS},
    "seg": {"n_seg": (0, N_SEG + 1), **{f"s{i}_{k}": b for i in range(N_SEG)
            for k, b in (("x", (0, M)), ("y", (0, M)), ("o", (0, 2)), ("len", (1, M)))}, **_POS},
}


def _i(v, hi):
    return int(min(max(np.floor(v), 0), hi - 1))


def empty_grid():
    g = np.ones((N, N), np.int32)
    g[1:-1, 1:-1] = 0
    return g


def build(space: str, p: dict):
    """Point -> (grid, start, theta, goal, invalid_reason or None); cells in full-grid coords."""
    inner = np.zeros((M, M), np.int32)
    if space == "dr":
        cells = np.random.default_rng(int(p["wall_seed"])).permutation(M * M)
        inner.flat[cells[:_i(p["n_walls"], N_FILL)]] = 1
    elif space == "seg":
        for i in range(_i(p["n_seg"], N_SEG + 1)):
            x, y, ln = _i(p[f"s{i}_x"], M), _i(p[f"s{i}_y"], M), _i(p[f"s{i}_len"], M + 1)
            if _i(p[f"s{i}_o"], 2) == 0:
                inner[y, x:x + ln] = 1
            else:
                inner[y:y + ln, x] = 1
    else:
        raise ValueError(f"unknown space {space!r}; choose from {tuple(SPACES)}")
    grid = empty_grid()
    grid[1:-1, 1:-1] = inner
    start = (_i(p["start_x"], M) + 1, _i(p["start_y"], M) + 1)
    goal = (_i(p["goal_x"], M) + 1, _i(p["goal_y"], M) + 1)
    grid[start[1], start[0]] = grid[goal[1], goal[0]] = 0
    theta = float(np.clip(p["theta"], -np.pi, np.pi))
    return grid, start, theta, goal, ("agent_on_goal" if start == goal else None)


def bfs_path(grid, a, b):
    """Shortest 4-connected path of (x, y) cells from a to b on grid[y, x] (0 = free), or None."""
    grid = np.asarray(grid)
    h, w = grid.shape
    a, b = tuple(a), tuple(b)
    if grid[a[1], a[0]] or grid[b[1], b[0]]:
        return None
    prev = {a: None}
    q = deque([a])
    while q:
        c = q.popleft()
        if c == b:
            path = []
            while c is not None:
                path.append(c)
                c = prev[c]
            return path[::-1]
        x, y = c
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= n[0] < w and 0 <= n[1] < h and not grid[n[1], n[0]] and n not in prev:
                prev[n] = c
                q.append(n)
    return None


def corners(path):
    """Cells where the path changes direction, plus the last cell (the start is dropped)."""
    out = []
    for i in range(1, len(path) - 1):
        d0 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d1 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d0 != d1:
            out.append(path[i])
    out.append(path[-1])
    return out


def descriptors(grid, start, theta, goal) -> dict:
    """Interpretable level features (ints, binned //4 by the Recorder)."""
    free = np.asarray(grid) == 0
    f = np.pad(free, 1)
    nb = f[:-2, 1:-1].astype(int) + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:]
    path = bfs_path(grid, start, goal)
    man = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    turn = None
    if path is not None and len(path) > 1:
        dx, dy = path[1][0] - path[0][0], path[1][1] - path[0][1]
        e = (np.arctan2(dy, dx) - theta + np.pi) % (2 * np.pi) - np.pi
        turn = int(abs(e) // (np.pi / 8))       # 0..7: initial turn needed, in pi/8 steps
    return {"n_walls": int((~free[1:-1, 1:-1]).sum()),
            "path_len": None if path is None else len(path) - 1, "manhattan": man,
            "n_corners": None if path is None else len(corners(path)) - 1,
            "dead_ends": int(((nb == 1) & free).sum()), "turn_oct": turn}


DESC_KEYS = ("n_walls", "path_len", "manhattan", "n_corners", "dead_ends", "turn_oct")


def to_str(grid, start, theta, goal) -> str:
    """Grid rows ('#' wall, 'S' start, 'G' goal; row 0 = y 0) plus the heading, replayable
    with from_str."""
    rows = [["#" if c else "." for c in row] for row in np.asarray(grid)]
    rows[goal[1]][goal[0]] = "G"
    rows[start[1]][start[0]] = "S"
    return "\n".join("".join(r) for r in rows) + f"\ntheta={theta:.6f}"


def from_str(s: str):
    *rows, th = s.strip().split("\n")
    grid = np.array([[1 if ch == "#" else 0 for ch in r] for r in rows], np.int32)
    find = lambda ch: next((x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == ch)  # noqa: E731
    return grid, find("S"), float(th.split("=")[1]), find("G")
