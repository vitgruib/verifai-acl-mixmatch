"""Falsification spaces over JaxUED maze levels (13x13 interior, no border walls; the env
adds the border). Pure numpy, so it runs without JAX.

A space maps a point of a VerifAI box (`bounds`, a dict name -> (lo, hi)) to a level:
(walls (13, 13) bool, agent (x, y), agent_dir, goal (x, y)). Two spaces:

  dr   the training distribution's own knobs: wall count, a wall seed, agent, goal and
       direction. Wall positions come from the seed, so that dimension is not smooth.
  seg  structured walls: up to 10 straight segments (x, y, orientation, length) plus agent,
       goal and direction. Smooth in every dimension, and reaches corridor and room
       layouts that uniform random walls almost never produce.

Validity is a hard constraint: walls under the agent or goal are removed, agent != goal,
and the goal must be reachable (cell BFS; turning is free). Invalid points are reported,
never evaluated.
"""
from __future__ import annotations

from collections import deque

import numpy as np

N = 13
_POS = {"agent_x": (0, N), "agent_y": (0, N), "goal_x": (0, N), "goal_y": (0, N), "agent_dir": (0, 4)}
N_SEG = 10
SPACES = {
    "dr": {"n_walls": (0, 61), "wall_seed": (0, 1e6), **_POS},
    "seg": {"n_seg": (0, N_SEG + 1), **{f"s{i}_{k}": b for i in range(N_SEG)
            for k, b in (("x", (0, N)), ("y", (0, N)), ("o", (0, 2)), ("len", (1, N)))}, **_POS},
}


def _i(v, hi):
    return int(min(max(np.floor(v), 0), hi - 1))


def build(space: str, p: dict):
    """Point -> (walls, agent, agent_dir, goal, invalid_reason or None)."""
    walls = np.zeros((N, N), dtype=bool)
    if space == "dr":
        cells = np.random.default_rng(int(p["wall_seed"])).permutation(N * N)
        walls.flat[cells[:_i(p["n_walls"], 61)]] = True
    elif space == "seg":
        for i in range(_i(p["n_seg"], N_SEG + 1)):
            x, y, ln = _i(p[f"s{i}_x"], N), _i(p[f"s{i}_y"], N), _i(p[f"s{i}_len"], N + 1)
            if _i(p[f"s{i}_o"], 2) == 0:
                walls[y, x:x + ln] = True
            else:
                walls[y:y + ln, x] = True
    else:
        raise ValueError(f"unknown space {space!r}; choose from {tuple(SPACES)}")
    agent = (_i(p["agent_x"], N), _i(p["agent_y"], N))
    goal = (_i(p["goal_x"], N), _i(p["goal_y"], N))
    walls[agent[1], agent[0]] = walls[goal[1], goal[0]] = False
    reason = None
    if agent == goal:
        reason = "agent_on_goal"
    elif shortest_path(walls, agent, goal) is None:
        reason = "unreachable"
    return walls, agent, _i(p["agent_dir"], 4), goal, reason


def shortest_path(walls, a, b):
    """Cell-step BFS distance from a to b (None if unreachable)."""
    dist = {a: 0}
    q = deque([a])
    while q:
        c = q.popleft()
        if c == b:
            return dist[c]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < N and 0 <= n[1] < N and not walls[n[1], n[0]] and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return None


def descriptors(walls, agent, goal) -> dict:
    """Interpretable level features used to locate a failure relative to training."""
    open_ = np.pad(~walls, 1)
    nb = open_[:-2, 1:-1].astype(int) + open_[2:, 1:-1] + open_[1:-1, :-2] + open_[1:-1, 2:]
    spl = shortest_path(walls, agent, goal)
    man = abs(agent[0] - goal[0]) + abs(agent[1] - goal[1])
    return {"n_walls": int(walls.sum()), "path_len": spl, "manhattan": man,
            "detour": (spl / man) if spl and man else None,
            "dead_ends": int(((nb == 1) & ~walls).sum()),
            "corridor_cells": int(((nb == 2) & ~walls).sum())}


DESC_KEYS = ("n_walls", "path_len", "manhattan", "dead_ends", "corridor_cells")


def to_str(walls, agent, agent_dir, goal) -> str:
    """JaxUED `Level.from_str` format, so any record can be replayed as a level."""
    rows = [["#" if w else "." for w in row] for row in walls]
    rows[goal[1]][goal[0]] = "G"
    rows[agent[1]][agent[0]] = ">v<^"[agent_dir]
    return "\n".join("".join(r) for r in rows)
