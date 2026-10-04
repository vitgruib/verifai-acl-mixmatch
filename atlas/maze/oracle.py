"""Exact solvability oracle for JaxUED maze levels.

A level counts as a counterexample only if the agent fails on a level that *can* be solved within
the episode the agent actually gets. Cell reachability (space.shortest_path) is not enough: the
JaxUED agent turns in place (left/right cost one step each), the goal is reached by moving
forward *into* it, and the episode ends after `max_steps_in_episode` (250) actions. A goal that
is reachable but needs more actions than that is unsolvable for every policy.

`solve` runs a breadth-first search over (x, y, dir) with JaxUED's exact transition rule
(`jaxued/environments/maze/env.py: Maze._step_agent`): forward moves one cell along DIR_TO_VEC,
clamped to the grid, blocked by walls; the goal is entered only by a forward action, which ends
the episode with reward. BFS is exhaustive over 13*13*4 states, so the answer is exact:

  SOLVABLE    min_steps <= max_steps; `cert` is a shortest action string ("L", "R", "F")
  UNSOLVABLE  goal not enterable at all, or only after more than max_steps actions
  UNKNOWN     never returned here; kept so other envs' oracles share the same record format

`replay` checks a certificate by re-simulating it (an independent loop over the same rule).
`atlas.maze.policy.Policy.replay` replays it in the real JaxUED env (tests/test_atlas.py).
"""
from __future__ import annotations

from collections import deque

import numpy as np

SOLVABLE, UNSOLVABLE, UNKNOWN = "solvable", "unsolvable", "unknown"
MAX_STEPS = 250                                   # Maze.default_params.max_steps_in_episode
DIR_TO_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))   # right, down, left, up (JaxUED order)
ACTIONS = {"L": 0, "R": 1, "F": 2}                # JaxUED Actions.left/right/forward


def _step(walls, pos, d, a, goal):
    """One JaxUED transition -> (pos, dir, reached_goal)."""
    h, w = walls.shape
    if a == "F":
        dx, dy = DIR_TO_VEC[d]
        nx, ny = min(max(pos[0] + dx, 0), w - 1), min(max(pos[1] + dy, 0), h - 1)
        if (nx, ny) == tuple(goal):
            return pos, d, True
        if not walls[ny, nx]:
            pos = (nx, ny)
        return pos, d, False
    return pos, (d + (1 if a == "R" else -1)) % 4, False


def solve(walls, agent, agent_dir, goal, max_steps: int = MAX_STEPS) -> dict:
    """-> {"status", "min_steps", "cert", "max_steps"}; min_steps counts the final forward."""
    walls = np.asarray(walls, bool)
    start = (tuple(int(v) for v in agent), int(agent_dir))
    goal = tuple(int(v) for v in goal)
    out = {"max_steps": max_steps, "min_steps": None, "cert": None}
    if start[0] == goal:
        return {**out, "status": UNSOLVABLE}
    parent = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        for a in "FLR":
            pos, d, hit = _step(walls, s[0], s[1], a, goal)
            if hit:
                acts = [a]
                while parent[s] is not None:
                    s, prev_a = parent[s]
                    acts.append(prev_a)
                cert = "".join(reversed(acts))
                status = SOLVABLE if len(cert) <= max_steps else UNSOLVABLE
                return {**out, "status": status, "min_steps": len(cert), "cert": cert}
            n = (pos, d)
            if n not in parent:
                parent[n] = (s, a)
                q.append(n)
    return {**out, "status": UNSOLVABLE}


def replay(walls, agent, agent_dir, goal, cert: str, max_steps: int = MAX_STEPS) -> bool:
    """True iff the action string enters the goal within max_steps."""
    walls = np.asarray(walls, bool)
    pos, d = tuple(int(v) for v in agent), int(agent_dir)
    for t, a in enumerate(cert):
        if t >= max_steps:
            return False
        pos, d, hit = _step(walls, pos, d, a, goal)
        if hit:
            return True
    return False
