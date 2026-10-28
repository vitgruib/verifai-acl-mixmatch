"""Maze exam sections (random and custom), stored like `acl_bench/exam/sets.py`: one .npz per
section plus a manifest.json of sha256 digests that loading verifies.

A level is the tuple `space.build` returns: (walls (13,13) bool indexed [y, x], agent (x, y),
agent_dir, goal (x, y)). The random section needs JAX and a JaxUED checkout (JAXUED_DIR)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

from atlas.maze import oracle, space

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAXUED_DIR = os.environ.get("JAXUED_DIR", os.path.join(ROOT, "third_party", "jaxued"))
N = space.N
EXAM_SEED = 20261028
NAMED = ["SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
         "StandardMaze", "StandardMaze2", "StandardMaze3"]
EXTREMES = ["empty", "longest_path", "dead_end_goal", "corridor", "walls30", "walls50"]


@dataclass
class MazeSet:
    walls: np.ndarray                           # (N, 13, 13) bool, [y, x]
    agent: np.ndarray                           # (N, 2) (x, y)
    agent_dir: np.ndarray                       # (N,)
    goal: np.ndarray                            # (N, 2) (x, y)
    labels: list[str] | None = None             # custom section only
    ref_fail_frac: np.ndarray | None = None     # (N,) VerifAI section only

    def __post_init__(self):
        self.walls = np.asarray(self.walls, bool)
        self.agent = np.asarray(self.agent, np.int64)
        self.agent_dir = np.asarray(self.agent_dir, np.int64)
        self.goal = np.asarray(self.goal, np.int64)
        if self.ref_fail_frac is not None:
            self.ref_fail_frac = np.asarray(self.ref_fail_frac, np.float64)

    def __len__(self) -> int:
        return len(self.walls)

    def digest(self) -> str:
        h = hashlib.sha256()
        for arr in (self.walls, self.agent, self.agent_dir, self.goal):
            h.update(np.ascontiguousarray(arr).tobytes())
        if self.ref_fail_frac is not None:
            h.update(np.ascontiguousarray(self.ref_fail_frac).tobytes())
        if self.labels is not None:
            h.update("\n".join(self.labels).encode())
        return h.hexdigest()


def _from_levels(levels, labels=None) -> MazeSet:
    return MazeSet(np.stack([l[0] for l in levels]), np.array([l[1] for l in levels]),
                   np.array([l[2] for l in levels]), np.array([l[3] for l in levels]), labels)


def _jaxued():
    sys.path.insert(0, os.path.join(JAXUED_DIR, "src"))
    import jax
    from jaxued.environments.maze import Level
    from jaxued.environments.maze.util import make_level_generator
    return jax, Level, make_level_generator


def random_section(n: int = 300, seed: int = EXAM_SEED) -> MazeSet:
    """The first `n` BFS-solvable levels JaxUED's 25-wall generator draws from PRNGKey(seed)."""
    jax, _, make_level_generator = _jaxued()
    draw = jax.jit(jax.vmap(make_level_generator(N, N, 25)))
    levels, key = [], jax.random.PRNGKey(seed)
    while len(levels) < n:
        key, sub = jax.random.split(key)
        lv = draw(jax.random.split(sub, 2 * n))
        for w, a, d, g in zip(np.asarray(lv.wall_map), np.asarray(lv.agent_pos),
                              np.asarray(lv.agent_dir), np.asarray(lv.goal_pos)):
            l = (w.astype(bool), tuple(int(v) for v in a), int(d), tuple(int(v) for v in g))
            if oracle.solve(*l)["status"] == oracle.SOLVABLE and len(levels) < n:
                levels.append(l)
    return _from_levels(levels)


def _named(name: str):
    _, Level, _ = _jaxued()
    p = Level.load_prefabs([name])
    return (np.asarray(p.wall_map[0], bool), tuple(int(v) for v in p.agent_pos[0]),
            int(p.agent_dir[0]), tuple(int(v) for v in p.goal_pos[0]))


def _serpentine():
    """Free rows 0, 2, ..., 12 joined by one gap per wall row, alternating sides: the longest path."""
    w = np.zeros((N, N), bool)
    for i, y in enumerate(range(1, N, 2)):
        w[y, :] = True
        w[y, N - 1 if i % 2 == 0 else 0] = False
    return w, (0, 0), 0, (N - 1, N - 1)


def _spiral():
    """A one-cell-wide corridor carved in a spiral from the corner to the middle."""
    w = np.ones((N, N), bool)
    x = y = 0
    dx, dy = 1, 0
    w[0, 0] = False
    turns = 0
    while turns < 2:
        nx, ny = x + dx, y + dy
        ahead2 = (x + 2 * dx, y + 2 * dy)
        if 0 <= nx < N and 0 <= ny < N and w[ny, nx] and not (
                0 <= ahead2[0] < N and 0 <= ahead2[1] < N and not w[ahead2[1], ahead2[0]]):
            x, y, turns = nx, ny, 0
            w[y, x] = False
        else:
            dx, dy, turns = -dy, dx, turns + 1
    return w, (0, 0), 0, (x, y)


def _random_walls(k: int):
    """k random walls, redrawn until solvable from corner (0, 0) to corner (12, 12)."""
    rng = np.random.default_rng(EXAM_SEED + k)
    while True:
        w = np.zeros(N * N, bool)
        w[rng.choice(N * N - 2, k, replace=False) + 1] = True   # first and last cells stay free
        l = (w.reshape(N, N), (0, 0), 0, (N - 1, N - 1))
        if oracle.solve(*l)["status"] == oracle.SOLVABLE:
            return l


def custom_section() -> MazeSet:
    """The papers' held-out mazes (JaxUED prefabs) plus six constructed extremes."""
    dead = np.zeros((N, N), bool)
    dead[6, 5] = dead[6, 7] = dead[7, 6] = True                 # goal (6, 6) open only to the north
    levels = [_named(name) for name in NAMED]
    levels += [(np.zeros((N, N), bool), (0, 0), 0, (N - 1, N - 1)),
               _serpentine(), (dead, (0, 0), 0, (6, 6)), _spiral(), _random_walls(30), _random_walls(50)]
    return _from_levels(levels, NAMED + EXTREMES)


def save_set(path: str, name: str, ms: MazeSet) -> None:
    """Write `<path>/<name>.npz` and merge the section's count and digest into manifest.json."""
    os.makedirs(path, exist_ok=True)
    arrays = dict(walls=ms.walls, agent=ms.agent, agent_dir=ms.agent_dir, goal=ms.goal)
    if ms.labels is not None:
        arrays["labels"] = np.array(ms.labels)
    if ms.ref_fail_frac is not None:
        arrays["ref_fail_frac"] = ms.ref_fail_frac
    np.savez_compressed(os.path.join(path, f"{name}.npz"), **arrays)
    mpath = os.path.join(path, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"sets": {}}
    manifest["sets"][name] = {"n": len(ms), "sha256": ms.digest()}
    json.dump(manifest, open(mpath, "w"), indent=2, sort_keys=True)


def load_sets(path: str) -> dict[str, MazeSet]:
    """Load every section in manifest.json and verify its digest; raises if one changed."""
    manifest = json.load(open(os.path.join(path, "manifest.json")))
    sets = {}
    for name, meta in manifest["sets"].items():
        z = np.load(os.path.join(path, f"{name}.npz"))
        ms = MazeSet(z["walls"], z["agent"], z["agent_dir"], z["goal"],
                     [str(s) for s in z["labels"]] if "labels" in z.files else None,
                     z["ref_fail_frac"] if "ref_fail_frac" in z.files else None)
        if ms.digest() != meta["sha256"]:
            raise ValueError(f"section {name} does not match its manifest checksum")
        sets[name] = ms
    return sets
