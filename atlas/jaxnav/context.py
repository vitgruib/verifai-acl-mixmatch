"""Where a JaxNav level sits relative to training (the `train` field of a record).

Reference: 4000 levels from the env's own training generator (GridMapPolygonAgents.sample_map +
grid_sample_test_case with valid_path_check False): n_walls ~ randint(0, 48) interior cells
without replacement, start then goal uniform over free cells, heading uniform. Every algorithm
in the SFL repo sees this distribution (DR directly; PLR/minimax/SFL draw from it). SFL
checkpoints carry no level buffer, so only the DR reference is available.

Fields: dr_knn (mean scaled feature distance to the 10 nearest reference levels; large = rarely
seen in training).
"""
from __future__ import annotations

import numpy as np

from atlas.jaxnav import space

K = 10


def dr_levels(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        inner = np.zeros(space.M * space.M, np.int32)
        inner[rng.permutation(space.M * space.M)[:rng.integers(0, space.N_FILL)]] = 1
        free = np.flatnonzero(inner == 0)
        s, g = rng.choice(free, 2, replace=False)
        grid = space.empty_grid()
        grid[1:-1, 1:-1] = inner.reshape(space.M, space.M)
        cell = lambda i: (int(i % space.M) + 1, int(i // space.M) + 1)  # noqa: E731
        out.append((grid, cell(s), float(rng.uniform(-np.pi, np.pi)), cell(g)))
    return out


def featurize(grid, start, theta, goal):
    d = space.descriptors(grid, start, theta, goal)
    return np.array([-1 if d[k] is None else d[k] for k in space.DESC_KEYS], float)


class TrainContext:
    def __init__(self, n_ref: int = 4000):
        self.dr = np.stack([featurize(*lv) for lv in dr_levels(n_ref)])
        self.scale = self.dr.std(0) + 1e-6
        self.dr_unsolvable = float((self.dr[:, 1] < 0).mean())

    def __call__(self, grid, start, theta, goal) -> dict:
        d = np.linalg.norm((self.dr - featurize(grid, start, theta, goal)) / self.scale, axis=1)
        return {"dr_knn": round(float(np.sort(d)[:K].mean()), 4)}

    def summary(self) -> dict:
        return {"dr_ref_n": len(self.dr), "dr_unreachable": round(self.dr_unsolvable, 4),
                "buffer_n": None,
                "desc_scale": dict(zip(space.DESC_KEYS, self.scale.round(3).tolist()))}
