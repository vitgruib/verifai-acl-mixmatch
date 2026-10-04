"""Where a level sits relative to what the agent trained on (the `train` field of a record).

Every level is reduced to its feature vector (`space.DESC_KEYS`), scaled by the spread of
the DR reference set. Two references:

  dr_*      4000 levels from JaxUED's DR generator (same distribution as
            `make_level_generator`: n_walls cells drawn with replacement, agent then goal
            uniform over free cells). Every algorithm sees DR levels, so this is the base
            training distribution. Unsolvable DR levels are kept, as in training.
  buf_*     the PLR/ACCEL level buffer saved in the checkpoint, with its scores (absent
            for DR and minimax).

Fields: dr_knn (mean distance to the 10 nearest DR levels; large = rarely seen),
buf_knn (same against the buffer), buf_score_pct (percentile, within the buffer, of the
mean score of those 10 neighbours: low = the curriculum ranked this region low, the
question of hypothesis H2), buf_age (mean staleness of the neighbours, in buffer
timestamps behind the newest).
"""
from __future__ import annotations

import numpy as np

from atlas.maze import space

K = 10


def dr_levels(n: int, n_walls: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        walls = np.zeros(space.N * space.N, bool)
        walls[rng.integers(0, space.N * space.N, n_walls)] = True
        free = np.flatnonzero(~walls)
        a, g = rng.choice(free, 2, replace=False)
        out.append((walls.reshape(space.N, space.N), (a % space.N, a // space.N), (g % space.N, g // space.N)))
    return out


def featurize(walls, agent, goal):
    d = space.descriptors(walls, tuple(int(v) for v in agent), tuple(int(v) for v in goal))
    # unreachable training levels get path_len = -1 (a distinct, far-away value)
    return np.array([-1 if d[k] is None else d[k] for k in space.DESC_KEYS], float)


class TrainContext:
    def __init__(self, n_walls: int, buffer: dict | None, n_ref: int = 4000):
        ref = dr_levels(n_ref, n_walls)
        self.dr = np.stack([featurize(*lv) for lv in ref])
        self.scale = self.dr.std(0) + 1e-6
        self.dr_unsolvable = float((self.dr[:, 1] < 0).mean())
        self.buf = None
        if buffer is not None:
            self.buf = np.stack([featurize(w, a, g) for w, a, g in
                                 zip(buffer["walls"], buffer["agent"], buffer["goal"])])
            self.buf_scores, self.buf_ts = buffer["scores"], buffer["timestamps"]

    def __call__(self, walls, agent, goal) -> dict:
        x = featurize(walls, agent, goal)
        d = np.linalg.norm((self.dr - x) / self.scale, axis=1)
        out = {"dr_knn": round(float(np.sort(d)[:K].mean()), 4)}
        if self.buf is not None:
            d = np.linalg.norm((self.buf - x) / self.scale, axis=1)
            nn = np.argsort(d)[:K]
            s = self.buf_scores[nn].mean()
            out.update(buf_knn=round(float(d[nn].mean()), 4),
                       buf_score_pct=round(float((self.buf_scores < s).mean()), 4),
                       buf_age=round(float(self.buf_ts.max() - self.buf_ts[nn].mean()), 1))
        return out

    def summary(self) -> dict:
        return {"dr_ref_n": len(self.dr), "dr_unsolvable": round(self.dr_unsolvable, 4),
                "buffer_n": None if self.buf is None else len(self.buf),
                "desc_scale": dict(zip(space.DESC_KEYS, self.scale.round(3).tolist()))}
