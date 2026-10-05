"""Where a BipedalWalker level sits relative to training (the `train` field of a record).

Reference: 2000 levels from the training box (PARAM_RANGES_FULL uniform plus a uniform seed,
as reset_random / rand_int_seed draw them). DR trains on this distribution directly,
PLR/Robust PLR/ACCEL/SFL draw their candidates from it (ACCEL then mutates them inside the box)
and minimax's adversary proposes in the same box. Checkpoints carry no level buffer, so only
the DR reference is available.

Fields: dr_knn (mean scaled descriptor distance to the 10 nearest reference levels; large =
rarely seen in training).
"""
from __future__ import annotations

import numpy as np

from atlas.bipedal import space

K = 10


def dr_levels(n: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        p = {k: rng.uniform(lo, hi) for k, (lo, hi) in space.SPACES["dr"].items()}
        out.append(space.build("dr", p)[0])
    return out


def featurize(level):
    d = space.descriptors(level)
    return np.array([d[k] for k in space.DESC_KEYS], float)


class TrainContext:
    def __init__(self, n_ref: int = 2000):
        self.dr = np.stack([featurize(lv) for lv in dr_levels(n_ref)])
        self.scale = np.maximum(self.dr.std(0), 1.0)

    def __call__(self, level) -> dict:
        d = np.linalg.norm((self.dr - featurize(level)) / self.scale, axis=1)
        return {"dr_knn": round(float(np.sort(d)[:K].mean()), 4)}

    def summary(self) -> dict:
        return {"dr_ref_n": len(self.dr), "buffer_n": None,
                "desc_mean": dict(zip(space.DESC_KEYS, self.dr.mean(0).round(3).tolist())),
                "desc_scale": dict(zip(space.DESC_KEYS, self.scale.round(3).tolist()))}
