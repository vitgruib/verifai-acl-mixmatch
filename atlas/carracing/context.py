"""Where a CarRacing track sits relative to training (the `train` field of a record).

Reference: 2000 tracks from the env's own training generator (reset_random:
bezier.get_random_points(n=12, scale=playfield), start at track index 0). DR trains on this
distribution directly and PLR/Robust PLR/ACCEL/SFL draw their candidates from it (ACCEL then
edits them on the sketch grid); minimax trains on adversary sketches instead. Checkpoints
carry no level buffer, so only the DR reference is available.

Fields: dr_knn (mean scaled descriptor distance to the 10 nearest reference tracks; large =
rarely seen in training).
"""
from __future__ import annotations

import numpy as np

from atlas.carracing import space

K = 10


def dr_levels(n: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    return [(space.bezier().get_random_points(n=space.N_CTRL, scale=space.PLAYFIELD, np_random=rng), None)
            for _ in range(n)]


def featurize(points, start_alpha=None):
    d = space.descriptors(points, start_alpha)
    return np.array([d[k] for k in space.DESC_KEYS], float)


class TrainContext:
    def __init__(self, n_ref: int = 2000):
        self.dr = np.stack([featurize(*lv) for lv in dr_levels(n_ref)])
        # floor of 1 descriptor unit: n_ctrl is constant (12) under DR, std 0
        self.scale = np.maximum(self.dr.std(0), 1.0)

    def __call__(self, points, start_alpha=None) -> dict:
        d = np.linalg.norm((self.dr - featurize(points, start_alpha)) / self.scale, axis=1)
        return {"dr_knn": round(float(np.sort(d)[:K].mean()), 4)}

    def summary(self) -> dict:
        return {"dr_ref_n": len(self.dr), "buffer_n": None,
                "desc_mean": dict(zip(space.DESC_KEYS, self.dr.mean(0).round(3).tolist())),
                "desc_scale": dict(zip(space.DESC_KEYS, self.scale.round(3).tolist()))}
