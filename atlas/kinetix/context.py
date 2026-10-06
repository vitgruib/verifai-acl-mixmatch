"""Where a Kinetix level sits relative to training (the `train` field of a record).

Reference: levels from the env's own training generator (Kinetix train_level_mode "random":
make_reset_fn_from_config -> sample_kinetix_level, random shapes/joints/thrusters in the S
arena). DR samples it directly and PLR/Robust PLR/SFL draw their candidates from it; ACCEL also
mutates them. Checkpoints carry no level buffer, so only the DR reference is available.

Fields: dr_knn (mean scaled descriptor distance to the K nearest reference levels, the level
index excluded; large = rarely seen in training).
"""
from __future__ import annotations

import jax
import numpy as np

from atlas.kinetix import space
from kinetix.environment.ued.ued import make_reset_fn_from_config

K = 10
FEAT_KEYS = tuple(k for k in space.DESC_KEYS if k != "level")


def featurize(level) -> np.ndarray:
    d = space.descriptors(level)
    return np.array([-1 if d[k] is None else d[k] for k in FEAT_KEYS], float)


def dr_levels(policy, n: int, seed: int = 0):
    """n training-distribution levels, batched (leading axis n)."""
    reset = make_reset_fn_from_config(policy.config, policy.env_params, policy.static_env_params,
                                      physics_engine=policy.env.physics_engine)
    return jax.jit(jax.vmap(reset))(jax.random.split(jax.random.PRNGKey(seed), n))


class TrainContext:
    def __init__(self, policy, n_ref: int = 2000):
        lv = dr_levels(policy, n_ref)
        self.dr = np.stack([featurize(jax.tree.map(lambda x: x[i], lv)) for i in range(n_ref)])
        self.scale = np.maximum(self.dr.std(0), 1.0)   # integer features; some are constant under DR (gravity, friction)

    def __call__(self, level) -> dict:
        d = np.linalg.norm((self.dr - featurize(level)) / self.scale, axis=1)
        return {"dr_knn": round(float(np.sort(d)[:K].mean()), 4)}

    def summary(self) -> dict:
        return {"dr_ref_n": len(self.dr), "buffer_n": None,
                "dr_no_goal_pair": round(float((self.dr[:, 0] < 0).mean()), 4),
                "desc_scale": dict(zip(FEAT_KEYS, self.scale.round(3).tolist()))}
