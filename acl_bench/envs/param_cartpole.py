"""CartPole-v1 with physical parameters exposed as a Scenic/VerifAI-samplable task."""
from __future__ import annotations

import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

PARAM_BOUNDS = {
    "length": (0.25, 1.5),       # half-length of the pole (m)
    "masspole": (0.05, 0.5),     # pole mass (kg)
    "masscart": (0.5, 2.0),      # cart mass (kg)
    "force_mag": (4.0, 16.0),    # magnitude of the applied push (N)
    "init_range": (0.05, 0.3),   # |initial state| ~ U(-init_range, init_range)
}
MAX_EPISODE_STEPS = 500
OBS_DIM = 4
ACTION_TYPE = "discrete"
ACTION_DIM = 2


class ParamCartPoleEnv(CartPoleEnv):
    def set_task(self, params: dict) -> None:
        self.length = float(params["length"])
        self.masspole = float(params["masspole"])
        self.masscart = float(params["masscart"])
        self.force_mag = float(params["force_mag"])
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self._init_range = float(params["init_range"])

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        low = -getattr(self, "_init_range", 0.05)
        high = getattr(self, "_init_range", 0.05)
        options.setdefault("low", low)
        options.setdefault("high", high)
        return super().reset(seed=seed, options=options)


def make_env(params: dict) -> ParamCartPoleEnv:
    env = ParamCartPoleEnv()
    env.set_task(params)
    return env


def normalize(params: dict) -> np.ndarray:
    vec = []
    for name, (lo, hi) in PARAM_BOUNDS.items():
        vec.append((params[name] - lo) / (hi - lo))
    return np.array(vec, dtype=np.float64)
