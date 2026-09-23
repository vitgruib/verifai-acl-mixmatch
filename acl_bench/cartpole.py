"""CartPole with its physical parameters exposed as a task that Scenic/VerifAI samples.

A task is five scalars (`PARAM_ORDER`); `cartpole.scenic` declares the same box as
`VerifaiRange` parameters so the samplers in `acl_bench.sampling` can draw from it.
Episodes are capped at `MAX_EPISODE_STEPS` (the raw gymnasium class has no cap).
"""
from __future__ import annotations

import os

from gymnasium.envs.classic_control.cartpole import CartPoleEnv

PARAM_BOUNDS = {
    "length": (0.25, 1.5),       # half-length of the pole (m)
    "masspole": (0.05, 0.5),     # pole mass (kg)
    "masscart": (0.5, 2.0),      # cart mass (kg)
    "force_mag": (4.0, 16.0),    # magnitude of the applied push (N)
    "init_range": (0.05, 0.3),   # |initial state| ~ U(-init_range, init_range)
}
PARAM_ORDER = tuple(PARAM_BOUNDS)
MAX_EPISODE_STEPS = 500
OBS_DIM, ACTION_DIM = 4, 2
SCENIC_FILE = os.path.join(os.path.dirname(__file__), "cartpole.scenic")


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
        options.setdefault("low", -self._init_range)
        options.setdefault("high", self._init_range)
        return super().reset(seed=seed, options=options)


def make_env(params: dict) -> ParamCartPoleEnv:
    env = ParamCartPoleEnv()
    env.set_task(params)
    return env
