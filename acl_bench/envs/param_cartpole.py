"""CartPole-v1 with physical parameters exposed as a VerifAI-samplable task.

Mirrors the role Scenic/MetaDrive plays in SIPACL (github.com/vitgruib/SIPACL):
there, a Scenic program samples a driving *scene* (traffic, geometry) that the
policy is trained/evaluated on. Here, a VerifAI FeatureSpace samples a
*physics configuration* for CartPole that plays the same role, at a fraction
of the compute cost, so many sampler x learning-potential combinations can
actually be run to completion in one sitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


@dataclass(frozen=True)
class CartPoleParams:
    """One point in the task space a sampler can propose."""

    length: float = 0.5          # half-length of the pole (m)
    masspole: float = 0.1        # pole mass (kg)
    masscart: float = 1.0        # cart mass (kg)
    force_mag: float = 10.0      # magnitude of the applied push (N)
    init_range: float = 0.05     # |initial state| ~ U(-init_range, init_range)

    @staticmethod
    def bounds() -> dict:
        """(low, high) for each field, used to build the VerifAI FeatureSpace."""
        return {
            "length": (0.25, 1.5),
            "masspole": (0.05, 0.5),
            "masscart": (0.5, 2.0),
            "force_mag": (4.0, 16.0),
            "init_range": (0.05, 0.3),
        }

    @staticmethod
    def default() -> "CartPoleParams":
        return CartPoleParams()


class ParamCartPoleEnv(CartPoleEnv):
    """CartPoleEnv whose physics + initial-state distribution are set per-episode.

    Larger `length`, larger `masspole`, smaller `force_mag`, and larger
    `init_range` all make balancing harder -- these five dimensions give
    samplers/curricula a genuine easy<->hard task manifold to explore,
    analogous to weather/traffic-density knobs in a Scenic scenario.
    """

    def set_task(self, params: CartPoleParams) -> None:
        self.length = float(params.length)
        self.masspole = float(params.masspole)
        self.masscart = float(params.masscart)
        self.force_mag = float(params.force_mag)
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self._init_range = float(params.init_range)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        low = -getattr(self, "_init_range", 0.05)
        high = getattr(self, "_init_range", 0.05)
        options.setdefault("low", low)
        options.setdefault("high", high)
        return super().reset(seed=seed, options=options)


def make_env(params: CartPoleParams | None = None) -> ParamCartPoleEnv:
    env = ParamCartPoleEnv()
    env.set_task(params or CartPoleParams.default())
    return env


def normalize(params: CartPoleParams) -> np.ndarray:
    """Map params into [0, 1]^5 for difficulty-heuristic / plotting use."""
    b = CartPoleParams.bounds()
    vec = []
    for name in ("length", "masspole", "masscart", "force_mag", "init_range"):
        lo, hi = b[name]
        v = getattr(params, name)
        vec.append((v - lo) / (hi - lo))
    return np.array(vec, dtype=np.float64)
