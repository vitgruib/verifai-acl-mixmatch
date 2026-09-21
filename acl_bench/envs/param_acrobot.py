"""Acrobot-v1 (chaotic double pendulum, discrete torque) with its link
dynamics exposed as a Scenic/VerifAI-samplable task. This is the "complex
feature space" environment in the grid: 6 continuous dimensions versus
CartPole's 5, and genuinely chaotic dynamics for extreme
parameter values, while remaining a classic_control env (no Box2D).
"""
from __future__ import annotations

import numpy as np
from gymnasium.envs.classic_control.acrobot import AcrobotEnv

PARAM_BOUNDS = {
    "link_length_1": (0.5, 1.5),
    "link_length_2": (0.5, 1.5),
    "link_mass_1": (0.5, 1.5),
    "link_mass_2": (0.5, 1.5),
    "link_moi": (0.5, 1.5),
    "torque_noise_max": (0.0, 0.3),
}
MAX_EPISODE_STEPS = 500
OBS_DIM = 6
ACTION_TYPE = "discrete"
ACTION_DIM = 3


class ParamAcrobotEnv(AcrobotEnv):
    def set_task(self, params: dict) -> None:
        self.LINK_LENGTH_1 = float(params["link_length_1"])
        self.LINK_LENGTH_2 = float(params["link_length_2"])
        self.LINK_MASS_1 = float(params["link_mass_1"])
        self.LINK_MASS_2 = float(params["link_mass_2"])
        self.LINK_MOI = float(params["link_moi"])
        self.torque_noise_max = float(params["torque_noise_max"])
        # Centers of mass follow the upstream convention (link midpoint);
        # sampling them independently risks nonphysical (com > length) states.
        self.LINK_COM_POS_1 = 0.5 * self.LINK_LENGTH_1
        self.LINK_COM_POS_2 = 0.5 * self.LINK_LENGTH_2


def make_env(params: dict) -> ParamAcrobotEnv:
    env = ParamAcrobotEnv()
    env.set_task(params)
    return env


def normalize(params: dict) -> np.ndarray:
    vec = []
    for name, (lo, hi) in PARAM_BOUNDS.items():
        vec.append((params[name] - lo) / (hi - lo))
    return np.array(vec, dtype=np.float64)
