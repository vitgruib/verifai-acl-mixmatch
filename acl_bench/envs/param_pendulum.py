"""Pendulum-v1 (continuous torque control) with its dynamics exposed as a
Scenic/VerifAI-samplable task. This is the one continuous-action environment
in the grid, alongside CartPole and Acrobot's discrete actions.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv

PARAM_BOUNDS = {
    "g": (5.0, 15.0),            # gravity
    "m": (0.5, 2.0),             # pendulum mass (kg)
    "l": (0.5, 2.0),             # pendulum length (m)
    "max_torque": (1.0, 4.0),    # actuator limit (N.m)
    "max_speed": (4.0, 12.0),    # angular velocity clamp (rad/s)
}
MAX_EPISODE_STEPS = 200
OBS_DIM = 3
ACTION_TYPE = "continuous"
ACTION_DIM = 1


class ParamPendulumEnv(PendulumEnv):
    def set_task(self, params: dict) -> None:
        self.g = float(params["g"])
        self.m = float(params["m"])
        self.l = float(params["l"])
        self.max_torque = float(params["max_torque"])
        self.max_speed = float(params["max_speed"])
        # max_torque/max_speed are baked into the cached action/observation
        # spaces at __init__ time; refresh them so a policy reading
        # env.action_space bounds (to rescale a tanh output) sees this task's
        # actual limits, not PendulumEnv's fixed defaults.
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)


def make_env(params: dict) -> ParamPendulumEnv:
    env = ParamPendulumEnv()
    env.set_task(params)
    return env


def normalize(params: dict) -> np.ndarray:
    vec = []
    for name, (lo, hi) in PARAM_BOUNDS.items():
        vec.append((params[name] - lo) / (hi - lo))
    return np.array(vec, dtype=np.float64)
