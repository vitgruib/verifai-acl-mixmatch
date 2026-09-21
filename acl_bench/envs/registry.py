"""Central registry: one entry per environment under test, tying together
the gym env factory, its task-parameter bounds, and the .scenic file that
exposes those bounds as VerifaiRange parameters."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from acl_bench.envs import param_acrobot, param_cartpole

_SCENIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scenic_scenarios")


@dataclass(frozen=True)
class EnvSpec:
    name: str
    make_env: Callable[[dict], object]
    normalize: Callable[[dict], object]
    param_bounds: dict
    max_episode_steps: int
    success_return: float  # episode return counted as "solved" (used by intermediate_difficulty)
    obs_dim: int
    action_type: str  # "discrete" | "continuous"
    action_dim: int
    scenic_file: str


ENV_SPECS: dict[str, EnvSpec] = {
    "cartpole": EnvSpec(
        name="cartpole", make_env=param_cartpole.make_env, normalize=param_cartpole.normalize,
        param_bounds=param_cartpole.PARAM_BOUNDS, max_episode_steps=param_cartpole.MAX_EPISODE_STEPS,
        obs_dim=param_cartpole.OBS_DIM, action_type=param_cartpole.ACTION_TYPE,
        action_dim=param_cartpole.ACTION_DIM,
        scenic_file=os.path.join(_SCENIC_DIR, "cartpole.scenic"),
        success_return=195.0,   # survive >= 195 of 500 steps
    ),
    "acrobot": EnvSpec(
        name="acrobot", make_env=param_acrobot.make_env, normalize=param_acrobot.normalize,
        param_bounds=param_acrobot.PARAM_BOUNDS, max_episode_steps=param_acrobot.MAX_EPISODE_STEPS,
        obs_dim=param_acrobot.OBS_DIM, action_type=param_acrobot.ACTION_TYPE,
        action_dim=param_acrobot.ACTION_DIM,
        scenic_file=os.path.join(_SCENIC_DIR, "acrobot.scenic"),
        success_return=-499.0,  # any return above the -500 step cap means the goal was reached
    ),
}

ENV_NAMES = tuple(ENV_SPECS.keys())
