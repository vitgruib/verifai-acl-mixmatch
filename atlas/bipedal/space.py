"""Falsification space over DCD BipedalWalker levels (BipedalWalker-Adversarial-v0, mode 'full':
8 parameters plus a terrain seed; DCD's level encoding is [8 floats..., int seed], see
BipedalWalkerAdversarialEnv.reset_to_level).

  dr   the training box: the 8 parameters uniform over PARAM_RANGES_FULL (DR's reset_random,
       minimax's step_adversary range, PLR/ACCEL's candidate generator), seed uniform over
       [0, 2^32) like rand_int_seed. ACCEL's edits stay inside this box (clipped mutations).

Parameters: roughness [0, 10]; pit gap a, b [0, 10]; stump height a, b [0, 5]; stair height a, b
[0, 5]; stair steps [1, 9]. The env sorts each pair into a (lo, hi) range and draws obstacle
sizes uniformly from it; an obstacle type is disabled when its hi is below a threshold (pits
0.8, stumps 0.2, stairs 0.2; get_config). A level is deterministic: the terrain and the walker's
initial push come from (parameters, seed).

Descriptors are ints (binned //4 by the Recorder): the parameters in half-terrain-step units
for the obstacle heights, 0 when the obstacle type is disabled, plus two features of the
generated terrain (obstacle bodies and height range, from a reset of a bare env).
"""
from __future__ import annotations

import os
import sys

import numpy as np

NAMES = ("roughness", "pit_a", "pit_b", "stump_a", "stump_b", "stair_a", "stair_b", "stair_steps")
RANGES = ((0, 10), (0, 10), (0, 10), (0, 5), (0, 5), (0, 5), (0, 5), (1, 9))   # PARAM_RANGES_FULL
SEED_HI = 2 ** 32
TERRAIN_STEP = 14 / 30.0
TERRAIN_LENGTH = 200

SPACES = {"dr": {**{k: (float(lo), float(hi)) for k, (lo, hi) in zip(NAMES, RANGES)},
                 "seed": (0.0, float(SEED_HI))}}


def build(space: str, p: dict):
    """Point -> (level [8 floats, int seed], invalid_reason or None)."""
    if space != "dr":
        raise ValueError(f"unknown space {space!r}; choose from {tuple(SPACES)}")
    vec = [float(np.clip(p[k], lo, hi)) for k, (lo, hi) in zip(NAMES, RANGES)]
    return vec + [int(min(max(np.floor(p["seed"]), 0), SEED_HI - 1))], None


def active(level) -> dict:
    """Which obstacle types the env generates (adversarial.get_config thresholds)."""
    v = level
    return {"pit": max(v[1], v[2]) >= 0.8, "stump": max(v[3], v[4]) >= 0.2,
            "stair": max(v[5], v[6]) >= 0.2}


_ENV = None


def bare_env():
    """A bare BipedalWalkerFull with rendering stubbed (DCD_DIR on sys.path)."""
    d = os.environ.get("DCD_DIR")
    if d and d not in sys.path:
        sys.path.insert(0, d)
    from envs.bipedalwalker.adversarial import BipedalWalkerFull
    env = BipedalWalkerFull()
    env.render = lambda *a, **k: None
    return env


def descriptors(level) -> dict:
    global _ENV
    if _ENV is None:
        _ENV = bare_env()
    _ENV.reset_to_level(list(level))
    y = np.array(_ENV.terrain_y) / TERRAIN_STEP
    on = active(level)
    return {"roughness": int(round(level[0])),
            "pit_hi": int(round(max(level[1:3]))) if on["pit"] else 0,
            "stump_hi2": int(round(2 * max(level[3:5]))) if on["stump"] else 0,
            "stair_hi2": int(round(2 * max(level[5:7]))) if on["stair"] else 0,
            "stair_steps": int(round(level[7])) if on["stair"] else 0,
            "obstacle_bodies": len(_ENV.terrain) - (TERRAIN_LENGTH - 1),
            "height_range": int(y.max() - y.min())}


DESC_KEYS = ("roughness", "pit_hi", "stump_hi2", "stair_hi2", "stair_steps", "obstacle_bodies",
             "height_range")
