"""The environments. Each is a module with the same contract:

  PARAM_BOUNDS, PARAM_ORDER   the task box Scenic/VerifAI sample (also in <env>.scenic)
  SCENIC_FILE                 that Scenic file
  MAX_EPISODE_STEPS, OBS_DIM, ACTION_DIM
  GOAL                        "survive" (pass = never terminate) or "reach" (pass = terminate)
  make_env(params)            the gymnasium env used for training
  sample_starts, step, observe, terminated
                              batched exam physics, checked against make_env (tests/test_grader.py)
  planning_cost(params)       ranking heuristic for the winnability certificate (exam/certify.py)

Each environment keeps its exam in frozen_sets/<env>/ and its results in results/<env>/.
"""
from __future__ import annotations

import importlib
import os

NAMES = ("cartpole", "acrobot", "mountaincar")


def get(name: str):
    if name not in NAMES:
        raise ValueError(f"unknown environment {name!r}; choose from {NAMES}")
    return importlib.import_module(f"acl_bench.envs.{name}")


def exam_dir(name: str) -> str:
    return os.path.join("frozen_sets", name)


def search_dir(name: str) -> str:
    return os.path.join("frozen_sets", name, "search")


def results_dir(name: str) -> str:
    return os.path.join("results", name)
