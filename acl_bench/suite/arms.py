"""The suite's arms (pre-registered settings) and per-environment training budgets.

PLR uses JaxUED's Maze defaults (replay 0.8, MaxMC, beta 0.3, staleness 0.3) with the classic
tasks' buffer of 1000. Robust PLR adds no-gradient-on-new-levels; ACCEL adds level edits with
the existing `accel` arm's mutation size. SFL is the existing SFL arm with its scouting charged
to the budget, so every arm spends the same environment steps. REF is DR on separate seeds,
the reference agents the exam's VerifAI section is built from.
"""
from __future__ import annotations

import dataclasses

from acl_bench.plr.levels import LevelConfig
from acl_bench.plr.screen import DR, SFL

_PLR = LevelConfig(replay_prob=0.8, score="maxmc", beta=0.3, staleness=0.3, buffer=1000)
_RPLR = dataclasses.replace(_PLR, robust=True)

ARMS: dict[str, LevelConfig] = {
    "DR": LevelConfig(**DR),
    "PLR": _PLR,
    "RPLR": _RPLR,
    "ACCEL": dataclasses.replace(_RPLR, accel=True),
    "SFL": LevelConfig(**SFL, charge_scouting=True),
    "REF": LevelConfig(**DR),
}

BUDGET: dict[str, int] = {}     # env steps per run, set from the calibration runs
