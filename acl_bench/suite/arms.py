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

# Env steps per run. CartPole keeps the earlier study's calibrated length (acl_bench/study/arms.py).
# LunarLander and FrozenLake: 3 DR seeds with 10-20 check-ins on a 100-question random section; the
# budget is the first check-in after which the mean pass rate rises less than 0.02 over the next two.
# LunarLander (to 20.48M): 0.36 at 8.2M, 0.63 at 10.2M, 0.69 at 11.26M, then 0.68, 0.61; 0.75 at 20.48M.
# FrozenLake (to 2.05M): 0.51, 0.56, 0.63 at 614k, then 0.65, 0.63; 0.67 at 2.05M.
# LunarLander is doubled from that rule's 11.26M (batch 21, amendment 3): at 11.26M, 6 of 10 REF
# agents never learned the task.
BUDGET: dict[str, int] = {"cartpole": 614_400, "lunarlander": 22_528_000, "frozenlake": 614_400}

# SFL's scouting per env, sized so that charged scouting takes about half of the budget
# (mean scouted_steps / budget in 0.4-0.6 over 2 seeds at the full budget). Procedure: keep
# sfl_k = 8 and sfl_every = 10, set sfl_top = max(10, sfl_n // 10) (the default's ratio) and
# change sfl_n; if sfl_n would fall below 20, fix it at 20 and raise sfl_every instead.
# Measure with an exam holding one small random section (train reports scouted_steps per row):
#   python -m acl_bench.suite.run --env <env> --arms SFL --seeds 1-2 --steps <BUDGET[env]> \
#       --checks 1 --workers 2 --allow-battery --out <scratch> --exam <dir with the section>
# CartPole (614_400 steps): sfl_n 20 at sfl_every 10 scouted 0.74; sfl_every 25/30/35/40 gave
# 0.58/0.54/0.48/0.51 (2-seed means, noise about 0.03); 40 kept.
# LunarLander (measured at 11_264_000 steps, kept at the doubled budget): sfl_n 100/50/20 at
# sfl_every 10 scouted 0.97/0.95/0.88 (long episodes); sfl_n 20 at sfl_every 60/70/80 gave
# 0.52/0.48/0.42; 60 kept.
# FrozenLake (614_400 steps): sfl_n 20/40/100 at sfl_every 10 scouted 0.16/0.26/0.48; 100 kept.
# An env with no entry has no SFL arm: arm_config raises rather than fall back to LevelConfig's
# defaults (sfl_n 1000, sfl_every 10), which spend nearly the whole budget scouting.
SFL_SCOUT: dict[str, dict[str, int]] = {
    "cartpole": {"sfl_n": 20, "sfl_top": 10, "sfl_every": 40},
    "lunarlander": {"sfl_n": 20, "sfl_top": 10, "sfl_every": 60},
    "frozenlake": {"sfl_n": 100, "sfl_top": 10, "sfl_every": 10},
}


def arm_config(arm: str, env: str) -> LevelConfig:
    """The arm's settings on `env`: SFL takes that env's scouting size."""
    if arm != "SFL":
        return ARMS[arm]
    if env not in SFL_SCOUT:
        raise KeyError(f"SFL has no SFL_SCOUT entry for {env}")
    return dataclasses.replace(ARMS["SFL"], **SFL_SCOUT[env])
