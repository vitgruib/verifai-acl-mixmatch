"""The study's arms (docs/ablation.md): the 2x2 component ablation, run with every
setting at its standard value, plus the reference agents that define the exam's
searched sections.

| arm       | sampler           | ACL (replay) | score      |
|-----------|-------------------|--------------|------------|
| N         | random            | off          | neg_return (consumed by nothing) |
| A         | random            | on           | pvl_gae    |
| S_<x>     | ce / mab / sa     | off          | pvl_gae    |
| B_<x>     | ce / mab / sa     | on           | pvl_gae    |
| REF       | random            | off          | neg_return (same config as N, independent seeds) |
"""
from __future__ import annotations

from dataclasses import dataclass

from acl_bench.sampling import ADAPTIVE_SAMPLERS

# Training length per environment, from its calibration runs (docs/exam.md): a multiple
# of 30 rollouts of 1024 steps, so the 30 check-ins (every BUDGET / 30 steps) are exact.
# cartpole: every run had taken off by 307k steps and mean success plateaus around
# 490k-610k (20 plain runs to 1.2M), so 600 rollouts.
# acrobot: every run took off by 41k steps; mean success plateaus near 0.93 from ~330k
# and mean steps to the goal from ~490k (10 plain runs to 1.2M), so 480 rollouts.
# mountaincar: every run took off by 164k and all ten froze by 246k at the same policy,
# always push right (10 plain runs to 1.2M), so 300 rollouts.
BUDGET = {"cartpole": 614_400, "acrobot": 491_520, "mountaincar": 307_200}
N_CHECKINS = 30
SCORE = "pvl_gae"                     # SIPACL's own score


def checkpoint_every(budget: int) -> int:
    return budget // N_CHECKINS


@dataclass(frozen=True)
class Arm:
    name: str
    sampler: str
    acl: bool
    score: str


def _build() -> dict[str, Arm]:
    arms = [Arm("REF", "random", False, "neg_return"),
            Arm("N", "random", False, "neg_return"),
            Arm("A", "random", True, SCORE)]
    for s in ADAPTIVE_SAMPLERS:
        arms += [Arm(f"S_{s}", s, False, SCORE), Arm(f"B_{s}", s, True, SCORE)]
    return {a.name: a for a in arms}


ARMS = _build()
MAIN_ARMS = ("N", "A") + tuple(f"S_{s}" for s in ADAPTIVE_SAMPLERS) + tuple(f"B_{s}" for s in ADAPTIVE_SAMPLERS)
