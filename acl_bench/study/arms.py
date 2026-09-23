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

# From the calibration run (20 plain runs to 1.2M steps): every run had taken off by 307k
# steps and the mean success plateaus around 490k-610k, so 600 rollouts of 1024 steps.
DEFAULT_BUDGET = 614_400
CHECKPOINT_EVERY = 20 * 1024          # a multiple of the 1024-step rollout, so checkpoints are exact
SCORE = "pvl_gae"                     # SIPACL's own score


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
