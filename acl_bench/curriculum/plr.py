"""Prioritized-replay curriculum (ACL) and its coupling to the VerifAI sampler.

Ports the PLR logic in SIPACL/Work/custom/custom_gym.py (`MetaDriveEnv`):
rank-based replay `P(i) ~ 1/rank_i**alpha` with SIPACL's own constants
(alpha=1.0, EMA beta=0.2, buffer 5000), a 1e10 placeholder so brand-new slots
rank first, and first-visit scores that skip the EMA. Deliberate differences:
the buffer is in memory rather than pickled Scenic scenes on disk, and see
`acl_bench.potential.functions` on truncation bootstrapping.

Two independent switches, which the grid crosses (see experiment.py):

  - ACL (`use_replay`): whether tasks are ever replayed. Off = every episode
    is a fresh draw from the sampler (SIPACL's `replay_resample_prob=-1`).
  - The sampler (acl_bench.scenic_sampling): `random`/`halton` ignore feedback;
    `ce`/`mab`/`sa` steer by it.

The scoring function (`potential_fn`) is the third factor and is always
computed. It has up to two consumers: it ranks tasks for replay when ACL is
on, and -- z-scored and negated into `rho` -- it is the feedback the sampler
receives after each NEW draw (VerifAI's convention: low rho = "counterexample",
worth more samples; every scoring function's high score = worth revisiting, so
rho = -z(score)). SIPACL itself keeps these separate: its Scenic feedback is
`feedback_fn(simulation.result)`, an identity function that ppo.py never
overrides, while PVL only drives replay. Sharing one score is this repo's
choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from acl_bench.scenic_sampling import ScenicTaskSampler

NEW = "new"
REPLAY = "replay"

_NEW_TASK_LP = 1e10   # placeholder so brand-new slots always look "worth trying"
_RANK_ALPHA = 1.0     # SIPACL DEFAULT_LP_RANK_ALPHA
_LP_EMA_BETA = 0.2    # SIPACL DEFAULT_LP_EMA_BETA


@dataclass
class _Slot:
    params: dict
    lp: float = _NEW_TASK_LP
    state: dict = field(default_factory=dict)  # scratch memory for potential_fn


class RunningNormalizer:
    """Welford online mean/std, turning an unbounded, scale-varying-by-env
    potential-function score into a roughly [-1, 1] falsification-style rho:
    the same z-scoring works whether the underlying scores come from
    CartPole's positive O(100s) returns or Acrobot's negative step-cost returns,
    with no per-env constant to hand-tune."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return (self.m2 / self.n) ** 0.5 if self.n > 1 else 1.0

    def to_rho(self, x: float) -> float:
        # Higher potential/return = more worth sampling more of that region =
        # should read as a "counterexample" to VerifAI's samplers, i.e. LOW rho.
        z = (x - self.mean) / (self.std + 1e-8)
        z = max(-3.0, min(3.0, z))
        return -z / 3.0  # in [-1, 1]


class PLRCurriculum:
    def __init__(self, task_sampler: ScenicTaskSampler, param_names: tuple[str, ...],
                 potential_fn: Callable, gamma: float, gae_lambda: float,
                 use_replay: bool = True, replay_prob: float = 0.5, buffer_max: int = 5000,
                 rng: Optional[np.random.Generator] = None):
        self.task_sampler = task_sampler
        self.param_names = param_names
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.replay_prob = replay_prob if use_replay else 0.0
        self.buffer_max = buffer_max
        self.rng = rng or np.random.default_rng()
        self.slots: list[_Slot] = []
        self.rho_norm = RunningNormalizer()

    def _replay_probs(self) -> np.ndarray:
        lp = np.array([s.lp for s in self.slots], dtype=np.float64)
        order = np.argsort(-lp, kind="stable")  # rank 1 = highest lp; ties -> lower index (as SIPACL)
        ranks = np.empty(len(lp), dtype=np.float64)
        ranks[order] = np.arange(1, len(lp) + 1, dtype=np.float64)
        weights = 1.0 / ranks ** _RANK_ALPHA
        return weights / weights.sum()

    def pick_task(self) -> tuple[dict, int, str]:
        if self.slots and self.rng.uniform() < self.replay_prob:
            probs = self._replay_probs()
            idx = int(self.rng.choice(len(self.slots), p=probs))
            return self.slots[idx].params, idx, REPLAY

        params = self.task_sampler.draw(self.param_names)
        if len(self.slots) >= self.buffer_max:
            self.slots.pop(0)  # FIFO eviction, mirrors SIPACL's _evict_oldest_if_full
        self.slots.append(_Slot(params=params))
        return params, len(self.slots) - 1, NEW

    def report_episode(self, idx: int, mode: str, rewards: list[float],
                        values: list[float], next_value: float) -> float:
        """Score the just-finished episode, update the replay buffer, and (for
        NEW draws) queue feedback for the sampler's *next* draw. Returns the
        raw score for logging."""
        slot = self.slots[idx]
        raw_score, slot.state = self.potential_fn(
            rewards, values, next_value, self.gamma, self.gae_lambda, slot.state,
        )
        slot.lp = raw_score if slot.lp >= 0.5 * _NEW_TASK_LP else (
            _LP_EMA_BETA * raw_score + (1 - _LP_EMA_BETA) * slot.lp
        )

        if mode == NEW:
            self.rho_norm.update(raw_score)
            self.task_sampler.give_feedback(self.rho_norm.to_rho(raw_score))
        return raw_score

    @property
    def buffer_lp(self) -> np.ndarray:
        return np.array([s.lp for s in self.slots], dtype=np.float64)
