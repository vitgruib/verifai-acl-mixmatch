"""Prioritized-replay curriculum: the ACL half of the mix-and-match grid,
now with the sampler feedback and the replay-priority score unified into one
function (see the README's "Is the feedback function shared?" section).

Two independent things can each be switched off, which is how the grid's
ablations are expressed:

  - `potential_fn=None` ("none" in the grid) turns ACL off entirely: no
    replay buffer (replay_prob forced to 0, so every episode is a fresh
    domain-randomization draw) and no learning-potential score -- the
    sampler still needs *some* scalar per VerifAI's API, so it falls back to
    raw (normalized) episode return. This isolates "sampler alone."
  - Using `sampler="random"` or `"halton"` (acl_bench.scenic_sampling) is
    "not using verifAI's adaptive sampling": both ignore whatever rho they
    are given. Combined with `potential_fn=None` that's the pure baseline
    (plain domain randomization, no curriculum sophistication at all);
    combined with a real potential_fn it's "ACL alone."

Everything else -- rank-based replay (`P(i) ~ 1/rank_i**0.9`, Prioritized
Level Replay, Jiang et al., 2021) and EMA-smoothing the raw per-visit score
into a slot's running priority -- is unchanged from SIPACL's
`MetaDriveEnv._compute_learning_progress`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from acl_bench.scenic_sampling import ScenicTaskSampler

NEW = "new"
REPLAY = "replay"

_NEW_TASK_LP = 1e10   # placeholder so brand-new slots always look "worth trying"
_RANK_ALPHA = 0.9     # PLR's rank-sampling temperature
_LP_EMA_BETA = 0.5    # EMA smoothing of the raw per-visit score into the slot


@dataclass
class _Slot:
    params: dict
    lp: float = _NEW_TASK_LP
    state: dict = field(default_factory=dict)  # scratch memory for potential_fn


class RunningNormalizer:
    """Welford online mean/std, turning an unbounded, scale-varying-by-env
    potential-function score into a roughly [-1, 1] falsification-style rho:
    the same z-scoring works whether the underlying scores come from
    CartPole's O(100s) returns or Pendulum's O(1000s)-magnitude penalties,
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
                 potential_fn: Optional[Callable], gamma: float, gae_lambda: float,
                 replay_prob: float = 0.5, buffer_max: int = 200,
                 rng: Optional[np.random.Generator] = None):
        self.task_sampler = task_sampler
        self.param_names = param_names
        self.potential_fn = potential_fn  # None => "no ACL" ablation
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.replay_prob = replay_prob if potential_fn is not None else 0.0
        self.buffer_max = buffer_max
        self.rng = rng or np.random.default_rng()
        self.slots: list[_Slot] = []
        self.rho_norm = RunningNormalizer()

    def _replay_probs(self) -> np.ndarray:
        lp = np.array([s.lp for s in self.slots], dtype=np.float64)
        ranks = lp.argsort()[::-1].argsort() + 1  # rank 1 = highest lp
        weights = 1.0 / (ranks.astype(np.float64) ** _RANK_ALPHA)
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
        raw score actually used (potential_fn's, or raw return under the
        "none" ablation) for logging."""
        slot = self.slots[idx]
        if self.potential_fn is not None:
            raw_score, slot.state = self.potential_fn(
                rewards, values, next_value, self.gamma, self.gae_lambda, slot.state,
            )
            slot.lp = raw_score if slot.lp >= 0.5 * _NEW_TASK_LP else (
                _LP_EMA_BETA * raw_score + (1 - _LP_EMA_BETA) * slot.lp
            )
        else:
            raw_score = float(np.sum(rewards))

        if mode == NEW:
            self.rho_norm.update(raw_score)
            self.task_sampler.give_feedback(self.rho_norm.to_rho(raw_score))
        return raw_score

    @property
    def buffer_lp(self) -> np.ndarray:
        return np.array([s.lp for s in self.slots], dtype=np.float64)
