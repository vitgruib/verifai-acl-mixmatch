"""ACL: prioritized replay of already-seen tasks, and its coupling to the sampler.

Ports the replay logic in SIPACL/Work/custom/custom_gym.py (`MetaDriveEnv`):
rank-based replay `P(i) ~ 1/rank_i**alpha` with SIPACL's constants (alpha=1.0, EMA
beta=0.2, buffer 5000, FIFO eviction), a 1e10 placeholder so unscored tasks rank
first, and a first visit that skips the EMA. Differences from SIPACL are listed in
docs/sipacl.md; the main one is that a replay repeats the task's *parameters* (the
episode still starts from a fresh random state), not a saved scene.

ACL on/off (`use_replay`) is independent of the sampler: off means every episode is a
fresh draw from the sampler (SIPACL's `replay_resample_prob=-1`).

The task score (acl_bench.scoring) has up to two consumers: it ranks tasks for replay
(ACL on), and after every NEW draw -- never after a replay -- it becomes the sampler's
feedback: z-scored against all feedback so far in the run, clipped to +-3, divided by
3 and negated, so `rho` is in [-1, 1] and a high score reads as a VerifAI
"counterexample" (low rho = worth sampling more of). SIPACL never steered a sampler
(its scenarios use Halton, which ignores feedback), so this mapping is this repo's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from acl_bench.sampling import ScenicTaskSampler

NEW = "new"
REPLAY = "replay"

_NEW_TASK_LP = 1e10   # placeholder so brand-new slots always look "worth trying"
_RANK_ALPHA = 1.0     # SIPACL DEFAULT_LP_RANK_ALPHA
_LP_EMA_BETA = 0.2    # SIPACL DEFAULT_LP_EMA_BETA


@dataclass
class _Slot:
    params: dict
    lp: float = _NEW_TASK_LP


class RunningNormalizer:
    """Welford online mean/std, turning an unbounded task score into a [-1, 1]
    falsification-style rho with no scale constant to hand-tune."""

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
        # Higher score = more worth sampling more of that region =
        # should read as a "counterexample" to VerifAI's samplers, i.e. LOW rho.
        z = (x - self.mean) / (self.std + 1e-8)
        z = max(-3.0, min(3.0, z))
        return -z / 3.0  # in [-1, 1]


class PLRCurriculum:
    def __init__(self, task_sampler: ScenicTaskSampler, param_names: tuple[str, ...],
                 score_fn: Callable, gamma: float, gae_lambda: float,
                 use_replay: bool = True, replay_prob: float = 0.5, buffer_max: int = 5000,
                 rank_alpha: float = _RANK_ALPHA, ema_beta: float = _LP_EMA_BETA,
                 rng: Optional[np.random.Generator] = None):
        self.rank_alpha = rank_alpha
        self.ema_beta = ema_beta
        self.task_sampler = task_sampler
        self.param_names = param_names
        self.score_fn = score_fn
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
        weights = 1.0 / ranks ** self.rank_alpha
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
                        values: list[float], next_value: float) -> None:
        """Score the just-finished episode, update the replay buffer, and (for
        NEW draws) queue feedback for the sampler's *next* draw."""
        slot = self.slots[idx]
        raw_score = self.score_fn(rewards, values, next_value, self.gamma, self.gae_lambda)
        slot.lp = raw_score if slot.lp >= 0.5 * _NEW_TASK_LP else (
            self.ema_beta * raw_score + (1 - self.ema_beta) * slot.lp
        )

        if mode == NEW:
            self.rho_norm.update(raw_score)
            self.task_sampler.give_feedback(self.rho_norm.to_rho(raw_score))
