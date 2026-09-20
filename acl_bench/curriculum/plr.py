"""Prioritized-replay curriculum: the ACL half of the mix-and-match grid.

Ports the buffer logic in SIPACL/Work/custom/custom_gym.py (`MetaDriveEnv`)
almost line-for-line, but keeps everything in memory (no scene pickling to
disk -- CartPole tasks are five floats, not a MetaDrive world) and makes the
scoring function ("learning potential") and the new-task proposal mechanism
("sampler") independently pluggable, which is the actual point of this repo:

  - `sampler`  answers "what NEW task should we try?" -- one of VerifAI's
    random / Halton / multi-armed-bandit samplers over the CartPole task
    space (acl_bench/samplers.py).
  - `potential_fn` answers "how worth REPLAYING is a task we've already
    seen?" -- one of the functions in acl_bench/potential/functions.py.

Each episode, `pick_task` either draws a fresh task from `sampler` (which
also gets `update()`-d with a falsification-style `rho`, exactly as VerifAI
expects) or replays a buffered task, chosen by rank-based sampling on its
learning-potential score (Prioritized Level Replay, Jiang et al., 2021):
P(i) ∝ 1/rank_i**alpha, rank 1 = highest score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from acl_bench.envs.param_cartpole import CartPoleParams
from acl_bench.samplers import point_to_params

NEW = "new"
REPLAY = "replay"

_NEW_TASK_LP = 1e10  # placeholder so brand-new slots always look "worth trying"
_RANK_ALPHA = 0.9    # PLR's rank-sampling temperature
_LP_EMA_BETA = 0.5   # EMA smoothing of the raw per-visit score into the slot


@dataclass
class _Slot:
    params: CartPoleParams
    sampler_info: object
    lp: float = _NEW_TASK_LP
    state: dict = field(default_factory=dict)  # scratch memory for potential_fn


class PLRCurriculum:
    def __init__(self, sampler, potential_fn: Callable, gamma: float,
                 gae_lambda: float, replay_prob: float = 0.5,
                 buffer_max: int = 200, max_return: float = 500.0,
                 rng: Optional[np.random.Generator] = None):
        self.sampler = sampler
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.replay_prob = replay_prob
        self.buffer_max = buffer_max
        self.max_return = max_return
        self.rng = rng or np.random.default_rng()
        self.slots: list[_Slot] = []

    def _replay_probs(self) -> np.ndarray:
        lp = np.array([s.lp for s in self.slots], dtype=np.float64)
        ranks = lp.argsort()[::-1].argsort() + 1  # rank 1 = highest lp
        weights = 1.0 / (ranks.astype(np.float64) ** _RANK_ALPHA)
        return weights / weights.sum()

    def pick_task(self) -> tuple[CartPoleParams, int, str]:
        if self.slots and self.rng.uniform() < self.replay_prob:
            probs = self._replay_probs()
            idx = int(self.rng.choice(len(self.slots), p=probs))
            return self.slots[idx].params, idx, REPLAY

        sample, info = self.sampler.getSample()
        params = point_to_params(sample)
        if len(self.slots) >= self.buffer_max:
            self.slots.pop(0)  # FIFO eviction, mirrors SIPACL's _evict_oldest_if_full
        self.slots.append(_Slot(params=params, sampler_info=info))
        return params, len(self.slots) - 1, NEW

    def report_episode(self, idx: int, mode: str, rewards: list[float],
                        values: list[float], next_value: float) -> float:
        """Score the just-finished episode and update both the sampler and
        the replay buffer. Returns the raw (pre-EMA) potential score."""
        slot = self.slots[idx]
        raw_score, slot.state = self.potential_fn(
            rewards, values, next_value, self.gamma, self.gae_lambda, slot.state,
        )
        slot.lp = raw_score if slot.lp >= 0.5 * _NEW_TASK_LP else (
            _LP_EMA_BETA * raw_score + (1 - _LP_EMA_BETA) * slot.lp
        )

        if mode == NEW:
            episode_return = float(np.sum(rewards))
            rho = (episode_return / self.max_return) * 2.0 - 1.0  # in [-1, 1]
            self.sampler.update(None, slot.sampler_info, rho)
        return raw_score

    @property
    def buffer_lp(self) -> np.ndarray:
        return np.array([s.lp for s in self.slots], dtype=np.float64)
