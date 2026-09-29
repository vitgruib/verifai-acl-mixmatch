"""A learnability-steered VerifAI picker (docs/wrapper_methodology.md, candidate 1): new
training tasks come from a VerifAI sampler (cross-entropy or bandit, through Scenic, as in
acl_bench.sampling), mixed with uniform draws, and the sampler is steered by learnability
instead of the task-blind critic's PVL.

Learnability is estimated from ordinary training episodes only (no extra simulation): a
running pass rate per bucket of each task parameter (5 buckets per dimension, VerifAI's
own bucketing, with exponential forgetting so it follows the improving policy), pooled
over every episode. A task's learnability is the mean over dimensions of p(1 - p) of the
buckets it falls in. Its feedback to the sampler is rho = -clip(z, -3, 3) / 3, z = the
learnability z-scored against the run's feedback so far: above-average learnability
reads as a VerifAI counterexample, a region to sample more of (acl_bench.acl's mapping).

Each parallel environment slot has its own sampler, so every VerifAI draw is followed by
the feedback for that same draw, as VerifAI's propose / evaluate / feed back loop expects.
"""
from __future__ import annotations

import random

import numpy as np

from acl_bench.acl import RunningNormalizer


class BucketLearnability:
    def __init__(self, bounds: np.ndarray, buckets: int = 5, decay: float = 0.999, prior: float = 1.0):
        self.lo, self.span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
        self.buckets, self.decay = buckets, decay
        self.wins = np.full((len(bounds), buckets), 0.5 * prior)
        self.tries = np.full((len(bounds), buckets), prior)

    def _index(self, params: np.ndarray) -> np.ndarray:
        return np.clip(((params - self.lo) / self.span * self.buckets).astype(int), 0, self.buckets - 1)

    def update(self, params: np.ndarray, success: bool) -> None:
        self.wins *= self.decay
        self.tries *= self.decay
        idx = self._index(params)
        dims = np.arange(len(idx))
        self.wins[dims, idx] += float(success)
        self.tries[dims, idx] += 1.0

    def learnability(self, params: np.ndarray) -> float:
        idx = self._index(params)
        p = self.wins[np.arange(len(idx)), idx] / self.tries[np.arange(len(idx)), idx]
        return float(np.mean(p * (1 - p)))


class LearnabilityPicker:
    def __init__(self, env, sampler_name: str, n_slots: int, uniform_share: float,
                 rng: np.random.Generator, seed: int):
        from acl_bench.sampling import ScenicTaskSampler
        random.seed(seed)                        # Scenic / VerifAI draw from these globals
        np.random.seed(seed % (1 << 32))
        self.env, self.rng, self.uniform_share = env, rng, uniform_share
        self.bounds = np.array([env.PARAM_BOUNDS[k] for k in env.PARAM_ORDER], dtype=np.float64)
        self.samplers = [ScenicTaskSampler.load(sampler_name, env.SCENIC_FILE) for _ in range(n_slots)]
        self.tracker = BucketLearnability(self.bounds)
        self.norm = RunningNormalizer()
        self.from_sampler = np.zeros(n_slots, dtype=bool)

    def draw(self, k: int) -> np.ndarray:
        if self.rng.uniform() < self.uniform_share:
            self.from_sampler[k] = False
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            return lo + (hi - lo) * self.rng.uniform(size=len(lo))
        self.from_sampler[k] = True
        task = self.samplers[k].draw(self.env.PARAM_ORDER)
        return np.array([task[name] for name in self.env.PARAM_ORDER], dtype=np.float64)

    def report(self, k: int, params: np.ndarray, success: bool) -> None:
        self.tracker.update(params, success)
        if self.from_sampler[k]:
            learn = self.tracker.learnability(params)
            self.norm.update(learn)
            self.samplers[k].give_feedback(self.norm.to_rho(learn))
