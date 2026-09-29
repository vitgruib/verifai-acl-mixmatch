"""Proposers: each picks the next training task one way (docs/library.md).

  Uniform            plain domain randomization.
  SIR                sampling-importance-resampling from uniform in proportion to
                     score(task)^alpha, score = the pass model's learnability; knows its own
                     density, so it can be importance-corrected (Curriculum.is_power).
  Replay             a PLR-style buffer of visited tasks, replayed in proportion to their
                     posterior learnability E[p(1-p)] (Beta posterior whose prior mean is the
                     pass model's prediction), with forgetting as the policy changes.
  Mutate             ACCEL-style edits: a Gaussian perturbation of a task Replay would pick.
  VerifAISurrogate   a VerifAI sampler (through Scenic) steered by the pass model's
                     learnability; feedback is computed from the model, so the sampler can
                     take several search steps per training task at no simulation cost.
"""
from __future__ import annotations

import random

import numpy as np

from acl_bench.curriculum.core import Episode, Proposal, TaskSpace
from acl_bench.curriculum.estimators import PassModel


class Proposer:
    name = "base"

    def propose(self, k: int) -> Proposal:
        raise NotImplementedError

    def report(self, k: int, ep: Episode) -> None:
        pass

    def density(self, params: np.ndarray) -> float:
        raise NotImplementedError(f"{self.name} has no density; it cannot be importance-corrected")


class Uniform(Proposer):
    name = "uniform"

    def __init__(self, space: TaskSpace):
        self.space = space

    def propose(self, k: int) -> Proposal:
        return Proposal(self.space.uniform(1)[0], self.name, 1.0)

    def density(self, params: np.ndarray) -> float:
        return 1.0


class SIR(Proposer):
    """q(task) = uniform(task) * s(task)^alpha / Z, s = learnability + floor, drawn by
    resampling `n_candidates` uniform tasks. Z = E_uniform[s^alpha], tracked as a running
    mean of the candidate sets' mean weight (reset whenever the model refits)."""

    def __init__(self, space: TaskSpace, model: PassModel, rng: np.random.Generator, alpha: float = 1.0,
                 n_candidates: int = 64, floor: float = 0.01, name: str = "sir"):
        self.space, self.model, self.rng = space, model, rng
        self.alpha, self.n_candidates, self.floor, self.name = alpha, n_candidates, floor, name
        self._z_sum, self._z_n, self._version = 0.0, 0, -1

    def _weights(self, params: np.ndarray) -> np.ndarray:
        return (self.model.learnability(params) + self.floor) ** self.alpha

    def _z(self) -> float:
        if self._version != self.model.version:
            self._version, self._z_sum, self._z_n = self.model.version, 0.0, 0
            cand = self.space.uniform(4096)
            self._z_sum, self._z_n = float(self._weights(cand).sum()), len(cand)
        return self._z_sum / self._z_n

    def propose(self, k: int) -> Proposal:
        if not self.model.ready:
            return Proposal(self.space.uniform(1)[0], self.name, 1.0)
        cand = self.space.uniform(self.n_candidates)
        w = self._weights(cand)
        i = int(self.rng.choice(len(cand), p=w / w.sum()))
        return Proposal(cand[i], self.name)

    def density(self, params: np.ndarray) -> float:
        if not self.model.ready:
            return 1.0
        return float(self._weights(params)[0] / self._z())


class Replay(Proposer):
    name = "replay"

    def __init__(self, space: TaskSpace, model: PassModel | None, rng: np.random.Generator, size: int = 1000,
                 alpha: float = 4.0, prior_strength: float = 2.0, forget: float = 0.8, min_fill: int = 100):
        self.space, self.model, self.rng = space, model, rng
        self.size, self.alpha, self.m, self.forget, self.min_fill = size, alpha, prior_strength, forget, min_fill
        self.params = np.zeros((size, space.dim))
        self.wins, self.tries = np.zeros(size), np.zeros(size)
        self.n, self.head = 0, 0
        self.slot_of = {}                   # env slot -> buffer index being replayed
        self._prior, self._version = np.full(size, 0.5), -1

    def scores(self) -> np.ndarray:
        """Posterior E[p(1-p)] per stored task: Beta(a, b) with a = m p0 + wins,
        b = m (1 - p0) + losses; E[p(1-p)] = ab / ((a+b)(a+b+1))."""
        n = self.n
        if self.model is not None and self.model.version != self._version and n:
            self._version = self.model.version
            self._prior[:n] = self.model.p(self.params[:n])
        a = self.m * self._prior[:n] + self.wins[:n]
        b = self.m * (1 - self._prior[:n]) + self.tries[:n] - self.wins[:n]
        return a * b / ((a + b) * (a + b + 1))

    def sample_index(self) -> int | None:
        if self.n < self.min_fill:
            return None
        s = self.scores() ** self.alpha
        return int(self.rng.choice(self.n, p=s / s.sum()))

    def propose(self, k: int) -> Proposal:
        i = self.sample_index()
        if i is None:
            self.slot_of.pop(k, None)
            return Proposal(self.space.uniform(1)[0], "uniform")
        self.slot_of[k] = i
        return Proposal(self.params[i].copy(), self.name)

    def report(self, k: int, ep: Episode) -> None:
        if ep.source == self.name and k in self.slot_of:
            i = self.slot_of.pop(k)
            self.wins[i] = self.forget * self.wins[i] + float(ep.success)
            self.tries[i] = self.forget * self.tries[i] + 1.0
            return
        i = self.head
        self.head = (self.head + 1) % self.size
        self.n = min(self.n + 1, self.size)
        self.params[i], self.wins[i], self.tries[i] = ep.params, float(ep.success), 1.0
        self._prior[i] = self.model.p(ep.params)[0] if self.model is not None else 0.5


class Mutate(Proposer):
    name = "mutate"

    def __init__(self, space: TaskSpace, replay: Replay, rng: np.random.Generator, sigma: float = 0.05):
        self.space, self.replay, self.rng, self.sigma = space, replay, rng, sigma

    def propose(self, k: int) -> Proposal:
        i = self.replay.sample_index()
        if i is None:
            return Proposal(self.space.uniform(1)[0], "uniform")
        child = self.replay.params[i] + self.rng.normal(0, self.sigma, self.space.dim) * self.space.span
        return Proposal(self.space.clip(child), self.name)


class VerifAISurrogate(Proposer):
    name = "verifai"

    def __init__(self, env, space: TaskSpace, model: PassModel, n_slots: int, seed: int,
                 sampler: str = "ce", inner: int = 4):
        from acl_bench.acl import RunningNormalizer
        from acl_bench.sampling import ScenicTaskSampler
        random.seed(seed)                        # Scenic / VerifAI draw from these globals
        np.random.seed(seed % (1 << 32))
        self.env, self.space, self.model, self.inner = env, space, model, inner
        self.samplers = [ScenicTaskSampler.load(sampler, env.SCENIC_FILE) for _ in range(n_slots)]
        self.norm = RunningNormalizer()

    def _draw(self, k: int) -> np.ndarray:
        task = self.samplers[k].draw(self.env.PARAM_ORDER)
        return np.array([task[n] for n in self.env.PARAM_ORDER], dtype=np.float64)

    def propose(self, k: int) -> Proposal:
        steps = self.inner if self.model.ready else 1
        for _ in range(steps):
            params = self._draw(k)
            if self.model.ready:
                learn = float(self.model.learnability(params)[0])
                self.norm.update(learn)
                self.samplers[k].give_feedback(self.norm.to_rho(learn))
        return Proposal(params, self.name)
