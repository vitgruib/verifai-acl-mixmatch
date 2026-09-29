"""The curriculum library's core: a task space, the episode record every component learns
from, and the `Curriculum` that mixes proposers (docs/library.md).

Environment-agnostic by construction: a component sees only the task-parameter box, the
parameters of each episode's task and its outcome (plus, for model-specific components,
the learner's own per-step values). Nothing names an environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class TaskSpace:
    """The box of task parameters; components work in normalized coordinates [0, 1]^d."""

    def __init__(self, bounds: np.ndarray, rng: np.random.Generator):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.lo, self.span = self.bounds[:, 0], self.bounds[:, 1] - self.bounds[:, 0]
        self.dim, self.rng = len(self.bounds), rng

    def uniform(self, n: int) -> np.ndarray:
        return self.lo + self.span * self.rng.uniform(size=(n, self.dim))

    def unit(self, params: np.ndarray) -> np.ndarray:
        return (np.asarray(params) - self.lo) / self.span

    def clip(self, params: np.ndarray) -> np.ndarray:
        return np.clip(params, self.bounds[:, 0], self.bounds[:, 1])


@dataclass
class Episode:
    params: np.ndarray
    success: bool
    ret: float
    length: int
    source: str = ""                       # which proposer chose the task
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    next_value: float = 0.0


@dataclass
class Proposal:
    params: np.ndarray
    source: str
    density: float | None = None           # q(task) / uniform(task), when the proposer knows it


class Curriculum:
    """Draws each new task from one of its proposers (fixed mixture weights), reports every
    finished episode to all of them and to the shared estimators, and optionally returns an
    importance weight per episode: w = (uniform / q)^is_power, which with is_power = 1 makes
    the policy gradient an unbiased estimate of plain domain randomization's (docs/library.md,
    "tempered importance correction"). Requires every proposer in the mix to know its density."""

    def __init__(self, space: TaskSpace, proposers: dict, weights: dict, estimators: list,
                 rng: np.random.Generator, is_power: float = 0.0, max_weight: float = 10.0, reweighter=None):
        self.space, self.proposers, self.estimators, self.rng = space, proposers, estimators, rng
        names = list(weights)
        self.names = names
        self.mix = np.array([weights[n] for n in names], dtype=np.float64)
        self.mix /= self.mix.sum()
        self.is_power, self.max_weight = is_power, max_weight
        # reweighting instead of resampling: tasks stay as proposed, each episode's policy
        # loss is weighted by reweighter.density(task) (e.g. an SIR's s^alpha / Z), which in
        # expectation equals sampling from that SIR without narrowing coverage
        self.reweighter = reweighter
        self.sim_steps = 0                 # extra simulation, charged to the budget
        self.counts = {n: 0 for n in names}

    def propose(self, k: int) -> tuple[np.ndarray, str, float]:
        """(params, source, importance weight) for environment slot k."""
        name = self.names[int(self.rng.choice(len(self.names), p=self.mix))]
        prop = self.proposers[name].propose(k)
        self.counts[name] += 1
        w = 1.0
        if self.is_power > 0:
            q = sum(m * self.proposers[n].density(prop.params) for n, m in zip(self.names, self.mix))
            w = float(min((1.0 / max(q, 1e-12)) ** self.is_power, self.max_weight))
        if self.reweighter is not None:
            w = float(min(w * self.reweighter.density(prop.params), self.max_weight))
        return prop.params, prop.source, w

    def report(self, k: int, ep: Episode) -> None:
        for e in self.estimators:
            e.update(ep)
        for p in self.proposers.values():
            p.report(k, ep)

    def on_rollout(self, agent, obs, params, act, adv, done, src) -> None:
        """Model-specific estimators see each rollout (T, K) after its advantages are known."""
        for e in self.estimators:
            if hasattr(e, "on_rollout"):
                e.on_rollout(agent, obs, params, act, adv, done, src)

    def stats(self) -> dict:
        """One fixed column whatever the mix, so every arm's rows share a CSV."""
        return {"lib_counts": ";".join(f"{n}={c}" for n, c in self.counts.items())}
