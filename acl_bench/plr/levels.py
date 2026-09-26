"""A level sampler with the knobs of the PLR papers, all in one place.

  - Jiang, Grefenstette, Rocktaschel. Prioritized Level Replay. ICML 2021.
    Rank prioritization P_S ~ (1/rank)^(1/beta), beta = 0.1; staleness mix
    P = (1 - rho) P_S + rho P_C with P_C ~ (episodes since last visit), rho = 0.1;
    a replayed level's score is replaced by the newest one; scores: L1 value loss
    (mean |GAE|) was best on Procgen.
  - Jiang, Dennis, Parker-Holder, Foerster, Grefenstette, Rocktaschel. Replay-Guided
    Adversarial Environment Design (Robust PLR, PLR-perp). NeurIPS 2021.
    Replay with a fixed probability once the buffer holds enough levels; a new level
    enters a full buffer only by beating its lowest score; positive value loss (PVL);
    PLR-perp trains only on replayed levels (new levels are scored, not learned from).
  - Parker-Holder et al. Evolving Curricula with Regret-Based Environment Design
    (ACCEL). ICML 2022. MaxMC score: mean over the episode of (best return ever seen on
    the level - V(s_t)); replay rate 0.8-0.9.
  - Rutherford et al. No Regrets: Investigating and Improving Regret Approximations for
    Curriculum Discovery (SFL). NeurIPS 2024. For binary outcomes, learnability p(1-p),
    p = the level's success rate, beats PVL and MaxMC.

This repo's original (SIPACL) configuration is `LevelConfig(beta=1.0, staleness=0.0,
score_ema=0.2, buffer=5000, admit="fifo", min_fill=1)`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEW, REPLAY = 0, 1


@dataclass
class LevelConfig:
    replay_prob: float = 0.0       # 0 = plain domain randomization
    beta: float = 0.1              # rank temperature: P_S ~ (1/rank)^(1/beta)
    staleness: float = 0.1         # rho in P = (1 - rho) P_S + rho P_C
    score_ema: float = 1.0         # weight of the newest score (1 = replace)
    buffer: int = 1000
    admit: str = "min"             # "min": a full buffer evicts its lowest score if beaten; "fifo"
    min_fill: int = 0              # replay only once this many levels are stored (0 = buffer // 10)
    score: str = "pvl"             # pvl | l1 | maxmc | learn (p(1-p)) | neg_return
    robust: bool = False           # PLR-perp: no gradient from episodes on new levels
    prior: float = 0.5             # learnability: p's prior mean (one pseudo-observation)
    # SFL (Rutherford et al. 2024): every `sfl_every` updates, roll out `sfl_n` random levels
    # `sfl_k` times each with the current stochastic policy and keep the `sfl_top` with the
    # highest p(1 - p); replays (replay_prob) then draw uniformly from those.
    sfl: bool = False
    sfl_n: int = 1000
    sfl_k: int = 8
    sfl_top: int = 100
    sfl_every: int = 10


class LevelSampler:
    def __init__(self, cfg: LevelConfig, bounds: np.ndarray, rng: np.random.Generator):
        self.cfg, self.bounds, self.rng = cfg, bounds, rng
        n = cfg.buffer
        self.params = np.zeros((n, len(bounds)))
        self.scores = np.zeros(n)
        self.last = np.zeros(n)              # episode counter at the last visit
        self.best = np.full(n, -np.inf)      # best return seen (MaxMC)
        self.wins = np.zeros(n)
        self.tries = np.zeros(n)
        self.size = 0
        self.fifo = 0
        self.clock = 0
        self.min_fill = cfg.min_fill or max(1, n // 10)

    # ---------------------------------------------------------------- choosing
    def draw_new(self) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + (hi - lo) * self.rng.uniform(size=len(lo))

    def replay_probs(self) -> np.ndarray:
        s = self.scores[:self.size]
        order = np.argsort(-s, kind="stable")
        ranks = np.empty(self.size)
        ranks[order] = np.arange(1, self.size + 1)
        w = (1.0 / ranks) ** (1.0 / self.cfg.beta)
        p = w / w.sum()
        if self.cfg.staleness > 0:
            stale = self.clock - self.last[:self.size]
            pc = stale / stale.sum() if stale.sum() > 0 else np.full(self.size, 1.0 / self.size)
            p = (1 - self.cfg.staleness) * p + self.cfg.staleness * pc
        return p

    def set_sfl(self, params: np.ndarray) -> None:
        self.sfl_levels = params

    def pick(self) -> tuple[np.ndarray, int, int]:
        """(params, slot or -1, NEW/REPLAY)."""
        if self.cfg.sfl:
            levels = getattr(self, "sfl_levels", None)
            if levels is not None and self.rng.uniform() < self.cfg.replay_prob:
                return levels[self.rng.integers(len(levels))].copy(), -1, REPLAY
            return self.draw_new(), -1, NEW
        if self.size >= self.min_fill and self.rng.uniform() < self.cfg.replay_prob:
            i = int(self.rng.choice(self.size, p=self.replay_probs()))
            return self.params[i].copy(), i, REPLAY
        return self.draw_new(), -1, NEW

    # ---------------------------------------------------------------- scoring
    def score(self, rewards, values, next_value, gamma, lam, success: bool, slot: int) -> float:
        kind = self.cfg.score
        if kind == "learn":
            w = (self.wins[slot] if slot >= 0 else 0.0) + float(success) + self.cfg.prior
            n = (self.tries[slot] if slot >= 0 else 0.0) + 1.0 + 1.0
            p = w / n
            return p * (1 - p)
        if kind == "neg_return":
            return -float(np.sum(rewards))
        if kind == "maxmc":
            ret = float(np.sum(rewards))
            best = max(ret, self.best[slot]) if slot >= 0 else ret
            # discounted return-to-go at each step, against the best return seen
            return float(np.mean(best - np.asarray(values)))
        adv = gae(np.asarray(rewards), np.asarray(values), next_value, gamma, lam)
        if kind == "pvl":
            return float(np.mean(np.clip(adv, 0.0, None)))
        if kind == "l1":
            return float(np.mean(np.abs(adv)))
        raise ValueError(kind)

    def report(self, params, slot: int, mode: int, score: float, ret: float, success: bool) -> None:
        self.clock += 1
        if self.cfg.sfl:
            return
        if mode == REPLAY:
            a = self.cfg.score_ema
            self.scores[slot] = (1 - a) * self.scores[slot] + a * score
            self.last[slot] = self.clock
            self.best[slot] = max(self.best[slot], ret)
            self.wins[slot] += success
            self.tries[slot] += 1
            return
        if self.cfg.replay_prob <= 0:
            return
        if self.size < self.cfg.buffer:
            i = self.size
            self.size += 1
        elif self.cfg.admit == "fifo":
            i = self.fifo
            self.fifo = (self.fifo + 1) % self.cfg.buffer
        else:
            i = int(np.argmin(self.scores))
            if score <= self.scores[i]:
                return
        self.params[i], self.scores[i], self.last[i] = params, score, self.clock
        self.best[i], self.wins[i], self.tries[i] = ret, float(success), 1.0


def gae(rewards, values, next_value, gamma, lam) -> np.ndarray:
    T = len(rewards)
    adv = np.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else next_value
        last = rewards[t] + gamma * v_next - values[t] + gamma * lam * last
        adv[t] = last
    return adv
