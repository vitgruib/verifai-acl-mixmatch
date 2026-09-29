"""Estimators shared by proposers: what the curriculum knows about the task space.

PassModel: a pass-rate model over the task space fit to ordinary training episodes (one
per task, pooled; GoalGAN-style, Florensa et al. 2018). In the offline signal screen it was
the best frontier signal on all four dev environments (docs/plr.md, `model_1ep`).

GradSignal (model-specific): each episode's actual policy-gradient contribution through
the actor's last layer, in closed form, and a regression model of it over the task space:
its size (for variance-optimal sampling) or its alignment with the gradient of the uniform
episodes in the same rollout (for sampling tasks whose update also improves DR's objective).
"""
from __future__ import annotations

import numpy as np
import torch

from acl_bench.curriculum.core import Episode, TaskSpace


def _mlp(d: int, hidden: int, seed: int) -> torch.nn.Sequential:
    """A small tanh MLP initialized from its own generator, leaving torch's global random
    state (which PPO's action sampling uses) untouched."""
    g = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng():
        net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden),
                                  torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
    with torch.no_grad():
        for m in net:
            if isinstance(m, torch.nn.Linear):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) / np.sqrt(m.in_features))
                m.bias.zero_()
    return net


class PassModel:
    """Keeps the latest `window` (task, passed) results and refits a small MLP classifier
    every `refit_every` episodes (warm-started). `ready` once `warmup` results are in;
    before that `p()` returns 0.5 everywhere. `version` counts refits, so consumers can
    cache predictions."""

    def __init__(self, space: TaskSpace, seed: int, window: int = 4096, refit_every: int = 256,
                 warmup: int = 512, hidden: int = 32, steps: int = 100):
        self.space, self.window, self.refit_every, self.warmup, self.steps = space, window, refit_every, warmup, steps
        self.x, self.y, self.seen, self.version = [], [], 0, 0
        self.net = _mlp(space.dim, hidden, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-2, weight_decay=1e-4)

    @property
    def ready(self) -> bool:
        return self.version > 0

    def _z(self, params: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.space.unit(params) * 2 - 1, dtype=torch.float32)

    def update(self, ep: Episode) -> None:
        self.x.append(np.array(ep.params, dtype=np.float64))
        self.y.append(float(ep.success))
        del self.x[:-self.window], self.y[:-self.window]
        self.seen += 1
        if self.seen >= self.warmup and self.seen % self.refit_every == 0:
            xt, yt = self._z(np.array(self.x)), torch.as_tensor(np.array(self.y), dtype=torch.float32)
            for _ in range(self.steps):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net(xt).squeeze(1), yt)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.version += 1

    @torch.no_grad()
    def p(self, params: np.ndarray) -> np.ndarray:
        params = np.atleast_2d(params)
        if not self.ready:
            return np.full(len(params), 0.5)
        return torch.sigmoid(self.net(self._z(params)).squeeze(1)).numpy().astype(np.float64)

    def learnability(self, params: np.ndarray) -> np.ndarray:
        p = self.p(params)
        return p * (1 - p)


class ProgressModel(PassModel):
    """Learning progress from the pass model's own history (a model-based ALP-GMM, Portelas
    et al. 2019): score = |p_now(task) - p_then(task)|, p_then = the model `lag` refits ago.
    A shorter window (1,024 episodes) keeps p_now current."""

    def __init__(self, space: TaskSpace, seed: int, lag: int = 2, window: int = 1024, **kw):
        super().__init__(space, seed, window=window, **kw)
        import copy
        self._copy, self.lag, self.history = copy.deepcopy, lag, []

    def update(self, ep: Episode) -> None:
        v = self.version
        super().update(ep)
        if self.version != v:
            self.history.append(self._copy(self.net))
            del self.history[:-(self.lag + 1)]

    @torch.no_grad()
    def score(self, params: np.ndarray) -> np.ndarray:
        params = np.atleast_2d(params)
        if len(self.history) <= self.lag:
            return np.zeros(len(params))
        z = self._z(params)
        now, then = (torch.sigmoid(n(z).squeeze(1)).numpy() for n in (self.history[-1], self.history[0]))
        return np.abs(now - then).astype(np.float64)


class EnsembleModel:
    """`n` pass models, each fit on a Poisson(1) bootstrap of the episodes (each episode is
    given to member i Poisson(1) times). score = mean p(1-p) + bonus * std over members of p:
    learnability plus an exploration bonus where the models disagree."""

    def __init__(self, space: TaskSpace, seed: int, n: int = 4, bonus: float = 1.0):
        self.members = [PassModel(space, seed * 31 + i) for i in range(n)]
        self.rng, self.bonus = np.random.default_rng([seed, 5]), bonus

    @property
    def version(self) -> int:
        return min(m.version for m in self.members)

    @property
    def ready(self) -> bool:
        return all(m.ready for m in self.members)

    def update(self, ep: Episode) -> None:
        for m in self.members:
            for _ in range(self.rng.poisson(1.0)):
                m.update(ep)

    def p(self, params: np.ndarray) -> np.ndarray:
        return np.mean([m.p(params) for m in self.members], axis=0)

    def score(self, params: np.ndarray) -> np.ndarray:
        ps = np.stack([m.p(params) for m in self.members])
        return (ps * (1 - ps)).mean(0) + self.bonus * ps.std(0)


class TaskRegressor:
    """An MLP regression y(task) over the latest `window` (task, y) pairs, refit every
    `refit_every` pairs once `warmup` are in; y is standardized for the fit."""

    def __init__(self, space: TaskSpace, seed: int, window: int = 4096, refit_every: int = 256,
                 warmup: int = 512, hidden: int = 32, steps: int = 100):
        self.space, self.window, self.refit_every, self.warmup, self.steps = space, window, refit_every, warmup, steps
        self.x, self.y, self.seen, self.version = [], [], 0, 0
        self.net = _mlp(space.dim, hidden, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-2, weight_decay=1e-4)
        self.mu, self.sd = 0.0, 1.0

    @property
    def ready(self) -> bool:
        return self.version > 0

    def _z(self, params: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.space.unit(params) * 2 - 1, dtype=torch.float32)

    def add(self, params: np.ndarray, y: float) -> None:
        self.x.append(np.array(params, dtype=np.float64))
        self.y.append(float(y))
        del self.x[:-self.window], self.y[:-self.window]
        self.seen += 1
        if self.seen >= self.warmup and self.seen % self.refit_every == 0:
            y = np.array(self.y)
            self.mu, self.sd = float(y.mean()), float(y.std() + 1e-8)
            xt, yt = self._z(np.array(self.x)), torch.as_tensor((y - self.mu) / self.sd, dtype=torch.float32)
            for _ in range(self.steps):
                loss = ((self.net(xt).squeeze(1) - yt) ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.version += 1

    @torch.no_grad()
    def predict_z(self, params: np.ndarray) -> np.ndarray:
        """Prediction in standardized units (0 = the mean of the recent data)."""
        params = np.atleast_2d(params)
        if not self.ready:
            return np.zeros(len(params))
        return self.net(self._z(params)).squeeze(1).numpy().astype(np.float64)


class GradSignal:
    """kind "norm": y = log ||G_ep||, G_ep = sum over the episode's steps of
    A_t (onehot(a_t) - pi_t) [h_t, 1]^T, the gradient of the policy-gradient objective with
    respect to the actor's last layer (A_t = the rollout-normalized GAE advantage, h_t = the
    last hidden layer). kind "align": y = sum over the episode's rollout segments of
    <G_seg, g_u> / ||g_u||, g_u = the mean per-step contribution of the rollout's uniform
    episodes. `score(params)` is a positive sampling weight for SIR."""

    def __init__(self, space: TaskSpace, seed: int, n_slots: int, kind: str = "norm"):
        self.kind, self.model = kind, TaskRegressor(space, seed)
        self.acc = [None] * n_slots        # running G (norm) or alignment sum (align) per slot

    @property
    def version(self) -> int:
        return self.model.version

    @property
    def ready(self) -> bool:
        return self.model.ready

    def update(self, ep) -> None:          # episodes are finalized in on_rollout instead
        pass

    def on_rollout(self, agent, obs, params, act, adv, done, src) -> None:
        T, K = act.shape
        with torch.no_grad():
            x = torch.as_tensor(obs.reshape(T * K, -1))
            h = agent.actor[:-1](x)                                # last hidden layer
            pi = torch.softmax(agent.actor[-1](h), dim=1)
            a = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
            a = (a - a.mean()) / (a.std() + 1e-8)
            e = torch.nn.functional.one_hot(torch.as_tensor(act.reshape(-1)), pi.shape[1]).float()
            hb = torch.cat([h, torch.ones(len(h), 1)], dim=1)
            c = (a[:, None, None] * (e - pi)[:, :, None] * hb[:, None, :]).reshape(T, K, -1).numpy()
        if self.kind == "align":
            uni = np.array([[s == "uniform" for s in row] for row in src])
            g = c[uni].mean(0) if uni.any() else c.reshape(T * K, -1).mean(0)
            g = g / (np.linalg.norm(g) + 1e-8)
            c = c @ g                                                # (T, K) projections
        for k in range(K):
            for t in range(T):
                self.acc[k] = c[t, k] if self.acc[k] is None else self.acc[k] + c[t, k]
                if done[t, k]:
                    y = np.log(np.linalg.norm(self.acc[k]) + 1e-6) if self.kind == "norm" else float(self.acc[k])
                    self.model.add(params[t, k], y)
                    self.acc[k] = None

    def score(self, params: np.ndarray) -> np.ndarray:
        z = self.model.predict_z(params)
        if self.kind == "norm":
            return np.exp(z * self.model.sd)                          # ||G|| relative to its geometric mean
        return np.clip(z, 0.0, None)                                  # aligned beyond average


class VarModel:
    """Outcome-variance learnability for any outcome, not only pass/fail: a heteroscedastic
    Gaussian fit (mean and log-variance heads, NLL loss) of the episode return (standardized
    over the window) over the task space. score = the predicted standard deviation of the
    outcome on the task. For a binary outcome Var = p(1-p), so this generalizes the pass
    model's learnability to returns; tasks where the same policy sometimes does well and
    sometimes badly are the frontier. A short window keeps it about the current policy."""

    def __init__(self, space: TaskSpace, seed: int, window: int = 512, refit_every: int = 128,
                 warmup: int = 256, hidden: int = 32, steps: int = 100, outcome: str = "ret"):
        self.space, self.window, self.refit_every, self.warmup, self.steps = space, window, refit_every, warmup, steps
        self.outcome = outcome
        self.x, self.y, self.seen, self.version = [], [], 0, 0
        g = torch.Generator().manual_seed(seed)
        self.body = _mlp(space.dim, hidden, seed)[:-1]
        self.head = torch.nn.Linear(hidden, 2)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(self.head.weight.shape, generator=g) / np.sqrt(hidden) * 0.1)
            self.head.bias.zero_()
        self.opt = torch.optim.Adam(list(self.body.parameters()) + list(self.head.parameters()), lr=1e-2,
                                    weight_decay=1e-4)
        self.mu, self.sd = 0.0, 1.0

    @property
    def ready(self) -> bool:
        return self.version > 0

    def _z(self, params: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.space.unit(params) * 2 - 1, dtype=torch.float32)

    def update(self, ep: Episode) -> None:
        self.x.append(np.array(ep.params, dtype=np.float64))
        self.y.append(float(ep.ret if self.outcome == "ret" else ep.success))
        del self.x[:-self.window], self.y[:-self.window]
        self.seen += 1
        if self.seen >= self.warmup and self.seen % self.refit_every == 0:
            y = np.array(self.y)
            self.mu, self.sd = float(y.mean()), float(y.std() + 1e-8)
            xt, yt = self._z(np.array(self.x)), torch.as_tensor((y - self.mu) / self.sd, dtype=torch.float32)
            for _ in range(self.steps):
                out = self.head(self.body(xt))
                m, lv = out[:, 0], out[:, 1].clamp(-8, 4)
                loss = (0.5 * (lv + (yt - m) ** 2 / lv.exp())).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.version += 1

    @torch.no_grad()
    def _out(self, params: np.ndarray) -> np.ndarray:
        return self.head(self.body(self._z(np.atleast_2d(params)))).numpy().astype(np.float64)

    def mean(self, params: np.ndarray) -> np.ndarray:
        if not self.ready:
            return np.zeros(len(np.atleast_2d(params)))
        return self._out(params)[:, 0]

    def score(self, params: np.ndarray) -> np.ndarray:
        if not self.ready:
            return np.ones(len(np.atleast_2d(params)))
        return np.exp(0.5 * np.clip(self._out(params)[:, 1], -8, 4))
