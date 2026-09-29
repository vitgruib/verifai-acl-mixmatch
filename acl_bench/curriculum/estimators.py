"""Estimators shared by proposers: what the curriculum knows about the task space.

PassModel: a pass-rate model over the task space fit to ordinary training episodes (one
per task, pooled; GoalGAN-style, Florensa et al. 2018). In the offline signal screen it was
the best frontier signal on all four dev environments (docs/plr.md, `model_1ep`).
"""
from __future__ import annotations

import numpy as np
import torch

from acl_bench.curriculum.core import Episode, TaskSpace


class PassModel:
    """Keeps the latest `window` (task, passed) results and refits a small MLP classifier
    every `refit_every` episodes (warm-started). `ready` once `warmup` results are in;
    before that `p()` returns 0.5 everywhere. `version` counts refits, so consumers can
    cache predictions."""

    def __init__(self, space: TaskSpace, seed: int, window: int = 4096, refit_every: int = 256,
                 warmup: int = 512, hidden: int = 32, steps: int = 100):
        self.space, self.window, self.refit_every, self.warmup, self.steps = space, window, refit_every, warmup, steps
        self.x, self.y, self.seen, self.version = [], [], 0, 0
        g = torch.Generator().manual_seed(seed)
        d = space.dim
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, torch.nn.Linear):
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) / np.sqrt(m.in_features))
                    m.bias.zero_()
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
