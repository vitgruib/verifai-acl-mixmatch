"""Solvability oracle for Kinetix levels: find an action sequence that solves the level, then
replay it open-loop to certify it.

The physics (jax2d) is deterministic and Kinetix's step ignores its key without auto-reset, so
an action sequence that solves a level from its reset state always solves it: a replay is an
exact certificate, not a statistical one (checked by `replay` on every witness). Witness sources,
cheapest first:
  self    the tested policy's own solved attempts (already rolled out by the falsifier)
  ref     solved attempts of reference policies (other checkpoints, --ref)
  search  open-loop action sequences: first PROBE uniform-random sequences with mixed hold
          lengths (1/8/32 steps; on the 10 S eval levels hold 8 alone solves 3.5-70% of random
          sequences), then if none solves, cross-entropy search with actions held for HOLD
          steps, per-decision categoricals, POP candidates x ITERS rounds, fitness = solved
          first, then episode return (Kinetix's dense distance shaping)
No witness -> UNKNOWN (the level is recorded as invalid, never as a counterexample).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

SOLVABLE, UNKNOWN = "solvable", "unknown"


class Oracle:
    name = "kinetix-witness (self / ref policies / CEM open-loop search, replay-verified)"

    def __init__(self, env, env_params, max_steps: int, refs=(), hold: int = 8, probe: int = 64,
                 pop: int = 256, iters: int = 6, elite: float = 0.1, ref_attempts: int = 8, seed: int = 0):
        self.env, self.ep, self.max_steps = env, env_params, max_steps
        self.refs, self.ref_attempts = list(refs), ref_attempts
        self.probe = probe
        self.hold, self.pop, self.iters, self.n_elite = hold, pop, iters, max(2, int(pop * elite))
        self.key = jax.random.PRNGKey(seed)
        sizes = np.asarray(env.action_type.number_of_dims_per_distribution)
        self.sizes, self.nmax = sizes, int(sizes.max())
        self.mask = jnp.asarray(np.arange(self.nmax)[None] < sizes[:, None])   # (D, nmax) valid values
        self.n_dec = -(-max_steps // hold)
        self._replay = jax.jit(jax.vmap(self._replay_one, (None, 0)))
        self._cem = jax.jit(self._cem_round)
        self._probe = jax.jit(self._probe_round)

    def params(self) -> dict:
        return {"probe": self.probe, "hold": self.hold, "pop": self.pop, "iters": self.iters, "elite": self.n_elite,
                "refs": len(self.refs), "ref_attempts": self.ref_attempts, "max_steps": self.max_steps}

    def _replay_one(self, level, acts):
        """acts (T, D) -> (solved, return, length); stops counting at the first done."""
        _, state = self.env.reset(jax.random.PRNGKey(0), self.ep, level)

        def step(c, a):
            state, alive = c
            _, state, r, d, info = self.env.step(jax.random.PRNGKey(0), state, a, self.ep)
            return (state, alive & ~d), (r * alive, alive, alive & d & info["GoalR"])

        _, (r, alive, solved) = jax.lax.scan(step, (state, jnp.bool_(True)), acts)
        return solved.any(), r.sum(), alive.sum()

    def replay(self, level, acts) -> tuple[bool, float, int]:
        s, r, n = self._replay(level, jnp.asarray(acts)[None])
        return bool(s[0]), float(r[0]), int(n[0])

    def _probe_round(self, level, key):
        """probe random sequences, a third each held 1, 8 and 32 steps -> (acts of first solver, any)."""
        hi = jnp.asarray(self.sizes)
        seqs = []
        for i, h in enumerate((1, 8, 32)):
            n = self.probe // 3 + (i < self.probe % 3)
            d = jax.random.randint(jax.random.fold_in(key, h), (n, -(-self.max_steps // h), len(self.sizes)), 0, hi)
            seqs.append(jnp.repeat(d, h, axis=1)[:, :self.max_steps])
        acts = jnp.concatenate(seqs)
        solved, _, _ = jax.vmap(self._replay_one, (None, 0))(level, acts)
        return acts[jnp.argmax(solved)], solved.any()

    def _cem_round(self, level, logits, key):
        k1, k2 = jax.random.split(key)
        lg = jnp.where(self.mask, logits, -jnp.inf)                                # (K, D, nmax)
        dec = jax.random.categorical(k1, lg, axis=-1, shape=(self.pop,) + lg.shape[:-1])  # (P, K, D)
        acts = jnp.repeat(dec, self.hold, axis=1)[:, :self.max_steps]
        solved, ret, _ = jax.vmap(self._replay_one, (None, 0))(level, acts)
        fit = ret + 1000.0 * solved
        top = jnp.argsort(-fit)[:self.n_elite]
        freq = jax.nn.one_hot(dec[top], self.nmax).mean(0)                         # (K, D, nmax)
        new = 0.7 * jnp.log(freq + 0.02) + 0.3 * logits
        best = top[0]
        return new, acts[best], solved[best], fit[best], k2

    def search(self, level):
        logits = jnp.zeros((self.n_dec, len(self.sizes), self.nmax))
        self.key, key, kp = jax.random.split(self.key, 3)
        acts, solved = self._probe(level, kp)
        if bool(solved):
            return np.asarray(acts), 0
        best = None
        for it in range(self.iters):
            logits, acts, solved, fit, key = self._cem(level, logits, key)
            best = (np.asarray(acts), float(fit), it + 1)
            if bool(solved):
                return best[0], it + 1
        return None, self.iters

    def check(self, level, own=None) -> dict:
        """own: (solved (A,), actions (T, A, D)) of the tested policy, or None."""
        cands = []
        if own is not None:
            cands.append(("self", own))
        for i, ref in enumerate(self.refs):
            cands.append((f"ref{i}", lambda ref=ref: ref.run(level, self.ref_attempts, 7)[2:]))
        for src, c in cands:
            solved, acts = c() if callable(c) else c
            for a in np.flatnonzero(solved)[:1]:
                ok, ret, n = self.replay(level, acts[:, a])
                if ok:
                    return {"verdict": SOLVABLE, "source": src, "witness_len": n,
                            "witness": acts[:n, a].astype(np.int8)}
        acts, its = self.search(level)
        if acts is not None:
            ok, ret, n = self.replay(level, acts)
            if ok:
                return {"verdict": SOLVABLE, "source": "search", "search_iters": its, "witness_len": n,
                        "witness": acts[:n].astype(np.int8)}
        return {"verdict": UNKNOWN, "source": None, "search_iters": its}
