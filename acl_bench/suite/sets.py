"""Locked exam sections for the CartPole suite (docs/cartpole_suite.md).

A section is a fixed list of questions, each a (task, start) pair: the physics
(`params`, in evaluator.PARAM_ORDER) and the exact initial state (`s0`). The static
sections E0-E5 live in `frozen_sets/cartpole_v2/`; they were generated from fixed
seeds and filtered once to questions a hand-built expert could win, then that expert
was removed (the manifest records the commit it can be recovered from). Nothing here
depends on it.

E6 (hard), E7 (easy) and POOL are built by `build_pool_sets` from reference agents,
judging winnability from the agents' own results instead of an expert.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from acl_bench.envs.param_cartpole import PARAM_BOUNDS
from acl_bench.suite.evaluator import PARAM_ORDER, rollout_steps

MAX_STEPS = 500
FAIL_BINS = (0.0, 0.1, 0.5, 0.9, 1.0001)   # reference-failure-fraction bins for POOL reporting


@dataclass
class PairSet:
    name: str
    params: np.ndarray                          # (N, 5) in PARAM_ORDER
    s0: np.ndarray                              # (N, 4) initial states
    ref_fail_frac: np.ndarray | None = None     # (N,) share of reference agents that fail (pool-derived sets)

    def __len__(self) -> int:
        return len(self.params)

    def digest(self) -> str:
        h = hashlib.sha256()
        for arr in (self.params, self.s0):
            h.update(np.ascontiguousarray(arr).tobytes())
        if self.ref_fail_frac is not None:
            h.update(np.ascontiguousarray(self.ref_fail_frac).tobytes())
        return h.hexdigest()


def draw_pairs(n: int, rng: np.random.Generator, ranges: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`ranges` overrides the (lo, hi) of named parameters; the rest are uniform over
    the full box. Starts are uniform within each task's init_range."""
    box = {**PARAM_BOUNDS, **(ranges or {})}
    params = np.array([[rng.uniform(*box[k]) for k in PARAM_ORDER] for _ in range(n)])
    init = params[:, PARAM_ORDER.index("init_range")]
    s0 = np.array([rng.uniform(-r, r, 4) for r in init])
    return params, s0


def build_pool_sets(reference_agents, seed: int = 20260922, pool_size: int = 5000,
                    hard_frac: float = 0.5, hard_target: int = 250,
                    easy_target: int = 100) -> tuple[dict[str, PairSet], dict]:
    """POOL, E6 and E7 from reference agents, with no expert.

    POOL: `pool_size` random questions (some are impossible), each annotated with
    the share of reference agents that fail it, for difficulty-binned reporting.
    E6 (hard): questions at least `hard_frac` of the reference agents fail AND at
    least one passes; the "at least one passes" is the only evidence of winnability
    available, so it also drops the very hardest winnable questions.
    E7 (easy): questions every reference agent passes.
    `reference_agents` must be trained separately from anything being evaluated.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    params, s0 = draw_pairs(pool_size, rng)
    success = np.array([rollout_steps(a, params, s0, MAX_STEPS) == MAX_STEPS for a in reference_agents])
    fail_frac = 1.0 - success.mean(axis=0)
    pool = PairSet("POOL", params, s0, fail_frac)

    def subset(name, mask, cap):
        idx = np.flatnonzero(mask)[:cap]
        return PairSet(name, params[idx], s0[idx], fail_frac[idx])

    hard = (fail_frac >= hard_frac) & (fail_frac < 1.0)
    easy = fail_frac == 0.0
    stats = {"pool_size": pool_size, "n_reference_agents": len(reference_agents),
             "hard_available": int(hard.sum()), "easy_available": int(easy.sum()),
             "nobody_passes": int((fail_frac == 1.0).sum())}
    return {"POOL": pool, "E6": subset("E6", hard, hard_target), "E7": subset("E7", easy, easy_target)}, stats


def evaluate_sets(agent, sets: dict[str, PairSet]) -> dict[str, float]:
    """Flat metrics dict: per section, the number of questions, the share passed
    (`success`) and the mean steps survived; plus success by reference-difficulty bin
    on POOL if present."""
    out = {}
    for name, ps in sets.items():
        if len(ps) == 0:                                   # e.g. no easy questions found: report, don't warn
            out.update({f"{name}/n": 0, f"{name}/success": float("nan"), f"{name}/mean_steps": float("nan")})
            continue
        steps = rollout_steps(agent, ps.params, ps.s0, MAX_STEPS)
        ok = steps == MAX_STEPS
        out[f"{name}/n"] = len(ps)
        out[f"{name}/success"] = float(ok.mean())
        out[f"{name}/mean_steps"] = float(steps.mean())
        if name == "POOL" and ps.ref_fail_frac is not None:
            for lo, hi in zip(FAIL_BINS[:-1], FAIL_BINS[1:]):
                m = (ps.ref_fail_frac >= lo) & (ps.ref_fail_frac < hi)
                key = f"POOL/bin_{lo:g}_{min(hi, 1.0):g}"
                out[f"{key}/success"] = float(ok[m].mean()) if m.any() else float("nan")
                out[f"{key}/n"] = int(m.sum())
    return out


def save_sets(sets: dict[str, PairSet], directory: str, extra: dict | None = None) -> dict:
    """Write each section as an .npz and merge its count and checksum into manifest.json."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "manifest.json")
    manifest = json.load(open(path)) if os.path.exists(path) else {"sets": {}}
    for name, ps in sets.items():
        arrays = dict(params=ps.params, s0=ps.s0)
        if ps.ref_fail_frac is not None:
            arrays["ref_fail_frac"] = ps.ref_fail_frac
        np.savez(os.path.join(directory, f"{name}.npz"), **arrays)
        manifest["sets"].setdefault(name, {}).update({"n": len(ps), "sha256": ps.digest()})
    manifest.update(extra or {})
    json.dump(manifest, open(path, "w"), indent=2, sort_keys=True)
    return manifest


def load_sets(directory: str, names=None) -> dict[str, PairSet]:
    """Load sections and verify them against the manifest; raises if one changed."""
    manifest = json.load(open(os.path.join(directory, "manifest.json")))
    sets = {}
    for name, meta in manifest["sets"].items():
        if names is not None and name not in names:
            continue
        z = np.load(os.path.join(directory, f"{name}.npz"))
        ps = PairSet(name, z["params"], z["s0"], z["ref_fail_frac"] if "ref_fail_frac" in z.files else None)
        if ps.digest() != meta["sha256"]:
            raise ValueError(f"section {name} does not match its manifest checksum")
        sets[name] = ps
    return sets
