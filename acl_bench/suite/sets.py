"""Frozen (task, start) pair sets for the CartPole suite (docs/cartpole_suite.md).

Each set is a fixed list of pairs plus oracle labels (acl_bench.oracle), generated
once from a seed and stored, so every agent is scored on identical pairs. Sets
that need trained reference agents (E6 hard-but-solvable, E7 easy, and the
difficulty-binned POOL) are built by `build_pool_sets` in Stage 0.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass

import numpy as np

from acl_bench.envs.param_cartpole import PARAM_BOUNDS
from acl_bench.oracle import infeasible_start, survival_steps
from acl_bench.suite.evaluator import PARAM_ORDER, rollout_steps

MAX_STEPS = 500
VERSION = "cartpole_v1"
FAIL_BINS = (0.0, 0.1, 0.5, 0.9, 1.0001)   # reference-failure-fraction bins for POOL reporting


@dataclass
class PairSet:
    name: str
    params: np.ndarray                 # (N, 5) in PARAM_ORDER
    s0: np.ndarray                     # (N, 4) initial states
    infeasible: np.ndarray             # (N,) bool: guaranteed one-step failure
    oracle_steps: np.ndarray           # (N,) steps the LQR oracle survives
    ref_fail_frac: np.ndarray | None = None   # (N,) share of reference agents that fail

    @property
    def learnable(self) -> np.ndarray:
        return ~self.infeasible & (self.oracle_steps >= MAX_STEPS)

    def __len__(self) -> int:
        return len(self.params)

    def digest(self) -> str:
        h = hashlib.sha256()
        for arr in (self.params, self.s0, self.infeasible, self.oracle_steps):
            h.update(np.ascontiguousarray(arr).tobytes())
        if self.ref_fail_frac is not None:
            h.update(np.ascontiguousarray(self.ref_fail_frac).tobytes())
        return h.hexdigest()


def label(params: np.ndarray, s0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oracle labels. Infeasible starts end on step 1 whatever is done (tested in
    tests/test_oracle.py), so the oracle rollout is skipped for them."""
    infeasible = np.array([infeasible_start(s) for s in s0])
    steps = np.array([1 if inf else survival_steps(dict(zip(PARAM_ORDER, p)), s, MAX_STEPS)
                      for p, s, inf in zip(params, s0, infeasible)], dtype=np.int64)
    return infeasible, steps


def _draw_start(rng, init_range: float, feasible_only: bool) -> np.ndarray:
    while True:
        s = rng.uniform(-init_range, init_range, 4)
        if not (feasible_only and infeasible_start(s)):
            return s


def draw_pairs(n: int, rng: np.random.Generator, ranges: dict | None = None,
               feasible_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """`ranges` overrides the (lo, hi) of named parameters; the rest are uniform
    over the full box. Starts are uniform within each task's init_range."""
    box = {**PARAM_BOUNDS, **(ranges or {})}
    params = np.array([[rng.uniform(*box[k]) for k in PARAM_ORDER] for _ in range(n)])
    s0 = np.array([_draw_start(rng, p[PARAM_ORDER.index("init_range")], feasible_only) for p in params])
    return params, s0


def _make(name, params, s0) -> PairSet:
    infeasible, steps = label(params, s0)
    return PairSet(name, params, s0, infeasible, steps)


def _quarter(name, top):
    lo, hi = PARAM_BOUNDS[name]
    w = hi - lo
    return (lo + 0.75 * w, hi) if top else (lo, lo + 0.25 * w)


def build_static_sets(seed: int = 20260921) -> dict[str, PairSet]:
    """E0-E5: sets that need no trained agent."""
    specs = {
        "E0": dict(n=300),
        "E1": dict(n=150, ranges={"force_mag": _quarter("force_mag", top=False)}),
        "E1b": dict(n=100, ranges={"force_mag": _quarter("force_mag", top=False),
                                   "masscart": _quarter("masscart", top=True)}),
        "E2": dict(n=100, ranges={"masscart": _quarter("masscart", top=True)}),
        "E3a": dict(n=100, ranges={"length": _quarter("length", top=True)}),
        "E3b": dict(n=100, ranges={"masspole": _quarter("masspole", top=True)}),
        "E4": dict(n=100, ranges={"init_range": _quarter("init_range", top=True)}, feasible_only=True),
    }
    children = np.random.SeedSequence(seed).spawn(len(specs) + 1)
    sets = {}
    for (name, spec), ss in zip(specs.items(), children):
        params, s0 = draw_pairs(spec["n"], np.random.default_rng(ss), spec.get("ranges"),
                                spec.get("feasible_only", False))
        sets[name] = _make(name, params, s0)

    rng = np.random.default_rng(children[-1])          # E5: all 32 vertices x 10 starts
    corners = np.array(list(itertools.product(*[PARAM_BOUNDS[k] for k in PARAM_ORDER])))
    params = np.repeat(corners, 10, axis=0)
    s0 = np.array([_draw_start(rng, p[PARAM_ORDER.index("init_range")], False) for p in params])
    sets["E5"] = _make("E5", params, s0)
    return sets


def build_pool_sets(reference_agents, seed: int = 20260922, pool_size: int = 5000,
                    hard_frac: float = 0.5, hard_target: int = 250,
                    easy_target: int = 100) -> tuple[dict[str, PairSet], dict]:
    """POOL (difficulty-binned reporting), E6 (oracle-solvable pairs that at least
    `hard_frac` of the reference agents fail) and E7 (pairs every reference agent
    solves). `reference_agents` must be trained on seeds no evaluated agent uses."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    params, s0 = draw_pairs(pool_size, rng)
    pool = _make("POOL", params, s0)
    success = np.array([rollout_steps(a, params, s0, MAX_STEPS) == MAX_STEPS for a in reference_agents])
    pool.ref_fail_frac = 1.0 - success.mean(axis=0)

    def subset(name, mask, cap):
        idx = np.flatnonzero(mask)[:cap]
        return PairSet(name, pool.params[idx], pool.s0[idx], pool.infeasible[idx],
                       pool.oracle_steps[idx], pool.ref_fail_frac[idx])

    learnable = pool.learnable
    hard = learnable & (pool.ref_fail_frac >= hard_frac)
    easy = learnable & (pool.ref_fail_frac == 0.0)
    stats = {"pool_size": pool_size, "n_reference_agents": len(reference_agents),
             "hard_available": int(hard.sum()), "easy_available": int(easy.sum()),
             "learnable_in_pool": int(learnable.sum())}
    return {"POOL": pool, "E6": subset("E6", hard, hard_target), "E7": subset("E7", easy, easy_target)}, stats


def evaluate_sets(agent, sets: dict[str, PairSet]) -> dict[str, float]:
    """Flat metrics dict: per set, success on learnable pairs, raw success on all
    pairs, mean steps on learnable pairs, and n_learnable; plus success by
    reference-difficulty bin on POOL if present."""
    out = {}
    for name, ps in sets.items():
        steps = rollout_steps(agent, ps.params, ps.s0, MAX_STEPS)
        ok, lrn = steps == MAX_STEPS, ps.learnable
        out[f"{name}/n_learnable"] = int(lrn.sum())
        out[f"{name}/success"] = float(ok[lrn].mean()) if lrn.any() else float("nan")
        out[f"{name}/success_all"] = float(ok.mean())
        out[f"{name}/mean_steps"] = float(steps[lrn].mean()) if lrn.any() else float("nan")
        if name == "POOL" and ps.ref_fail_frac is not None:
            for lo, hi in zip(FAIL_BINS[:-1], FAIL_BINS[1:]):
                m = lrn & (ps.ref_fail_frac >= lo) & (ps.ref_fail_frac < hi)
                out[f"POOL/bin_{lo:g}_{min(hi, 1.0):g}/success"] = float(ok[m].mean()) if m.any() else float("nan")
                out[f"POOL/bin_{lo:g}_{min(hi, 1.0):g}/n"] = int(m.sum())
    return out


def save_sets(sets: dict[str, PairSet], directory: str, seed_info: dict | None = None) -> dict:
    os.makedirs(directory, exist_ok=True)
    manifest = {"version": VERSION, "seeds": seed_info or {}, "sets": {}}
    manifest["oracle_source_sha256"] = hashlib.sha256(
        open(os.path.join(os.path.dirname(__file__), "..", "oracle.py"), "rb").read()).hexdigest()
    for name, ps in sets.items():
        arrays = dict(params=ps.params, s0=ps.s0, infeasible=ps.infeasible, oracle_steps=ps.oracle_steps)
        if ps.ref_fail_frac is not None:
            arrays["ref_fail_frac"] = ps.ref_fail_frac
        np.savez(os.path.join(directory, f"{name}.npz"), **arrays)
        manifest["sets"][name] = {"n": len(ps), "n_learnable": int(ps.learnable.sum()), "sha256": ps.digest()}
    path = os.path.join(directory, "manifest.json")
    if os.path.exists(path):                           # merge, so static and pool sets can be saved separately
        old = json.load(open(path))
        old["sets"].update(manifest["sets"]); old["seeds"].update(manifest["seeds"])
        manifest["sets"], manifest["seeds"] = old["sets"], old["seeds"]
    json.dump(manifest, open(path, "w"), indent=2, sort_keys=True)
    return manifest


def load_sets(directory: str, names=None) -> dict[str, PairSet]:
    """Load and verify against the manifest checksums; raises if a set changed."""
    manifest = json.load(open(os.path.join(directory, "manifest.json")))
    sets = {}
    for name, meta in manifest["sets"].items():
        if names is not None and name not in names:
            continue
        z = np.load(os.path.join(directory, f"{name}.npz"))
        ps = PairSet(name, z["params"], z["s0"], z["infeasible"], z["oracle_steps"],
                     z["ref_fail_frac"] if "ref_fail_frac" in z.files else None)
        if ps.digest() != meta["sha256"]:
            raise ValueError(f"set {name} does not match its manifest checksum")
        sets[name] = ps
    return sets
