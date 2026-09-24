"""Locked exam sections (docs/exam.md).

A section is a fixed list of questions, each a (task, start) pair: the physics
(`params`, in the environment's PARAM_ORDER) and the exact initial state (`s0`), optionally
with the share of reference agents that fail it. Sections are saved as .npz files
with a manifest of counts and SHA-256 fingerprints; loading verifies them.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from acl_bench.exam.grader import grade


@dataclass
class PairSet:
    name: str
    params: np.ndarray                          # (N, n_params) in PARAM_ORDER
    s0: np.ndarray                              # (N, state_dim) initial states
    ref_fail_frac: np.ndarray | None = None     # (N,) share of reference agents that fail

    def __len__(self) -> int:
        return len(self.params)

    def digest(self) -> str:
        h = hashlib.sha256()
        for arr in (self.params, self.s0):
            h.update(np.ascontiguousarray(arr).tobytes())
        if self.ref_fail_frac is not None:
            h.update(np.ascontiguousarray(self.ref_fail_frac).tobytes())
        return h.hexdigest()


def evaluate_sets(env, agent, sets: dict[str, PairSet]) -> dict[str, float]:
    """Flat metrics dict: per section, the number of questions, the share passed
    (`success`) and the mean episode length in steps (`mean_steps`: steps survived for a
    survive goal, higher is better; steps to the goal for a reach goal, lower is better)."""
    out = {}
    for name, ps in sets.items():
        if len(ps) == 0:                                   # e.g. no easy questions found: report, don't warn
            out.update({f"{name}/n": 0, f"{name}/success": float("nan"), f"{name}/mean_steps": float("nan")})
            continue
        ok, steps = grade(env, agent, ps.params, ps.s0)
        out[f"{name}/n"] = len(ps)
        out[f"{name}/success"] = float(ok.mean())
        out[f"{name}/mean_steps"] = float(steps.mean())
    return out


def save_sets(sets: dict[str, PairSet], directory: str, extra: dict | None = None) -> dict:
    """Write each section as an .npz and merge its count and checksum into manifest.json
    (sections not in `sets` are kept)."""
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


def remove_sets(names, directory: str) -> None:
    """Delete sections and their manifest entries."""
    path = os.path.join(directory, "manifest.json")
    manifest = json.load(open(path))
    for name in names:
        manifest["sets"].pop(name, None)
        f = os.path.join(directory, f"{name}.npz")
        if os.path.exists(f):
            os.remove(f)
    json.dump(manifest, open(path, "w"), indent=2, sort_keys=True)


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
