"""The `custom` exam section: hand-picked parameter settings around the training range.

Cases are the defaults, then each setting alone swept over its two edges and 10% / 50% of
the range beyond each edge (the rest at their defaults); settings the environment rejects
(`valid`) are dropped. Labels: "default", "<setting>:<min|max>" for an edge and
"<setting>:<min|max>+10%" / "+50%" beyond it. FrozenLake instead goes beyond its range by
explicit values, labelled "<setting>:<min|max>+beyond(<value>)" by the side they lie on, and
adds the held-out maps ("held_out:<name>"); each of its cases is played on 3 fixed random
maps (label suffix "/map<k>") and a map is kept only if it is winnable (`certify_exact`,
`optimal_success` >= 0.5). Case labels belong in the manifest: pass
`extra={"custom_cases": [label for label, _ in custom_cases(env)]}` to `save_sets`.
"""
from __future__ import annotations

import numpy as np

from acl_bench.exam.sets import PairSet

OFFSETS = (0.0, 0.1, 0.5)
LAYOUT_SEEDS = (101, 102, 103)
MIN_OPTIMAL_SUCCESS = 0.5
FROZENLAKE_BEYOND = {"size": (11, 15), "hole_density": (0.32, 0.4, 0.08, 0.0), "slip": (0.73, 0.9)}


def _is_frozenlake(env) -> bool:
    return "layout_seed" in env.PARAM_BOUNDS


def _label(setting: str, side: str, offset: float) -> str:
    return f"{setting}:{side}" if offset == 0 else f"{setting}:{side}+{round(offset * 100)}%"


def _swept(env) -> list[tuple[str, dict]]:
    out = [("default", dict(env.DEFAULTS))]
    for setting, (lo, hi) in env.PARAM_BOUNDS.items():
        if setting == "layout_seed":
            continue
        for side, edge, sign in (("min", lo, -1), ("max", hi, 1)):
            for offset in OFFSETS:
                if _is_frozenlake(env) and offset:
                    continue
                out.append((_label(setting, side, offset), {**env.DEFAULTS, setting: edge + sign * offset * (hi - lo)}))
    if _is_frozenlake(env):
        for setting, values in FROZENLAKE_BEYOND.items():
            lo, hi = env.PARAM_BOUNDS[setting]
            for v in values:
                side = "min" if v < lo else "max"
                out.append((f"{setting}:{side}+beyond({v:g})", {**env.DEFAULTS, setting: v}))
    return [(label, p) for label, p in out if env.valid(p)]


def custom_cases(env) -> list[tuple[str, dict]]:
    """(label, params) per case, in the order the section lists them."""
    cases = _swept(env)
    if not _is_frozenlake(env):
        return cases
    kept = []
    for label, p in cases:
        for k, seed in enumerate(LAYOUT_SEEDS, 1):
            q = {**p, "layout_seed": seed}
            row = np.array([q[s] for s in env.PARAM_ORDER], dtype=np.float64)
            s0 = env.sample_starts(row, 1, np.random.default_rng(0))
            if env.certify_exact(row[None], s0).all() and env.optimal_success(row) >= MIN_OPTIMAL_SUCCESS:
                kept.append((f"{label}/map{k}", q))
    for name, (row, _) in env.held_out_questions().items():
        q = dict(zip(env.PARAM_ORDER, row.tolist()))
        if env.optimal_success(row) >= MIN_OPTIMAL_SUCCESS:
            kept.append((f"held_out:{name}", q))
    return kept


def custom_suite(env, starts: int = 10, seed: int = 0) -> PairSet:
    """One question per case: its params played from `starts` start states."""
    rng = np.random.default_rng(seed)
    rows = [np.array([p[s] for s in env.PARAM_ORDER], dtype=np.float64) for _, p in custom_cases(env)]
    params = np.repeat(np.stack(rows), starts, axis=0)
    s0 = np.concatenate([env.sample_starts(row, starts, rng) for row in rows])
    return PairSet("custom", params, s0)
