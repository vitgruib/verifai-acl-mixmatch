"""Turn a VerifAI falsification search (find_edge_cases.py) into locked, named
exam sections, replacing docs/cartpole_suite.md's quartile-assumed E1-E5
("weak push", "heavy cart", ...) with clusters actually found to be hard.
E0 (general), E6 (hard-by-measurement) and E7 (easy) carry over from v2
unchanged -- they were already evidence-based (reference agents), not assumed.

Every hard-but-winnable question (ref_fail_frac in [0.5, 1)) found by any of
the three samplers is pooled, clustered in normalized 5D task-parameter space,
and each cluster becomes one locked section, named by its most extreme
parameters (e.g. "force_mag=very low, masscart=very high").

    python -m acl_bench.suite.build_edge_sections \\
        --search frozen_sets/cartpole_v3_search --v2 frozen_sets/cartpole_v2 \\
        --out frozen_sets/cartpole_v3 --n-clusters 5 --section-size 100
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.cluster import KMeans

from acl_bench.envs.param_cartpole import PARAM_BOUNDS
from acl_bench.suite.evaluator import PARAM_ORDER
from acl_bench.suite.sets import PairSet, load_sets, save_sets

QUARTILE_LABELS = {0: "very low", 1: "low", 2: "high", 3: "very high"}


def normalize(params: np.ndarray) -> np.ndarray:
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_ORDER])
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_ORDER])
    return (params - lo) / (hi - lo)


def describe_cluster(params: np.ndarray) -> str:
    """Names a cluster by its most extreme parameters (quartile of the mean)."""
    norm = normalize(params).mean(axis=0)
    quartile = np.clip((norm * 4).astype(int), 0, 3)
    extremity = np.abs(norm - 0.5)
    order = np.argsort(-extremity)
    parts = [f"{PARAM_ORDER[i]}={QUARTILE_LABELS[quartile[i]]}"
             for i in order[:2] if extremity[i] > 0.15]
    return ", ".join(parts) if parts else "mixed / no dominant parameter"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", default="frozen_sets/cartpole_v3_search")
    parser.add_argument("--v2", default="frozen_sets/cartpole_v2", help="carries over E0, E6, E7")
    parser.add_argument("--out", default="frozen_sets/cartpole_v3")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--section-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    search = load_sets(args.search)
    v2 = load_sets(args.v2, names=["E0", "E6", "E7"])

    params = np.concatenate([ps.params for ps in search.values()])
    s0 = np.concatenate([ps.s0 for ps in search.values()])
    fail_frac = np.concatenate([ps.ref_fail_frac for ps in search.values()])
    hard = (fail_frac >= 0.5) & (fail_frac < 1.0)
    params, s0, fail_frac = params[hard], s0[hard], fail_frac[hard]
    print(f"{len(params)} hard-but-winnable questions pooled from {len(search)} samplers")

    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    labels = km.fit_predict(normalize(params))

    sections = dict(v2)
    manifest_notes = {}
    for c in range(args.n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            continue
        idx = np.flatnonzero(mask)
        rng = np.random.default_rng(args.seed + c)
        idx = rng.permutation(idx)[:args.section_size]
        name = f"E{c + 1}v"
        sections[name] = PairSet(name, params[idx], s0[idx], fail_frac[idx])
        desc = describe_cluster(params[mask])
        manifest_notes[name] = {"n_found": int(mask.sum()), "description": desc,
                                 "mean_fail_frac": float(fail_frac[mask].mean())}
        print(f"{name}: {mask.sum()} found, kept {len(idx)}, mean fail_frac "
              f"{fail_frac[mask].mean():.2f} -- {desc}")

    save_sets(sections, args.out, extra={"edge_clusters": manifest_notes,
                                          "note": "E1v..Env replace v2's quartile-assumed E1-E5; "
                                                  "found by VerifAI falsification search "
                                                  "(find_edge_cases.py), not assumed. "
                                                  "E0/E6/E7 carried over unchanged from v2."})
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
