"""Turn a VerifAI falsification search (find_edge_cases.py) into locked, named
exam sections, replacing docs/cartpole_suite.md's quartile-assumed E1-E5
("weak push", "heavy cart", ...) with clusters actually found to be hard.
E0 (general), E6 (hard-by-measurement) and E7 (easy) carry over from v2
unchanged -- they were already evidence-based (reference agents), not assumed.

Every hard question (at least `--min-fail-frac` of the reference agents fail
it; default 0.6, i.e. 6+ of 10, including questions every agent fails) found by
any of the three samplers is pooled, keeping only questions PROVEN winnable (a
surviving action sequence was found and replayed; acl_bench/suite/feasibility.py),
so a section never contains an impossible question or one of unknown status. The
kept questions are clustered in normalized 5D task-parameter space, and each cluster becomes one locked section, named by its most extreme
parameters (e.g. "force_mag=very low, masscart=very high").

    python -m acl_bench.suite.build_edge_sections \\
        --search frozen_sets/cartpole_v3_search --v2 frozen_sets/cartpole_v2 \\
        --out frozen_sets/cartpole_v3 --n-clusters 5 --section-size 100
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from sklearn.cluster import KMeans

from acl_bench.envs.param_cartpole import PARAM_BOUNDS
from acl_bench.suite.evaluator import PARAM_ORDER
from acl_bench.suite.feasibility import IMPOSSIBLE, UNKNOWN, WINNABLE, classify
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
    parser.add_argument("--min-section-size", type=int, default=30,
                        help="drop clusters smaller than this (too few questions to score)")
    parser.add_argument("--min-fail-frac", type=float, default=0.6,
                        help="keep questions at least this share of reference agents fail (1.0 included)")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    search = load_sets(args.search)
    v2 = load_sets(args.v2, names=["E0", "E6", "E7"])

    params = np.concatenate([ps.params for ps in search.values()])
    s0 = np.concatenate([ps.s0 for ps in search.values()])
    fail_frac = np.concatenate([ps.ref_fail_frac for ps in search.values()])
    hard = fail_frac >= args.min_fail_frac - 1e-9       # fail_frac is k/n_agents; tolerate float error
    params, s0, fail_frac = params[hard], s0[hard], fail_frac[hard]
    print(f"{len(params)} questions with fail_frac >= {args.min_fail_frac} from {len(search)} samplers; "
          f"classifying winnability (a few minutes)...", flush=True)
    status = classify(params, s0)["status"]
    counts = {k: {"fail_6_9": int(((status == k) & (fail_frac < 1)).sum()),
                  "fail_all": int(((status == k) & (fail_frac == 1)).sum())}
              for k in (WINNABLE, IMPOSSIBLE, UNKNOWN)}
    print(f"feasibility: {counts}")
    keep = status == WINNABLE
    params, s0, fail_frac = params[keep], s0[keep], fail_frac[keep]
    print(f"{len(params)} proven winnable kept ({int((fail_frac == 1.0).sum())} failed by every reference agent)")

    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    labels = km.fit_predict(normalize(params))

    sections = dict(v2)
    manifest_notes, dropped = {}, []
    by_size = sorted(range(args.n_clusters), key=lambda c: -(labels == c).sum())   # E1v = largest
    for c in by_size:
        mask = labels == c
        if mask.sum() < args.min_section_size:
            dropped.append({"n_found": int(mask.sum()), "description": describe_cluster(params[mask])
                            if mask.any() else ""})
            print(f"dropped a cluster of {mask.sum()} (< {args.min_section_size}): {dropped[-1]['description']}")
            continue
        idx = np.flatnonzero(mask)
        rng = np.random.default_rng(args.seed + c)
        idx = rng.permutation(idx)[:args.section_size]
        name = f"E{len(manifest_notes) + 1}v"
        sections[name] = PairSet(name, params[idx], s0[idx], fail_frac[idx])
        desc = describe_cluster(params[mask])
        manifest_notes[name] = {"n_found": int(mask.sum()), "description": desc,
                                 "mean_fail_frac": float(fail_frac[mask].mean()),
                                 "share_all_fail": float((fail_frac[mask] == 1.0).mean())}
        print(f"{name}: {mask.sum()} found, kept {len(idx)}, mean fail_frac "
              f"{fail_frac[mask].mean():.2f} -- {desc}")

    if os.path.isdir(args.out):          # a full rebuild: save_sets merges, so clear stale sections first
        shutil.rmtree(args.out)
    save_sets(sections, args.out, extra={"edge_clusters": manifest_notes, "min_fail_frac": args.min_fail_frac,
                                          "feasibility": counts,
                                          "dropped_small_clusters": dropped,
                                          "note": "E1v..Env replace v2's quartile-assumed E1-E5; "
                                                  "found by VerifAI falsification search "
                                                  "(find_edge_cases.py), not assumed. "
                                                  "E0/E6/E7 carried over unchanged from v2. Edge sections include "
                                                  "questions every reference agent fails, but only "
                                                  "questions proven winnable (feasibility.py)."})
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
