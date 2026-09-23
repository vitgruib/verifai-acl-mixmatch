"""Build the exam's searched sections from the raw search pools (acl_bench.exam.search).
Every question in them is proven winnable (acl_bench.exam.feasibility), so no section
holds an impossible question or one of unknown status. E0 (general) is locked and
never rebuilt.

  - Hard questions: at least `--min-fail-frac` of the reference agents fail it (default
    0.6, i.e. 6+ of 10, including questions every agent fails), from every sampler's pool.
  - Edge sections E1v, E2v, ...: the hard questions clustered (k-means, normalized 5D
    task space), one section per cluster of at least `--min-section-size`, largest
    first, each named by its most extreme parameters.
  - E6 (hard): `--hard-size` hard questions not used by any edge section, drawn
    uniformly, so the hard score covers the whole failure landscape the search found.
  - E7 (easy): `--easy-size` questions every reference agent passes, drawn from the
    random sampler's pool (uniform over the task space), so it checks whether a method
    got better at hard questions by getting worse at ordinary ones.

    python -m acl_bench.exam.build
"""
from __future__ import annotations

import argparse
import re

import numpy as np
from sklearn.cluster import KMeans

from acl_bench.cartpole import PARAM_BOUNDS, PARAM_ORDER
from acl_bench.exam.feasibility import IMPOSSIBLE, UNKNOWN, WINNABLE, classify
from acl_bench.exam.search import EXAM_DIR, SEARCH_DIR
from acl_bench.exam.sets import PairSet, load_sets, remove_sets, save_sets

QUARTILE_LABELS = {0: "very low", 1: "low", 2: "high", 3: "very high"}


def normalize(params: np.ndarray) -> np.ndarray:
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_ORDER])
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_ORDER])
    return (params - lo) / (hi - lo)


def describe_cluster(params: np.ndarray) -> str:
    """Names a cluster by every parameter whose mean sits in the outer quarter of its
    range, most extreme first (the hard clusters share a corner, so a shorter name
    would hide what they have in common)."""
    norm = normalize(params).mean(axis=0)
    quartile = np.clip((norm * 4).astype(int), 0, 3)
    extremity = np.abs(norm - 0.5)
    parts = [f"{PARAM_ORDER[i]}={QUARTILE_LABELS[quartile[i]]}"
             for i in np.argsort(-extremity) if extremity[i] > 0.25]
    return ", ".join(parts) if parts else "mixed / no dominant parameter"


def pooled(search: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = sorted(search)
    source = np.concatenate([[n] * len(search[n]) for n in names])
    return (np.concatenate([search[n].params for n in names]), np.concatenate([search[n].s0 for n in names]),
            np.concatenate([search[n].ref_fail_frac for n in names]), source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", default=SEARCH_DIR)
    parser.add_argument("--exam", default=EXAM_DIR)
    parser.add_argument("--min-fail-frac", type=float, default=0.6)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--section-size", type=int, default=100)
    parser.add_argument("--min-section-size", type=int, default=30)
    parser.add_argument("--hard-size", type=int, default=250)
    parser.add_argument("--easy-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    params, s0, fail_frac, source = pooled(load_sets(args.search))

    # --- hard pool: proven winnable, 6+ of 10 fail
    hard = np.flatnonzero(fail_frac >= args.min_fail_frac - 1e-9)   # fail_frac is k/n_agents
    print(f"{len(hard)} questions with fail_frac >= {args.min_fail_frac}; proving winnability...", flush=True)
    status = classify(params[hard], s0[hard])["status"]
    feasibility = {k: {"fail_some": int(((status == k) & (fail_frac[hard] < 1)).sum()),
                       "fail_all": int(((status == k) & (fail_frac[hard] == 1)).sum())}
                   for k in (WINNABLE, IMPOSSIBLE, UNKNOWN)}
    print(f"feasibility: {feasibility}")
    hard = hard[status == WINNABLE]

    # --- edge sections: clusters of the hard pool
    labels = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10).fit_predict(
        normalize(params[hard]))
    sections, edge_notes, dropped, used = {}, {}, [], set()
    for c in sorted(range(args.n_clusters), key=lambda c: -(labels == c).sum()):
        members = hard[labels == c]
        desc = describe_cluster(params[members]) if len(members) else ""
        if len(members) < args.min_section_size:
            dropped.append({"n_found": len(members), "description": desc})
            continue
        idx = np.random.default_rng(args.seed + c).permutation(members)[:args.section_size]
        name = f"E{len(edge_notes) + 1}v"
        sections[name] = PairSet(name, params[idx], s0[idx], fail_frac[idx])
        used.update(idx.tolist())
        edge_notes[name] = {"n_found": len(members), "description": desc,
                            "mean_fail_frac": float(fail_frac[members].mean()),
                            "share_all_fail": float((fail_frac[members] == 1.0).mean())}

    # --- E6: the rest of the hard pool, drawn uniformly
    rest = np.array([i for i in hard if i not in used])
    idx = np.sort(rng.permutation(rest)[:args.hard_size])
    sections["E6"] = PairSet("E6", params[idx], s0[idx], fail_frac[idx])

    # --- E7: every agent passes, from the random sampler's pool, proven winnable
    easy = rng.permutation(np.flatnonzero((fail_frac == 0.0) & (source == "SEARCH_random")))
    picked = []
    for start in range(0, len(easy), 2 * args.easy_size):              # prove in batches until enough
        batch = easy[start:start + 2 * args.easy_size]
        picked += batch[classify(params[batch], s0[batch])["status"] == WINNABLE].tolist()
        if len(picked) >= args.easy_size:
            break
    idx = np.sort(np.array(picked[:args.easy_size]))
    sections["E7"] = PairSet("E7", params[idx], s0[idx], fail_frac[idx])

    old = [n for n in load_sets(args.exam) if n in ("E6", "E7") or re.fullmatch(r"E\d+v", n)]
    remove_sets(old, args.exam)                                          # E0 stays; everything else is rebuilt
    save_sets(sections, args.exam, extra={"searched_sections": {
        "min_fail_frac": args.min_fail_frac, "seed": args.seed, "feasibility_of_hard_candidates": feasibility,
        "edge": edge_notes, "dropped_small_clusters": dropped,
        "E6": {"n_hard_winnable": len(hard), "n_after_edge": len(rest),
               "mean_fail_frac": float(sections["E6"].ref_fail_frac.mean()),
               "share_all_fail": float((sections["E6"].ref_fail_frac == 1).mean())},
        "E7": {"source": "SEARCH_random", "rule": "every reference agent passes; proven winnable"},
        "note": "Every searched question is proven winnable (feasibility.py). E0 is locked and not rebuilt."}})
    for name, ps in sections.items():
        note = edge_notes.get(name, {}).get("description", "")
        print(f"{name}: {len(ps)} questions, mean fail_frac {ps.ref_fail_frac.mean():.2f} {note}")
    print(f"dropped clusters: {dropped}\nsaved -> {args.exam}")


if __name__ == "__main__":
    main()
