"""Build the exam's VerifAI suite from the raw search pools (acl_bench.exam.search).

The exam has two suites:

  - `random`: 282 random questions from fixed seeds, cleaned once of impossible
    questions (locked; never rebuilt).
  - `verifai`: every question the VerifAI samplers (ce, mab, sa) discovered that at
    least `--min-fail-frac` of the reference agents fail (default 0.6, i.e. 6+ of 10,
    including questions every agent fails) and that is proven winnable
    (acl_bench.exam.feasibility), so it holds no impossible or unresolved question.

    python -m acl_bench.exam.build
"""
from __future__ import annotations

import argparse

import numpy as np

from acl_bench.exam.feasibility import IMPOSSIBLE, UNKNOWN, WINNABLE, classify
from acl_bench.exam.search import EXAM_DIR, SEARCH_DIR
from acl_bench.exam.sets import PairSet, load_sets, remove_sets, save_sets
from acl_bench.sampling import ADAPTIVE_SAMPLERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", default=SEARCH_DIR)
    parser.add_argument("--exam", default=EXAM_DIR)
    parser.add_argument("--min-fail-frac", type=float, default=0.6)
    args = parser.parse_args()

    pools = load_sets(args.search, names=[f"SEARCH_{s}" for s in ADAPTIVE_SAMPLERS])
    params = np.concatenate([p.params for p in pools.values()])
    s0 = np.concatenate([p.s0 for p in pools.values()])
    fail_frac = np.concatenate([p.ref_fail_frac for p in pools.values()])

    hard = np.flatnonzero(fail_frac >= args.min_fail_frac - 1e-9)     # fail_frac is k/n_agents
    print(f"{len(hard)} questions with fail_frac >= {args.min_fail_frac}; proving winnability...", flush=True)
    status = classify(params[hard], s0[hard])["status"]
    feasibility = {k: {"fail_some": int(((status == k) & (fail_frac[hard] < 1)).sum()),
                       "fail_all": int(((status == k) & (fail_frac[hard] == 1)).sum())}
                   for k in (WINNABLE, IMPOSSIBLE, UNKNOWN)}
    keep = hard[status == WINNABLE]
    suite = PairSet("verifai", params[keep], s0[keep], fail_frac[keep])

    remove_sets([n for n in load_sets(args.exam) if n != "random"], args.exam)
    save_sets({"verifai": suite}, args.exam, extra={"verifai": {
        "samplers": list(ADAPTIVE_SAMPLERS), "min_fail_frac": args.min_fail_frac,
        "feasibility_of_candidates": feasibility, "mean_fail_frac": float(suite.ref_fail_frac.mean()),
        "share_failed_by_all": float((suite.ref_fail_frac == 1).mean())}})
    print(f"feasibility: {feasibility}\nverifai: {len(suite)} questions -> {args.exam}")


if __name__ == "__main__":
    main()
