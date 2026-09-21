"""Build the hard (E6), easy (E7) and POOL sections from saved reference agents.

    python -m acl_bench.suite.build_pool --snapshots results/snapshots --arm REF

E6 and E7 are saved in the main exam; the POOL reporting sample goes to a separate folder
and is graded from saved snapshots after a run, not at every check-in.

Each reference run is represented by its best check-in among the last few (highest
General-exam score, ties to the later one): runs sometimes dip temporarily after
reaching their plateau, and a collapsed check-in would misjudge difficulty. Selecting
reference agents by their score on E0 is fine because they are not among the methods
being compared. The choices are recorded in the manifest.
"""
from __future__ import annotations

import argparse

from acl_bench.suite.evaluator import rollout_steps
from acl_bench.suite.sets import MAX_STEPS, build_pool_sets, load_sets, save_sets
from acl_bench.suite.snapshots import find_runs, load_run

LAST_K = 5


def pick_reference(agents_by_step: dict, e0, last_k: int = LAST_K):
    """(step, agent): the best-scoring of the last `last_k` check-ins on E0."""
    steps = sorted(agents_by_step)[-last_k:]
    scores = {s: float((rollout_steps(agents_by_step[s], e0.params, e0.s0, MAX_STEPS) == MAX_STEPS).mean())
              for s in steps}
    best = max(steps, key=lambda s: (scores[s], s))
    return best, agents_by_step[best], scores[best]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--arm", default="REF")
    parser.add_argument("--sets", default="frozen_sets/cartpole_v2")
    parser.add_argument("--pool-size", type=int, default=80_000)
    parser.add_argument("--report-size", type=int, default=10_000)
    parser.add_argument("--pool-dir", default="frozen_sets/cartpole_v2_pool")
    parser.add_argument("--hard-target", type=int, default=250)
    parser.add_argument("--easy-target", type=int, default=100)
    parser.add_argument("--min-hard", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    e0 = load_sets(args.sets, names=["E0"])["E0"]
    agents, picks = [], {}
    for arm, replicate, path in find_runs(args.snapshots):
        if arm != args.arm:
            continue
        step, agent, score = pick_reference(load_run(path), e0)
        agents.append(agent)
        picks[replicate] = {"step": step, "E0_success": round(score, 4)}
    if len(agents) < 5:
        raise SystemExit(f"need at least 5 reference runs of arm {args.arm}, found {len(agents)}")

    pool_size = args.pool_size
    while True:
        sets, stats = build_pool_sets(agents, args.seed, pool_size, hard_target=args.hard_target,
                                      easy_target=args.easy_target, report_size=args.report_size)
        print(f"pool {pool_size}: hard available {stats['hard_available']}, easy available "
              f"{stats['easy_available']}, nobody passes {stats['nobody_passes']}")
        if stats["hard_available"] >= args.min_hard or pool_size >= 320_000:
            break
        pool_size *= 2                                   # too few hard questions: draw a bigger pool
    info = {"arm": args.arm, "n_agents": len(agents), "picks": picks, "pool_size": pool_size,
            "pool_seed": args.seed, "stats": stats,
            "note": "E6 = fail_frac in [0.5, 1); E7 = fail_frac == 0; questions nobody passes are excluded from both"}
    save_sets({k: v for k, v in sets.items() if k != "POOL"}, args.sets, extra={"reference": info})
    save_sets({"POOL": sets["POOL"]}, args.pool_dir, extra={"reference": info})
    print({name: len(ps) for name, ps in sets.items()}, "->", args.sets, "and", args.pool_dir)


if __name__ == "__main__":
    main()
