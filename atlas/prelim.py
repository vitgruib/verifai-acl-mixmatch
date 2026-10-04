"""Rank falsifiers from a directory of atlas run records.

Reads every <root>/<algo>/s<seed>/<space>_<sampler>_r<k>/ and scores each sampler, per space and
per algo, against random search at the same budget:

  cex_rate   counterexamples per valid level (search efficiency)
  modes      distinct coarse failure modes among counterexamples (cells of DESC bins; diversity)
  first_i    index of the first counterexample (time to first failure; budget-normalised)
  invalid    share of the budget wasted on invalid levels
  min_rho    worst robustness value found

The headline score is `modes` relative to random, since the atlas wants many kinds of failure, not
the same failure resampled. A sampler that raises cex_rate without raising modes is exploiting
one failure. Also tests H2 (the PLR buffer under-covers failures) by comparing buf_score_pct and
dr_knn for counterexamples vs non-counterexamples, and prints which level features separate
failing from passing levels (random sampler only, so the comparison is not shaped by a sampler).

  python -m atlas.prelim runs/maze_prelim --csv results/atlas/prelim_maze.csv
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from atlas.record import load

# Coarse failure-mode bins: (key, width). Width chosen so a 13x13 maze has ~4-8 bins per key.
MODE_BINS = (("n_walls", 10), ("path_len", 6), ("dead_ends", 5))


def mode_of(desc):
    return tuple(None if desc.get(k) is None else int(desc[k]) // w for k, w in MODE_BINS)


def run_stats(rows, budget):
    valid = [r for r in rows if r["valid"]]
    cex = [r for r in valid if r["cex"]]
    first = cex[0]["i"] if cex else budget
    return {
        "cex_rate": len(cex) / max(len(valid), 1),
        "modes": len({mode_of(r["desc"]) for r in cex}),
        "first_i": first,
        "invalid": 1 - len(valid) / max(len(rows), 1),
        "min_rho": min((r["rho"] for r in valid), default=np.nan),
    }


def h2_rows(rows):
    """(key, value, is_cex) for train-context fields of valid levels."""
    for r in rows:
        if r["valid"]:
            for k, v in (r.get("train") or {}).items():
                if v is not None:
                    yield k, v, r["cex"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    stats = defaultdict(list)          # (algo, space, sampler) -> [run_stats]
    h2 = defaultdict(lambda: ([], []))  # (algo, key) -> (cex values, ok values)
    desc = defaultdict(lambda: ([], []))  # (space, key) -> (cex values, ok values), random only
    for d in sorted(glob.glob(os.path.join(a.root, "*", "s*", "*_r*"))):
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        header, rows, _ = load(d)
        key = (header["algo"], header["space"]["name"], header["falsifier"]["name"])
        stats[key].append(run_stats(rows, header["budget"]))
        for k, v, is_cex in h2_rows(rows):
            h2[(header["algo"], k)][0 if is_cex else 1].append(v)
        if header["falsifier"]["name"] == "random":
            for r in rows:
                if r["valid"]:
                    for k, v in r["desc"].items():
                        desc[(header["space"]["name"], k)][0 if r["cex"] else 1].append(v)

    metrics = ("cex_rate", "modes", "first_i", "invalid", "min_rho")
    table = []
    for (algo, sp, smp), runs in sorted(stats.items()):
        row = {"algo": algo, "space": sp, "sampler": smp, "runs": len(runs)}
        for m in metrics:
            row[m] = float(np.mean([r[m] for r in runs]))
        table.append(row)
    # ratios vs random within (algo, space)
    base = {(r["algo"], r["space"]): r for r in table if r["sampler"] == "random"}
    for r in table:
        b = base.get((r["algo"], r["space"]))
        r["cex_x"] = r["cex_rate"] / b["cex_rate"] if b and b["cex_rate"] else np.nan
        r["modes_x"] = r["modes"] / b["modes"] if b and b["modes"] else np.nan

    cols = ["algo", "space", "sampler", "runs", *metrics, "cex_x", "modes_x"]
    print(" ".join(f"{c:>9}" for c in cols))
    for r in table:
        print(" ".join(f"{r[c]:>9.3f}" if isinstance(r[c], float) else f"{r[c]:>9}" for c in cols))

    # overall ranking per space: geometric mean of modes_x across algos
    print("\nper-space ranking (geo-mean over algos of modes vs random; cex vs random):")
    by = defaultdict(list)
    for r in table:
        by[(r["space"], r["sampler"])].append(r)
    for sp in sorted({r["space"] for r in table}):
        ranked = []
        for (s2, smp), rs in by.items():
            if s2 == sp:
                gm = lambda k: float(np.exp(np.nanmean(np.log([max(x[k], 1e-3) for x in rs]))))
                ranked.append((gm("modes_x"), gm("cex_x"), smp, np.mean([x["invalid"] for x in rs])))
        for mx, cx, smp, inv in sorted(ranked, reverse=True):
            print(f"  {sp:>4} {smp:>7}: modes x{mx:.2f}  cex x{cx:.2f}  invalid {inv:.3f}")

    print("\nH2 train-context (mean for cex vs ok, pooled over spaces/samplers):")
    for (algo, k), (c, o) in sorted(h2.items()):
        if c and o:
            print(f"  {algo:>6} {k:>14}: cex {np.mean(c):8.3f} (n={len(c)})  ok {np.mean(o):8.3f} (n={len(o)})")

    print("\nlevel features, cex vs ok (random sampler, pooled over algos):")
    for (sp, k), (c, o) in sorted(desc.items()):
        if c and o and np.mean(o):
            print(f"  {sp:>4} {k:>14}: cex {np.mean(c):6.1f}  ok {np.mean(o):6.1f}  x{np.mean(c) / np.mean(o):.2f}")

    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in table:
                w.writerow({c: round(r[c], 4) if isinstance(r[c], float) else r[c] for c in cols})
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
