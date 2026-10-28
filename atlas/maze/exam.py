"""Build and freeze the Maze exam: random, custom (atlas.maze.suite) and VerifAI sections.

VerifAI section: one cross-entropy search over the 25-wall `dr25` space per reference DR agent
(seeds 100-109), spec as in atlas.maze.falsify (solve rate over 10 plays >= 0.5). Every
oracle-solvable level the searches visit is pooled, deduplicated and replayed by all 10 agents;
a level is kept when at least 3 of them fail it (solve rate < 0.5). The kept pool is capped at
3,000 by a seeded uniform subset; the full pool is written next to the exam.

Freezing never rebuilds a section that is already in the --out manifest, so the random and
custom sections built once on one machine stay byte-identical on another (JAX's random draws
may differ across backends). A frozen VerifAI section records the reference checkpoints it was
built from, and a rerun with different ones refuses rather than overwrite it.

  python -m atlas.maze.exam --sections random custom              # build the backend-free part
  python -m atlas.maze.exam --ref-dir checkpoints/<ref run name>   # add VerifAI (GPU box)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np

from atlas.maze import oracle, space, suite
from atlas.maze.falsify import make_scenario

SPACE = "dr25"
REF_SEEDS = range(100, 110)
N_REF = len(REF_SEEDS)
PLAYS = 10
TAU = 0.5
CAP = 3000
SECTIONS = ("random", "custom", "verifai")


def _seed(*parts) -> int:
    return int(np.random.SeedSequence(list(parts)).generate_state(1)[0] % 2**31)


def _level(ms, i):
    return ms.walls[i], tuple(int(v) for v in ms.agent[i]), int(ms.agent_dir[i]), tuple(int(v) for v in ms.goal[i])


def solve_rates(policy, ms: suite.MazeSet, plays: int = PLAYS, seed: int = suite.EXAM_SEED) -> np.ndarray:
    """Fraction of `plays` stochastic episodes solved (return > 0) on each level of `ms`."""
    return np.array([(policy.run(*_level(ms, i), plays, _seed(seed, i))[0] > 0).mean()
                     for i in range(len(ms))])


def _search(policy, iters: int, seed: int) -> list:
    """One CE search against `policy`; -> the oracle-solvable levels it visited."""
    random.seed(seed)
    np.random.seed(seed)
    scen = make_scenario(SPACE, "ce", seed)
    levels, fb = [], None
    for n in range(iters):
        scene, _ = scen.generate(feedback=fb)
        walls, agent, d, goal, why = space.build(SPACE, {k: float(scene.params[k]) for k in space.SPACES[SPACE]})
        if why or oracle.solve(walls, agent, d, goal)["status"] != oracle.SOLVABLE:
            fb = 1 - TAU
            continue
        rets = policy.run(walls, agent, d, goal, PLAYS, _seed(seed, n))[0]
        fb = float((rets > 0).mean()) - TAU
        levels.append((walls, agent, d, goal))
    return levels


def verifai_section(policies: list, iters: int = 3000, seed: int = suite.EXAM_SEED,
                    min_fail: int = 3) -> tuple[suite.MazeSet, suite.MazeSet]:
    """-> (section capped at CAP levels, full pool of levels >= `min_fail` of `policies` fail)."""
    pool = {}
    for k, pol in enumerate(policies):
        for lv in _search(pol, iters, _seed(seed, 1, k)):
            pool.setdefault(lv[0].tobytes() + bytes([*lv[1], lv[2], *lv[3]]), lv)
    ms = suite._from_levels(list(pool.values()))
    rates = np.stack([solve_rates(pol, ms, PLAYS, _seed(seed, 3, j)) for j, pol in enumerate(policies)])
    fails = (rates < TAU).sum(0)
    keep = fails >= min_fail
    full = suite.MazeSet(ms.walls[keep], ms.agent[keep], ms.agent_dir[keep], ms.goal[keep],
                         ref_fail_frac=fails[keep] / len(policies))
    if len(full) <= CAP:
        return full, full
    idx = np.sort(np.random.default_rng(np.random.SeedSequence([seed, 2])).choice(len(full), CAP, replace=False))
    capped = suite.MazeSet(full.walls[idx], full.agent[idx], full.agent_dir[idx], full.goal[idx],
                           ref_fail_frac=full.ref_fail_frac[idx])
    return capped, full


def _final_step(cfg: dict) -> int:
    """The last orbax step a finished JaxUED run writes: it saves eval steps
    0 .. num_updates // eval_freq - 1, every checkpoint_save_interval-th one (maze_dr.py)."""
    every = cfg["checkpoint_save_interval"]
    return (cfg["num_updates"] // cfg["eval_freq"] - 1) // every * every


def _ref_runs(root: str) -> list[tuple[str, int]]:
    """-> (seed dir, final step) for each of REF_SEEDS; stops unless every run finished training."""
    runs, problems = [], []
    for s in REF_SEEDS:
        d = os.path.join(root, str(s))
        cfg_path = os.path.join(d, "config.json")
        if not os.path.exists(cfg_path):
            problems.append(f"{s}: missing")
            continue
        with open(cfg_path) as f:
            step = _final_step(json.load(f))
        if not os.path.isdir(os.path.join(d, "models", str(step))):
            problems.append(f"{s}: no checkpoint at its final step {step} (training unfinished)")
        runs.append((d, step))
    if problems:
        sys.exit(f"the VerifAI section needs all {N_REF} reference agents (seeds {REF_SEEDS[0]}-"
                 f"{REF_SEEDS[-1]}) fully trained in {root}; " + "; ".join(problems))
    return runs


def _fingerprint(dirs: list[str]) -> dict:
    """Per reference checkpoint: its config.json digest and saved steps (no JAX needed)."""
    out = {}
    for d in dirs:
        with open(os.path.join(d, "config.json"), "rb") as f:
            cfg = hashlib.sha256(f.read()).hexdigest()
        out[os.path.basename(d)] = {"config_sha256": cfg, "steps": sorted(os.listdir(os.path.join(d, "models")))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", help="checkpoints/<ref run name>, holding the 10 reference DR seed dirs")
    ap.add_argument("--out", default="frozen_sets/suite/maze")
    ap.add_argument("--pool-out", default="frozen_sets/suite_pool/maze")
    ap.add_argument("--iters", type=int, default=3000, help="CE iterations per reference agent")
    ap.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    a = ap.parse_args(argv)
    if "verifai" in a.sections:
        if not a.ref_dir:
            ap.error("--ref-dir is required to build the VerifAI section")
        refs = _ref_runs(a.ref_dir)
    frozen = suite.load_sets(a.out) if os.path.exists(os.path.join(a.out, "manifest.json")) else {}
    for name, build in (("random", suite.random_section), ("custom", suite.custom_section)):
        if name in a.sections and name not in frozen:
            suite.save_set(a.out, name, build())
            print(f"froze {name}", flush=True)
    if "verifai" not in a.sections:
        return
    fp = _fingerprint([d for d, _ in refs])
    fp_path = os.path.join(a.out, "verifai_refs.json")
    if "verifai" in frozen:
        stored = None
        if os.path.exists(fp_path):
            with open(fp_path) as f:
                stored = json.load(f)
        if stored != fp:
            sys.exit(f"{a.out} holds a VerifAI section built from other reference checkpoints "
                     f"({fp_path}); refusing to rebuild a frozen section")
        print("exam already frozen; nothing to do", flush=True)
        return
    from atlas.maze.policy import Policy
    capped, full = verifai_section([Policy(d, step) for d, step in refs], a.iters)
    suite.save_set(a.pool_out, "verifai", full)
    with open(fp_path, "w") as f:
        json.dump(fp, f, indent=2, sort_keys=True)
    suite.save_set(a.out, "verifai", capped)
    print(f"froze verifai: {len(capped)} of {len(full)} levels", flush=True)


if __name__ == "__main__":
    main()
