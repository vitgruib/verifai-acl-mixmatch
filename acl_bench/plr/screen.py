"""Screen named PLR configurations on the fast harness (acl_bench.plr.fast).

    python -m acl_bench.plr.screen --env acrobot --configs DR plr_paper --seeds 1-8 \\
        --frac 0.5 --out results/acrobot/plr/screen.csv

`--frac` shortens training to that share of the environment's budget. Grades both exam
suites at `--checks` evenly spaced check-ins. One CSV row per (config, seed, check-in).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import time

import pandas as pd

from acl_bench import envs
from acl_bench.plr.levels import LevelConfig

DR = dict(replay_prob=0.0)
SIPACL = dict(replay_prob=0.5, beta=1.0, staleness=0.0, score_ema=0.2, buffer=5000, admit="fifo", min_fill=1)
PAPER = dict(replay_prob=0.5, beta=0.1, staleness=0.1, score_ema=1.0, buffer=1000, admit="min")

# name -> (LevelConfig overrides, FastConfig overrides)
CONFIGS: dict[str, tuple[dict, dict]] = {
    "DR": (DR, {}),
    "REF": (DR, {}),          # reference agents for building an exam: DR on independent seeds
    "sipacl": (SIPACL, {}),
    "paper": (PAPER, {}),
}


def register(name: str, levels: dict, base: dict = PAPER, **fast) -> None:
    CONFIGS[name] = ({**base, **levels}, fast)


# ---- round 1: one direction each, from the PLR/Robust-PLR defaults
register("paper_l1", {"score": "l1"})
register("paper_maxmc", {"score": "maxmc"})
register("paper_robust", {"robust": True})
register("paper_p09", {"replay_prob": 0.9})
register("paper_p02", {"replay_prob": 0.2})
register("paper_beta03", {"beta": 0.3})
register("paper_buf200", {"buffer": 200})
register("paper_negret", {"score": "neg_return"})
register("sipacl_beta01", {"beta": 0.1}, base=SIPACL)
SFL = dict(replay_prob=0.5, sfl=True)
register("sfl", {}, base=SFL)

# ---- round 2: SFL's knobs (round 1 found no PVL-family setting better than DR on Acrobot)
register("sfl_p09", {"replay_prob": 0.9}, base=SFL)
register("sfl_top20", {"sfl_top": 20}, base=SFL)
register("sfl_cheap", {"sfl_n": 500, "sfl_every": 20}, base=SFL)

# ---- round 3: headroom (oracle: half the episodes on the VerifAI suite's own tasks) and
# SFL's MountainCar curve gain with more seeds
register("oracle50", {"replay_prob": 0.5, "oracle": "verifai"}, base=DR)
register("sfl_p09_top20", {"replay_prob": 0.9, "sfl_top": 20}, base=SFL)

# ---- round 6: CartPole, around the leaders (sfl, paper, paper_maxmc)
register("sfl_top50", {"sfl_top": 50}, base=SFL)
register("paper_maxmc_buf200", {"score": "maxmc", "buffer": 200})

# ---- round 7: surprise in either direction (L1), averaged over visits, and negative only
register("l1_avg", {"score": "l1", "score_ema": 0.3})
register("nvl", {"score": "nvl"})

# ---- round 8: make PVL-ranked PLR work. Root cause: a task-blind critic makes PVL track
# easiness. Fixes: a privileged critic (_pc), task-aware networks (_po), exact-start
# replay (_st). Each network change has its own DR baseline.
for _sfx, _fast in (("pc", {"critic_params": True}), ("po", {"obs_params": True})):
    CONFIGS[f"DR_{_sfx}"] = (DR, _fast)
    register(f"paper_{_sfx}", {}, **_fast)
    register(f"sipacl_{_sfx}", {}, base=SIPACL, **_fast)
register("paper_st", {"replay_start": True})
register("paper_pc_st", {"replay_start": True}, critic_params=True)
register("sipacl_st", {"replay_start": True}, base=SIPACL)
register("sipacl_pc_st", {"replay_start": True}, base=SIPACL, critic_params=True)
# ---- round 10: from the literature, aimed at the measured failure (PVL tracks easiness)
register("pvl_learn", {"score": "pvl_learn"})                     # PVL gated to the frontier
register("sipacl_learn", {"score": "pvl_learn"}, base=SIPACL)
register("pvl_resid", {"score": "pvl_resid"})                     # PVL decorrelated from return
register("entropy", {"score": "entropy"})                         # PLR paper: policy entropy
register("vds", {"score": "vds", "n_value_ens": 5})               # value disagreement (VDS)
register("accel", {"accel": True, "replay_prob": 0.8})            # ACCEL: PVL + level edits
register("accel_maxmc", {"accel": True, "replay_prob": 0.8, "score": "maxmc"})
register("accel_learn", {"accel": True, "replay_prob": 0.8, "score": "pvl_learn"})
register("paper_rho03", {"staleness": 0.3})                       # Robust PLR / ACCEL staleness
register("pvl_learn_pc", {"score": "pvl_learn"}, critic_params=True)
# ---- round 11: planner regret (PLR's own theory, minimax regret, with the exam's physics
# planner estimating the best achievable result instead of the task-blind critic)
register("regret", {"score": "regret", "replay_start": True})
register("sipacl_regret", {"score": "regret", "replay_start": True}, base=SIPACL)
# ---- round 12: SFL's scouting with PLR's score -- is PVL the problem, or is it that a
# PLR buffer only ranks levels training happened to draw?
register("scout_pvl", {"sfl_score": "pvl"}, base=SFL)
register("scout_pvl_top20", {"sfl_score": "pvl", "sfl_top": 20}, base=SFL)
# ---- candidate 1 (docs/wrapper_methodology.md): learnability-steered VerifAI picker,
# learnability from training episodes only (no extra simulation)
register("vlearn_ce", {"picker": "ce"}, base=DR)
register("vlearn_mab", {"picker": "mab"}, base=DR)
register("vlearn_ce25", {"picker": "ce", "picker_uniform": 0.25}, base=DR)

# ---- candidate 2: a PLR buffer ranked by learnability from training episodes
register("lbuf", {"score": "learn_bucket"})                 # pooled per parameter bucket
register("lbuf_level", {"score": "learn"})                  # each level's own visits

# ---- candidate 5: a pass-rate model over the task space from training episodes
register("fmodel", {"picker": "model"}, base=DR)
register("fmodel25", {"picker": "model", "picker_uniform": 0.25}, base=DR)
register("fmodel_c16", {"picker": "model", "picker_candidates": 16}, base=DR)

# ---- candidate 3: budget-fair SFL (its scouting counts against the budget)
register("sfl_fair", {"charge_scouting": True}, base=SFL)
register("sfl_fair_small", {"charge_scouting": True, "sfl_n": 100, "sfl_k": 4, "sfl_every": 20,
                            "sfl_top": 20}, base=SFL)
# ---- library batch 8 (Amendment 3: scouting uncharged): SFL with a smaller replay share,
# to take less time from the broad distribution (SFL's Acrobot cost)
register("sfl_p25", {"replay_prob": 0.25}, base=SFL)
# ---- library batch 9 (Amendment 3): scouting is free, so search a wider pool -- the hard
# suite's corner holds ~0.3% of uniform mass, so 1000 candidates see ~3 of its tasks
register("sfl_n4k", {"sfl_n": 4000, "sfl_k": 4}, base=SFL)
register("sfl_n8k", {"sfl_n": 8000, "sfl_k": 4}, base=SFL)

# ---- library batch 10: mix and match (docs/literature.md). SFL's scouted frontier combined
# with ACCEL's local edits (reach small regions by hill-climbing from the last frontier) and
# with var_low's tilt toward hard tasks (here on the scouted pass rate, not a model)
register("sfl_mut", {"sfl_mut": 0.5}, base=SFL)
register("sfl_tilt", {"sfl_tilt": 1.0}, base=SFL)
register("sfl_mut_tilt", {"sfl_mut": 0.5, "sfl_tilt": 1.0}, base=SFL)
# robustness check of the confirmed arm (batch 11): other learners, each against its own DR
register("sfl_tilt_po", {"sfl_tilt": 1.0}, base=SFL, obs_params=True)   # vs DR_po
register("DR_lr1e3", {}, base=DR, lr=1e-3)
register("sfl_tilt_lr1e3", {"sfl_tilt": 1.0}, base=SFL, lr=1e-3)
# ---- library batch 12: build on SFL from its follow-ups (docs/literature.md, lineage)
register("sfl_tilt_soft", {"sfl_tilt": 1.0, "sfl_soft": True}, base=SFL)    # NCC: score-proportional replay
register("sfl_tilt_carry", {"sfl_tilt": 1.0, "sfl_carry": True}, base=SFL)  # persistent frontier
register("sfl_tilt_verify", {"sfl_tilt": 1.0, "sfl_verify": 200}, base=SFL)  # fresh-rollout shortlist
register("sfl_tilt_amort", {"sfl_tilt": 1.0, "sfl_amort": 10}, base=SFL)    # k-NN pre-screened scouting
# batch 14 (docs/library_log.md)
register("sfl_tilt_sc", {"sfl_tilt": 1.0, "sfl_soft": True, "sfl_carry": True}, base=SFL)   # soft + carry
register("sfl_halving", {"sfl_tilt": 1.0, "sfl_n": 4000, "sfl_k": 2, "sfl_halving": 2}, base=SFL)  # best-arm id
register("sfl_ghost", {"sfl_tilt": 1.0, "sfl_ghost": 0.5}, base=SFL)          # + learning progress
register("sfl_spread", {"sfl_tilt": 1.0, "sfl_spread": 3}, base=SFL)          # diverse frontier
register("sfl_bisect", {"sfl_tilt": 1.0, "sfl_bisect": 300}, base=SFL)        # boundary bisection
# batch 15: spare Acrobot by stabilizing / diversifying the frontier, or self-tune the intensity
register("sfl_spread_carry", {"sfl_tilt": 1.0, "sfl_spread": 3, "sfl_carry": True}, base=SFL)
register("sfl_spread_verify", {"sfl_tilt": 1.0, "sfl_spread": 3, "sfl_verify": 200}, base=SFL)
register("sfl_spread0", {"sfl_spread": 3}, base=SFL)                          # no tilt
register("sfl_auto", {"sfl_tilt": 1.0, "sfl_auto": 500}, base=SFL)            # bandit replay prob
register("sfl_carry_mem", {"sfl_tilt": 1.0, "sfl_carry": True, "sfl_memory": 0.5}, base=SFL)  # pooled evidence
register("sfl_carry0", {"sfl_carry": True}, base=SFL)                            # carry, no tilt
register("sfl_spread_carry0", {"sfl_spread": 3, "sfl_carry": True}, base=SFL)    # spread + carry, no tilt
register("sfl_carry_mem25", {"sfl_tilt": 1.0, "sfl_carry": True, "sfl_memory": 0.25}, base=SFL)  # weaker memory
register("sfl_spread2_carry", {"sfl_tilt": 1.0, "sfl_spread": 2, "sfl_carry": True}, base=SFL)  # milder spread
register("sfl_spread_carry_mem", {"sfl_tilt": 1.0, "sfl_spread": 3, "sfl_carry": True, "sfl_memory": 0.25}, base=SFL)
register("sfl_states", {"sfl_tilt": 1.0, "sfl_states": 0.5}, base=SFL)   # start-state SFL: half the candidates are visited states
register("sfl_states_carry", {"sfl_tilt": 1.0, "sfl_states": 0.5, "sfl_carry": True}, base=SFL)

# ---- oracle variants (docs/wrapper_methodology.md, section 3): "unlearnable" or "wrong dose"?
register("oracle20", {"replay_prob": 0.2, "oracle": "verifai"}, base=DR)
register("oracle80", {"replay_prob": 0.8, "oracle": "verifai"}, base=DR)
register("oracle_front50", {"replay_prob": 0.5, "oracle": "frontier"}, base=DR)
register("lp", {"score": "lp"})
register("lp_st", {"score": "lp", "replay_start": True})

# ---- the curriculum library (acl_bench.curriculum, docs/library.md): every arm by name
from acl_bench.curriculum import ARMS as _LIB_ARMS  # noqa: E402
for _name in _LIB_ARMS:
    register(_name, {"lib": _name}, base=DR)
# batch 7 (c): a second learner, task-aware PPO (policy and critic see the task), against DR_po
register("var_low_po", {"lib": "var_low"}, base=DR, obs_params=True)


def run_seed(env_name: str, config: str, replicate: int) -> int:
    return int(hashlib.sha256(f"plr:{config}:{replicate}".encode()).hexdigest()[:8], 16)


HARD_SUITES = ("verifai", "calib")     # split into dev / test halves (docs/protocol.md, section 1)


def split_dev_test(sets: dict) -> dict:
    """Each hard suite becomes `<name>_dev` (even positions) and `<name>_test` (odd), a fixed
    split of the frozen section; other sections are kept whole."""
    from acl_bench.exam.sets import PairSet
    out = {}
    for name, ps in sets.items():
        if name not in HARD_SUITES:
            out[name] = ps
            continue
        for tag, sl in (("dev", slice(0, None, 2)), ("test", slice(1, None, 2))):
            ref = ps.ref_fail_frac[sl] if ps.ref_fail_frac is not None else None
            out[f"{name}_{tag}"] = PairSet(f"{name}_{tag}", ps.params[sl], ps.s0[sl], ref)
    return out


def code_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def job_fn(job):
    import torch
    torch.set_num_threads(1)
    from acl_bench.exam.sets import load_sets
    from acl_bench.plr.fast import FastConfig, evaluate, train
    env_name, config, replicate, steps, n_checks, snap_dir = (tuple(job) + (None,))[:6]
    env = envs.get(env_name)
    sets = split_dev_test(load_sets(envs.exam_dir(env_name)))
    lv, fast = CONFIGS[config]
    cfg = FastConfig(steps=steps, seed=run_seed(env_name, config, replicate),
                     levels=dataclasses.replace(LevelConfig(), **lv), **fast)
    snapshots = {}

    def on_check(step, agent):
        if snap_dir:
            snapshots[step] = {k: v.detach().cpu().numpy().copy() for k, v in agent.state_dict().items()}
        return evaluate(env, agent, sets)

    t0 = time.time()
    _, checks, stats = train(env_name, env, cfg, n_checks, on_check)
    wall = time.time() - t0
    if snap_dir:
        from acl_bench.snapshots import save_run
        save_run(os.path.join(snap_dir, config, f"{replicate}.npz"), snapshots)
    commit = code_commit()
    return [{"config": config, "seed": replicate, **c, **stats, "wall_time_sec": wall, "commit": commit}
            for c in checks]


def main():
    from acl_bench.study.arms import BUDGET
    from acl_bench.study.run import append_rows, parse_seeds
    from acl_bench.study.safety import Limits, Watchdog, run_jobs
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=envs.NAMES)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", default="1-8")
    ap.add_argument("--frac", type=float, default=1.0, help="share of the environment's training budget")
    ap.add_argument("--checks", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--min-battery-pct", type=int, default=25)
    ap.add_argument("--snapshots", action="store_true",
                    help="save each run's agent at every check-in under results/<env>/snapshots/<config>/<seed>.npz "
                         "(the study's format, so exam search and regrade read them)")
    args = ap.parse_args()
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; known: {sorted(CONFIGS)}")
    steps = int(BUDGET[args.env] * args.frac) // (1024 * args.checks) * 1024 * args.checks
    done = set()
    if args.resume and os.path.exists(args.out):
        prior = pd.read_csv(args.out, usecols=["config", "seed", "step"])
        prior = prior[prior.step == prior.groupby(["config", "seed"]).step.transform("max")]
        done = {(c, s) for c, s, st in zip(prior.config, prior.seed, prior.step) if st == steps}
    snap = os.path.join(envs.results_dir(args.env), "snapshots") if args.snapshots else None
    jobs = [(args.env, c, s, steps, args.checks, snap) for s in parse_seeds(args.seeds) for c in args.configs
            if (c, s) not in done]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    watchdog = Watchdog(Limits(require_ac=not args.allow_battery, min_battery_pct=args.min_battery_pct),
                        disk_path=os.path.dirname(os.path.abspath(args.out)))
    print(f"{len(jobs)} runs of {steps} steps on {args.workers} workers", flush=True)
    t0, n = time.time(), [0]

    def on_result(rows):
        append_rows(args.out, rows)
        n[0] += 1
        if n[0] % 8 == 0 or n[0] == len(jobs):
            el = time.time() - t0
            print(f"  {n[0]}/{len(jobs)} done, {el / 60:.1f} min, ~{el / n[0] * (len(jobs) - n[0]) / 60:.0f} min left",
                  flush=True)

    failed = run_jobs(jobs, job_fn, args.workers, on_result, watchdog, stop_file="results/STOP",
                      log=lambda m: print(m, flush=True))
    print(f"finished: {n[0]} runs, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
