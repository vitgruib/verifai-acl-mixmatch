"""Falsify a trained DCD BipedalWalker agent (DR, PLR, Robust PLR, ACCEL, minimax, SFL) with a
VerifAI sampler (through Scenic, as in acl_bench.sampling) and write a failure-record directory
(atlas.record).

Spec: the agent reaches the end of the terrain (info['finish'], within 2000 steps) in at least
`tau` of `attempts` stochastic episodes. rho = solve_rate - tau; rho < 0 is a counterexample.
Counterexamples count only on levels shown solvable by a replayed witness (atlas.bipedal.oracle):
the scripted walker (flat/mildly rough levels), a reference checkpoint (--ref-ckpt), or the
policy itself finishing one of its own attempts (oracle.source 'self'). Hence 'hard'
counterexamples (solve rate 0) need a scripted or reference witness; with no --ref-ckpt they
are limited to near-flat levels. Uncertified levels are recorded as invalid (oracle_unknown,
with the policy's results kept) and fed back as rho = 1 - tau so adaptive samplers move away.

  python -m atlas.bipedal.falsify --ckpt $RUNS/dcd/bw_accel/s0 --algo accel \
      --ref-ckpt $RUNS/dcd/bw_dr/s0 $RUNS/dcd/bw_plr/s0 --sampler ce --budget 2000 \
      --out runs/bipedal
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import scenic

from acl_bench.sampling import DEFAULT_SAMPLER_PARAMS, _scenario_params
from atlas import record
from atlas.bipedal import oracle, space
from atlas.bipedal.context import TrainContext
from atlas.bipedal.policy import POS_EVERY, Policy

SAMPLERS = ("random", "ce", "mab")


def make_scenario(space_name: str, sampler: str):
    src = "".join(f"param {k} = VerifaiRange({lo}, {hi})\n" for k, (lo, hi) in space.SPACES[space_name].items())
    src += "ego = new Object at (0, 0)\n"
    params = _scenario_params(sampler, DEFAULT_SAMPLER_PARAMS.get(sampler, {}))
    return scenic.scenarioFromString(src, params=params, mode2D=True)


def _results(r):
    return {"returns": r["returns"].round(4), "lengths": r["lengths"],
            "solve_rate": float(r["finished"].mean()), "fell_rate": float(r["fell"].mean()),
            "progress": r["progress"].round(4), "mean_progress": round(float(r["progress"].mean()), 4),
            "mean_return": round(float(r["returns"].mean()), 4)}


def run(policy: Policy, orc: oracle.Oracle, ctx: TrainContext, args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    scen = make_scenario(args.space, args.sampler)
    header = {
        "env": "bipedal", "algo": args.algo, "train_seed": policy.config.get("seed"),
        "checkpoint": policy.ckpt_dir, "checkpoint_step": policy.step,
        "train_config": policy.config,
        "upstream": {"repo": "https://github.com/facebookresearch/dcd",
                     "commit": policy.config.get("dcd_commit")},
        "atlas_commit": record.git_commit(os.path.dirname(__file__)),
        "space": {"name": args.space, "bounds": space.SPACES[args.space]},
        "falsifier": {"name": args.sampler, "params": DEFAULT_SAMPLER_PARAMS.get(args.sampler, {}),
                      "via": "scenic.scenarioFromString + VerifAI"},
        "spec": {"metric": "solve_rate (end of terrain reached, info['finish'])", "tau": args.tau,
                 "attempts": args.attempts, "rho": "solve_rate - tau",
                 "cex": "rho < 0 on a witness-certified solvable level"},
        "oracle": {"name": orc.name, "max_steps": orc.max_steps, "module": "atlas.bipedal.oracle",
                   "script_rough": oracle.SCRIPT_ROUGH, "script_budget": orc.script_budget,
                   "refs": [r.ckpt_dir for r in orc.refs], "ref_attempts": orc.ref_attempts,
                   "self_certify": True},
        "budget": args.budget, "seed": args.seed, "train_context": ctx.summary(),
    }
    rec = record.Recorder(args.out_dir, header, space.DESC_KEYS)
    fb = None
    for n in range(args.budget):
        scene, _ = scen.generate(feedback=fb)
        p = {k: float(scene.params[k]) for k in space.SPACES[args.space]}
        level, why = space.build(args.space, p)
        rseed = args.seed * 1_000_003 + n
        if why:
            fb = 1 - args.tau
            rec.add({"params": p, "valid": False, "invalid_reason": why, "level": level,
                     "oracle": {"verdict": None}})
            continue
        orc_out = orc.check(level, rseed)
        r = policy.run(level, args.attempts, rseed, keep_actions=True)
        if orc_out["verdict"] != oracle.SOLVABLE:
            for i in np.flatnonzero(r["finished"]):
                if orc.replay(level, r["actions"][i]):
                    orc_out.update(verdict=oracle.SOLVABLE, source="self", witness=r["actions"][i])
                    break
        witness = orc_out.pop("witness", None)
        if orc_out["verdict"] != oracle.SOLVABLE:
            fb = 1 - args.tau
            rec.add({"params": p, "valid": False, "invalid_reason": f"oracle_{orc_out['verdict']}",
                     "level": level, "oracle": orc_out, **_results(r)})
            continue
        res = _results(r)
        fb = res["solve_rate"] - args.tau
        orc_out["witness_len"] = len(witness)
        T = int(r["lengths"].max()) // POS_EVERY + 1
        rec.add({"params": p, "valid": True, "invalid_reason": None, "level": level,
                 "desc": space.descriptors(level), **res, "rho": fb, "cex": fb < 0,
                 "hard": res["solve_rate"] == 0, "oracle_source": orc_out["source"],
                 "oracle": orc_out, "train": ctx(level)},
                trace={"pos": r["pos"][:T].astype(np.float16), "len": r["lengths"],
                       "witness": witness.astype(np.float32)})
    return rec.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="DCD run dir holding meta.json and model.tar")
    ap.add_argument("--step", type=int, default=-1, help="kept for the header (DCD keeps the latest model.tar)")
    ap.add_argument("--algo", required=True, help="label: dr, plr, rplr, accel, minimax, sfl")
    ap.add_argument("--ref-ckpt", nargs="*", default=[], help="other run dirs used as solvability witnesses")
    ap.add_argument("--ref-attempts", type=int, default=4)
    ap.add_argument("--script-budget", type=int, default=oracle.SCRIPT_BUDGET)
    ap.add_argument("--space", choices=tuple(space.SPACES), default="dr")
    ap.add_argument("--sampler", choices=SAMPLERS, nargs="+", default=["random"])
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--n-ref", type=int, default=2000, help="training-context reference levels")
    ap.add_argument("--out", default="runs/bipedal")
    a = ap.parse_args()
    pol = Policy(a.ckpt, a.step)
    refs = [Policy(c) for c in a.ref_ckpt if os.path.abspath(c) != pol.ckpt_dir]
    orc = oracle.Oracle(refs, a.ref_attempts, a.script_budget)
    ctx = TrainContext(a.n_ref)
    for sampler in a.sampler:
        for seed in a.seed:
            args = argparse.Namespace(**{**vars(a), "sampler": sampler, "seed": seed})
            args.out_dir = os.path.join(a.out, a.algo, f"s{pol.config.get('seed')}",
                                        f"{a.space}_{sampler}_r{seed}")
            s = run(pol, orc, ctx, args)
            print(f"{a.algo} {a.space} {sampler} r{seed}: cex {s['n_cex']}/{s['n_valid']} "
                  f"first@{s['first_cex_i']} cells {s['distinct_cex_cells']} "
                  f"invalid {s['invalid_rate']} {s['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
