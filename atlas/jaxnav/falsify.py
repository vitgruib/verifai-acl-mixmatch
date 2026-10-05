"""Falsify a trained SFL-repo JaxNav agent (single agent) with a VerifAI sampler (through Scenic,
as in acl_bench.sampling) and write a failure-record directory (atlas.record).

Spec: the agent reaches the goal in at least `tau` of `attempts` stochastic episodes.
rho = solve_rate - tau; rho < 0 is a counterexample. Every level first goes through the
solvability oracle (atlas.jaxnav.oracle): only levels where a scripted controller reached the
goal in the real env within the episode limit are evaluated, so every counterexample comes with
a witness action sequence. Invalid levels (start on goal, unreachable, oracle unknown) are
recorded but not evaluated, and fed back as rho = 1 - tau so adaptive samplers move away.

  python -m atlas.jaxnav.falsify --ckpt $RUNS/sfl/jaxnav_sfl/s0 --algo sfl \
      --space seg --sampler ce --budget 2000 --out runs/jaxnav
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import scenic

from acl_bench.sampling import DEFAULT_SAMPLER_PARAMS, _scenario_params
from atlas import record
from atlas.jaxnav import oracle, space
from atlas.jaxnav.context import TrainContext
from atlas.jaxnav.policy import Policy, instance

SAMPLERS = ("random", "ce", "mab")


def make_scenario(space_name: str, sampler: str):
    src = "".join(f"param {k} = VerifaiRange({lo}, {hi})\n" for k, (lo, hi) in space.SPACES[space_name].items())
    src += "ego = new Object at (0, 0)\n"
    params = _scenario_params(sampler, DEFAULT_SAMPLER_PARAMS.get(sampler, {}))
    return scenic.scenarioFromString(src, params=params, mode2D=True)


def run(policy: Policy, orc: oracle.Oracle, ctx: TrainContext, args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    scen = make_scenario(args.space, args.sampler)
    header = {
        "env": "jaxnav", "algo": args.algo, "train_seed": policy.config.get("seed"),
        "checkpoint": policy.ckpt_dir, "checkpoint_step": policy.step,
        "train_config": policy.config,
        "upstream": {"repo": "https://github.com/amacrutherford/sampling-for-learnability",
                     "commit": policy.config.get("sfl_commit")},
        "atlas_commit": record.git_commit(os.path.dirname(__file__)),
        "space": {"name": args.space, "bounds": space.SPACES[args.space]},
        "falsifier": {"name": args.sampler, "params": DEFAULT_SAMPLER_PARAMS.get(args.sampler, {}),
                      "via": "scenic.scenarioFromString + VerifAI"},
        "spec": {"metric": "solve_rate (goal_reached)", "tau": args.tau, "attempts": args.attempts,
                 "rho": "solve_rate - tau", "cex": "rho < 0 on an oracle-certified solvable level"},
        "oracle": {"name": orc.name, "max_steps": orc.max_steps, "module": "atlas.jaxnav.oracle"},
        "budget": args.budget, "seed": args.seed, "train_context": ctx.summary(),
    }
    rec = record.Recorder(args.out_dir, header, space.DESC_KEYS)
    fb = None
    for n in range(args.budget):
        scene, _ = scen.generate(feedback=fb)
        p = {k: float(scene.params[k]) for k in space.SPACES[args.space]}
        grid, start, theta, goal, why = space.build(args.space, p)
        level = space.to_str(grid, start, theta, goal)
        inst = instance(grid, np.add(start, 0.5), theta, np.add(goal, 0.5))
        orc_out = {"verdict": None} if why else orc.check(inst, start, goal)
        if not why and orc_out["verdict"] != oracle.SOLVABLE:
            why = f"oracle_{orc_out['verdict']}"
        if why:
            fb = 1 - args.tau
            rec.add({"params": p, "valid": False, "invalid_reason": why, "level": level, "oracle": orc_out})
            continue
        rets, lens, succ, crash, pos = policy.run(inst, args.attempts, args.seed * 1_000_003 + n)
        solve = float(succ.mean())
        fb = solve - args.tau
        rec.add({"params": p, "valid": True, "invalid_reason": None, "level": level,
                 "desc": space.descriptors(grid, start, theta, goal),
                 "returns": rets.round(4), "lengths": lens, "solve_rate": solve,
                 "crash_rate": float(crash.mean()),
                 "mean_return": round(float(rets.mean()), 4), "rho": fb, "cex": fb < 0,
                 "hard": solve == 0, "oracle": orc_out, "train": ctx(grid, start, theta, goal)},
                trace={"pos": pos[:lens.max()].astype(np.float16), "len": lens})
    return rec.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="SFL run dir holding model.safetensors")
    ap.add_argument("--step", type=int, default=-1, help="kept for the header (SFL saves final params only)")
    ap.add_argument("--algo", required=True, help="label: dr, plr, rplr, accel, minimax, sfl")
    ap.add_argument("--space", choices=tuple(space.SPACES), default="seg")
    ap.add_argument("--sampler", choices=SAMPLERS, nargs="+", default=["random"])
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/jaxnav")
    a = ap.parse_args()
    pol = Policy(a.ckpt, a.step)
    orc = oracle.Oracle(pol.env)
    ctx = TrainContext()
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
