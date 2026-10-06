"""Falsify a trained Kinetix agent (FLAIROx/Kinetix, symbolic obs, S levels) with a VerifAI
sampler (through Scenic, as in acl_bench.sampling) and write a failure-record directory
(atlas.record).

Spec: the agent solves the level (green touches blue) in at least `tau` of `attempts`
stochastic episodes. rho = solve_rate - tau; rho < 0 is a counterexample. A counterexample also
needs a solvability witness (atlas.kinetix.oracle): an action sequence that solves the level
when replayed open-loop in the real, deterministic env (the agent's own solved attempt, or one
found by search). Levels the Checker rejects (overlapping shapes) or the oracle cannot certify
are recorded as invalid, not evaluated, and fed back as rho = 1 - tau so adaptive samplers move
away. The policy is rolled out first, since its own solved attempts are the cheapest witness.
Headless: never renders.

  python -m atlas.kinetix.falsify --ckpt $RUNS/kinetix/plr/s0 --algo plr \
      --space pos --sampler ce mab --budget 1000 --out runs/kinetix_falsify
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import scenic

from acl_bench.sampling import DEFAULT_SAMPLER_PARAMS, _scenario_params
from atlas import record
from atlas.kinetix import oracle, space
from atlas.kinetix.context import TrainContext
from atlas.kinetix.policy import Policy

SAMPLERS = ("random", "ce", "mab")
UPSTREAM = {"repo": "https://github.com/FLAIROx/Kinetix",
            "commit": os.environ.get("KINETIX_COMMIT", "80ee9c292837c825cafaf9889323fdcc212efd1b")}


def make_scenario(space_name: str, sampler: str):
    src = "".join(f"param {k} = VerifaiRange({lo}, {hi})\n" for k, (lo, hi) in space.SPACES[space_name].items())
    src += "ego = new Object at (0, 0)\n"
    params = _scenario_params(sampler, DEFAULT_SAMPLER_PARAMS.get(sampler, {}))
    return scenic.scenarioFromString(src, params=params, mode2D=True)


def run(policy: Policy, orc: oracle.Oracle, checker: space.Checker, ctx: TrainContext, args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    scen = make_scenario(args.space, args.sampler)
    header = {
        "env": "kinetix", "algo": args.algo, "train_seed": policy.config.get("seed"),
        "checkpoint": policy.model_path, "checkpoint_step": policy.step,
        "train_config": json.loads(json.dumps(policy.config, default=str)), "upstream": UPSTREAM,
        "atlas_commit": record.git_commit(os.path.dirname(__file__)),
        "space": {"name": args.space, "bounds": space.SPACES[args.space], "base_levels": policy.eval_names},
        "falsifier": {"name": args.sampler, "params": DEFAULT_SAMPLER_PARAMS.get(args.sampler, {}),
                      "via": "scenic.scenarioFromString + VerifAI"},
        "spec": {"metric": "solve_rate (done with GoalR)", "tau": args.tau, "attempts": args.attempts,
                 "rho": "solve_rate - tau", "cex": "rho < 0 on an oracle-certified solvable level"},
        "oracle": {"name": orc.name, "params": orc.params(), "module": "atlas.kinetix.oracle"},
        "budget": args.budget, "seed": args.seed, "train_context": ctx.summary(),
    }
    rec = record.Recorder(args.out_dir, header, space.DESC_KEYS)
    fb = None
    for n in range(args.budget):
        scene, _ = scen.generate(feedback=fb)
        p = {k: float(scene.params[k]) for k in space.SPACES[args.space]}
        idx = space.level_index(p)
        level, applied = space.build(args.space, p, policy.eval_levels)
        desc = space.descriptors(level, idx)
        row = {"params": p, "applied": applied, "level": space.to_str(policy.eval_names[idx], args.space, p),
               "base": policy.eval_names[idx]}
        why = checker(level, idx)
        if why is None:
            rets, lens, solved, acts = policy.run(level, args.attempts, args.seed * 1_000_003 + n)
            orc_out = orc.check(level, own=(solved, acts))
            if orc_out["verdict"] != oracle.SOLVABLE:
                why = f"oracle_{orc_out['verdict']}"
        else:
            orc_out = {"verdict": None}
        if why:
            fb = 1 - args.tau
            rec.add({**row, "valid": False, "invalid_reason": why, "desc": desc, "oracle": orc_out})
            continue
        witness = orc_out.pop("witness")
        solve = float(solved.mean())
        fb = solve - args.tau
        rec.add({**row, "valid": True, "invalid_reason": None, "desc": desc,
                 "returns": rets.round(4), "lengths": lens, "solve_rate": solve,
                 "mean_return": round(float(rets.mean()), 4), "rho": fb, "cex": fb < 0,
                 "hard": solve == 0, "oracle": orc_out, "train": ctx(level)},
                trace={"witness": witness, "acts": acts[:lens.max()].astype(np.int8), "len": lens})
    return rec.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Kinetix run dir (holds .hydra/config.yaml and */full_model.pbz2)")
    ap.add_argument("--config", default=None, help="Hydra config.yaml if not at or above --ckpt")
    ap.add_argument("--algo", required=True, help="label: dr, plr, rplr, accel, sfl")
    ap.add_argument("--space", choices=tuple(space.SPACES), default="pos")
    ap.add_argument("--sampler", choices=SAMPLERS, nargs="+", default=["random"])
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/kinetix_falsify")
    a = ap.parse_args()
    pol = Policy(a.ckpt, a.config)
    orc = oracle.Oracle(pol.env, pol.env_params, pol.max_steps)
    checker = space.Checker(pol.env, pol.env_params, pol.eval_levels)
    ctx = TrainContext(pol)
    for sampler in a.sampler:
        for seed in a.seed:
            args = argparse.Namespace(**{**vars(a), "sampler": sampler, "seed": seed})
            args.out_dir = os.path.join(a.out, a.algo, f"s{pol.config.get('seed')}",
                                        f"{a.space}_{sampler}_r{seed}")
            s = run(pol, orc, checker, ctx, args)
            print(f"{a.algo} {a.space} {sampler} r{seed}: cex {s['n_cex']}/{s['n_valid']} "
                  f"first@{s['first_cex_i']} cells {s['distinct_cex_cells']} "
                  f"invalid {s['invalid_rate']} {s['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
