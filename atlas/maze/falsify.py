"""Falsify a trained JaxUED maze agent with a VerifAI sampler (through Scenic, as in
acl_bench.sampling) and write a failure-record directory (atlas.record).

Spec: the agent solves the level in at least `tau` of `attempts` stochastic episodes.
rho = solve_rate - tau; rho < 0 is a counterexample. Every level first goes through the exact
solvability oracle (atlas.maze.oracle): only levels certified solvable within the episode's step
limit are evaluated, so every counterexample comes with a witness action sequence proving the
level was solvable. Invalid levels (agent on goal, oracle-unsolvable) are recorded but not
evaluated, and fed back as rho = 1 - tau (the best possible value), so adaptive
samplers move away from them.

  python -m atlas.maze.falsify --ckpt checkpoints/<run>/<seed> --algo rplr \
      --space seg --sampler ce --budget 2000 --out runs/maze
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import scenic
from dotmap import DotMap

from acl_bench.sampling import DEFAULT_SAMPLER_PARAMS, _scenario_params
from atlas import record
from atlas.maze import oracle, space
from atlas.maze.context import TrainContext
from atlas.maze.policy import JAXUED_DIR, Policy

SAMPLERS = ("random", "halton", "ce", "mab", "sa")


# Halton in ~50 dimensions: the first ~p_d^2 points of the large-prime bases move in lockstep
# (agent_x == goal_x == ... for the first hundred points), so start deep in the sequence.
HALTON_START = 100_000


def make_scenario(space_name: str, sampler: str, seed: int):
    src = "".join(f"param {k} = VerifaiRange({lo}, {hi})\n" for k, (lo, hi) in space.SPACES[space_name].items())
    src += "ego = new Object at (0, 0)\n"
    params = _scenario_params(sampler, DEFAULT_SAMPLER_PARAMS.get(sampler, {}))
    if sampler == "halton":
        params["verifaiSamplerParams"] = DotMap(sample_index=HALTON_START * (1 + seed), bases_skipped=0)
    return scenic.scenarioFromString(src, params=params, mode2D=True)


def run(policy: Policy, ctx: TrainContext, args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    scen = make_scenario(args.space, args.sampler, args.seed)
    header = {
        "env": "jaxued_maze", "algo": args.algo, "train_seed": policy.config.get("seed"),
        "checkpoint": policy.ckpt_dir, "checkpoint_step": policy.step,
        "train_config": policy.config,
        "upstream": {"repo": "https://github.com/DramaCow/jaxued", "commit": record.git_commit(JAXUED_DIR)},
        "atlas_commit": record.git_commit(os.path.dirname(__file__)),
        "space": {"name": args.space, "bounds": space.SPACES[args.space]},
        "falsifier": {"name": args.sampler, "params": DEFAULT_SAMPLER_PARAMS.get(args.sampler, {}),
                      "via": "scenic.scenarioFromString + VerifAI"},
        "spec": {"metric": "solve_rate", "tau": args.tau, "attempts": args.attempts,
                 "rho": "solve_rate - tau", "cex": "rho < 0 on an oracle-certified solvable level"},
        "oracle": {"name": "exact BFS over (x, y, dir), JaxUED Maze._step_agent rule",
                   "max_steps": policy.env_params.max_steps_in_episode, "module": "atlas.maze.oracle"},
        "budget": args.budget, "seed": args.seed, "train_context": ctx.summary(),
    }
    rec = record.Recorder(args.out_dir, header, space.DESC_KEYS)
    fb = None
    for n in range(args.budget):
        scene, _ = scen.generate(feedback=fb)
        p = {k: float(scene.params[k]) for k in space.SPACES[args.space]}
        walls, agent, d, goal, why = space.build(args.space, p)
        level = space.to_str(walls, agent, d, goal)
        orc = oracle.solve(walls, agent, d, goal, policy.env_params.max_steps_in_episode)
        if orc["status"] != oracle.SOLVABLE:
            why = why or f"oracle_{orc['status']}"
        if why:
            fb = 1 - args.tau
            rec.add({"params": p, "valid": False, "invalid_reason": why, "level": level, "oracle": orc})
            continue
        rets, lens, pos = policy.run(walls, agent, d, goal, args.attempts, args.seed * 1_000_003 + n)
        solve = float((rets > 0).mean())
        fb = solve - args.tau
        rec.add({"params": p, "valid": True, "invalid_reason": None, "level": level,
                 "desc": space.descriptors(walls, agent, goal),
                 "returns": rets.round(4), "lengths": lens, "solve_rate": solve,
                 "mean_return": round(float(rets.mean()), 4), "rho": fb, "cex": fb < 0,
                 "hard": solve == 0, "oracle": orc, "train": ctx(walls, agent, goal)},
                trace={"pos": pos[:lens.max()].astype(np.int8), "len": lens})
    return rec.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoints/<run_name>/<seed>")
    ap.add_argument("--step", type=int, default=-1, help="checkpoint step (-1 = latest)")
    ap.add_argument("--algo", required=True, help="label: dr, plr, rplr, accel, paired")
    ap.add_argument("--space", choices=tuple(space.SPACES), default="seg")
    ap.add_argument("--sampler", choices=SAMPLERS, nargs="+", default=["random"])
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/maze")
    a = ap.parse_args()
    pol = Policy(a.ckpt, a.step)
    ctx = TrainContext(pol.config["n_walls"], pol.buffer)
    for sampler in a.sampler:
        for seed in a.seed:
            args = argparse.Namespace(**{**vars(a), "sampler": sampler, "seed": seed})
            args.out_dir = os.path.join(a.out, a.algo, f"s{pol.config.get('seed')}",
                                        f"{a.space}_{sampler}_r{seed}")
            s = run(pol, ctx, args)
            print(f"{a.algo} {a.space} {sampler} r{seed}: cex {s['n_cex']}/{s['n_valid']} "
                  f"first@{s['first_cex_i']} cells {s['distinct_cex_cells']} "
                  f"invalid {s['invalid_rate']} {s['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
