"""Run one environment for a long budget and log a learning curve, to find how
many environment steps each env actually needs before the grid's fixed budget
stops being the bottleneck.

    python -m acl_bench.convergence --env acrobot --seed 1 \
        --configs random:neg_return:off random:pvl_gae:on sa:pvl_gae:on \
        --total-timesteps 1000000 --eval-every 25000 --out results/convergence_acrobot_s1.csv

One row per (config, checkpoint): held-out eval return on the same fixed task
set the grid uses (configs are `sampler:function:acl`, acl = on|off), plus the mean of the last 20 training episodes.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import torch

from acl_bench.envs.registry import ENV_SPECS
from acl_bench.experiment import make_fixed_eval_set
from acl_bench.potential.functions import resolve_potential_fn
from acl_bench.ppo import PPOConfig, run_training
from acl_bench.scenic_sampling import ScenicTaskSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=list(ENV_SPECS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--configs", nargs="+", default=["random:neg_return:off", "random:pvl_gae:on", "sa:pvl_gae:on"],
                        help="sampler:potential_fn:acl triples (acl = on|off)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)  # several of these run side by side; avoid oversubscribing cores
    env_spec = ENV_SPECS[args.env]
    eval_set = make_fixed_eval_set(args.env)

    rows = []
    for config in args.configs:
        sampler_name, potential_name, acl = config.split(":")
        cfg = PPOConfig(total_timesteps=args.total_timesteps, seed=args.seed, acl=(acl == "on"))
        task_sampler = ScenicTaskSampler.load(env_spec.scenic_file, sampler_name)
        t0 = time.time()
        log, _ = run_training(env_spec, task_sampler, resolve_potential_fn(potential_name, env_spec.success_return), cfg,
                              eval_set=eval_set, eval_every_steps=args.eval_every)
        elapsed = time.time() - t0
        for step, eval_return, train_last20 in log.checkpoints:
            rows.append({"env": args.env, "sampler": sampler_name, "acl": acl, "potential_fn": potential_name,
                         "seed": args.seed, "step": step, "eval_return": eval_return,
                         "train_return_last20": train_last20})
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"{args.env} seed={args.seed} {config}: {len(log.checkpoints)} checkpoints, "
              f"final eval={log.checkpoints[-1][1]:.1f}, {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
