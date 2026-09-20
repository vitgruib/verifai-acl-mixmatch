"""Grid runner: every (VerifAI sampler) x (learning-potential function)
combination, over several seeds, on ParamCartPole. Writes one row per run to
results/grid_results.csv.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from acl_bench.envs.param_cartpole import CartPoleParams, normalize
from acl_bench.potential.functions import POTENTIAL_FUNCTIONS
from acl_bench.ppo import PPOConfig, evaluate_agent, run_training
from acl_bench.samplers import SAMPLER_NAMES, make_feature_space, make_sampler


def make_fixed_eval_set(n: int = 15, seed: int = 12345) -> list[CartPoleParams]:
    """Sampler-independent held-out task set (plain uniform grid via a fixed
    RNG) used to score every combination on the same yardstick."""
    rng = np.random.default_rng(seed)
    bounds = CartPoleParams.bounds()
    tasks = []
    for _ in range(n):
        kwargs = {name: rng.uniform(lo, hi) for name, (lo, hi) in bounds.items()}
        tasks.append(CartPoleParams(**kwargs))
    return tasks


def run_one(sampler_name: str, potential_name: str, seed: int,
            cfg: PPOConfig, eval_set: list[CartPoleParams]) -> dict:
    cfg = PPOConfig(**{**cfg.__dict__, "seed": seed})
    space = make_feature_space()
    sampler = make_sampler(sampler_name, space)
    potential_fn = POTENTIAL_FUNCTIONS[potential_name]

    t0 = time.time()
    log, agent = run_training(sampler, potential_fn, cfg)
    wall_time = time.time() - t0

    eval_return = evaluate_agent(agent, eval_set, seed=seed)

    returns = np.array(log.episode_returns)
    last_k = returns[-20:] if len(returns) >= 20 else returns
    param_matrix = np.array([normalize(p) for p in log.episode_params])
    task_diversity = float(param_matrix.std(axis=0).mean()) if len(param_matrix) else 0.0
    n_new = sum(1 for m in log.episode_modes if m == "new")
    n_replay = len(log.episode_modes) - n_new

    return {
        "sampler": sampler_name,
        "potential_fn": potential_name,
        "seed": seed,
        "n_episodes": len(returns),
        "train_return_mean_last20": float(last_k.mean()) if len(last_k) else float("nan"),
        "train_return_mean_all": float(returns.mean()) if len(returns) else float("nan"),
        "eval_return_mean": eval_return,
        "task_param_diversity": task_diversity,
        "n_new_episodes": n_new,
        "n_replay_episodes": n_replay,
        "final_buffer_size": len(log.episode_task_ids) and len(set(log.episode_task_ids)),
        "wall_time_sec": wall_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--total-timesteps", type=int, default=40_000)
    parser.add_argument("--out", type=str, default="results/grid_results.csv")
    args = parser.parse_args()

    cfg = PPOConfig(total_timesteps=args.total_timesteps)
    eval_set = make_fixed_eval_set()

    rows = []
    combos = [(s, p) for s in SAMPLER_NAMES for p in POTENTIAL_FUNCTIONS]
    for i, (sampler_name, potential_name) in enumerate(combos):
        for seed in args.seeds:
            print(f"[{i + 1}/{len(combos)}] sampler={sampler_name:6s} "
                  f"potential={potential_name:24s} seed={seed}", flush=True)
            row = run_one(sampler_name, potential_name, seed, cfg, eval_set)
            print(f"    -> eval_return={row['eval_return_mean']:.1f}  "
                  f"train_last20={row['train_return_mean_last20']:.1f}  "
                  f"({row['wall_time_sec']:.1f}s)", flush=True)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out, index=False)  # incremental save

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
