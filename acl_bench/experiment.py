"""Grid runner: every (env) x (VerifAI/Scenic sampler) x (learning-potential
function, incl. the "none" ACL-off ablation), over several seeds. Writes one
row per run to results/grid_results.csv.
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from acl_bench.envs.registry import ENV_NAMES, ENV_SPECS
from acl_bench.potential.functions import POTENTIAL_FUNCTION_NAMES, resolve_potential_fn
from acl_bench.ppo import PPOConfig, evaluate_agent, run_training
from acl_bench.scenic_sampling import SAMPLER_NAMES, ScenicTaskSampler

warnings.filterwarnings("ignore", category=UserWarning, module="GPyOpt")


def make_fixed_eval_set(env_name: str, n: int = 15, seed: int = 12345) -> list[dict]:
    """Sampler-independent held-out task set (plain uniform grid via a fixed
    RNG) used to score every combination on the same yardstick."""
    rng = np.random.default_rng(seed)
    bounds = ENV_SPECS[env_name].param_bounds
    return [{name: rng.uniform(lo, hi) for name, (lo, hi) in bounds.items()} for _ in range(n)]


def run_one(env_name: str, sampler_name: str, potential_name: str, seed: int,
            cfg: PPOConfig, eval_set: list[dict]) -> dict:
    cfg = PPOConfig(**{**cfg.__dict__, "seed": seed})
    env_spec = ENV_SPECS[env_name]
    task_sampler = ScenicTaskSampler.load(env_spec.scenic_file, sampler_name)
    potential_fn = resolve_potential_fn(potential_name)

    t0 = time.time()
    log, agent = run_training(env_spec, task_sampler, potential_fn, cfg)
    wall_time = time.time() - t0

    eval_return = evaluate_agent(agent, env_spec, eval_set, seed=seed)

    returns = np.array(log.episode_returns)
    last_k = returns[-20:] if len(returns) >= 20 else returns
    param_matrix = np.array([env_spec.normalize(p) for p in log.episode_params])
    task_diversity = float(param_matrix.std(axis=0).mean()) if len(param_matrix) else 0.0
    n_new = sum(1 for m in log.episode_modes if m == "new")
    n_replay = len(log.episode_modes) - n_new

    return {
        "env": env_name,
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
        "final_buffer_size": len(set(log.episode_task_ids)) if log.episode_task_ids else 0,
        "wall_time_sec": wall_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=str, nargs="+", default=list(ENV_NAMES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--out", type=str, default="results/grid_results.csv")
    args = parser.parse_args()

    cfg = PPOConfig(total_timesteps=args.total_timesteps)

    rows = []
    combos = [(e, s, p) for e in args.envs for s in SAMPLER_NAMES for p in POTENTIAL_FUNCTION_NAMES]
    eval_sets = {e: make_fixed_eval_set(e) for e in args.envs}
    for i, (env_name, sampler_name, potential_name) in enumerate(combos):
        for seed in args.seeds:
            print(f"[{i + 1}/{len(combos)}] env={env_name:9s} sampler={sampler_name:6s} "
                  f"potential={potential_name:24s} seed={seed}", flush=True)
            row = run_one(env_name, sampler_name, potential_name, seed, cfg, eval_sets[env_name])
            print(f"    -> eval_return={row['eval_return_mean']:.1f}  "
                  f"train_last20={row['train_return_mean_last20']:.1f}  "
                  f"({row['wall_time_sec']:.1f}s)", flush=True)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out, index=False)  # incremental save

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
