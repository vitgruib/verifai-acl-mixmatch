"""Offline signal screen (docs/wrapper_methodology.md, section 4): does a candidate task
signal find the frontier, judged on saved agents without training anything?

For saved agents at several points in training, draw random tasks and compute
  - ground truth from `--k-truth` stochastic-policy rollouts per task: the pass rate p and
    learnability p(1 - p);
  - each candidate signal from what it would see during training (one or a few episodes);
and report, per signal: Spearman with learnability, precision@10% (the share of the
signal's top 10% of tasks that are on the frontier, 0.1 < p < 0.9), and Spearman with the
pass rate (its easiness bias: positive = it prefers tasks the agent already passes).

    python -m acl_bench.plr.signals --env cartpole --arm N --seeds 1-6
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from acl_bench import envs
from acl_bench.plr.fast import REWARD
from acl_bench.snapshots import load_run

FRONTIER = (0.1, 0.9)


@torch.no_grad()
def rollouts(env, agent, params: np.ndarray, rng: np.random.Generator, gamma=0.99, lam=0.95) -> dict:
    """One stochastic-policy episode per row of `params`, batched. Returns per episode:
    success, return, PVL, L1 (mean |GAE|), mean policy entropy."""
    n = len(params)
    state = np.concatenate([env.sample_starts(p, 1, rng) for p in params])
    alive, ended = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
    streak = np.zeros(n, dtype=int)
    r_step, r_term = REWARD.get(env.__name__.rsplit(".", 1)[-1], (0.0, 0.0))
    rews, vals, alives, terms, ents = [], [], [], [], []
    for _ in range(env.MAX_EPISODE_STEPS):
        obs = torch.as_tensor(env.observe(state).astype(np.float32))
        logp = torch.log_softmax(agent.actor(obs), dim=1)
        a = torch.multinomial(logp.exp(), 1).squeeze(1).numpy()
        vals.append(agent.critic(obs).squeeze(1).numpy())
        ents.append((-(logp.exp() * logp).sum(1)).numpy())
        alives.append(alive.copy())
        r_env = env.reward(state, a, params) if hasattr(env, "reward") else None
        state = np.where(alive[:, None], env.step(state, a, params), state)
        if env.GOAL == "balance":
            streak = np.where(env.upright(state), streak + 1, 0)
        e = alive & env.terminated(state)
        rews.append(r_env if r_env is not None else np.where(e, r_term, r_step))
        terms.append(e)
        ended |= e
        alive &= ~e
        if not alive.any():
            break
    v_end = agent.critic(torch.as_tensor(env.observe(state).astype(np.float32))).squeeze(1).numpy()
    rew, val, al, term, ent = map(np.array, (rews, vals, alives, terms, ents))
    T = len(rew)
    pos, absa, last = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in reversed(range(T)):
        nxt = np.where(term[t], 0.0, val[t + 1] if t + 1 < T else v_end)
        delta = rew[t] + gamma * nxt - val[t]
        last = np.where(al[t], delta + gamma * lam * np.where(term[t], 0.0, last), 0.0)
        pos += np.where(al[t], np.clip(last, 0.0, None), 0.0)
        absa += np.where(al[t], np.abs(last), 0.0)
    length = np.maximum(al.sum(0), 1)
    if env.GOAL == "balance":
        success = streak >= env.HOLD_STEPS
    else:
        success = ended if env.GOAL == "reach" else ~ended
    return {"success": success.astype(float), "return": (rew * al).sum(0), "pvl": pos / length,
            "l1": absa / length, "entropy": (ent * al).sum(0) / length}


def signals_for(env, agent, tasks: np.ndarray, k_truth: int, rng: np.random.Generator) -> pd.DataFrame:
    n = len(tasks)
    truth = rollouts(env, agent, np.repeat(tasks, k_truth, axis=0), rng)
    p = truth["success"].reshape(n, k_truth).mean(1)
    # what training sees: a separate, independent handful of episodes per task
    few = rollouts(env, agent, np.repeat(tasks, 4, axis=0), rng)
    per = {k: v.reshape(n, 4) for k, v in few.items()}
    out = pd.DataFrame({"p": p, "learn_true": p * (1 - p)})
    out["pvl_1"] = per["pvl"][:, 0]
    out["l1_1"] = per["l1"][:, 0]
    out["entropy_1"] = per["entropy"][:, 0]
    out["negret_1"] = -per["return"][:, 0]
    out["pvl_4"] = per["pvl"].mean(1)
    for k in (2, 4):
        q = per["success"][:, :k].mean(1)
        out[f"learn_{k}"] = q * (1 - q)
    q2 = per["success"][:, :2].mean(1)
    out["pvl_x_learn_2"] = out["pvl_1"] * 4 * q2 * (1 - q2)
    return out


def score_signals(df: pd.DataFrame) -> pd.DataFrame:
    frontier = (df.p > FRONTIER[0]) & (df.p < FRONTIER[1])
    jitter = 1e-9 * np.random.default_rng(0).random(len(df))           # break ties at random
    true_top = (df.learn_true + jitter) >= np.quantile(df.learn_true + jitter, 0.9)
    rows = []
    for s in [c for c in df.columns if c not in ("p", "learn_true")]:
        x = df[s] + jitter
        top = x >= np.quantile(x, 0.9)
        rows.append({"signal": s,
                     "rho_learn": spearmanr(x, df.learn_true)[0],
                     "overlap_10": (top & true_top).sum() / max(top.sum(), 1),
                     "prec_at_10": frontier[top].mean(),
                     "rho_pass": spearmanr(x, df.p)[0],
                     "top_pass_rate": df.p[top].mean()})
    rows.append({"signal": "(random pick)", "overlap_10": 0.1, "prec_at_10": frontier.mean(), "top_pass_rate": df.p.mean()})
    return pd.DataFrame(rows).set_index("signal")


def main():
    from acl_bench.study.run import parse_seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=envs.NAMES)
    ap.add_argument("--arm", default="N")
    ap.add_argument("--seeds", default="1-6")
    ap.add_argument("--points", default="0.2,0.5,0.8", help="shares of training at which to take the saved agent")
    ap.add_argument("--n-tasks", type=int, default=2000)
    ap.add_argument("--k-truth", type=int, default=16)
    ap.add_argument("--out", default=None, help="CSV of every (agent, task) row")
    args = ap.parse_args()
    torch.set_num_threads(1)
    env = envs.get(args.env)
    lo, hi = (np.array([env.PARAM_BOUNDS[k][i] for k in env.PARAM_ORDER]) for i in (0, 1))
    frames = []
    for seed in parse_seeds(args.seeds):
        agents = load_run(f"{envs.results_dir(args.env)}/snapshots/{args.arm}/{seed}.npz")
        steps = sorted(agents)
        for point in map(float, args.points.split(",")):
            step = steps[min(len(steps) - 1, round(point * (len(steps) - 1)))]
            rng = np.random.default_rng([seed, step])
            tasks = rng.uniform(lo, hi, size=(args.n_tasks, len(lo)))
            df = signals_for(env, agents[step], tasks, args.k_truth, rng)
            df["seed"], df["point"] = seed, point
            frames.append(df)
            print(f"seed {seed} point {point}: frontier share {((df.p > 0.1) & (df.p < 0.9)).mean():.2f}", flush=True)
    all_df = pd.concat(frames, ignore_index=True)
    if args.out:
        all_df.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    for point, g in all_df.groupby("point"):
        print(f"\n== {args.env}, agents at {point:.0%} of training ({g.seed.nunique()} seeds, {len(g)} tasks)")
        print(score_signals(g.drop(columns=["seed", "point"])).round(3).to_string())


if __name__ == "__main__":
    main()
