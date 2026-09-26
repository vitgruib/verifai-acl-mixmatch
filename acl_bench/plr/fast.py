"""Vectorized PPO for screening: `n_envs` copies of the environment step together through
its batched physics (the same code the grader uses, checked against gymnasium in
tests/test_grader.py), so one policy forward pass serves all of them.

Same PPO as acl_bench/ppo.py (network, losses, 1024 samples per update, 4 minibatches,
4 epochs, truncation treated as terminal) except that the 1024 samples come from
`n_envs` parallel episodes of 1024 / n_envs steps. Results from here are compared only
with results from here; a finding is confirmed in the study pipeline (acl_bench.study).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from acl_bench.plr.levels import NEW, REPLAY, LevelConfig, LevelSampler
from acl_bench.ppo import Agent

REWARD = {"cartpole": (1.0, 1.0), "acrobot": (-1.0, 0.0), "mountaincar": (-1.0, -1.0)}   # (step, terminal step)


@dataclass
class FastConfig:
    steps: int = 491_520
    n_envs: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    minibatches: int = 4
    epochs: int = 4
    clip: float = 0.2
    ent: float = 0.01
    vf: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 1
    levels: LevelConfig = field(default_factory=LevelConfig)


def train(env_name: str, env, cfg: FastConfig, n_checks: int = 10, on_check=None):
    """Returns (agent, checks, stats). `on_check(step, agent)` runs at step 0 and after
    every steps / n_checks steps."""
    lv_ss, env_ss, sh_ss = np.random.SeedSequence(cfg.seed).spawn(3)
    torch.manual_seed(cfg.seed)
    env_rng, shuffle_rng = np.random.default_rng(env_ss), np.random.default_rng(sh_ss)
    bounds = np.array([env.PARAM_BOUNDS[k] for k in env.PARAM_ORDER], dtype=np.float64)
    levels = LevelSampler(cfg.levels, bounds, np.random.default_rng(lv_ss))
    r_step, r_term = REWARD[env_name]

    K, T = cfg.n_envs, 1024 // cfg.n_envs
    agent = Agent(env.OBS_DIM, env.ACTION_DIM)
    opt = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    params = np.zeros((K, len(bounds)))
    state = None
    slot = np.full(K, -1)
    mode = np.zeros(K, dtype=int)
    t_ep = np.zeros(K, dtype=int)
    ep_r = [[] for _ in range(K)]
    ep_v = [[] for _ in range(K)]
    ep_train = np.ones(K, dtype=bool)         # PLR-perp: does this episode's data train the agent?

    def start(k):
        p, i, m = levels.pick()
        params[k], slot[k], mode[k] = p, i, m
        s = env.sample_starts(p, 1, env_rng)[0]
        t_ep[k] = 0
        ep_r[k], ep_v[k] = [], []
        ep_train[k] = not (cfg.levels.robust and m == NEW)
        return s

    state = np.stack([start(k) for k in range(K)])
    checks, stats = [], {"episodes": 0, "replays": 0}
    every = cfg.steps // n_checks
    if on_check:
        checks.append({"step": 0, **on_check(0, agent)})

    n_iter = cfg.steps // (K * T)
    for it in range(n_iter):
        b_obs = np.zeros((T, K, env.OBS_DIM), dtype=np.float32)
        b_act = np.zeros((T, K), dtype=np.int64)
        b_logp = np.zeros((T, K), dtype=np.float32)
        b_val = np.zeros((T, K), dtype=np.float32)
        b_rew = np.zeros((T, K), dtype=np.float32)
        b_done = np.zeros((T, K), dtype=np.float32)
        b_mask = np.zeros((T, K), dtype=np.float32)
        for t in range(T):
            obs = env.observe(state).astype(np.float32)
            with torch.no_grad():
                x = torch.as_tensor(obs)
                logits = agent.actor(x)
                logp_all = torch.log_softmax(logits, dim=1)
                a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                v = agent.critic(x).squeeze(1)
            a_np = a.numpy()
            b_obs[t], b_act[t] = obs, a_np
            b_logp[t] = logp_all.gather(1, a[:, None]).squeeze(1).numpy()
            b_val[t] = v.numpy()
            b_mask[t] = ep_train
            state = env.step(state, a_np, params)
            term = env.terminated(state)
            t_ep += 1
            trunc = t_ep >= env.MAX_EPISODE_STEPS
            rew = np.where(term, r_term, r_step)
            b_rew[t] = rew
            done = term | trunc
            b_done[t] = done
            for k in range(K):
                ep_r[k].append(float(rew[k]))
                ep_v[k].append(float(b_val[t, k]))
                if done[k]:
                    if term[k]:
                        nv = 0.0
                    else:
                        with torch.no_grad():
                            nv = float(agent.critic(torch.as_tensor(env.observe(state[k:k + 1]).astype(np.float32))))
                    success = bool(term[k]) if env.GOAL == "reach" else not term[k]
                    ret = float(sum(ep_r[k]))
                    sc = levels.score(ep_r[k], ep_v[k], nv, cfg.gamma, cfg.lam, success, slot[k])
                    levels.report(params[k].copy(), slot[k], mode[k], sc, ret, success)
                    stats["episodes"] += 1
                    stats["replays"] += int(mode[k] == REPLAY)
                    state[k] = start(k)

        with torch.no_grad():
            nv = agent.critic(torch.as_tensor(env.observe(state).astype(np.float32))).squeeze(1).numpy()
        adv = np.zeros((T, K), dtype=np.float32)
        last = np.zeros(K, dtype=np.float32)
        for t in reversed(range(T)):
            nnt = 1.0 - b_done[t]
            nxt = nv if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + cfg.gamma * nxt * nnt - b_val[t]
            last = delta + cfg.gamma * cfg.lam * nnt * last
            adv[t] = last
        ret_ = adv + b_val

        f = lambda x: torch.as_tensor(x.reshape(T * K, *x.shape[2:]))
        o, ac, lp, ad, rt, mk = map(f, (b_obs, b_act, b_logp, adv, ret_, b_mask))
        keep = np.flatnonzero(mk.numpy() > 0)
        if len(keep) >= 64:
            mb = max(1, len(keep) // cfg.minibatches)
            for _ in range(cfg.epochs):
                shuffle_rng.shuffle(keep)
                for s0 in range(0, len(keep) - mb + 1, mb):
                    i = torch.as_tensor(keep[s0:s0 + mb])
                    logits = agent.actor(o[i])
                    logp = torch.log_softmax(logits, dim=1)
                    newlp = logp.gather(1, ac[i][:, None]).squeeze(1)
                    ent = -(logp.exp() * logp).sum(1).mean()
                    ratio = (newlp - lp[i]).exp()
                    a_ = ad[i]
                    a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                    pg = torch.max(-a_ * ratio, -a_ * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
                    vl = 0.5 * ((agent.critic(o[i]).squeeze(1) - rt[i]) ** 2).mean()
                    loss = pg - cfg.ent * ent + cfg.vf * vl
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    opt.step()

        if cfg.levels.sfl and it % cfg.levels.sfl_every == 0:
            levels.set_sfl(sfl_select(env, agent, levels, cfg.levels, env_rng))

        done_steps = (it + 1) * K * T
        if on_check and (done_steps % every < K * T or it == n_iter - 1):
            checks.append({"step": done_steps, **on_check(done_steps, agent)})
    return agent, checks, stats


@torch.no_grad()
def sfl_select(env, agent, levels: LevelSampler, lc: LevelConfig, rng: np.random.Generator) -> np.ndarray:
    """SFL's buffer: the `sfl_top` of `sfl_n` random levels by p(1 - p), p = the share of
    `sfl_k` stochastic-policy rollouts that pass."""
    cand = np.stack([levels.draw_new() for _ in range(lc.sfl_n)])
    params = np.repeat(cand, lc.sfl_k, axis=0)
    state = np.concatenate([env.sample_starts(p, 1, rng) for p in params])
    alive = np.ones(len(params), dtype=bool)
    ended = np.zeros(len(params), dtype=bool)
    for _ in range(env.MAX_EPISODE_STEPS):
        logits = agent.actor(torch.as_tensor(env.observe(state).astype(np.float32)))
        a = torch.multinomial(torch.softmax(logits, dim=1), 1).squeeze(1).numpy()
        state = np.where(alive[:, None], env.step(state, a, params), state)
        e = alive & env.terminated(state)
        ended |= e
        alive &= ~e
        if not alive.any():
            break
    success = ended if env.GOAL == "reach" else ~ended
    p = success.reshape(lc.sfl_n, lc.sfl_k).mean(1)
    learn = p * (1 - p)
    top = np.argsort(-learn, kind="stable")[:lc.sfl_top]
    return cand[top]
