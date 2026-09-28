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
from acl_bench.ppo import Agent, _mlp

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
    # Task-aware networks (the policy and critic otherwise see only the physical state):
    # critic_params gives the critic the task parameters (asymmetric actor-critic; the
    # policy stays blind), obs_params gives them to both.
    critic_params: bool = False
    obs_params: bool = False
    levels: LevelConfig = field(default_factory=LevelConfig)


def train(env_name: str, env, cfg: FastConfig, n_checks: int = 10, on_check=None):
    """Returns (agent, checks, stats). `on_check(step, agent)` runs at step 0 and after
    every steps / n_checks steps."""
    lv_ss, env_ss, sh_ss = np.random.SeedSequence(cfg.seed).spawn(3)
    torch.manual_seed(cfg.seed)
    env_rng, shuffle_rng = np.random.default_rng(env_ss), np.random.default_rng(sh_ss)
    bounds = np.array([env.PARAM_BOUNDS[k] for k in env.PARAM_ORDER], dtype=np.float64)
    levels = LevelSampler(cfg.levels, bounds, np.random.default_rng(lv_ss))
    r_step, r_term = REWARD.get(env_name, (0.0, 0.0))    # unused for an env with its own `reward`
    has_reward, balance = hasattr(env, "reward"), env.GOAL == "balance"
    if cfg.levels.oracle:
        from acl_bench import envs
        from acl_bench.exam.sets import load_sets
        levels.set_sfl(load_sets(envs.exam_dir(env_name), names=[cfg.levels.oracle])[cfg.levels.oracle].params)

    K, T = cfg.n_envs, 1024 // cfg.n_envs
    agent = make_agent(env, bounds, cfg)
    xa, xc = agent.inputs
    opt = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    # value-disagreement ensemble (vds score): extra critics used only for scoring levels,
    # with their own optimizer so they never touch the PPO update
    n_ens = cfg.levels.n_value_ens
    ens = nn.ModuleList([_mlp(agent.c_in, 1, out_std=1.0) for _ in range(n_ens)]) if n_ens else None
    ens_opt = optim.Adam(ens.parameters(), lr=cfg.lr, eps=1e-5) if n_ens else None

    params = np.zeros((K, len(bounds)))
    state = None
    slot = np.full(K, -1)
    mode = np.zeros(K, dtype=int)
    t_ep = np.zeros(K, dtype=int)
    ep_r = [[] for _ in range(K)]
    ep_v = [[] for _ in range(K)]
    ep_ent = [[] for _ in range(K)]           # policy entropy per step (entropy score)
    ep_dis = [[] for _ in range(K)]           # ensemble value std per step (vds score)
    ep_train = np.ones(K, dtype=bool)         # PLR-perp: does this episode's data train the agent?
    ep_s0 = [None] * K
    streak = np.zeros(K, dtype=int)           # consecutive upright steps ("balance" goal)

    def start(k):
        p, i, m = levels.pick()
        params[k], slot[k], mode[k] = p, i, m
        if m == REPLAY and cfg.levels.replay_start and i >= 0:
            s = levels.starts[i].copy()
        else:
            s = env.sample_starts(p, 1, env_rng)[0]
        ep_s0[k] = s.copy()
        streak[k] = 0
        t_ep[k] = 0
        ep_r[k], ep_v[k], ep_ent[k], ep_dis[k] = [], [], [], []
        ep_train[k] = not (cfg.levels.robust and m == NEW)
        return s

    state = np.stack([start(k) for k in range(K)])
    checks, stats = [], {"episodes": 0, "replays": 0}
    every = cfg.steps // n_checks
    if on_check:
        checks.append({"step": 0, **on_check(0, agent)})

    n_iter = cfg.steps // (K * T)
    for it in range(n_iter):
        b_obs = np.zeros((T, K, agent.a_in), dtype=np.float32)
        b_cobs = np.zeros((T, K, agent.c_in), dtype=np.float32)
        b_act = np.zeros((T, K), dtype=np.int64)
        b_logp = np.zeros((T, K), dtype=np.float32)
        b_val = np.zeros((T, K), dtype=np.float32)
        b_rew = np.zeros((T, K), dtype=np.float32)
        b_done = np.zeros((T, K), dtype=np.float32)
        b_mask = np.zeros((T, K), dtype=np.float32)
        for t in range(T):
            raw = env.observe(state).astype(np.float32)
            obs, cobs = xa(raw, params), xc(raw, params)
            with torch.no_grad():
                logits = agent.actor(torch.as_tensor(obs))
                logp_all = torch.log_softmax(logits, dim=1)
                a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                v = agent.critic(torch.as_tensor(cobs)).squeeze(1)
                ent_t = (-(logp_all.exp() * logp_all).sum(1)).numpy()
                dis_t = (torch.stack([m(torch.as_tensor(cobs)).squeeze(1) for m in ens]).std(0).numpy()
                         if ens is not None else None)
            a_np = a.numpy()
            b_obs[t], b_cobs[t], b_act[t] = obs, cobs, a_np
            b_logp[t] = logp_all.gather(1, a[:, None]).squeeze(1).numpy()
            b_val[t] = v.numpy()
            b_mask[t] = ep_train
            rew_env = env.reward(state, a_np, params) if has_reward else None
            state = env.step(state, a_np, params)
            term = env.terminated(state)
            if balance:
                streak[:] = np.where(env.upright(state), streak + 1, 0)
            t_ep += 1
            trunc = t_ep >= env.MAX_EPISODE_STEPS
            rew = rew_env if has_reward else np.where(term, r_term, r_step)
            b_rew[t] = rew
            done = term | trunc
            b_done[t] = done
            for k in range(K):
                ep_r[k].append(float(rew[k]))
                ep_v[k].append(float(b_val[t, k]))
                ep_ent[k].append(float(ent_t[k]))
                if dis_t is not None:
                    ep_dis[k].append(float(dis_t[k]))
                if done[k]:
                    if term[k]:
                        nv = 0.0
                    else:
                        with torch.no_grad():
                            c = xc(env.observe(state[k:k + 1]).astype(np.float32), params[k:k + 1])
                            nv = float(agent.critic(torch.as_tensor(c)))
                    if balance:
                        success = bool(streak[k] >= env.HOLD_STEPS)
                    else:
                        success = bool(term[k]) if env.GOAL == "reach" else not term[k]
                    ret = float(sum(ep_r[k]))
                    sc = levels.score(ep_r[k], ep_v[k], nv, cfg.gamma, cfg.lam, success, slot[k],
                                      entropy=float(np.mean(ep_ent[k])),
                                      disagreement=float(np.mean(ep_dis[k])) if ep_dis[k] else None)
                    levels.report(params[k].copy(), slot[k], mode[k], sc, ret, success, ep_s0[k])
                    stats["episodes"] += 1
                    stats["replays"] += int(mode[k] == REPLAY)
                    state[k] = start(k)

        with torch.no_grad():
            nv = agent.critic(torch.as_tensor(xc(env.observe(state).astype(np.float32), params))).squeeze(1).numpy()
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
        o, oc, ac, lp, ad, rt, mk = map(f, (b_obs, b_cobs, b_act, b_logp, adv, ret_, b_mask))
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
                    vl = 0.5 * ((agent.critic(oc[i]).squeeze(1) - rt[i]) ** 2).mean()
                    loss = pg - cfg.ent * ent + cfg.vf * vl
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    opt.step()
                    if ens is not None:
                        # each member on its own random half of the minibatch, for diversity
                        halves = torch.as_tensor(shuffle_rng.random((len(ens), len(i))) < 0.5)
                        el = sum(0.5 * (((m(oc[i]).squeeze(1) - rt[i]) ** 2) * h).sum() / h.sum().clamp(min=1)
                                 for m, h in zip(ens, halves))
                        ens_opt.zero_grad()
                        el.backward()
                        ens_opt.step()

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
    streak = np.zeros(len(params), dtype=int)
    for _ in range(env.MAX_EPISODE_STEPS):
        logits = agent.actor(torch.as_tensor(agent.inputs[0](env.observe(state).astype(np.float32), params)))
        a = torch.multinomial(torch.softmax(logits, dim=1), 1).squeeze(1).numpy()
        state = np.where(alive[:, None], env.step(state, a, params), state)
        if env.GOAL == "balance":
            streak = np.where(env.upright(state), streak + 1, 0)
        e = alive & env.terminated(state)
        ended |= e
        alive &= ~e
        if not alive.any():
            break
    if env.GOAL == "balance":
        success = streak >= env.HOLD_STEPS
    else:
        success = ended if env.GOAL == "reach" else ~ended
    p = success.reshape(lc.sfl_n, lc.sfl_k).mean(1)
    learn = p * (1 - p)
    top = np.argsort(-learn, kind="stable")[:lc.sfl_top]
    return cand[top]


def make_agent(env, bounds: np.ndarray, cfg: FastConfig) -> Agent:
    """The study's Agent, with the task parameters (scaled to [-1, 1]) appended to the
    critic's and/or the actor's input. `agent.inputs` = (actor input fn, critic input fn),
    each (raw obs (N, OBS_DIM), params (N, P)) -> network input."""
    n_p = len(bounds)
    a_in = env.OBS_DIM + (n_p if cfg.obs_params else 0)
    c_in = env.OBS_DIM + (n_p if cfg.obs_params or cfg.critic_params else 0)
    agent = Agent(a_in, env.ACTION_DIM)
    if c_in != a_in:
        agent.critic = _mlp(c_in, 1, out_std=1.0)
    lo, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]

    def with_params(raw, params):
        return np.concatenate([raw, ((params - lo) / span * 2 - 1).astype(np.float32)], axis=1)

    def plain(raw, params):
        return raw

    agent.inputs = (with_params if cfg.obs_params else plain,
                    with_params if (cfg.obs_params or cfg.critic_params) else plain)
    agent.a_in, agent.c_in = a_in, c_in
    return agent


class _TaskActor:
    """What the grader calls as `agent.actor(obs)`, for a policy that also reads the task:
    appends the questions' own parameters (set per section) to each observation."""

    def __init__(self, agent):
        self.agent, self.params = agent, None

    def __call__(self, obs):
        x = self.agent.inputs[0](obs.numpy(), self.params)
        return self.agent.actor(torch.as_tensor(x))


class _Graded:
    def __init__(self, actor):
        self.actor = actor


def evaluate(env, agent, sets) -> dict:
    """evaluate_sets, passing each question's parameters to a task-aware policy."""
    from acl_bench.exam.sets import evaluate_sets
    if agent.a_in == env.OBS_DIM:
        return evaluate_sets(env, agent, sets)
    shim = _Graded(_TaskActor(agent))
    out = {}
    for name, ps in sets.items():
        shim.actor.params = ps.params
        out.update(evaluate_sets(env, shim, {name: ps}))
    return out
