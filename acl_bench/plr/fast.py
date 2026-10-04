"""Vectorized PPO for screening: `n_envs` copies of the environment step together through
its batched physics (the same code the grader uses, checked against gymnasium in
tests/test_grader.py), so one policy forward pass serves all of them.

Same PPO as acl_bench/ppo.py (network, losses, 1024 samples per update, 4 minibatches,
4 epochs, truncation treated as terminal) except that the 1024 samples come from
`n_envs` parallel episodes of 1024 / n_envs steps. Results from here are compared only
with results from here; a finding is confirmed in the study pipeline (acl_bench.study).
"""
from __future__ import annotations

import copy
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
    regret, pending = cfg.levels.score == "regret", []
    if cfg.levels.oracle:
        levels.set_sfl(oracle_tasks(env_name, cfg.levels.oracle))

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

    cur = None
    ep_src, ep_w = [""] * K, np.ones(K, dtype=np.float32)
    ep_max = np.full(K, env.MAX_EPISODE_STEPS)     # per-episode horizon (library `cap` arms)
    if cfg.levels.lib:
        from acl_bench.curriculum import build
        cur = build(cfg.levels.lib, env, bounds, K, cfg.seed)

    picker = None
    if cfg.levels.picker == "model":
        from acl_bench.plr.picker import FrontierModelPicker
        picker = FrontierModelPicker(env, K, cfg.levels.picker_uniform, np.random.default_rng([cfg.seed, 7]),
                                     cfg.seed, n_candidates=cfg.levels.picker_candidates)
    elif cfg.levels.picker != "uniform":
        from acl_bench.plr.picker import LearnabilityPicker
        picker = LearnabilityPicker(env, cfg.levels.picker, K, cfg.levels.picker_uniform,
                                    np.random.default_rng([cfg.seed, 7]), cfg.seed)

    def start(k):
        p, i, m = levels.pick()
        if picker is not None and m == NEW:
            p = picker.draw(k)
        if cur is not None:
            p, ep_src[k], ep_w[k] = cur.propose(k)
            i, m = -1, NEW
            if cur.cap is not None and ep_src[k] != "uniform":
                ep_max[k] = max(1, int(cur.cap * env.MAX_EPISODE_STEPS))
            else:
                ep_max[k] = env.MAX_EPISODE_STEPS
        params[k], slot[k], mode[k] = p, i, m
        if m == REPLAY and cfg.levels.replay_start and i >= 0:
            s = levels.starts[i].copy()
        elif m == REPLAY and getattr(levels, "next_start", None) is not None:
            s = levels.next_start.copy()     # start-state SFL: replay from an archived state
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

    # With charge_scouting, SFL's scouting rollouts count against the budget: training stops
    # when training + scouting steps reach it, and check-ins fall on that total.
    charge = cfg.levels.charge_scouting
    n_iter = cfg.steps // (K * T)
    scouted, next_check = 0, every
    for it in range(n_iter):
        if cur is not None:
            cur.progress = it / n_iter      # fraction of the training budget used (for schedules)
        b_obs = np.zeros((T, K, agent.a_in), dtype=np.float32)
        b_cobs = np.zeros((T, K, agent.c_in), dtype=np.float32)
        b_act = np.zeros((T, K), dtype=np.int64)
        b_logp = np.zeros((T, K), dtype=np.float32)
        b_val = np.zeros((T, K), dtype=np.float32)
        b_rew = np.zeros((T, K), dtype=np.float32)
        b_done = np.zeros((T, K), dtype=np.float32)
        b_mask = np.zeros((T, K), dtype=np.float32)
        b_w = np.ones((T, K), dtype=np.float32)
        b_par = np.zeros((T, K, len(bounds)))
        b_src = [[""] * K for _ in range(T)]
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
            b_w[t] = ep_w
            b_par[t] = params
            b_src[t] = list(ep_src)
            rew_env = env.reward(state, a_np, params) if has_reward else None
            state = env.step(state, a_np, params)
            term = env.terminated(state)
            if balance:
                streak[:] = np.where(env.upright(state), streak + 1, 0)
            t_ep += 1
            trunc = t_ep >= ep_max
            if cfg.levels.sfl_states > 0:
                _archive(levels, cfg.levels, params, state, ~(term | trunc) & (t_ep % cfg.levels.sfl_archive_every == 0))
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
                        if ep_max[k] < env.MAX_EPISODE_STEPS:
                            b_rew[t, k] += cfg.gamma * nv      # capped snippet: bootstrap, not terminal
                    if balance:
                        success = bool(streak[k] >= env.HOLD_STEPS)
                    else:
                        success = bool(term[k]) if env.GOAL == "reach" else not term[k]
                    ret = float(sum(ep_r[k]))
                    if picker is not None:
                        picker.report(k, params[k], success)
                    snip = not term[k] and ep_max[k] < env.MAX_EPISODE_STEPS   # outcome unknown
                    if cur is not None and not snip:
                        from acl_bench.curriculum import Episode
                        cur.report(k, Episode(params[k].copy(), success, float(sum(ep_r[k])), int(t_ep[k]),
                                              ep_src[k], ep_r[k], ep_v[k], nv))
                    if regret:
                        # planner regret: scored in one batch at the end of the rollout
                        pending.append((params[k].copy(), ep_s0[k], int(slot[k]), int(mode[k]), ret, success))
                    else:
                        sc = levels.score(ep_r[k], ep_v[k], nv, cfg.gamma, cfg.lam, success, slot[k],
                                          entropy=float(np.mean(ep_ent[k])),
                                          disagreement=float(np.mean(ep_dis[k])) if ep_dis[k] else None)
                        levels.report(params[k].copy(), slot[k], mode[k], sc, ret, success, ep_s0[k])
                    stats["episodes"] += 1
                    stats["replays"] += int(mode[k] == REPLAY)
                    state[k] = start(k)

        if regret and pending:
            flush_regret(env, levels, pending, has_reward)
            pending.clear()

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
        if cur is not None:
            cur.on_rollout(agent, b_obs, b_par, b_act, adv, b_done, b_src)

        f = lambda x: torch.as_tensor(x.reshape(T * K, *x.shape[2:]))
        o, oc, ac, lp, ad, rt, mk, wt = map(f, (b_obs, b_cobs, b_act, b_logp, adv, ret_, b_mask, b_w))
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
                    pg_i = torch.max(-a_ * ratio, -a_ * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip))
                    w_i = wt[i]                        # importance weights (library arms), else 1
                    pg = (w_i * pg_i).sum() / w_i.sum()
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
            top, sim, w = sfl_select(env, agent, levels, cfg.levels, env_rng)
            levels.set_sfl(top, w, getattr(levels, "sfl_top_starts", None))
            scouted += sim

        done_steps = (it + 1) * K * T
        consumed = done_steps + (scouted if charge else 0)
        last = it == n_iter - 1 or consumed >= cfg.steps
        if on_check and (consumed >= next_check or last):
            checks.append({"step": min(consumed, cfg.steps), **on_check(consumed, agent)})
            while next_check <= consumed:
                next_check += every
        if last:
            break
    stats["scouted_steps"] = scouted + (cur.sim_steps if cur is not None else 0)
    if cur is not None:
        stats.update(cur.stats())
    return agent, checks, stats


@torch.no_grad()
def sfl_select(env, agent, levels: LevelSampler, lc: LevelConfig, rng: np.random.Generator):
    """SFL's buffer: the `sfl_top` of `sfl_n` random levels by p(1 - p), p = the share of
    `sfl_k` stochastic-policy rollouts that pass. With `lc.sfl_score == "pvl"` the same
    scouted levels are ranked by PVL instead (mean over the k rollouts of the episode's
    mean positive GAE advantage under the current critic): SFL's search with PLR's score.
    Returns (levels, simulated steps, replay weights or None for uniform)."""
    sim_auto = _auto_intensity(env, agent, levels, lc, rng) if lc.sfl_auto > 0 else 0
    cand = np.stack([levels.draw_new() for _ in range(lc.sfl_n)])
    hist = getattr(levels, "sfl_hist", [])
    if lc.sfl_amort > 0 and hist:               # amortized proposer: pre-screen by predicted score
        cand[:lc.sfl_n // 2] = _prescreen(levels, hist, lc, lc.sfl_n // 2)
    prev = getattr(levels, "sfl_levels", None)
    if lc.sfl_mut > 0 and prev is not None:     # local search around the last frontier (ACCEL)
        n_mut = int(lc.sfl_mut * lc.sfl_n)
        parents = prev[rng.integers(len(prev), size=n_mut)]
        cand[:n_mut] = [levels._mutate(q) for q in parents]
    st_s, st_has = None, None
    arch = getattr(levels, "archive", None)
    if lc.sfl_states > 0 and arch is not None and arch["n"] > 0:
        n_st = int(lc.sfl_states * lc.sfl_n)     # (level, visited state) candidates
        j = rng.integers(arch["n"], size=n_st)
        cand[:n_st] = arch["p"][j]
        st_s = np.zeros((lc.sfl_n, arch["s"].shape[1]))
        st_has = np.zeros(lc.sfl_n, dtype=bool)
        st_s[:n_st], st_has[:n_st] = arch["s"][j], True
    assert st_s is None or not (lc.sfl_halving or lc.sfl_bisect), "sfl_states with halving/bisect"
    if lc.sfl_carry and prev is not None:       # the last buffer competes again with fresh draws
        cand[-len(prev):] = prev
        old = getattr(levels, "sfl_starts", None)
        if st_s is not None and old is not None:
            st_s[-len(prev):], st_has[-len(prev):] = old
    params = np.repeat(cand, lc.sfl_k, axis=0)
    state = np.concatenate([env.sample_starts(p, 1, rng) for p in params])
    if st_s is not None:
        rep = np.repeat(st_has, lc.sfl_k)
        state[rep] = np.repeat(st_s, lc.sfl_k, axis=0)[rep]
    alive = np.ones(len(params), dtype=bool)
    ended = np.zeros(len(params), dtype=bool)
    streak = np.zeros(len(params), dtype=int)
    pvl = lc.sfl_score == "pvl"
    rews, vals, alives, terms = [], [], [], []
    r_step, r_term = REWARD.get(env.__name__.rsplit(".", 1)[-1], (0.0, 0.0))
    xa, xc = agent.inputs
    sim = 0                                        # environment steps simulated here
    for _ in range(env.MAX_EPISODE_STEPS):
        sim += int(alive.sum())
        raw = env.observe(state).astype(np.float32)
        logits = agent.actor(torch.as_tensor(xa(raw, params)))
        a = torch.multinomial(torch.softmax(logits, dim=1), 1).squeeze(1).numpy()
        if pvl:
            vals.append(agent.critic(torch.as_tensor(xc(raw, params))).squeeze(1).numpy())
            alives.append(alive.copy())
            r_env = env.reward(state, a, params) if hasattr(env, "reward") else None
        state = np.where(alive[:, None], env.step(state, a, params), state)
        if env.GOAL == "balance":
            streak = np.where(env.upright(state), streak + 1, 0)
        e = alive & env.terminated(state)
        if pvl:
            rews.append(r_env if r_env is not None else np.where(e, r_term, r_step))
            terms.append(e)
        ended |= e
        alive &= ~e
        if not alive.any():
            break
    if pvl:
        v_end = agent.critic(torch.as_tensor(xc(env.observe(state).astype(np.float32), params))).squeeze(1).numpy()
        score = _scout_pvl(np.array(rews), np.array(vals), np.array(alives), np.array(terms), v_end)
    else:
        if env.GOAL == "balance":
            success = streak >= env.HOLD_STEPS
        else:
            success = ended if env.GOAL == "reach" else ~ended
        p = success.reshape(lc.sfl_n, lc.sfl_k).mean(1)
        if lc.sfl_memory > 0:                   # pool a carried level's discounted earlier rollouts
            wins, tries = p * lc.sfl_k, np.full(len(p), float(lc.sfl_k))
            old = getattr(levels, "sfl_counts", None)
            if lc.sfl_carry and prev is not None and old is not None:
                wins[-len(prev):] += lc.sfl_memory * old[0]
                tries[-len(prev):] += lc.sfl_memory * old[1]
                p = wins / tries
        if lc.sfl_amort > 0:
            levels.sfl_hist = (hist + [(cand, p)])[-3:]
        if lc.sfl_halving > 0:
            cand, p, sim2 = _halving(env, agent, cand, p, lc, rng)
            sim += sim2
        if lc.sfl_bisect > 0:
            mid, pm, sim2 = _bisect(env, agent, levels, cand, p, lc, rng)
            cand, p, sim = np.concatenate([cand, mid]), np.concatenate([p, pm]), sim + sim2
        score = p * (1 - p) * (1 - p) ** lc.sfl_tilt
        if lc.sfl_ghost > 0:
            old = getattr(levels, "sfl_ghost_agent", None)
            if old is not None:
                p_old, sim2 = _pass_rate(env, old, cand, lc.sfl_k, rng)
                sim += sim2
                score = score + lc.sfl_ghost * np.abs(p - p_old)
            levels.sfl_ghost_agent = copy.deepcopy(agent)
        if lc.sfl_verify > 0:                   # re-rank the shortlist on 4x the rollouts
            short = np.argsort(-score, kind="stable")[:lc.sfl_verify]
            p2, sim2 = _pass_rate(env, agent, cand[short], 3 * lc.sfl_k, rng)
            sim += sim2
            q = (p[short] + 3 * p2) / 4
            score = np.full(len(cand), -1.0)
            score[short] = q * (1 - q) * (1 - q) ** lc.sfl_tilt
    if pvl:
        score = score.reshape(lc.sfl_n, lc.sfl_k).mean(1)
    sim += sim_auto
    if lc.sfl_soft:
        keep = score > 0
        levels.sfl_top_starts = None if st_s is None else (st_s[keep], st_has[keep])
        if keep.any():
            return cand[keep], sim, score[keep]
        return None, sim, None                  # nothing learnable scouted: replay nothing
    top = np.argsort(-score, kind="stable")[:lc.sfl_top]
    if lc.sfl_spread > 0:
        pool = np.argsort(-score, kind="stable")[:lc.sfl_spread * lc.sfl_top]
        top = pool[_farthest(levels, cand[pool], lc.sfl_top)]
    if lc.sfl_memory > 0 and not pvl:
        levels.sfl_counts = (wins[top], tries[top])
    levels.sfl_top_starts = None if st_s is None else (st_s[top], st_has[top])
    return cand[top], sim, None


def _archive(levels: LevelSampler, lc: LevelConfig, params: np.ndarray, state: np.ndarray, keep: np.ndarray) -> None:
    """Push the kept (level, visited state) pairs into a FIFO archive for start-state SFL."""
    idx = np.flatnonzero(keep)
    if not len(idx):
        return
    a = levels.__dict__.get("archive")
    if a is None:
        a = levels.archive = {"p": np.zeros((lc.sfl_archive, params.shape[1])),
                              "s": np.zeros((lc.sfl_archive, state.shape[1])), "n": 0, "i": 0}
    pos = (a["i"] + np.arange(len(idx))) % lc.sfl_archive
    a["p"][pos], a["s"][pos] = params[idx], state[idx]
    a["i"] = int((a["i"] + len(idx)) % lc.sfl_archive)
    a["n"] = min(lc.sfl_archive, a["n"] + len(idx))


AUTO_ARMS = (0.0, 0.25, 0.5, 0.75)


def _auto_intensity(env, agent, levels: LevelSampler, lc: LevelConfig, rng, eps: float = 0.3,
                    alpha: float = 0.3) -> int:
    """Pick this interval's replay probability (`levels.sfl_rp`) from AUTO_ARMS. The agent's
    robustness, mean + worst-quarter pass rate on `lc.sfl_auto` uniform levels drawn once, is
    measured every scout; its change since the last scout rewards the arm played in between
    (recency-weighted average `alpha`, for a moving learner). Each arm is tried once, then
    epsilon-greedy. Returns the simulated steps."""
    st = levels.__dict__.get("auto")
    if st is None:
        st = levels.auto = {"probe": np.stack([levels.draw_new() for _ in range(lc.sfl_auto)]),
                            "q": np.zeros(len(AUTO_ARMS)), "n": np.zeros(len(AUTO_ARMS), int),
                            "arm": None, "rob": None, "trace": []}
    p, sim = _pass_rate(env, agent, st["probe"], lc.sfl_k, rng)
    rob = p.mean() + np.sort(p)[:max(1, len(p) // 4)].mean()
    if st["arm"] is not None:
        a, r = st["arm"], rob - st["rob"]
        st["q"][a] = r if st["n"][a] == 0 else st["q"][a] + alpha * (r - st["q"][a])
        st["n"][a] += 1
    untried = np.flatnonzero(st["n"] == 0)
    if len(untried):
        a = int(untried[0])
    elif rng.uniform() < eps:
        a = int(rng.integers(len(AUTO_ARMS)))
    else:
        a = int(np.argmax(st["q"]))
    st["arm"], st["rob"] = a, rob
    st["trace"].append(a)
    levels.sfl_rp = AUTO_ARMS[a]
    return sim


def _halving(env, agent, cand, p, lc: LevelConfig, rng, eta: int = 4):
    """Successive halving on the scouted levels: each of `lc.sfl_halving` rounds keeps the
    best 1/eta (at least `sfl_top`) by the score of the Beta(1,1) posterior mean pass rate,
    then rolls them out until each has eta x its previous rollouts. Returns the survivors,
    their pooled pass rates and the steps simulated."""
    succ, n_roll = p * lc.sfl_k, np.full(len(cand), float(lc.sfl_k))
    idx, sim = np.arange(len(cand)), 0
    for r in range(lc.sfl_halving):
        q = (succ[idx] + 1) / (n_roll[idx] + 2)
        s = q * (1 - q) * (1 - q) ** lc.sfl_tilt
        idx = idx[np.argsort(-s, kind="stable")[:max(lc.sfl_top, len(idx) // eta)]]
        extra = lc.sfl_k * eta ** r * (eta - 1)
        p2, sim2 = _pass_rate(env, agent, cand[idx], extra, rng)
        succ[idx] += p2 * extra
        n_roll[idx] += extra
        sim += sim2
    return cand[idx], succ[idx] / n_roll[idx], sim


def _bisect(env, agent, levels: LevelSampler, cand, p, lc: LevelConfig, rng, steps: int = 3):
    """Edge-of-competence search (VerifAI-style boundary sampling): pair `lc.sfl_bisect`
    random failed levels (p <= 1/8) with their nearest solved one (p >= 7/8) and bisect each
    segment `steps` times toward the score's peak p* = 1/(2 + tilt). Returns the last
    midpoints, their pass rates and the steps simulated."""
    from scipy.spatial import cKDTree
    lo, span = levels.bounds[:, 0], levels.bounds[:, 1] - levels.bounds[:, 0]
    solved, failed = cand[p >= 7 / 8], cand[p <= 1 / 8]
    if not len(solved) or not len(failed):
        return cand[:0], p[:0], 0
    f = failed[rng.integers(len(failed), size=lc.sfl_bisect)]
    _, j = cKDTree((solved - lo) / span).query((f - lo) / span)
    s, target, sim = solved[j], 1 / (2 + lc.sfl_tilt), 0
    for _ in range(steps):
        mid = (s + f) / 2
        pm, sim2 = _pass_rate(env, agent, mid, lc.sfl_k, rng)
        sim += sim2
        easy = (pm > target)[:, None]
        s, f = np.where(easy, mid, s), np.where(easy, f, mid)
    return mid, pm, sim


def _farthest(levels: LevelSampler, x: np.ndarray, m: int) -> np.ndarray:
    """Indices of `m` rows of `x` by greedy farthest-point traversal from row 0 (the best),
    in box-normalized coordinates."""
    if len(x) <= m:
        return np.arange(len(x))
    lo, span = levels.bounds[:, 0], levels.bounds[:, 1] - levels.bounds[:, 0]
    z = (x - lo) / span
    chosen, d = [0], np.linalg.norm(z - z[0], axis=1)
    for _ in range(m - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(z - z[i], axis=1))
    return np.array(chosen)


def _pass_rate(env, agent, cand: np.ndarray, k: int, rng: np.random.Generator):
    """Share of `k` stochastic-policy rollouts on each level in `cand` that pass, and the
    environment steps simulated (the success test of `sfl_select`)."""
    params = np.repeat(cand, k, axis=0)
    state = np.concatenate([env.sample_starts(q, 1, rng) for q in params])
    alive = np.ones(len(params), dtype=bool)
    ended = np.zeros(len(params), dtype=bool)
    streak = np.zeros(len(params), dtype=int)
    xa, _ = agent.inputs
    sim = 0
    for _ in range(env.MAX_EPISODE_STEPS):
        sim += int(alive.sum())
        logits = agent.actor(torch.as_tensor(xa(env.observe(state).astype(np.float32), params)))
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
    return success.reshape(len(cand), k).mean(1), sim


def _prescreen(levels: LevelSampler, hist, lc: LevelConfig, m: int, nn_k: int = 16) -> np.ndarray:
    """The `m` best of `lc.sfl_amort * lc.sfl_n` uniform levels by the score of p predicted
    as the mean measured p of the `nn_k` nearest scouted levels (box-normalized distance)."""
    from scipy.spatial import cKDTree
    lo, span = levels.bounds[:, 0], levels.bounds[:, 1] - levels.bounds[:, 0]
    x = np.concatenate([c for c, _ in hist])
    y = np.concatenate([q for _, q in hist])
    pool = lo + span * levels.rng.uniform(size=(lc.sfl_amort * lc.sfl_n, len(lo)))
    _, idx = cKDTree((x - lo) / span).query((pool - lo) / span, k=min(nn_k, len(x)))
    p = y[idx].reshape(len(pool), -1).mean(1)
    score = p * (1 - p) * (1 - p) ** lc.sfl_tilt
    return pool[np.argsort(-score, kind="stable")[:m]]


def _scout_pvl(rew, val, alive, term, v_end, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
    """Per rollout, mean over its steps of max(GAE advantage, 0), for (T, N) arrays of a
    batched rollout: termination ends an episode (next value 0), and one still running at
    the step cap bootstraps from the critic, as the training loop's PVL does."""
    T, n = rew.shape
    adv_pos, last = np.zeros(n), np.zeros(n)
    for t in reversed(range(T)):
        nxt = np.where(term[t], 0.0, val[t + 1] if t + 1 < T else v_end)
        delta = rew[t] + gamma * nxt - val[t]
        last = np.where(alive[t], delta + gamma * lam * np.where(term[t], 0.0, last), 0.0)
        adv_pos += np.where(alive[t], np.clip(last, 0.0, None), 0.0)
    length = np.maximum(alive.sum(0), 1)
    return adv_pos / length


def oracle_tasks(env_name: str, which: str) -> np.ndarray:
    """Task parameters for a diagnostic oracle: an exam section's own tasks ("verifai"), or
    "frontier": the VerifAI search pools' tasks that 30-70% of the reference agents fail."""
    from acl_bench import envs
    from acl_bench.exam.sets import load_sets
    from acl_bench.sampling import ADAPTIVE_SAMPLERS
    if which != "frontier":
        return load_sets(envs.exam_dir(env_name), names=[which])[which].params
    pools = load_sets(envs.search_dir(env_name), names=[f"SEARCH_{s}" for s in ADAPTIVE_SAMPLERS]).values()
    params = np.concatenate([p.params for p in pools])
    fail = np.concatenate([p.ref_fail_frac for p in pools])
    return params[(fail >= 0.3 - 1e-9) & (fail <= 0.7 + 1e-9)]


def plan_values(env, params: np.ndarray, s0: np.ndarray, beam: int = 64) -> np.ndarray:
    """What the physics planner (the exam's beam search, acl_bench.exam.certify) achieves
    on each (task, start): the return of its sequence for an env with a reward, else 1.0
    if its replayed sequence passes; -inf / 0.0 where it found nothing."""
    from acl_bench.exam.certify import beam_search, replay_passes
    won, _, actions = beam_search(env, params, s0, beam=beam)
    if not hasattr(env, "reward"):
        return (won & replay_passes(env, params, s0, actions)).astype(float)
    state, ret = np.array(s0, dtype=np.float64), np.zeros(len(params))
    for t in range(actions.shape[1]):
        a = np.maximum(actions[:, t], 0)
        ret += env.reward(state, a, params)
        state = env.step(state, a, params)
    return np.where(won, ret, -np.inf)


def flush_regret(env, levels: LevelSampler, pending: list, continuous: bool) -> None:
    """Score and report a rollout's finished episodes by planner regret: max(planner,
    best outcome seen on the level, this outcome) - this outcome, where the outcome is the
    return (continuous) or the pass indicator. New levels are planned once, in one batch;
    replays reuse their level's stored plan (they repeat the exact start: replay_start)."""
    new = [j for j, p in enumerate(pending) if p[3] == NEW]
    plans = {}
    if new:
        vals = plan_values(env, np.stack([pending[j][0] for j in new]), np.stack([pending[j][1] for j in new]))
        plans = dict(zip(new, vals))
    for j, (prm, s0, slot, mode, ret, success) in enumerate(pending):
        outcome = ret if continuous else float(success)
        if mode == NEW:
            plan, seen = plans[j], -np.inf
        else:
            plan = levels.plan[slot]
            seen = levels.best[slot] if continuous else (1.0 if levels.wins[slot] > 0 else 0.0)
        sc = max(plan, seen, outcome) - outcome
        levels.report(prm, slot, mode, sc, ret, success, s0, plan=plan)


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
