"""Minimal discrete PPO, structured after SIPACL's Work/policy/ppo.py
(itself a CleanRL-style implementation) but for CartPole's Discrete(2)
action space instead of MetaDrive's continuous control, and driven by a
`PLRCurriculum` instead of a disk-backed scene buffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from acl_bench.curriculum.plr import PLRCurriculum, NEW
from acl_bench.envs.param_cartpole import CartPoleParams, make_env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


@dataclass
class PPOConfig:
    total_timesteps: int = 40_000
    num_steps: int = 1024
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    max_episode_steps: int = 500
    replay_prob: float = 0.5
    buffer_max: int = 200
    seed: int = 1


@dataclass
class RunLog:
    episode_returns: list = field(default_factory=list)
    episode_modes: list = field(default_factory=list)     # "new" / "replay"
    episode_task_ids: list = field(default_factory=list)
    episode_params: list = field(default_factory=list)    # CartPoleParams
    lp_scores: list = field(default_factory=list)


def run_training(sampler, potential_fn, cfg: PPOConfig, device="cpu") -> tuple[RunLog, Agent]:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    curriculum = PLRCurriculum(
        sampler=sampler, potential_fn=potential_fn, gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda, replay_prob=cfg.replay_prob,
        buffer_max=cfg.buffer_max, max_return=float(cfg.max_episode_steps),
        rng=rng,
    )

    obs_dim, n_actions = 4, 2
    agent = Agent(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    log = RunLog()

    params, task_idx, mode = curriculum.pick_task()
    env = make_env(params)
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    ep_rewards: list[float] = []
    ep_values: list[float] = []

    num_iterations = cfg.total_timesteps // cfg.num_steps
    global_step = 0
    for _iteration in range(num_iterations):
        b_obs = torch.zeros((cfg.num_steps, obs_dim))
        b_actions = torch.zeros(cfg.num_steps, dtype=torch.long)
        b_logprobs = torch.zeros(cfg.num_steps)
        b_rewards = torch.zeros(cfg.num_steps)
        b_dones = torch.zeros(cfg.num_steps)
        b_values = torch.zeros(cfg.num_steps)

        for t in range(cfg.num_steps):
            global_step += 1
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            b_obs[t] = obs_t
            b_actions[t] = action
            b_logprobs[t] = logprob
            b_values[t] = value.flatten()

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            b_rewards[t] = reward
            b_dones[t] = float(done)
            ep_rewards.append(float(reward))
            ep_values.append(float(value.item()))

            obs = next_obs
            if done:
                next_value = 0.0 if terminated else float(
                    agent.get_value(torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0))
                )
                raw_lp = curriculum.report_episode(task_idx, mode, ep_rewards, ep_values, next_value)

                log.episode_returns.append(sum(ep_rewards))
                log.episode_modes.append(mode)
                log.episode_task_ids.append(task_idx)
                log.episode_params.append(params)
                log.lp_scores.append(raw_lp)

                params, task_idx, mode = curriculum.pick_task()
                env = make_env(params)
                obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
                ep_rewards, ep_values = [], []

        with torch.no_grad():
            next_value = agent.get_value(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            ).reshape(1)
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0.0
            for t in reversed(range(cfg.num_steps)):
                nextnonterminal = 1.0 - b_dones[t]
                nextvalues = next_value if t == cfg.num_steps - 1 else b_values[t + 1]
                delta = b_rewards[t] + cfg.gamma * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + b_values

        b_inds = np.arange(cfg.num_steps)
        minibatch_size = cfg.num_steps // cfg.num_minibatches
        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.num_steps, minibatch_size):
                mb = b_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                mb_adv = advantages[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.flatten() - returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

    return log, agent


@torch.no_grad()
def evaluate_agent(agent: Agent, eval_params: list[CartPoleParams],
                    episodes_per_task: int = 3, max_steps: int = 500,
                    seed: int = 0) -> float:
    """Mean return over a fixed, sampler-independent set of task params --
    the generalization metric used to compare mix-and-match combinations."""
    rng = np.random.default_rng(seed)
    returns = []
    for params in eval_params:
        env = make_env(params)
        for _ in range(episodes_per_task):
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
            total = 0.0
            for _ in range(max_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_t)
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
    return float(np.mean(returns))
