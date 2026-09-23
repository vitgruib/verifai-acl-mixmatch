"""Minimal CleanRL-style PPO on the task-parameterized CartPole, driven by the ACL
curriculum (acl_bench.acl). Independent of SIPACL: its Work/policy/ppo.py is not
carried over. Like SIPACL's and CleanRL's PPO, the update treats a time-limit
truncation as terminal; the curriculum's task score does not (docs/sipacl.md).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from acl_bench import cartpole
from acl_bench.acl import PLRCurriculum


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_dim: int, out_dim: int, out_std: float) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, 64)), nn.Tanh(),
        layer_init(nn.Linear(64, 64)), nn.Tanh(),
        layer_init(nn.Linear(64, out_dim), std=out_std),
    )


class Agent(nn.Module):
    """Actor and critic, each two 64-unit tanh layers; a Categorical over discrete actions."""

    def __init__(self, obs_dim: int = cartpole.OBS_DIM, action_dim: int = cartpole.ACTION_DIM):
        super().__init__()
        self.critic = _mlp(obs_dim, 1, out_std=1.0)
        self.actor = _mlp(obs_dim, action_dim, out_std=0.01)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        dist = Categorical(logits=self.actor(x))
        action = dist.sample() if action is None else action
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


@dataclass
class PPOConfig:
    total_timesteps: int = 100_000
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
    acl: bool = True          # False = never replay (SIPACL's replay_resample_prob=-1)
    replay_prob: float = 0.5  # SIPACL's replay_resample_prob when ACL is on
    buffer_max: int = 5000    # SIPACL DEFAULT_BUFFER_MAX
    rank_alpha: float = 1.0   # SIPACL DEFAULT_LP_RANK_ALPHA
    ema_beta: float = 0.2     # SIPACL DEFAULT_LP_EMA_BETA
    seed: int = 1


@dataclass
class RunLog:
    episode_returns: list = field(default_factory=list)
    episode_modes: list = field(default_factory=list)     # "new" / "replay"
    episode_params: list = field(default_factory=list)    # dict per episode
    # one dict per on_checkpoint(step, agent) call: {"step": ..., **metrics}; includes step 0 (untrained)
    checkpoint_metrics: list = field(default_factory=list)


def run_training(task_sampler, score_fn, cfg: PPOConfig, checkpoint_every: int | None = None,
                 on_checkpoint=None) -> tuple[RunLog, Agent]:
    """Train one agent. Every `checkpoint_every` env steps (and at step 0 and the end)
    `on_checkpoint(step, agent)` is called and its returned dict is logged."""
    # One independent stream per component, so a seed fully determines a run (same
    # config and seed reproduce exactly) and no component's draws depend on how many
    # another consumed.
    curriculum_ss, env_ss, shuffle_ss, sampler_ss = np.random.SeedSequence(cfg.seed).spawn(4)
    rng = np.random.default_rng(curriculum_ss)          # replay choices
    env_rng = np.random.default_rng(env_ss)             # per-episode initial-state seeds
    shuffle_rng = np.random.default_rng(shuffle_ss)     # PPO minibatch order
    torch.manual_seed(cfg.seed)                          # network init, then action sampling
    sampler_state = int(sampler_ss.generate_state(1)[0])
    random.seed(sampler_state)                           # Scenic/VerifAI samplers draw from
    np.random.seed(sampler_state)                        # these globals; nothing else here does

    curriculum = PLRCurriculum(
        task_sampler=task_sampler, param_names=cartpole.PARAM_ORDER, score_fn=score_fn,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, use_replay=cfg.acl,
        replay_prob=cfg.replay_prob, buffer_max=cfg.buffer_max, rank_alpha=cfg.rank_alpha,
        ema_beta=cfg.ema_beta, rng=rng,
    )

    obs_dim = cartpole.OBS_DIM
    agent = Agent()
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    log = RunLog()
    if on_checkpoint is not None:
        log.checkpoint_metrics.append({"step": 0, **on_checkpoint(0, agent)})

    params, task_idx, mode = curriculum.pick_task()
    env = cartpole.make_env(params)
    obs, _ = env.reset(seed=int(env_rng.integers(1 << 30)))
    ep_rewards: list[float] = []
    ep_values: list[float] = []
    ep_steps = 0

    num_iterations = cfg.total_timesteps // cfg.num_steps
    for _iteration in range(num_iterations):
        b_obs = torch.zeros((cfg.num_steps, obs_dim))
        b_actions = torch.zeros(cfg.num_steps, dtype=torch.long)
        b_logprobs = torch.zeros(cfg.num_steps)
        b_rewards = torch.zeros(cfg.num_steps)
        b_dones = torch.zeros(cfg.num_steps)
        b_values = torch.zeros(cfg.num_steps)

        for t in range(cfg.num_steps):
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            b_obs[t] = obs_t
            b_actions[t] = action
            b_logprobs[t] = logprob
            b_values[t] = value.flatten()

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            ep_steps += 1
            truncated = truncated or ep_steps >= cartpole.MAX_EPISODE_STEPS
            done = terminated or truncated
            b_rewards[t] = reward
            b_dones[t] = float(done)
            ep_rewards.append(float(reward))
            ep_values.append(float(value.item()))

            obs = next_obs
            if done:
                if terminated:
                    next_value = 0.0
                else:
                    with torch.no_grad():
                        next_value = float(agent.get_value(
                            torch.as_tensor(np.asarray(next_obs), dtype=torch.float32).unsqueeze(0)
                        ))
                curriculum.report_episode(task_idx, mode, ep_rewards, ep_values, next_value)

                log.episode_returns.append(sum(ep_rewards))
                log.episode_modes.append(mode)
                log.episode_params.append(params)

                params, task_idx, mode = curriculum.pick_task()
                env = cartpole.make_env(params)
                obs, _ = env.reset(seed=int(env_rng.integers(1 << 30)))
                ep_rewards, ep_values = [], []
                ep_steps = 0

        with torch.no_grad():
            next_value = agent.get_value(
                torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
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
            shuffle_rng.shuffle(b_inds)
            for start in range(0, cfg.num_steps, minibatch_size):
                mb = b_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
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

        steps_done = (_iteration + 1) * cfg.num_steps
        if on_checkpoint is not None and checkpoint_every and (
                steps_done % checkpoint_every < cfg.num_steps or _iteration == num_iterations - 1):
            log.checkpoint_metrics.append({"step": steps_done, **on_checkpoint(steps_done, agent)})

    return log, agent
