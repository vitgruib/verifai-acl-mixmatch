"""Minimal CleanRL-style PPO, running on any environment in
acl_bench.envs.registry (discrete or continuous actions), driven by a
PLRCurriculum. Independent of SIPACL: its Work/policy/ppo.py is deliberately not
carried over (only the ACL-carrying gym code in Work/custom/ is).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from acl_bench.curriculum.plr import PLRCurriculum
from acl_bench.envs.registry import EnvSpec
from acl_bench.potential.functions import resolve_potential_fn


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
    """`action_type="discrete"` -> Categorical over `action_dim` logits.
    `action_type="continuous"` -> Normal, tanh-squashed to [-1, 1] (with the
    standard change-of-variables log-prob correction), rescaled to the env's
    actual action bounds by the caller. Currently unexercised: no continuous-action
    environment remains in the registry (Pendulum was dropped)."""

    def __init__(self, obs_dim: int, action_dim: int, action_type: str):
        super().__init__()
        self.action_type = action_type
        self.critic = _mlp(obs_dim, 1, out_std=1.0)
        if action_type == "discrete":
            self.actor = _mlp(obs_dim, action_dim, out_std=0.01)
        else:
            self.actor_mean = _mlp(obs_dim, action_dim, out_std=0.01)
            self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -1.0))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, stored=None):
        """Returns (env_action, stored, logprob, entropy, value). `stored` is
        what must be replayed verbatim to recompute logprob during a PPO
        update: the discrete action itself, or the continuous pre-tanh
        sample (so `torch.tanh` stays consistent between rollout and update).
        """
        if self.action_type == "discrete":
            dist = Categorical(logits=self.actor(x))
            action = dist.sample() if stored is None else stored
            return action, action, dist.log_prob(action), dist.entropy(), self.critic(x)

        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        pretanh = dist.rsample() if stored is None else stored
        action = torch.tanh(pretanh)
        logprob = (dist.log_prob(pretanh) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, pretanh, logprob, entropy, self.critic(x)


def scale_to_action_space(unit_action: np.ndarray, env) -> np.ndarray:
    """[-1, 1]^d -> the env's actual action bounds."""
    low, high = env.action_space.low, env.action_space.high
    return low + (unit_action + 1.0) * 0.5 * (high - low)


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
    feedback_fn: str | None = None  # None: the scoring function also feeds the sampler; a name decouples them
    seed: int = 1


@dataclass
class RunLog:
    episode_returns: list = field(default_factory=list)
    episode_modes: list = field(default_factory=list)     # "new" / "replay"
    episode_task_ids: list = field(default_factory=list)
    episode_params: list = field(default_factory=list)    # dict per episode
    lp_scores: list = field(default_factory=list)
    # (env steps so far, held-out eval return, mean of last 20 training episodes)
    checkpoints: list = field(default_factory=list)


def run_training(env_spec: EnvSpec, task_sampler, potential_fn, cfg: PPOConfig,
                  device="cpu", eval_set: list[dict] | None = None,
                  eval_every_steps: int | None = None) -> tuple[RunLog, Agent]:
    """If `eval_set` and `eval_every_steps` are given, the policy is evaluated
    on the held-out tasks every `eval_every_steps` env steps (rounded up to a
    whole PPO iteration), producing a learning curve in `log.checkpoints`."""
    # One independent stream per component. A shared seed then means shared
    # randomness *per component* even when two arms consume the streams at
    # different rates -- which is what makes paired-by-seed comparisons work.
    curriculum_ss, env_ss, shuffle_ss, sampler_ss = np.random.SeedSequence(cfg.seed).spawn(4)
    rng = np.random.default_rng(curriculum_ss)          # replay choices
    env_rng = np.random.default_rng(env_ss)             # per-episode initial-state seeds
    shuffle_rng = np.random.default_rng(shuffle_ss)     # PPO minibatch order
    torch.manual_seed(cfg.seed)                          # network init, then action sampling
    sampler_state = int(sampler_ss.generate_state(1)[0])
    random.seed(sampler_state)                           # Scenic/VerifAI samplers draw from
    np.random.seed(sampler_state)                        # these globals; nothing else here does

    param_names = tuple(env_spec.param_bounds.keys())
    curriculum = PLRCurriculum(
        task_sampler=task_sampler, param_names=param_names, potential_fn=potential_fn,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, use_replay=cfg.acl,
        replay_prob=cfg.replay_prob, buffer_max=cfg.buffer_max, rank_alpha=cfg.rank_alpha,
        ema_beta=cfg.ema_beta,
        feedback_fn=(resolve_potential_fn(cfg.feedback_fn, env_spec.success_return)
                     if cfg.feedback_fn else None),
        rng=rng,
    )

    obs_dim, action_dim = env_spec.obs_dim, env_spec.action_dim
    agent = Agent(obs_dim, action_dim, env_spec.action_type).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    is_discrete = env_spec.action_type == "discrete"

    log = RunLog()

    params, task_idx, mode = curriculum.pick_task()
    env = env_spec.make_env(params)
    obs, _ = env.reset(seed=int(env_rng.integers(1 << 30)))
    ep_rewards: list[float] = []
    ep_values: list[float] = []
    ep_steps = 0

    num_iterations = cfg.total_timesteps // cfg.num_steps
    for _iteration in range(num_iterations):
        b_obs = torch.zeros((cfg.num_steps, obs_dim))
        b_actions = (torch.zeros(cfg.num_steps, dtype=torch.long) if is_discrete
                     else torch.zeros((cfg.num_steps, action_dim)))
        b_logprobs = torch.zeros(cfg.num_steps)
        b_rewards = torch.zeros(cfg.num_steps)
        b_dones = torch.zeros(cfg.num_steps)
        b_values = torch.zeros(cfg.num_steps)

        for t in range(cfg.num_steps):
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, stored, logprob, _, value = agent.get_action_and_value(obs_t)
            b_obs[t] = obs_t
            b_actions[t] = stored
            b_logprobs[t] = logprob
            b_values[t] = value.flatten()

            if is_discrete:
                env_action = int(action.item())
            else:
                env_action = scale_to_action_space(action.squeeze(0).numpy(), env)

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            ep_steps += 1
            truncated = truncated or ep_steps >= env_spec.max_episode_steps
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
                raw_lp = curriculum.report_episode(task_idx, mode, ep_rewards, ep_values, next_value)

                log.episode_returns.append(sum(ep_rewards))
                log.episode_modes.append(mode)
                log.episode_task_ids.append(task_idx)
                log.episode_params.append(params)
                log.lp_scores.append(raw_lp)

                params, task_idx, mode = curriculum.pick_task()
                env = env_spec.make_env(params)
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
                _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], stored=b_actions[mb]
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

        steps_done = (_iteration + 1) * cfg.num_steps
        if eval_set is not None and eval_every_steps and (
                steps_done % eval_every_steps < cfg.num_steps or _iteration == num_iterations - 1):
            recent = log.episode_returns[-20:]
            log.checkpoints.append((
                steps_done,
                evaluate_agent(agent, env_spec, eval_set, seed=cfg.seed),
                float(np.mean(recent)) if recent else float("nan"),
            ))

    return log, agent


@torch.no_grad()
def evaluate_agent(agent: Agent, env_spec: EnvSpec, eval_params: list[dict],
                    episodes_per_task: int = 3, seed: int = 0) -> float:
    """Mean return over a fixed, sampler-independent set of task params --
    the generalization metric used to compare mix-and-match combinations."""
    rng = np.random.default_rng(seed)
    torch_state = torch.get_rng_state()   # evaluation must not shift the training action stream
    is_discrete = env_spec.action_type == "discrete"
    returns = []
    for params in eval_params:
        env = env_spec.make_env(params)
        for _ in range(episodes_per_task):
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
            total = 0.0
            for _ in range(env_spec.max_episode_steps):
                obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
                action, _, _, _, _ = agent.get_action_and_value(obs_t)
                env_action = int(action.item()) if is_discrete else \
                    scale_to_action_space(action.squeeze(0).numpy(), env)
                obs, reward, terminated, truncated, _ = env.step(env_action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
    torch.set_rng_state(torch_state)
    return float(np.mean(returns))
