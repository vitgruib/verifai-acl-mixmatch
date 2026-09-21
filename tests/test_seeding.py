"""Paired-by-seed comparisons need a seed to fix the shared randomness."""
import numpy as np

from acl_bench.envs.registry import ENV_SPECS
from acl_bench.potential.functions import resolve_potential_fn
from acl_bench.ppo import PPOConfig, run_training
from acl_bench.scenic_sampling import ScenicTaskSampler

SPEC = ENV_SPECS["cartpole"]


def train(seed, acl=True, sampler="random", steps=3072):
    cfg = PPOConfig(total_timesteps=steps, seed=seed, acl=acl)
    task_sampler = ScenicTaskSampler.load(SPEC.scenic_file, sampler)
    log, _ = run_training(SPEC, task_sampler, resolve_potential_fn("pvl_gae", SPEC.success_return), cfg)
    return log


def test_same_seed_reproduces_exactly():
    a, b = train(7), train(7)
    assert a.episode_returns == b.episode_returns
    assert [list(p.values()) for p in a.episode_params] == [list(p.values()) for p in b.episode_params]


def test_different_seeds_differ():
    assert train(1).episode_returns != train(2).episode_returns


def test_same_seed_shares_the_new_task_stream_across_arms():
    """Common random numbers: the j-th brand-new task is identical whether or not
    ACL is on, because the sampler stream is only consumed by new draws."""
    def new_tasks(log):
        return [tuple(p.values()) for p, m in zip(log.episode_params, log.episode_modes) if m == "new"]
    off, on = new_tasks(train(3, acl=False)), new_tasks(train(3, acl=True))
    k = min(len(off), len(on))
    assert k > 10
    assert off[:k] == on[:k]

