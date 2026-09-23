"""A seed must fully determine a run, and each arm must get independent seeds."""
import numpy as np

from acl_bench.ppo import PPOConfig, run_training
from acl_bench.sampling import ScenicTaskSampler
from acl_bench.scoring import pvl_gae


def train(seed, acl=True, sampler="random", steps=3072):
    cfg = PPOConfig(total_timesteps=steps, seed=seed, acl=acl)
    log, _ = run_training(ScenicTaskSampler.load(sampler), pvl_gae, cfg)
    return log


def test_same_seed_reproduces_exactly():
    a, b = train(7), train(7)
    assert a.episode_returns == b.episode_returns
    assert [list(p.values()) for p in a.episode_params] == [list(p.values()) for p in b.episode_params]


def test_different_seeds_differ():
    assert train(1).episode_returns != train(2).episode_returns


def test_arm_seeds_are_independent_stable_and_distinct():
    from acl_bench.study.run import arm_seed
    assert arm_seed("N", 1) == arm_seed("N", 1)                       # stable across calls
    assert arm_seed("N", 1) != arm_seed("A", 1)                       # same replicate, different arm
    seeds = {arm_seed(a, r) for a in ("N", "A", "B_sa", "S_ce") for r in range(1, 201)}
    assert len(seeds) == 4 * 200                                     # no collisions
    assert all(0 <= x < 2 ** 32 for x in seeds)
