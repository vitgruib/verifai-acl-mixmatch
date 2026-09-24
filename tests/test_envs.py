"""Every environment honours the shared contract (acl_bench/envs/__init__.py)."""
import re

import numpy as np
import pytest

from acl_bench import envs
from acl_bench.exam.build import random_suite
from acl_bench.exam.certify import beam_search, certify, horizon, replay_passes
from acl_bench.sampling import SAMPLER_NAMES, ScenicTaskSampler


@pytest.mark.parametrize("name", envs.NAMES)
def test_scenic_file_declares_the_same_task_box(name):
    env = envs.get(name)
    declared = {k: (float(lo), float(hi)) for k, lo, hi in
                re.findall(r"param (\w+) = VerifaiRange\(([-\d.e]+), ([-\d.e]+)\)", open(env.SCENIC_FILE).read())}
    assert declared == {k: tuple(map(float, v)) for k, v in env.PARAM_BOUNDS.items()}
    assert tuple(declared) == env.PARAM_ORDER


@pytest.mark.parametrize("name", envs.NAMES)
@pytest.mark.parametrize("sampler", SAMPLER_NAMES)
def test_samplers_draw_inside_the_box(name, sampler):
    env = envs.get(name)
    s = ScenicTaskSampler.load(sampler, env.SCENIC_FILE)
    for _ in range(10):
        task = s.draw(env.PARAM_ORDER)
        s.give_feedback(0.5)
        for k, (lo, hi) in env.PARAM_BOUNDS.items():
            assert lo - 1e-9 <= task[k] <= hi + 1e-9


@pytest.mark.parametrize("name", ["acrobot", "mountaincar"])
def test_reach_certificates_replay_and_reach_on_time(name):
    env = envs.get(name)
    center = np.array([(lo + hi) / 2 for lo, hi in env.PARAM_BOUNDS.values()])
    params = np.tile(center, (8, 1))
    s0 = env.sample_starts(center, 8, np.random.default_rng(0))
    won, _, actions = beam_search(env, params, s0, beam=32)
    assert won.all() and replay_passes(env, params, s0, actions).all()
    assert actions.shape[1] == horizon(env) == env.MAX_EPISODE_STEPS
    cut = actions.copy()
    cut[:, 5:] = -1                                          # a truncated plan must not pass
    assert not replay_passes(env, params, s0, cut).any()


def test_mountaincar_too_weak_an_engine_is_not_certified():
    env = envs.get("mountaincar")
    params = np.array([[0.0005, 0.005, 0.05]])              # weakest engine, strongest gravity
    assert not certify(env, params, np.array([[-0.5, 0.0]]))[0]


@pytest.mark.parametrize("name", ["acrobot", "mountaincar"])
def test_random_suite_keeps_only_certified_questions(name):
    env = envs.get(name)
    suite, info = random_suite(env, 20, seed=1)
    assert len(suite) + info["n_not_certified"] == 20
    assert certify(env, suite.params, suite.s0).all()
