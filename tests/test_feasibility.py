import numpy as np

from acl_bench.envs import cartpole
from acl_bench.envs.cartpole import PARAM_BOUNDS, PARAM_ORDER, step_physics
from acl_bench.exam.certify import beam_search, replay_passes
from acl_bench.exam.feasibility import (IMPOSSIBLE, WINNABLE, classify, interval_doom_step, interval_step,
                                         split_interval_doom_step)
from acl_bench.exam.sets import load_sets

CENTER = np.array([(PARAM_BOUNDS[k][0] + PARAM_BOUNDS[k][1]) / 2 for k in PARAM_ORDER])
FORCE = PARAM_ORDER.index("force_mag")


def random_params(rng, n):
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_ORDER])
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_ORDER])
    return rng.uniform(lo, hi, size=(n, len(PARAM_ORDER)))


def test_interval_step_contains_every_point_successor():
    rng = np.random.default_rng(0)
    params = random_params(rng, 2000)
    center = rng.uniform([-2, -2, -0.2, -2], [2, 2, 0.2, 2], size=(2000, 4))
    half = rng.uniform(0, 0.05, size=(2000, 4))
    lo, hi = interval_step(center - half, center + half, params)
    for _ in range(20):                                   # random states in the box, random forces
        s = rng.uniform(center - half, center + half)
        f = rng.uniform(-1, 1, 2000) * params[:, FORCE]
        nxt = step_physics(s, f, params)
        assert ((nxt >= lo) & (nxt <= hi)).all()


def test_upright_start_is_winnable_with_a_replayable_certificate():
    params, s0 = CENTER[None, :], np.zeros((1, 4))
    survived, exhaustive, actions = beam_search(cartpole, params, s0, beam=16)
    assert survived[0] and exhaustive[0] == 0
    assert replay_passes(cartpole, params, s0, actions)[0]
    assert interval_doom_step(params, s0)[0] == 0         # no false proof of doom


def test_starting_past_saving_is_proven_impossible():
    params = np.tile(CENTER, (2, 1))
    s0 = np.array([[0.0, 0.0, 0.5, 1.0],                   # past the angle limit, falling further
                   [0.0, 0.0, 0.2, 3.0]])                  # inside the limit, but falling too fast to catch
    r = classify(params, s0, beam=16)
    assert list(r["status"]) == [IMPOSSIBLE, IMPOSSIBLE]


def test_split_interval_never_contradicts_a_certificate():
    rng = np.random.default_rng(1)
    params = random_params(rng, 200)
    s0 = rng.uniform(-0.05, 0.05, size=(200, 4))           # mostly easy, winnable starts
    survived, _, actions = beam_search(cartpole, params, s0, beam=32)
    certified = survived & replay_passes(cartpole, params, s0, actions)
    assert certified.mean() > 0.9
    assert (split_interval_doom_step(params[certified], s0[certified], depth=4) == 0).all()


def test_replay_rejects_a_sequence_that_fails():
    params, s0 = CENTER[None, :], np.array([[0.0, 0.0, 0.1, 0.0]])
    always_left = np.zeros((1, 499), dtype=np.int8)
    assert not replay_passes(cartpole, params, s0, always_left)[0]


def test_locked_general_questions_are_certified_winnable():
    suite = load_sets("frozen_sets/cartpole", names=["random"])["random"]     # cleaned once by an LQR expert
    r = classify(suite.params[:60], suite.s0[:60])
    assert (r["status"] == WINNABLE).all()
