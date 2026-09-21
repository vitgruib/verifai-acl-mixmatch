import numpy as np
import pytest

from acl_bench.envs.param_cartpole import PARAM_BOUNDS
from acl_bench.oracle import infeasible_start, lqr_gain, survival_steps

DEFAULT = dict(length=0.5, masspole=0.1, masscart=1.0, force_mag=10.0, init_range=0.05)


def random_task(rng):
    return {n: rng.uniform(lo, hi) for n, (lo, hi) in PARAM_BOUNDS.items()}


def test_oracle_balances_default_cartpole():
    assert survival_steps(DEFAULT, np.array([0.02, 0.0, 0.03, 0.0])) == 500


def test_closed_loop_linearization_is_stable():
    """LQR must stabilize the linear model it was designed for, across the task box."""
    from acl_bench.oracle import G, TAU
    rng = np.random.default_rng(0)
    for _ in range(50):
        t = random_task(rng)
        K = lqr_gain(t)
        mp, mc, l = t["masspole"], t["masscart"], t["length"]
        M, pml, le = mp + mc, mp * l, l * (4 / 3 - mp / (mp + mc))
        A = np.eye(4) + TAU * np.array([[0, 1, 0, 0], [0, 0, -pml * G / (M * le), 0],
                                         [0, 0, 0, 1], [0, 0, G / le, 0]])
        B = TAU * np.array([[0.0], [1 / M + pml / (M ** 2 * le)], [0.0], [-1 / (le * M)]])
        assert np.max(np.abs(np.linalg.eigvals(A - B @ K))) < 1.0


def test_infeasible_starts_are_never_survivable():
    """A guaranteed one-step failure must end on step 1 under the oracle too."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(400):
        t = random_task(rng)
        s0 = rng.uniform(-0.3, 0.3, 4)
        if infeasible_start(s0):
            assert survival_steps(t, s0) == 1
            checked += 1
    assert checked > 10


def test_start_past_threshold_is_not_automatically_infeasible():
    """Velocity pointing back inside can rescue a pole that starts past 12 degrees."""
    s0 = np.array([0.0, 0.0, 0.2105, -0.5])   # theta0 > 0.2095 but theta1 = 0.2105 - 0.01
    assert not infeasible_start(s0)
    assert infeasible_start(np.array([0.0, 0.0, 0.2105, 0.5]))
