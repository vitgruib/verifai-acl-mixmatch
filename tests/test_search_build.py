import numpy as np
import pytest

from acl_bench import envs
from acl_bench.envs.cartpole import PARAM_BOUNDS, PARAM_ORDER
from acl_bench.exam.grader import grade
from acl_bench.exam.search import search
from acl_bench.study.arms import ARMS
from acl_bench.study.run import train_arm


CARTPOLE = envs.get("cartpole")


@pytest.fixture(scope="module")
def agents():
    return [train_arm(CARTPOLE, ARMS["N"], seed=seed, steps=100_000)[1] for seed in (1, 2, 3)]


def test_search_returns_one_question_per_start(agents):
    pool = search(CARTPOLE, "ce", agents, n_iters=5, starts_per_task=3, rng=np.random.default_rng(0))
    assert len(pool) == 15
    assert pool.params.shape == (15, 5) and pool.s0.shape == (15, 4)
    assert pool.ref_fail_frac.shape == (15,)
    assert ((pool.ref_fail_frac >= 0.0) & (pool.ref_fail_frac <= 1.0)).all()


def test_search_fail_frac_matches_a_direct_rerun(agents):
    pool = search(CARTPOLE, "mab", agents, n_iters=5, starts_per_task=3, rng=np.random.default_rng(1))
    success = np.mean([grade(CARTPOLE, a, pool.params, pool.s0)[0] for a in agents], axis=0)
    np.testing.assert_allclose(pool.ref_fail_frac, 1 - success)


def test_search_questions_stay_within_task_param_bounds(agents):
    pool = search(CARTPOLE, "sa", agents, n_iters=5, starts_per_task=2, rng=np.random.default_rng(2))
    for i, name in enumerate(PARAM_ORDER):
        lo, hi = PARAM_BOUNDS[name]
        assert (pool.params[:, i] >= lo - 1e-9).all() and (pool.params[:, i] <= hi + 1e-9).all()
