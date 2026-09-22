import numpy as np
import pytest

from acl_bench.envs.registry import ENV_SPECS
from acl_bench.potential.functions import resolve_potential_fn
from acl_bench.ppo import PPOConfig, run_training
from acl_bench.scenic_sampling import ScenicTaskSampler
from acl_bench.suite.build_edge_sections import describe_cluster, normalize
from acl_bench.suite.evaluator import PARAM_ORDER, rollout_steps
from acl_bench.suite.find_edge_cases import search

SPEC = ENV_SPECS["cartpole"]


@pytest.fixture(scope="module")
def agents():
    trained = []
    for seed in (1, 2, 3):
        _, a = run_training(SPEC, ScenicTaskSampler.load(SPEC.scenic_file, "random"),
                            resolve_potential_fn("neg_return", SPEC.success_return),
                            PPOConfig(total_timesteps=100_000, seed=seed, acl=False))
        trained.append(a)
    return trained


def test_search_returns_one_question_per_start(agents):
    pool = search("ce", agents, n_iters=5, starts_per_task=3, rng=np.random.default_rng(0))
    assert len(pool) == 15
    assert pool.params.shape == (15, 5) and pool.s0.shape == (15, 4)
    assert pool.ref_fail_frac.shape == (15,)
    assert ((pool.ref_fail_frac >= 0.0) & (pool.ref_fail_frac <= 1.0)).all()


def test_search_fail_frac_matches_a_direct_rerun(agents):
    pool = search("mab", agents, n_iters=5, starts_per_task=3, rng=np.random.default_rng(1))
    success = np.mean([rollout_steps(a, pool.params, pool.s0) == 500 for a in agents], axis=0)
    np.testing.assert_allclose(pool.ref_fail_frac, 1 - success)


def test_search_questions_stay_within_task_param_bounds(agents):
    from acl_bench.envs.param_cartpole import PARAM_BOUNDS
    pool = search("sa", agents, n_iters=5, starts_per_task=2, rng=np.random.default_rng(2))
    for i, name in enumerate(PARAM_ORDER):
        lo, hi = PARAM_BOUNDS[name]
        assert (pool.params[:, i] >= lo - 1e-9).all() and (pool.params[:, i] <= hi + 1e-9).all()


def test_describe_cluster_names_the_extreme_parameter():
    from acl_bench.envs.param_cartpole import PARAM_BOUNDS
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_ORDER])
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_ORDER])
    # force_mag near its lower bound, everything else at the box center
    params = np.tile((lo + hi) / 2, (10, 1))
    params[:, PARAM_ORDER.index("force_mag")] = lo[PARAM_ORDER.index("force_mag")]
    desc = describe_cluster(params)
    assert "force_mag=very low" in desc


def test_normalize_maps_bounds_to_unit_interval():
    from acl_bench.envs.param_cartpole import PARAM_BOUNDS
    lo = np.array([[PARAM_BOUNDS[k][0] for k in PARAM_ORDER]])
    hi = np.array([[PARAM_BOUNDS[k][1] for k in PARAM_ORDER]])
    np.testing.assert_allclose(normalize(lo), 0.0)
    np.testing.assert_allclose(normalize(hi), 1.0)


def test_lost_on_first_step_flags_only_starts_past_the_angle_limit():
    from acl_bench.envs.param_cartpole import PARAM_BOUNDS
    from acl_bench.suite.build_edge_sections import lost_on_first_step
    center = np.array([[(PARAM_BOUNDS[k][0] + PARAM_BOUNDS[k][1]) / 2 for k in PARAM_ORDER]] * 2)
    s0 = np.array([[0.0, 0.0, 0.0, 0.0],      # upright: winnable
                   [0.0, 0.0, 0.5, 1.0]])     # 0.5 rad and falling further: gone whatever the push
    np.testing.assert_array_equal(lost_on_first_step(center, s0), [False, True])
