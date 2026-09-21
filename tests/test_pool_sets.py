import numpy as np
import pytest

from acl_bench.envs.registry import ENV_SPECS
from acl_bench.potential.functions import resolve_potential_fn
from acl_bench.ppo import Agent, PPOConfig, run_training
from acl_bench.scenic_sampling import ScenicTaskSampler
from acl_bench.suite.sets import FAIL_BINS, build_pool_sets, build_static_sets, evaluate_sets

SPEC = ENV_SPECS["cartpole"]


@pytest.fixture(scope="module")
def agents():
    trained = []
    for seed in (1, 2, 3):
        _, a = run_training(SPEC, ScenicTaskSampler.load(SPEC.scenic_file, "random"),
                            resolve_potential_fn("neg_return", SPEC.success_return),
                            PPOConfig(total_timesteps=100_000, seed=seed, acl=False))
        trained.append(a)
    import torch
    torch.manual_seed(0)
    return trained, Agent(4, 2, "discrete")            # references are trained; the untrained one is only a foil


@pytest.fixture(scope="module")
def pool(agents):
    return build_pool_sets(agents[0], seed=1, pool_size=500, hard_frac=0.5, hard_target=40, easy_target=15)


def test_hard_and_easy_sets_are_defined_by_reference_failure(pool):
    sets, stats = pool
    e6, e7 = sets["E6"], sets["E7"]
    assert 0 < len(e6) <= 40 and 0 < len(e7) <= 15, "vacuous if a set is empty"
    assert e6.learnable.all() and e7.learnable.all()
    assert (e6.ref_fail_frac >= 0.5).all()
    assert (e7.ref_fail_frac == 0.0).all()
    assert stats["hard_available"] >= len(e6) and stats["learnable_in_pool"] > 250


def test_sets_are_subsets_of_the_pool(pool):
    sets, _ = pool
    pool_rows = {tuple(np.concatenate([p, s])) for p, s in zip(sets["POOL"].params, sets["POOL"].s0)}
    for name in ("E6", "E7"):
        assert all(tuple(np.concatenate([p, s])) in pool_rows for p, s in zip(sets[name].params, sets[name].s0))


def test_reference_failure_fraction_is_consistent_with_a_direct_rerun(agents, pool):
    from acl_bench.suite.evaluator import rollout_steps
    ps = pool[0]["POOL"]
    success = np.mean([rollout_steps(a, ps.params, ps.s0) == 500 for a in agents[0]], axis=0)
    np.testing.assert_allclose(ps.ref_fail_frac, 1 - success)


def test_evaluate_sets_keys_and_bins(agents, pool):
    sets = {**build_static_sets(seed=9), **pool[0]}
    m = evaluate_sets(agents[0][0], sets)
    for name, ps in sets.items():
        assert m[f"{name}/n_learnable"] == int(ps.learnable.sum())
        assert len(ps) > 0 and 0.0 <= m[f"{name}/success"] <= 1.0
    counts = sum(v for k, v in m.items() if k.startswith("POOL/bin_") and k.endswith("/n"))
    assert counts == m["POOL/n_learnable"]                       # every learnable pool pair falls in exactly one bin
    assert len([k for k in m if k.startswith("POOL/bin_") and k.endswith("/success")]) == len(FAIL_BINS) - 1


def test_a_trained_agent_beats_an_untrained_one_on_the_uniform_set(agents):
    sets = build_static_sets(seed=9)
    assert evaluate_sets(agents[0][0], sets)["E0/success"] > evaluate_sets(agents[1], sets)["E0/success"]


def test_empty_set_is_reported_not_crashed(agents):
    from acl_bench.suite.sets import PairSet
    empty = PairSet("EMPTY", np.zeros((0, 5)), np.zeros((0, 4)), np.zeros(0, bool), np.zeros(0, int))
    m = evaluate_sets(agents[0][0], {"EMPTY": empty})
    assert m["EMPTY/n_learnable"] == 0 and np.isnan(m["EMPTY/success"])
