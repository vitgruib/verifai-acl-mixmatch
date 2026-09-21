import numpy as np
import pytest

from acl_bench.envs.registry import ENV_SPECS
from acl_bench.scenic_sampling import DEFAULT_SAMPLER_PARAMS, ScenicTaskSampler, resolve_sampler_params

SPEC = ENV_SPECS["cartpole"]
NAMES = tuple(SPEC.param_bounds)


def test_defaults_and_validation():
    assert resolve_sampler_params("ce", {"buckets": 4})["buckets"] == 4
    assert resolve_sampler_params("ce", None) == DEFAULT_SAMPLER_PARAMS["ce"]
    with pytest.raises(ValueError):
        resolve_sampler_params("ce", {"bukets": 4})        # typo must not be silently ignored
    with pytest.raises(ValueError):
        resolve_sampler_params("random", {"alpha": 0.5})   # nothing to tune


def test_overrides_reach_the_verifai_sampler():
    ce = ScenicTaskSampler.load(SPEC.scenic_file, "ce", {"buckets": 4, "alpha": 0.5, "thres": -0.3})
    inner = ce.scenario.externalSampler.sampler.domainSampler.cont_sampler
    assert (inner.buckets == 4).all() and inner.alpha == 0.5 and inner.thres == -0.3
    mab = ScenicTaskSampler.load(SPEC.scenic_file, "mab", {"buckets": 16, "thres": 0.3})
    inner = mab.scenario.externalSampler.sampler.domainSampler.cont_sampler
    assert (inner.buckets == 16).all() and inner.thres == 0.3
    sa = ScenicTaskSampler.load(SPEC.scenic_file, "sa", {"T": 3.0, "decay_rate": 0.95, "iterations": 40})
    s = sa.scenario.externalSampler.sampler.domainSampler.samplers[0]
    assert (s.T, s.decay_rate, s.iterations) == (3.0, 0.95, 40)


def test_ce_thres_changes_which_feedback_counts_as_a_counterexample():
    """thres decides whether a given rho moves the CE distribution at all."""
    def moved(thres):
        t = ScenicTaskSampler.load(SPEC.scenic_file, "ce", {"thres": thres})
        cont = t.scenario.externalSampler.sampler.domainSampler.cont_sampler
        before = [d.copy() for d in cont.dist]
        t.draw(NAMES); t.give_feedback(0.1); t.draw(NAMES)
        return any(not np.allclose(b, a) for b, a in zip(before, cont.dist))
    assert moved(0.3)        # rho=0.1 < 0.3 -> counterexample -> distribution updated
    assert not moved(-0.3)   # rho=0.1 >= -0.3 -> ignored
