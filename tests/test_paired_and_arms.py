import dataclasses

import numpy as np
import pandas as pd
import pytest

from acl_bench.ppo import PPOConfig
from acl_bench.scenic_sampling import resolve_sampler_params
from acl_bench.suite.ablation import ARMS, family, paired_comparisons
from acl_bench.suite.paired import derive_metrics, holm, seeds_needed, summarize_pair


def synthetic(rho, n=4000, seed=0, sd=0.1):
    rng = np.random.default_rng(seed)
    cov = sd ** 2 * np.array([[1, rho], [rho, 1]])
    x = rng.multivariate_normal([0.5, 0.6], cov, size=n)
    return pd.DataFrame({"arm": ["a"] * n + ["b"] * n, "seed": list(range(n)) * 2,
                         "m": np.concatenate([x[:, 0], x[:, 1]])})


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_summary_recovers_correlation_and_variance_removed(rho):
    out = summarize_pair(synthetic(rho), "a", "b", "m")
    assert out["rho"] == pytest.approx(rho, abs=0.05)
    assert out["variance_removed"] == pytest.approx(rho, abs=0.05)      # equal variances -> removed share = rho
    assert out["mean_diff"] == pytest.approx(-0.1, abs=0.01)


def test_pairing_shrinks_the_seed_requirement_only_when_correlated():
    hi, lo = summarize_pair(synthetic(0.9), "a", "b", "m"), summarize_pair(synthetic(0.0), "a", "b", "m")
    assert seeds_needed(hi["sd_diff_paired"], 0.05) < seeds_needed(lo["sd_diff_paired"], 0.05) / 4


def test_seeds_needed_matches_hand_calculation():
    # (z_a + z_b)^2 with alpha=0.0125, power 0.8 is 3.34^2 = 11.16; sd/delta = 2 -> 44.6 -> 45
    assert seeds_needed(0.10, 0.05) == 45


def test_holm_matches_textbook_example():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}


def test_auc_and_final_metrics():
    steps = [0, 100, 200]
    df = pd.DataFrame({"arm": "x", "seed": 1, "step": steps, "E0/success": [0.0, 0.5, 1.0],
                       "E1/success": [1.0, 1.0, 1.0]})
    m = derive_metrics(df).iloc[0]
    assert m["AUC_E0"] == pytest.approx(0.5)             # linear ramp 0 -> 1
    assert m["final_E0"] == pytest.approx(0.5)           # mean of the last three checkpoints
    assert m["S_edge"] == pytest.approx(1.0)


def test_every_arm_override_is_valid():
    fields = {f.name for f in dataclasses.fields(PPOConfig)}
    for arm in ARMS.values():
        assert set(arm.ppo) <= fields, arm.name
        resolve_sampler_params(arm.sampler, arm.sampler_params)      # raises on unknown keys
        assert arm.reference is None or arm.reference in ARMS


def test_registry_shape_and_variants_actually_differ_from_their_reference():
    assert {a.family for a in ARMS.values()} == {"primary", "acl", "sampler", "coupling"}
    assert len(family("acl")) == 8 + 2 + 2
    for arm in ARMS.values():
        if arm.reference:
            ref = ARMS[arm.reference]
            assert (arm.ppo, arm.sampler_params, arm.sampler, arm.acl) != (ref.ppo, ref.sampler_params, ref.sampler, ref.acl), arm.name
    assert len(paired_comparisons()) == len(set(paired_comparisons()))
