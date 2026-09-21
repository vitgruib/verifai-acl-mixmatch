import numpy as np
import pandas as pd
import pytest

from acl_bench.suite.compare import (compare_arms, derive_metrics, holm, min_detectable,
                                     seeds_needed)


def two_groups(mean_a, mean_b, sd, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"arm": ["a"] * n + ["b"] * n, "seed": list(range(n)) * 2,
                         "m": np.concatenate([rng.normal(mean_a, sd, n), rng.normal(mean_b, sd, n)])})


def test_compare_recovers_a_known_difference():
    out = compare_arms(two_groups(0.55, 0.50, 0.1), "a", "b", "m")
    assert out["mean_diff"] == pytest.approx(0.05, abs=0.02)
    assert out["ci_lo"] < 0.05 < out["ci_hi"] and out["p"] < 0.01


def test_no_difference_is_not_flagged():
    out = compare_arms(two_groups(0.5, 0.5, 0.1, seed=3), "a", "b", "m")
    assert out["ci_lo"] < 0 < out["ci_hi"]


def test_groups_are_independent_not_matched_by_seed():
    """Shuffling which row of `b` sits next to which row of `a` must not change anything."""
    df = two_groups(0.6, 0.5, 0.1, n=200)
    shuffled = df.copy()
    b = shuffled["arm"] == "b"
    shuffled.loc[b, "m"] = np.random.default_rng(9).permutation(shuffled.loc[b, "m"].to_numpy())
    assert compare_arms(df, "a", "b", "m")["mean_diff"] == pytest.approx(compare_arms(shuffled, "a", "b", "m")["mean_diff"])
    assert compare_arms(df, "a", "b", "m")["p"] == pytest.approx(compare_arms(shuffled, "a", "b", "m")["p"])


def test_seeds_needed_matches_hand_calculation_and_inverts_min_detectable():
    # (2.50 + 0.84)^2 = 11.16; sd 0.1 in both groups, delta 0.05 -> 11.16 * 0.02 / 0.0025 = 89.3 -> 90
    assert seeds_needed(0.1, 0.1, 0.05) == 90
    assert min_detectable(0.1, 0.1, 90) <= 0.05 < min_detectable(0.1, 0.1, 80)


def test_holm_matches_textbook_example():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}


def test_auc_and_final_metrics():
    df = pd.DataFrame({"arm": "x", "seed": 1, "step": [0, 100, 200], "E0/success": [0.0, 0.5, 1.0],
                       "E1/success": [1.0, 1.0, 1.0]})
    m = derive_metrics(df).iloc[0]
    assert m["AUC_E0"] == pytest.approx(0.5)             # linear ramp 0 -> 1
    assert m["final_E0"] == pytest.approx(0.5)           # mean of the last three checkpoints
    assert m["S_edge"] == pytest.approx(1.0)
