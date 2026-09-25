import math
import numpy as np
import pandas as pd
import pytest

from acl_bench.study.compare import compare_arms, derive_metrics, holm, min_detectable


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


def test_min_detectable_matches_hand_calculation():
    # z = 2.50 + 0.84 at alpha 0.0125, 80% power; sd 0.1 in both groups, 90 runs each
    assert min_detectable(0.1, 0.1, 90) == pytest.approx(3.339 * (0.02 / 90) ** 0.5, rel=1e-3)


def test_holm_matches_textbook_example():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}


def test_holm_ranks_nan_last_without_inflating_real_tests():
    # a NaN placed first used to sort unpredictably and push every later test to 1
    adj = holm({"n": float("nan"), "a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isnan(adj["n"])
    assert adj["a"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.09)
    assert adj["b"] == pytest.approx(0.09)


def test_derive_metrics_computes_final_and_curve_for_success_and_steps():
    df = pd.DataFrame({
        "arm": "x", "seed": 1, "step": [0, 100, 200],
        "random/success": [0.0, 0.5, 1.0], "random/mean_steps": [10.0, 250.0, 500.0],
        "verifai/success": [0.0, 0.0, 0.2], "verifai/mean_steps": [0.0, 100.0, 300.0],
    })
    m = derive_metrics(df).iloc[0]
    assert m["auc_random"] == pytest.approx(0.5)                     # linear ramp 0 -> 1
    assert m["final_random"] == pytest.approx(0.5)                   # mean of the last three check-ins
    assert m["final_steps_random"] == pytest.approx((10 + 250 + 500) / 3)
    assert m["auc_steps_random"] == pytest.approx((130 * 100 + 375 * 100) / 200)
    assert m["auc_verifai"] == pytest.approx(0.05)   # trapezoid: (0+0)/2*100 + (0+0.2)/2*100, /span 200
    assert m["final_verifai"] == pytest.approx(0.2 / 3)
    assert m["auc_steps_verifai"] == pytest.approx((50 * 100 + 200 * 100) / 200)


def test_checkpoint_grid_is_every_third_checkin_ending_at_the_final_model():
    from acl_bench.study.compare import checkpoint_grid
    steps = [20_480 * k for k in range(31)]            # step 0 plus 30 check-ins
    grid = checkpoint_grid(steps, 10)
    assert grid == [61_440 * k for k in range(1, 11)]
    assert checkpoint_grid(steps, 40) == steps[1:]     # asking for more than exist keeps them all
