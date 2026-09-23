import numpy as np
import pandas as pd
import pytest

from acl_bench.study.compare import benjamini_hochberg, compare_arms, derive_metrics, holm, min_detectable


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


def test_auc_and_final_metrics():
    df = pd.DataFrame({"arm": "x", "seed": 1, "step": [0, 100, 200], "E0/success": [0.0, 0.5, 1.0],
                       "E1/success": [1.0, 1.0, 1.0]})
    m = derive_metrics(df).iloc[0]
    assert m["AUC_E0"] == pytest.approx(0.5)             # linear ramp 0 -> 1
    assert m["final_E0"] == pytest.approx(0.5)           # mean of the last three checkpoints


def test_benjamini_hochberg_matches_the_textbook_example():
    adj = benjamini_hochberg({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.5})
    assert adj == {"a": pytest.approx(0.05), "b": pytest.approx(0.05), "c": pytest.approx(0.05),
                   "d": pytest.approx(0.05), "e": pytest.approx(0.5)}


def test_benjamini_hochberg_matches_scipy_reference():
    from scipy.stats import false_discovery_control
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 1, 25)
    keys = [f"k{i}" for i in range(len(pvals))]
    mine = benjamini_hochberg(dict(zip(keys, pvals)))
    ref = false_discovery_control(pvals, method="bh")
    for k, p, r in zip(keys, pvals, ref):
        assert mine[k] == pytest.approx(r), (k, p)


def test_benjamini_hochberg_is_never_stricter_than_holm():
    rng = np.random.default_rng(1)
    d = {f"k{i}": p for i, p in enumerate(rng.uniform(0, 0.2, 15))}
    bh, h = benjamini_hochberg(d), holm(d)
    for k in d:
        assert bh[k] <= h[k] + 1e-9


def test_derive_metrics_computes_every_section_and_metric_present():
    df = pd.DataFrame({
        "arm": "x", "seed": 1, "step": [0, 100, 200],
        "E0/success": [0.0, 0.5, 1.0], "E0/mean_steps": [10.0, 250.0, 500.0],
        "E6/success": [0.0, 0.0, 0.2],
    })
    m = derive_metrics(df).iloc[0]
    assert m["auc_E0"] == pytest.approx(0.5) and m["AUC_E0"] == pytest.approx(0.5)
    assert m["final_E0"] == pytest.approx(0.5)
    assert m["final_steps_E0"] == pytest.approx((10 + 250 + 500) / 3)
    assert m["auc_E6"] == pytest.approx(0.05)   # trapezoid: (0+0)/2*100 + (0+0.2)/2*100, /span 200
    assert m["final_E6"] == pytest.approx(0.2 / 3)


def test_checkpoint_grid_is_every_third_checkin_ending_at_the_final_model():
    from acl_bench.study.compare import checkpoint_grid
    steps = [20_480 * k for k in range(31)]            # step 0 plus 30 check-ins
    grid = checkpoint_grid(steps, 10)
    assert grid == [61_440 * k for k in range(1, 11)]
    assert checkpoint_grid(steps, 40) == steps[1:]     # asking for more than exist keeps them all
