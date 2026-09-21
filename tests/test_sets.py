"""The locked exam (frozen_sets/cartpole_v2): intact, on-spec, and free of impossible questions."""
import json
import os

import numpy as np
import pytest

from acl_bench.envs.param_cartpole import PARAM_BOUNDS
from acl_bench.suite.evaluator import PARAM_ORDER, TAU, THETA_THRESHOLD, X_THRESHOLD
from acl_bench.suite.sets import load_sets, save_sets

DIR = "frozen_sets/cartpole_v2"
IDX = {k: i for i, k in enumerate(PARAM_ORDER)}
EXPECTED = {"E0": 282, "E1": 138, "E1b": 93, "E2": 95, "E3a": 96, "E3b": 92, "E4": 99, "E5": 272}
STATIC = tuple(EXPECTED)
POOL_DERIVED = ("E6", "E7")
POOL_DIR = "frozen_sets/cartpole_v2_pool"


@pytest.fixture(scope="module")
def sets():
    return load_sets(DIR)          # raises if any section no longer matches its checksum


def test_sections_and_counts_match_the_manifest(sets):
    assert {k: len(sets[k]) for k in STATIC} == EXPECTED
    assert set(sets) == set(STATIC) | set(POOL_DERIVED)      # POOL is graded separately, from snapshots
    manifest = json.load(open(os.path.join(DIR, "manifest.json")))
    for name, meta in manifest["sets"].items():
        assert meta["n"] == len(sets[name])
    for name in STATIC:
        meta = manifest["sets"][name]
        assert meta["n"] == meta["n_before_filter"] - meta["n_removed_impossible"] - meta["n_removed_unclear"]


def test_no_question_is_impossible_from_the_first_step(sets):
    """An impossible question ends on step 1 whatever the agent does: after one Euler step
    the cart or pole is already out of bounds. That depends only on the start, not on the action."""
    for name in (*STATIC, "E6", "E7"):          # the POOL sample deliberately contains impossible questions
        ps = sets[name]
        x1 = ps.s0[:, 0] + TAU * ps.s0[:, 1]
        theta1 = ps.s0[:, 2] + TAU * ps.s0[:, 3]
        assert not ((np.abs(x1) > X_THRESHOLD) | (np.abs(theta1) > THETA_THRESHOLD)).any(), name


def test_sections_stay_inside_their_declared_ranges(sets):
    e = lambda name, k: sets[name].params[:, IDX[k]]
    assert e("E1", "force_mag").max() <= 7.0
    assert e("E1b", "force_mag").max() <= 7.0 and e("E1b", "masscart").min() >= 1.625
    assert e("E2", "masscart").min() >= 1.625
    assert e("E3a", "length").min() >= 1.1875 and e("E3b", "masspole").min() >= 0.3875
    assert e("E4", "init_range").min() >= 0.2375
    for name in STATIC:
        ps = sets[name]
        for k, (lo, hi) in PARAM_BOUNDS.items():
            assert lo <= ps.params[:, IDX[k]].min() and ps.params[:, IDX[k]].max() <= hi, (name, k)


def test_corners_section_covers_all_32_corners(sets):
    corners = {tuple(p) for p in sets["E5"].params}
    assert len(corners) == 32


def test_checksum_detects_tampering(sets, tmp_path):
    save_sets({"E1": sets["E1"]}, str(tmp_path))
    assert load_sets(str(tmp_path))["E1"].digest() == sets["E1"].digest()
    z = dict(np.load(tmp_path / "E1.npz"))
    z["s0"] = z["s0"] + 1e-3
    np.savez(tmp_path / "E1.npz", **z)
    with pytest.raises(ValueError):
        load_sets(str(tmp_path))


def test_hard_easy_and_pool_sections_are_consistent(sets):
    manifest = json.load(open(os.path.join(DIR, "manifest.json")))
    ref = manifest["reference"]
    assert ref["n_agents"] >= 5 and len(ref["picks"]) == ref["n_agents"]
    e6, e7 = sets["E6"], sets["E7"]
    assert len(e6) >= 150 and len(e7) >= 50, "sections too small to be useful"
    assert ((e6.ref_fail_frac >= 0.5) & (e6.ref_fail_frac < 1.0)).all()     # nobody-passes questions excluded
    assert (e7.ref_fail_frac == 0.0).all()
    pool = load_sets(POOL_DIR)["POOL"]                                       # separate folder, checksum verified
    assert len(pool) == ref["stats"]["report_size"] and pool.ref_fail_frac is not None
    assert json.load(open(os.path.join(POOL_DIR, "manifest.json")))["reference"] == ref
