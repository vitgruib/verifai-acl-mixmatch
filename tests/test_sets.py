"""The locked exam (frozen_sets/cartpole): intact, on-spec, and built by its stated rules."""
import json
import os

import numpy as np
import pytest

from acl_bench.cartpole import PARAM_BOUNDS, PARAM_ORDER
from acl_bench.exam.feasibility import WINNABLE, classify
from acl_bench.exam.sets import load_sets, save_sets

DIR = "frozen_sets/cartpole"


@pytest.fixture(scope="module")
def sets():
    return load_sets(DIR)          # raises if any suite no longer matches its checksum


def test_two_suites_and_counts_match_the_manifest(sets):
    assert set(sets) == {"random", "verifai"}
    assert len(sets["random"]) == 282 and len(sets["verifai"]) > 1000
    manifest = json.load(open(os.path.join(DIR, "manifest.json")))
    assert {n: m["n"] for n, m in manifest["sets"].items()} == {n: len(ps) for n, ps in sets.items()}


def test_verifai_questions_are_ones_6_or_more_of_10_reference_agents_fail(sets):
    assert (sets["verifai"].ref_fail_frac >= 0.6 - 1e-9).all()


def test_all_questions_inside_the_task_box(sets):
    for name, ps in sets.items():
        for i, k in enumerate(PARAM_ORDER):
            lo, hi = PARAM_BOUNDS[k]
            assert lo <= ps.params[:, i].min() and ps.params[:, i].max() <= hi, (name, k)


def test_a_sample_of_the_verifai_suite_is_proven_winnable(sets):
    ps = sets["verifai"]
    idx = np.random.default_rng(0).choice(len(ps), 30, replace=False)
    assert (classify(ps.params[idx], ps.s0[idx])["status"] == WINNABLE).all()


def test_checksum_detects_tampering(sets, tmp_path):
    save_sets({"random": sets["random"]}, str(tmp_path))
    assert load_sets(str(tmp_path))["random"].digest() == sets["random"].digest()
    z = dict(np.load(tmp_path / "random.npz"))
    z["s0"] = z["s0"] + 1e-3
    np.savez(tmp_path / "random.npz", **z)
    with pytest.raises(ValueError):
        load_sets(str(tmp_path))
