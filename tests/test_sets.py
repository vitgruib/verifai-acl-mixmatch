"""The locked exam (frozen_sets/cartpole): intact, on-spec, and built by its stated rules."""
import json
import os
import re

import numpy as np
import pytest

from acl_bench.cartpole import PARAM_BOUNDS, PARAM_ORDER
from acl_bench.exam.feasibility import WINNABLE, classify
from acl_bench.exam.sets import load_sets, save_sets

DIR = "frozen_sets/cartpole"


@pytest.fixture(scope="module")
def sets():
    return load_sets(DIR)          # raises if any section no longer matches its checksum


def edge(sets):
    return [n for n in sets if re.fullmatch(r"E\d+v", n)]


def test_sections_and_counts(sets):
    assert {"E0", "E6", "E7"} <= set(sets) and edge(sets)
    assert (len(sets["E0"]), len(sets["E6"]), len(sets["E7"])) == (282, 250, 100)
    assert all(30 <= len(sets[n]) <= 100 for n in edge(sets))
    manifest = json.load(open(os.path.join(DIR, "manifest.json")))
    assert {n: m["n"] for n, m in manifest["sets"].items()} == {n: len(ps) for n, ps in sets.items()}


def test_searched_sections_follow_their_rules(sets):
    for name in ["E6", *edge(sets)]:
        assert (sets[name].ref_fail_frac >= 0.6 - 1e-9).all(), name       # 6+ of 10 fail, all-10 included
    assert (sets["E7"].ref_fail_frac == 0.0).all()                          # every agent passes


def test_hard_and_edge_sections_share_no_question(sets):
    key = lambda ps: {tuple(np.round(np.r_[p, s], 12)) for p, s in zip(ps.params, ps.s0)}
    e6 = key(sets["E6"])
    for name in edge(sets):
        assert not (e6 & key(sets[name])), name


def test_all_questions_inside_the_task_box(sets):
    for name, ps in sets.items():
        for i, k in enumerate(PARAM_ORDER):
            lo, hi = PARAM_BOUNDS[k]
            assert lo <= ps.params[:, i].min() and ps.params[:, i].max() <= hi, (name, k)


def test_a_sample_of_every_searched_section_is_proven_winnable(sets):
    for name in ["E6", "E7", *edge(sets)]:
        ps = sets[name]
        assert (classify(ps.params[:15], ps.s0[:15])["status"] == WINNABLE).all(), name


def test_checksum_detects_tampering(sets, tmp_path):
    save_sets({"E7": sets["E7"]}, str(tmp_path))
    assert load_sets(str(tmp_path))["E7"].digest() == sets["E7"].digest()
    z = dict(np.load(tmp_path / "E7.npz"))
    z["s0"] = z["s0"] + 1e-3
    np.savez(tmp_path / "E7.npz", **z)
    with pytest.raises(ValueError):
        load_sets(str(tmp_path))
