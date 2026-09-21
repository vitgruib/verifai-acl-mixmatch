import json
import os

import numpy as np
import pytest

from acl_bench.envs.param_cartpole import PARAM_BOUNDS
from acl_bench.suite.evaluator import PARAM_ORDER
from acl_bench.suite.sets import build_static_sets, load_sets, save_sets

IDX = {k: i for i, k in enumerate(PARAM_ORDER)}


@pytest.fixture(scope="module")
def sets():
    return build_static_sets(seed=123)


def test_generation_is_reproducible():
    a, b = build_static_sets(seed=5), build_static_sets(seed=5)
    assert all(a[k].digest() == b[k].digest() for k in a)
    assert build_static_sets(seed=6)["E0"].digest() != a["E0"].digest()


def test_ranges_are_respected(sets):
    e1b = sets["E1b"].params
    assert e1b[:, IDX["force_mag"]].max() <= 7.0 and e1b[:, IDX["masscart"]].min() >= 1.625
    assert sets["E2"].params[:, IDX["masscart"]].min() >= 1.625
    assert sets["E3a"].params[:, IDX["length"]].min() >= 1.1875
    assert sets["E3b"].params[:, IDX["masspole"]].min() >= 0.3875
    assert sets["E4"].params[:, IDX["init_range"]].min() >= 0.2375
    for name, ps in sets.items():
        for k, (lo, hi) in PARAM_BOUNDS.items():
            col = ps.params[:, IDX[k]]
            assert col.min() >= lo and col.max() <= hi


def test_e4_has_only_feasible_starts_and_e5_covers_all_corners(sets):
    assert not sets["E4"].infeasible.any()
    corners = {tuple(p) for p in sets["E5"].params}
    assert len(corners) == 32 and len(sets["E5"]) == 320


def test_labels_are_consistent(sets):
    for ps in sets.values():
        assert (ps.oracle_steps[ps.infeasible] == 1).all()
        assert (ps.learnable == (~ps.infeasible & (ps.oracle_steps >= 500))).all()


def test_checksum_detects_tampering(sets, tmp_path):
    save_sets({"E1": sets["E1"]}, str(tmp_path))
    assert load_sets(str(tmp_path))["E1"].digest() == sets["E1"].digest()
    z = dict(np.load(tmp_path / "E1.npz"))
    z["s0"] = z["s0"] + 1e-3
    np.savez(tmp_path / "E1.npz", **z)
    with pytest.raises(ValueError):
        load_sets(str(tmp_path))
