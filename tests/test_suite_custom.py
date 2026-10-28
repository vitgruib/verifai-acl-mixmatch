import numpy as np
import pytest

from acl_bench.envs import cartpole, frozenlake, lunarlander
from acl_bench.suite.custom import custom_cases, custom_suite


def test_cartpole_has_default_plus_six_per_swept_setting():
    labels = [label for label, _ in custom_cases(cartpole)]
    assert labels[0] == "default" and len(labels) <= 1 + 5 * 6      # start spread is swept too
    assert "force_mag:max+50%" in labels and "init_range:max+50%" in labels
    assert len(labels) == len(set(labels))


def test_beyond_values_are_offsets_of_the_range():
    p = dict(custom_cases(cartpole))["force_mag:max+10%"]
    assert p["force_mag"] == pytest.approx(16 + 0.1 * 12)
    assert p["length"] == cartpole.DEFAULTS["length"]


def test_invalid_beyond_values_are_skipped():
    labels = [label for label, _ in custom_cases(lunarlander)]
    assert "gravity:min+10%" not in labels        # -13 is outside Gymnasium's (-12, 0)
    assert "wind_power:min+10%" not in labels     # negative wind
    assert "wind_power:max+10%" in labels


def test_frozenlake_custom_maps_are_all_solvable():
    s = custom_suite(frozenlake)
    assert len(s) > 0 and len(s) % 10 == 0
    assert all(frozenlake.optimal_success(p) >= 0.5 for p in np.unique(s.params, axis=0))
    assert frozenlake.certify_exact(s.params, s.s0).all()


def test_frozenlake_uses_three_maps_per_case_and_explicit_beyond_values():
    cases = custom_cases(frozenlake)
    assert not any(p["layout_seed"] < 0 for _, p in cases if not _.startswith("held_out"))
    assert any(label.startswith("slip:max+beyond(0.9)") for label, _ in cases)
    assert any(label.startswith("held_out:4x4") for label, _ in cases)


def test_suite_is_named_custom_and_reproducible():
    a, b = custom_suite(lunarlander, seed=3), custom_suite(lunarlander, seed=3)
    n = len(custom_cases(lunarlander))
    assert a.name == "custom" and len(a) == 10 * n
    assert a.digest() == b.digest()
