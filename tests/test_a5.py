"""Amendment 5: CVaR metric, calib dev/test split, multi-env primary."""
import numpy as np
import pandas as pd

from acl_bench.exam import sets as sets_mod
from acl_bench.exam.sets import ADV_STARTS, PairSet
from acl_bench.plr.decide import a5_primary, resolves
from acl_bench.plr.screen import split_dev_test


def test_cvar_is_mean_of_worst_tenth_of_levels(monkeypatch):
    n_levels = 20
    # 20 levels x 4 starts; level 0 passes 0/4, level 1 2/4, the rest 4/4: worst 10% = mean(0, 0.5)
    ok = np.concatenate([[False] * 4, [True, True, False, False], [True] * 72])

    def fake_grade(env, agent, params, s0, extras):
        return ok, np.zeros(len(ok))
    monkeypatch.setattr(sets_mod, "grade", fake_grade)
    ps = PairSet("adv", np.zeros((n_levels * ADV_STARTS, 1)), np.zeros((n_levels * ADV_STARTS, 1)))
    out = sets_mod.evaluate_sets(None, None, {"adv": ps})
    assert np.isclose(out["adv/cvar"], 0.25)
    assert np.isclose(out["adv/success"], ok.mean())


def test_calib_is_split_dev_test():
    ps = PairSet("calib", np.arange(10.)[:, None], np.zeros((10, 1)), np.linspace(0, 1, 10))
    out = split_dev_test({"calib": ps, "adv": ps})
    assert set(out) == {"calib_dev", "calib_test", "adv"}
    assert list(out["calib_dev"].params[:, 0]) == [0, 2, 4, 6, 8]


def _runs(arm_shift, rng, n=40):
    rows = [{"config": "DR", "seed": s, "fin_cd": v} for s, v in enumerate(rng.normal(0.4, 0.1, n))]
    rows += [{"config": "X", "seed": s, "fin_cd": v} for s, v in enumerate(rng.normal(0.4 + arm_shift, 0.1, n))]
    return pd.DataFrame(rows)


def test_a5_primary_standardizes_and_skips_floored_envs():
    rng = np.random.default_rng(0)
    floored = pd.DataFrame({"config": ["DR"] * 5 + ["X"] * 5, "seed": list(range(10)), "fin_cd": [0.0] * 10})
    tables = {"a": _runs(0.1, rng), "b": _runs(0.1, rng), "c": floored}
    z, p, zs = a5_primary(tables, "X", "fin_cd")
    assert set(zs) == {"a", "b"}
    assert 0.5 < z < 1.6 and p < 0.01
    z0, p0, _ = a5_primary({"a": _runs(0.0, rng, 200)}, "X", "fin_cd")
    assert abs(z0) < 0.3 and p0 > 0.01


def test_resolves():
    assert not resolves(pd.Series([0.0, 0.0, 0.0]))
    assert not resolves(pd.Series([0.99, 0.98, 0.97]))
    assert resolves(pd.Series([0.2, 0.4, 0.3]))
