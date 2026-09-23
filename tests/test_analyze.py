import pandas as pd
import pytest

from acl_bench.study.analyze import METRICS, comparisons, on_grid


def test_thirteen_predeclared_comparisons_all_distinct():
    comps = comparisons()
    assert len(comps) == 13
    assert len({(a, b) for _, a, b in comps}) == 13
    assert ("C1", "A", "N") in comps and ("C5_ce", "B_ce", "A") in comps


def test_four_metrics_per_suite():
    assert len(METRICS) == 8
    assert {"final_random", "auc_verifai", "final_steps_verifai", "auc_steps_random"} <= set(METRICS)


def test_on_grid_keeps_the_same_checkpoints_for_every_run_and_refuses_gaps():
    steps = [20_480 * k for k in range(31)]
    full = pd.DataFrame({"arm": "N", "seed": 1, "step": steps})
    grid_only = pd.DataFrame({"arm": "A", "seed": 1, "step": [61_440 * k for k in range(1, 11)]})
    out = on_grid(pd.concat([full, grid_only]))
    assert out.groupby("arm").size().tolist() == [10, 10]
    with pytest.raises(SystemExit):
        on_grid(pd.concat([full, grid_only.iloc[:-1]]))
