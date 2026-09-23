import pandas as pd
import pytest

from acl_bench.study.analyze import comparisons, edge_sections, on_grid


def test_thirteen_predeclared_comparisons_all_distinct():
    comps = comparisons()
    assert len(comps) == 13
    assert len({(a, b) for _, a, b in comps}) == 13
    assert ("C1", "A", "N") in comps and ("C5_ce", "B_ce", "A") in comps


def test_edge_sections_found_from_columns_in_numeric_order():
    df = pd.DataFrame(columns=["arm", "E10v/success", "E2v/success", "E2v/n", "E6/success", "E1v/success"])
    assert edge_sections(df) == ["E1v", "E2v", "E10v"]


def test_on_grid_keeps_the_same_checkpoints_for_every_run_and_refuses_gaps():
    steps = [20_480 * k for k in range(31)]
    full = pd.DataFrame({"arm": "N", "seed": 1, "step": steps})
    grid_only = pd.DataFrame({"arm": "A", "seed": 1, "step": [61_440 * k for k in range(1, 11)]})
    out = on_grid(pd.concat([full, grid_only]))
    assert out.groupby("arm").size().tolist() == [10, 10]
    with pytest.raises(SystemExit):
        on_grid(pd.concat([full, grid_only.iloc[:-1]]))
