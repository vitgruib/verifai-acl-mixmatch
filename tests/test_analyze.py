import pandas as pd
import pytest

from acl_bench.suite.analyze import comparisons, merge


def test_thirteen_predeclared_comparisons_all_distinct():
    comps = comparisons()
    assert len(comps) == 13
    assert len({(a, b) for _, a, b in comps}) == 13
    assert ("C1", "A", "N") in comps and ("C5_ce", "B_ce", "A") in comps


def test_merge_joins_on_run_and_step_and_refuses_ungraded_runs():
    main = pd.DataFrame({"arm": ["N", "N", "A"], "seed": [1, 1, 1], "step": [0, 10, 0],
                         "E0/success": [0.0, 0.5, 0.1], "E1/success": [0.0, 0.2, 0.0]})
    edge = pd.DataFrame({"arm": ["N", "N"], "seed": [1, 1], "step": [0, 10], "E1v/success": [0.0, 0.3]})
    with pytest.raises(SystemExit):
        merge(main, edge)                              # A/1 has no edge grades
    df = merge(main[main.arm == "N"], edge)
    assert list(df["E1v/success"]) == [0.0, 0.3]
    assert "E1/success" not in df.columns              # v2 edge sections dropped


def test_edge_sections_found_from_columns_in_numeric_order():
    from acl_bench.suite.analyze import edge_sections
    df = pd.DataFrame(columns=["arm", "E10v/success", "E2v/success", "E2v/n", "E6/success", "E1v/success"])
    assert edge_sections(df) == ["E1v", "E2v", "E10v"]
