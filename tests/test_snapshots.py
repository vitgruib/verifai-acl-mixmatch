import numpy as np
import pandas as pd

from acl_bench.suite.regrade import regrade
from acl_bench.suite.run_arms import run_job
from acl_bench.suite.snapshots import find_runs, load_run

SETS = "frozen_sets/cartpole_v2"


def test_saved_snapshots_regrade_to_exactly_the_live_grades(tmp_path):
    job = ("A", 3, 4096, 2048, SETS, {}, str(tmp_path))
    live = pd.DataFrame(run_job(job))
    assert [(a, r) for a, r, _ in find_runs(str(tmp_path))] == [("A", 3)]
    assert sorted(load_run(str(tmp_path / "A" / "3.npz"))) == [0, 2048, 4096]     # step 0 (untrained) included
    again = regrade(str(tmp_path), SETS)
    cols = [c for c in live.columns if "/" in c]
    pd.testing.assert_frame_equal(live.sort_values("step")[cols].reset_index(drop=True),
                                  again.sort_values("step")[cols].reset_index(drop=True))


def test_snapshots_can_be_graded_on_a_different_exam_later(tmp_path):
    snaps = tmp_path / "snaps"
    run_job(("N", 1, 2048, 2048, SETS, {}, str(snaps)))
    import json, os, shutil
    small = tmp_path / "exam"
    os.makedirs(small)
    manifest = json.load(open(os.path.join(SETS, "manifest.json")))
    manifest["sets"] = {"E1": manifest["sets"]["E1"]}
    shutil.copy(os.path.join(SETS, "E1.npz"), small / "E1.npz")
    json.dump(manifest, open(small / "manifest.json", "w"))
    only = regrade(str(snaps), str(small))
    assert "E1/success" in only.columns and "E0/success" not in only.columns


def test_snapshots_are_small(tmp_path):
    run_job(("N", 1, 4096, 2048, SETS, {}, str(tmp_path)))
    size = (tmp_path / "N" / "1.npz").stat().st_size
    assert size < 200_000, size          # three check-ins of two 64-unit MLPs
