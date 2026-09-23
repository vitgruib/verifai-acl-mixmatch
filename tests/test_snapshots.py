import json
import os
import shutil

import pandas as pd

from acl_bench.snapshots import find_runs, load_run
from acl_bench.study.regrade import grade_run
from acl_bench.study.run import run_job
from acl_bench.exam.sets import load_sets

EXAM = "frozen_sets/cartpole"


def test_saved_snapshots_regrade_to_exactly_the_live_grades(tmp_path):
    live = pd.DataFrame(run_job(("A", 3, 4096, 2048, EXAM, str(tmp_path))))
    assert [(a, r) for a, r, _ in find_runs(str(tmp_path))] == [("A", 3)]
    path = str(tmp_path / "A" / "3.npz")
    assert sorted(load_run(path)) == [0, 2048, 4096]          # step 0 (untrained) included
    again = pd.DataFrame(grade_run("A", 3, path, load_sets(EXAM)))
    cols = [c for c in live.columns if "/" in c]
    pd.testing.assert_frame_equal(live.sort_values("step")[cols].reset_index(drop=True),
                                  again.sort_values("step")[cols].reset_index(drop=True))


def test_regrade_on_a_checkpoint_grid_keeps_the_final_model(tmp_path):
    run_job(("N", 1, 6144, 1024, EXAM, str(tmp_path)))       # check-ins at 0, 1024, ..., 6144
    rows = grade_run("N", 1, str(tmp_path / "N" / "1.npz"), load_sets(EXAM), n_checkpoints=3)
    assert [r["step"] for r in rows] == [2048, 4096, 6144]


def test_snapshots_can_be_graded_on_a_different_exam_later(tmp_path):
    run_job(("N", 1, 2048, 2048, EXAM, str(tmp_path / "snaps")))
    small = tmp_path / "exam"
    os.makedirs(small)
    manifest = json.load(open(os.path.join(EXAM, "manifest.json")))
    manifest["sets"] = {"random": manifest["sets"]["random"]}
    shutil.copy(os.path.join(EXAM, "random.npz"), small / "random.npz")
    json.dump(manifest, open(small / "manifest.json", "w"))
    only = grade_run("N", 1, str(tmp_path / "snaps" / "N" / "1.npz"), load_sets(str(small)))
    assert all("random/success" in r and "verifai/success" not in r for r in only)


def test_snapshots_are_small(tmp_path):
    run_job(("N", 1, 4096, 2048, EXAM, str(tmp_path)))
    assert (tmp_path / "N" / "1.npz").stat().st_size < 200_000     # three check-ins of two 64-unit MLPs
