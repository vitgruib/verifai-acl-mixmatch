"""Maze exam VerifAI section: levels at least 3 of 10 reference agents fail, capped and frozen."""
import json

import numpy as np
import pytest

from atlas.maze import exam, suite


class WallCountAgent:
    """Fails (return 0) exactly the levels with at least `limit` walls; records every level it plays."""

    def __init__(self, limit):
        self.limit, self.seen = limit, set()

    def run(self, walls, agent, agent_dir, goal, attempts, seed):
        self.seen.add(np.asarray(walls, bool).tobytes() + bytes([*agent, agent_dir, *goal]))
        ret = 0.0 if np.asarray(walls).sum() >= self.limit else 1.0
        return np.full(attempts, ret), np.full(attempts, 10), np.zeros((250, attempts, 2))


def keys(ms):
    return {ms.walls[i].tobytes() + bytes([*ms.agent[i], ms.agent_dir[i], *ms.goal[i]])
            for i in range(len(ms))}


def test_keeps_levels_three_of_ten_fail():
    agents = [WallCountAgent(limit) for limit in range(8, 28, 2)]   # 8, 10, ..., 26 walls
    capped, full = exam.verifai_section(agents, iters=60, seed=1)
    seen = set().union(*(a.seen for a in agents))
    # agent k fails n_walls >= 8 + 2k, so a level fails >= 3 agents iff it has >= 12 walls
    expected = {k for k in seen if np.frombuffer(k[:169], bool).sum() >= 12}
    assert expected and keys(full) == expected
    assert keys(capped) == expected
    n_walls = full.walls.sum(axis=(1, 2))
    fails = np.array([sum(n >= a.limit for a in agents) for n in n_walls])
    assert np.array_equal(full.ref_fail_frac, fails / 10)
    assert n_walls.max() <= 25


def test_cap_is_seeded_subset():
    agents = [WallCountAgent(0) for _ in range(10)]                 # every level fails every agent
    capped, full = exam.verifai_section(agents, iters=320, seed=3)
    assert len(full) > 3000 and len(capped) == 3000
    assert keys(capped) <= keys(full)
    again, _ = exam.verifai_section([WallCountAgent(0) for _ in range(10)], iters=320, seed=3)
    assert again.digest() == capped.digest()


def test_refuses_without_ten_reference_agents(tmp_path):
    ref = tmp_path / "maze_dr_ref"
    for s in range(100, 109):
        (ref / str(s)).mkdir(parents=True)
    with pytest.raises(SystemExit) as e:
        exam.main(["--ref-dir", str(ref), "--out", str(tmp_path / "out")])
    assert "10" in str(e.value)
    assert not (tmp_path / "out").exists()


def test_rerun_of_frozen_exam_does_nothing_and_other_refs_refuse(tmp_path, capsys):
    ref, out = tmp_path / "maze_dr_ref", tmp_path / "out"
    for s in range(100, 110):
        (ref / str(s) / "models" / "30000").mkdir(parents=True)
        (ref / str(s) / "config.json").write_text(json.dumps({"seed": s}))
    custom = suite.custom_section()
    for name in ("random", "custom", "verifai"):
        suite.save_set(str(out), name, custom)
    (out / "verifai_refs.json").write_text(json.dumps(exam._fingerprint(
        [str(ref / str(s)) for s in range(100, 110)])))
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    exam.main(["--ref-dir", str(ref), "--out", str(out)])
    assert "nothing to do" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    (ref / "105" / "models" / "40000").mkdir()
    with pytest.raises(SystemExit, match="refusing"):
        exam.main(["--ref-dir", str(ref), "--out", str(out)])
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
