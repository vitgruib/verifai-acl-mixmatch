"""Maze exam sections: random 25-wall levels and the custom hand-picked set."""
import numpy as np
import pytest

from atlas.maze import oracle, space, suite

NAMED = ["SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
         "StandardMaze", "StandardMaze2", "StandardMaze3"]


def level(ms, i):
    return ms.walls[i], tuple(ms.agent[i]), int(ms.agent_dir[i]), tuple(ms.goal[i])


def test_random_section_is_solvable_and_in_box():
    ms = suite.random_section()
    assert ms.walls.shape == (300, 13, 13)
    for i in range(300):
        assert oracle.solve(*level(ms, i))["status"] == oracle.SOLVABLE
    assert ms.walls.sum(axis=(1, 2)).max() <= 25
    again = suite.random_section()
    for a in ("walls", "agent", "agent_dir", "goal"):
        assert np.array_equal(getattr(ms, a), getattr(again, a))


def test_custom_labels_and_solvable():
    ms = suite.custom_section()
    assert ms.labels == NAMED + ["empty", "longest_path", "dead_end_goal", "corridor", "walls30", "walls50"]
    lv = {lab: level(ms, i) for i, lab in enumerate(ms.labels)}
    for l in lv.values():
        assert oracle.solve(*l)["status"] == oracle.SOLVABLE
    assert lv["empty"][0].sum() == 0
    assert lv["walls30"][0].sum() == 30 and lv["walls50"][0].sum() == 50
    plen = {lab: space.shortest_path(w, a, g) for lab, (w, a, _, g) in lv.items()}
    assert plen["longest_path"] == max(plen.values())
    w, _, _, (gx, gy) = lv["dead_end_goal"]
    free = [(gx + dx, gy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= gx + dx < 13 and 0 <= gy + dy < 13 and not w[gy + dy, gx + dx]]
    assert len(free) == 1


def test_named_mazes_match_jaxued_prefabs():
    pytest.importorskip("jax")
    from atlas.maze import policy  # noqa: F401  (puts JaxUED on sys.path)
    from jaxued.environments.maze import Level
    ms = suite.custom_section()
    for name in NAMED:
        p = Level.load_prefabs([name])
        i = ms.labels.index(name)
        assert np.array_equal(ms.walls[i], np.asarray(p.wall_map[0]))
        assert tuple(ms.agent[i]) == tuple(int(v) for v in p.agent_pos[0])
        assert int(ms.agent_dir[i]) == int(p.agent_dir[0])
        assert tuple(ms.goal[i]) == tuple(int(v) for v in p.goal_pos[0])


def test_digest_mismatch_refuses(tmp_path):
    ms = suite.custom_section()
    suite.save_set(str(tmp_path), "custom", ms)
    assert suite.load_sets(str(tmp_path))["custom"].labels == ms.labels
    z = dict(np.load(tmp_path / "custom.npz"))
    z["walls"] = z["walls"].copy()
    z["walls"][0, 0, 0] ^= True
    np.savez_compressed(tmp_path / "custom.npz", **z)
    with pytest.raises(ValueError):
        suite.load_sets(str(tmp_path))
