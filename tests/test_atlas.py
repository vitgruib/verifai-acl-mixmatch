"""Atlas falsification spaces and failure records (numpy only; no JAX or checkpoints)."""
import json

import numpy as np

from atlas import record
from atlas.maze import oracle, space


def point(name, rng):
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in space.SPACES[name].items()}


def test_build_respects_constraints():
    rng = np.random.default_rng(0)
    for name in space.SPACES:
        n_valid = 0
        for _ in range(200):
            walls, agent, d, goal, why = space.build(name, point(name, rng))
            assert walls.shape == (space.N, space.N) and 0 <= d < 4
            assert not walls[agent[1], agent[0]] and not walls[goal[1], goal[0]]
            if why is None:
                n_valid += 1
                assert agent != goal and space.shortest_path(walls, agent, goal) is not None
        assert n_valid > 100, name


def test_build_is_deterministic_and_replayable():
    p = point("seg", np.random.default_rng(1))
    a, b = space.build("seg", p), space.build("seg", p)
    assert (a[0] == b[0]).all() and a[1:] == b[1:]
    s = space.to_str(*a[:4])
    assert s.count("G") == 1 and sum(s.count(c) for c in ">v<^") == 1


def test_descriptors_on_known_level():
    walls = np.zeros((space.N, space.N), bool)
    walls[1:, 1] = True                       # a wall column with a gap at the top
    d = space.descriptors(walls, (0, 12), (2, 12))
    assert d["n_walls"] == 12 and d["manhattan"] == 2
    assert d["path_len"] == 12 + 12 + 2       # up, across the gap, down
    assert d["detour"] == d["path_len"] / 2


def test_recorder_roundtrip(tmp_path):
    rec = record.Recorder(str(tmp_path), {"env": "maze", "algo": "dr"}, space.DESC_KEYS)
    desc = {k: 1 for k in space.DESC_KEYS}
    rec.add({"valid": False, "invalid_reason": "unreachable"})
    rec.add({"valid": True, "desc": desc, "rho": 0.3, "cex": False, "hard": False, "train": {"dr_knn": 1.0}})
    rec.add({"valid": True, "desc": desc, "rho": -0.5, "cex": True, "hard": True, "train": {"dr_knn": 3.0}},
            trace={"pos": np.zeros((5, 2, 2), np.int8)})
    s = rec.close()
    header, rows, s2 = record.load(str(tmp_path))
    assert header["schema_version"] == record.SCHEMA_VERSION and header["algo"] == "dr"
    assert [r["i"] for r in rows] == [0, 1, 2]
    assert s2 == json.loads(json.dumps(s))
    assert s["n"] == 3 and s["n_valid"] == 2 and s["n_cex"] == 1 and s["cex_rate"] == 0.5
    assert s["first_cex_i"] == 2 and s["min_rho"] == -0.5
    assert s["train_cex"]["dr_knn"] == 3.0 and s["train_ok"]["dr_knn"] == 1.0
    assert list(np.load(tmp_path / "traces.npz")) == ["i2_pos"]


def test_oracle_counts_turns_and_forward_into_goal():
    walls = np.zeros((space.N, space.N), bool)
    o = oracle.solve(walls, (0, 0), 0, (3, 0))            # facing right, goal 3 cells right
    assert o["status"] == oracle.SOLVABLE and o["cert"] == "FFF"
    o = oracle.solve(walls, (0, 0), 2, (3, 0))            # facing left: two turns first
    assert o["min_steps"] == 5 and o["cert"].count("F") == 3
    assert oracle.replay(walls, (0, 0), 2, (3, 0), o["cert"])
    assert not oracle.replay(walls, (0, 0), 2, (3, 0), o["cert"][:-1])


def test_oracle_step_cap_and_unreachable():
    walls = np.zeros((space.N, space.N), bool)
    walls[1:, 1] = True
    o = oracle.solve(walls, (0, 12), 3, (2, 12))
    assert o["status"] == oracle.SOLVABLE and o["min_steps"] > 26     # 26 moves + turns
    capped = oracle.solve(walls, (0, 12), 3, (2, 12), max_steps=o["min_steps"] - 1)
    assert capped["status"] == oracle.UNSOLVABLE and capped["min_steps"] == o["min_steps"]
    walls[0, 1] = True                                    # close the gap
    assert oracle.solve(walls, (0, 12), 3, (2, 12))["status"] == oracle.UNSOLVABLE


def test_oracle_agrees_with_cell_reachability():
    rng = np.random.default_rng(2)
    for name in space.SPACES:
        for _ in range(150):
            walls, agent, d, goal, why = space.build(name, point(name, rng))
            o = oracle.solve(walls, agent, d, goal)
            assert (o["min_steps"] is not None) == (why is None)
            if why is None:
                assert oracle.replay(walls, agent, d, goal, o["cert"])
                assert o["min_steps"] >= space.shortest_path(walls, agent, goal)


def test_oracle_certificate_solves_real_jaxued_env():
    import pytest
    jaxued = pytest.importorskip("jaxued.environments")
    from atlas.maze.policy import replay_actions
    env = jaxued.Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    rng = np.random.default_rng(3)
    for _ in range(20):
        walls, agent, d, goal, why = space.build("seg", point("seg", rng))
        if why is None:
            o = oracle.solve(walls, agent, d, goal, env.default_params.max_steps_in_episode)
            r, n = replay_actions(env, env.default_params, walls, agent, d, goal, o["cert"])
            assert r > 0 and n == o["min_steps"]
