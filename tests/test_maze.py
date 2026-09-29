"""The maze environment: movement, the egocentric view, exact winnability, held-out layouts."""
import numpy as np

from acl_bench.envs import maze as M
from acl_bench.envs.maze_layouts import HELD_OUT


def naive_view(s):
    """The 5x5 view one cell at a time: row 0 is farthest ahead, column 2 is straight ahead."""
    walls = s[M._W0:].reshape(M.SIZE, M.SIZE) > 0.5
    x, y, d = int(s[M._X]), int(s[M._Y]), int(s[M._DIR])
    f, r = M._FWD[d], M._RIGHT[d]
    wall, goal = np.zeros((5, 5)), np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            cx, cy = x + (4 - i) * f[0] + (j - 2) * r[0], y + (4 - i) * f[1] + (j - 2) * r[1]
            inside = 0 <= cx < M.SIZE and 0 <= cy < M.SIZE
            wall[i, j] = (not inside) or walls[cy, cx]
            goal[i, j] = (cx, cy) == (int(s[M._GX]), int(s[M._GY]))
    return np.concatenate([wall.ravel(), goal.ravel(), np.eye(4)[d]])


def random_states(n, seed):
    rng = np.random.default_rng(seed)
    tasks = np.stack([rng.uniform(0, 60, n), rng.uniform(0, 1e6, n)], axis=1)
    return tasks, np.concatenate([M.sample_starts(t, 1, rng) for t in tasks])


def test_view_matches_per_cell_reference():
    _, s = random_states(200, 0)
    rng = np.random.default_rng(1)
    for _ in range(20):
        np.testing.assert_array_equal(M.observe(s), np.stack([naive_view(r) for r in s]))
        s = M.step(s, rng.integers(0, 3, len(s)), None)


def test_moves_turns_and_walls():
    walls = np.ones((M.SIZE, M.SIZE), dtype=bool)
    walls[1:-1, 1:-1] = False
    walls[5, 4] = True                                    # a wall just right of (3, 5)
    s = M.make_state(walls, (3, 5), (10, 10), 0)[None]    # facing right
    assert tuple(M.step(s, [2], None)[0, :2]) == (3, 5)   # blocked
    s = M.step(s, [1], None)                              # turn right: now facing down
    assert s[0, M._DIR] == 1
    s = M.step(s, [2], None)
    assert tuple(s[0, :2]) == (3, 6) and s[0, M._T] == 2


def test_reward_only_on_arrival():
    walls = np.ones((M.SIZE, M.SIZE), dtype=bool)
    walls[1:-1, 1:-1] = False
    s = M.make_state(walls, (3, 5), (4, 5), 0)[None]
    assert M.reward(s, [0], None)[0] == 0.0
    r = M.reward(s, [2], None)[0]
    assert r == 1 - 0.9 * 1 / M.MAX_EPISODE_STEPS
    assert M.terminated(M.step(s, [2], None))[0]


def test_bfs_matches_a_greedy_rollout_when_one_exists():
    tasks, s0 = random_states(300, 2)
    ok = M.certify_exact(tasks, s0)
    assert 0.3 < ok.mean() < 1.0                           # some random layouts are unsolvable
    for s in s0[ok][:30]:                                  # follow the BFS distance downhill
        walls = s[M._W0:].reshape(M.SIZE, M.SIZE) > 0.5
        goal = (int(s[M._GX]), int(s[M._GY]))
        state = s[None]
        for _ in range(M.MAX_EPISODE_STEPS):
            if M.terminated(state)[0]:
                break
            dists = [M.shortest_steps(walls, tuple(int(v) for v in M.step(state, [a], None)[0, :2]), goal,
                                      int(M.step(state, [a], None)[0, M._DIR])) for a in range(3)]
            state = M.step(state, [int(np.argmin(dists))], None)
        assert M.terminated(state)[0]


def test_held_out_layouts():
    params, s0, labels = M.held_out_questions()
    assert len(s0) == 4 * len(HELD_OUT)
    for s in s0:
        walls = s[M._W0:].reshape(M.SIZE, M.SIZE) > 0.5
        assert walls[0].all() and walls[-1].all() and walls[:, 0].all() and walls[:, -1].all()
        assert not walls[int(s[M._Y]), int(s[M._X])] and not walls[int(s[M._GY]), int(s[M._GX])]


def test_gym_wrapper_matches_batched():
    env = M.make_env({"n_walls": 30, "layout_seed": 7})
    obs, _ = env.reset(seed=0)
    s = env.state.copy()[None]
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = int(rng.integers(0, 3))
        obs, r, term, _, _ = env.step(a)
        r_b = M.reward(s, [a], None)[0]
        s = M.step(s, [a], None)
        np.testing.assert_array_equal(obs, M.observe(s)[0].astype(np.float32))
        assert r == r_b and term == M.terminated(s)[0]
        if term:
            break
