"""FrozenLake: random maps, slip as in Gymnasium, egocentric view, exact winnability checks."""
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from acl_bench.envs import frozenlake as F
from acl_bench.exam.grader import grade
from acl_bench.ppo import Agent

LEFT, DOWN, RIGHT, UP = range(4)


def empty_map(size):
    return np.zeros((size, size), dtype=bool)


def fixed_agent(action):
    return types.SimpleNamespace(actor=lambda o: torch.nn.functional.one_hot(
        torch.full((len(o),), action), F.ACTION_DIM).float())


def test_start_and_goal_are_never_holes():
    for seed in range(200):
        holes, start, goal = F.layout(10, 0.3, seed)
        assert not holes[start] and not holes[goal] and start != goal


def test_no_slip_matches_deterministic_moves():
    s = F.make_state(empty_map(4), play_seed=0)[None]
    p = np.array([[4, 0.0, 0.0, 0]])
    s = F.step(s, [RIGHT], p)
    assert tuple(s[0, :3]) == (1, 0, 1)
    s = F.step(s, [UP], p)                                   # off the map: stays put
    assert tuple(s[0, :2]) == (1, 0)
    s = F.step(s, [DOWN], p)
    assert tuple(s[0, :2]) == (1, 1)


def test_slip_goes_perpendicular_with_equal_odds():
    n, slip = 20_000, 0.6
    s = np.stack([F.make_state(empty_map(4), play_seed=k) for k in range(n)])
    s[:, :2] = 1                                             # from (1, 1), all four moves stay on the map
    moved = F.step(s, np.full(n, RIGHT), np.tile([4, 0.0, slip, 0], (n, 1)))[:, :2] - 1
    frac = {d: np.mean((moved == d).all(axis=1)) for d in [(1, 0), (0, -1), (0, 1)]}
    assert frac[(1, 0)] == pytest.approx(1 - slip, abs=0.015)
    assert frac[(0, -1)] == pytest.approx(slip / 2, abs=0.015)
    assert frac[(0, 1)] == pytest.approx(slip / 2, abs=0.015)


@pytest.mark.parametrize("name, gym_value", [
    ("4x4", 0.74419),       # value iteration on Gymnasium FrozenLake-v1's own P table, 100 steps
    ("8x8", 0.64072),
])
def test_optimal_success_matches_gymnasium_slippery_maps(name, gym_value):
    p, _ = F.held_out_questions()[name]
    assert F.optimal_success(p) == pytest.approx(gym_value, abs=1e-4)


def test_unsolvable_map_is_not_certified():
    p = np.array([4, 0.6, 0.3, 1])                           # this seed walls the goal off
    holes, _, goal = F.layout(*p[[0, 1, 3]])
    assert holes[goal[1] - 1, goal[0]] and holes[goal[1], goal[0] - 1]
    s0 = F.sample_starts(p, 3, np.random.default_rng(0))
    assert not F.certify_exact(np.tile(p, (3, 1)), s0).any()
    assert F.optimal_success(p) == 0.0


def test_open_map_is_certified_and_slip_free_success_is_certain():
    p = np.array([6, 0.0, 0.0, 1])
    s0 = F.sample_starts(p, 2, np.random.default_rng(0))
    assert F.certify_exact(np.tile(p, (2, 1)), s0).all()
    assert F.optimal_success(p) == 1.0


def test_starts_have_distinct_play_seeds():
    s0 = F.sample_starts(np.array([8, 0.2, 0.67, 5]), 10, np.random.default_rng(0))
    assert len(set(s0[:, F._SEED])) == 10


def test_hole_ends_the_play_with_no_reward_and_fails_the_question():
    holes = empty_map(4)
    holes[0, 1] = True                                       # just right of the start
    s0 = F.make_state(holes, play_seed=0)[None]
    p = np.array([[4, 1 / 16, 0.0, 0]])
    assert F.reward(s0, [RIGHT], p)[0] == 0.0 and F.terminated(F.step(s0, [RIGHT], p))[0]
    success, steps = grade(F, fixed_agent(RIGHT), p, s0)
    assert not success[0] and steps[0] == 1


def test_goal_gives_time_discounted_reward():
    s0 = F.make_state(empty_map(2), play_seed=0)[None]
    p = np.array([[2, 0.0, 0.0, 0]])
    s1 = F.step(s0, [RIGHT], p)
    assert F.reward(s1, [DOWN], p)[0] == pytest.approx(1 - 0.9 * 2 / F.MAX_EPISODE_STEPS)
    assert F.terminated(F.step(s1, [DOWN], p))[0]


def test_reaching_the_goal_passes_the_question():
    right_is_off_map = 2 * 25 + 2 * 5 + 3                    # off-map channel, one cell right of the centre
    agent = types.SimpleNamespace(actor=lambda o: torch.nn.functional.one_hot(
        torch.where(o[:, right_is_off_map] > 0, DOWN, RIGHT), F.ACTION_DIM).float())
    s0 = F.make_state(empty_map(3), play_seed=0)[None]
    success, steps = grade(F, agent, np.array([[3, 0.0, 0.0, 0]]), s0)
    assert success[0] and steps[0] == 4


def test_view_is_centred_on_the_agent():
    holes = empty_map(4)
    holes[0, 1] = True
    obs = F.observe(F.make_state(holes, play_seed=0)[None])[0].reshape(3, 5, 5)   # channels hole, goal, off-map
    assert obs[0, 2, 3] == 1 and obs[0].sum() == 1           # the hole one cell right of the centre
    assert obs[1].sum() == 0                                 # the goal (3, 3) is outside the window
    assert obs[2, :2].all() and obs[2, :, :2].all() and obs[2, 2:, 2:].sum() == 0


@pytest.mark.parametrize("params, ok", [
    ({"size": 8, "hole_density": 0.2, "slip": 0.67}, True),
    ({"size": 15, "hole_density": 0.0, "slip": 1.0}, True),
    ({"size": 1, "hole_density": 0.2, "slip": 0.67}, False),
    ({"size": 16, "hole_density": 0.2, "slip": 0.67}, False),
    ({"size": 8, "hole_density": 1.0, "slip": 0.67}, False),
    ({"size": 8, "hole_density": -0.1, "slip": 0.67}, False),
    ({"size": 8, "hole_density": 0.2, "slip": 1.1}, False),
    ({"size": 8, "hole_density": 0.2, "slip": -0.1}, False),
])
def test_valid(params, ok):
    assert F.valid(params) is ok


def test_held_out_maps_are_solvable_questions():
    for name, (p, s0) in F.held_out_questions().items():
        assert F.certify_exact(np.tile(p, (len(s0), 1)), s0).all(), name
        assert F.optimal_success(p) >= 0.5, name


def graded(s0, p):
    torch.manual_seed(0)
    agent = Agent(F.OBS_DIM, F.ACTION_DIM)
    extras = {}
    success, steps = grade(F, agent, p, s0, extras=extras)
    return success.tolist(), steps.tolist(), extras["return"].tolist()


def exam_rows():
    p = np.array([[6, 0.15, 0.5, 11], [5, 0.1, 0.67, 3]])
    s0 = np.concatenate([F.sample_starts(r, 4, np.random.default_rng(3)) for r in p])
    return np.repeat(p, 4, axis=0), s0


def test_play_is_reproducible_from_its_start():
    p, s0 = exam_rows()
    first = graded(s0, p)
    assert graded(s0, p) == first
    assert len(set(first[1])) > 1                            # plays differ: not all trivial


def test_play_is_reproducible_in_another_process(tmp_path):
    p, s0 = exam_rows()
    np.savez(tmp_path / "q.npz", s0=s0, p=p)
    script = (
        "import json, sys, numpy as np\n"
        "from tests.test_frozenlake import graded\n"
        "d = np.load(sys.argv[1]); print(json.dumps(graded(d['s0'], d['p'])))\n")
    out = subprocess.run([sys.executable, "-c", script, str(tmp_path / "q.npz")], check=True,
                         capture_output=True, text=True, cwd=Path(__file__).parent.parent).stdout
    assert json.loads(out.splitlines()[-1]) == list(graded(s0, p))
