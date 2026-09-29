"""Point-mass navigation: the wall and its gap, the shaped reward, certification."""
import numpy as np

from acl_bench.envs import pointnav as P
from acl_bench.exam.certify import certify

RIGHT = 1                                  # action 1: thrust along +x


def task(gap_y=0.0, gap_width=0.2, wind=0.0, force=1.5, goal_y=0.0):
    return np.array([[gap_y, gap_width, wind, force, goal_y]])


def run_right(params, y0, steps=60):
    s = P.sample_starts(params[0], 1, np.random.default_rng(0))
    s[0, P._Y] = y0
    for _ in range(steps):
        s = P.step(s, [RIGHT], params)
    return s


def test_wall_blocks_outside_the_gap():
    s = run_right(task(gap_y=0.5, gap_width=0.2), y0=-0.5)
    assert s[0, P._X] < 0 and s[0, P._VX] == 0.0


def test_gap_lets_the_agent_through():
    s = run_right(task(gap_y=0.0, gap_width=0.4, goal_y=0.0), y0=0.0)
    assert P.terminated(s)[0] or s[0, P._X] > 0


def test_reward_is_route_progress_plus_arrival_bonus():
    params = task(gap_width=0.4)
    s = P.sample_starts(params[0], 1, np.random.default_rng(0))
    s[0, P._Y] = 0.0
    start_len, total = P.route_length(s)[0], 0.0
    for _ in range(P.MAX_EPISODE_STEPS):
        total += P.reward(s, [RIGHT], params)[0]
        s = P.step(s, [RIGHT], params)
        if P.terminated(s)[0]:
            break
    assert P.terminated(s)[0]
    assert np.isclose(total, 10 * (start_len - P.route_length(s)[0]) + 10)


def test_random_tasks_are_mostly_certifiable():
    rng = np.random.default_rng(1)
    lo, hi = (np.array([P.PARAM_BOUNDS[k][i] for k in P.PARAM_ORDER]) for i in (0, 1))
    params = rng.uniform(lo, hi, size=(100, 5))
    s0 = np.concatenate([P.sample_starts(p, 1, rng) for p in params])
    ok = certify(P, params, s0, beam=64)
    assert ok.mean() > 0.5
