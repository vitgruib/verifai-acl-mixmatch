"""Checks acl_bench's ACL against the reference implementation in
SIPACL/Work/custom/custom_gym.py (`MetaDriveEnv`).

The `_ref_*` functions below are transcribed from that file (constants
DEFAULT_LP_EMA_BETA=0.2, DEFAULT_LP_RANK_ALPHA=1.0, _NEW_SCENE_LEARNING_POTENTIAL
=1e10) so the comparison does not depend on an ad hoc reading of the code.
"""
import numpy as np
import pytest

from acl_bench.curriculum.plr import NEW, REPLAY, PLRCurriculum
from acl_bench.potential.functions import pvl_gae

BETA, ALPHA, PLACEHOLDER = 0.2, 1.0, 1e10


def _ref_replay_probs(lp):
    n = int(lp.size)
    order = np.argsort(-lp, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    w = 1.0 / np.power(ranks, ALPHA)
    return w / float(np.sum(w))


def _ref_compute_lp(old_lp, raw):
    if old_lp >= 0.5 * PLACEHOLDER:
        return float(raw)
    return BETA * raw + (1.0 - BETA) * old_lp


def _ref_lp_delta(ep_rewards, ep_values, gamma, lam):
    n = len(ep_rewards)
    if n < 1:
        return 0.0
    lastgaelam = 0.0
    advantages = [0.0] * n
    for t in reversed(range(n)):
        if t == n - 1:
            next_v, nextnonterminal = 0.0, 0.0
        else:
            next_v, nextnonterminal = ep_values[t + 1], 1.0
        delta = ep_rewards[t] + gamma * next_v * nextnonterminal - ep_values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[t] = max(advantages[t], 0.0)
    return sum(advantages) / len(advantages)


class FakeSampler:
    def __init__(self):
        self.draws, self.feedback = 0, []

    def draw(self, names):
        self.draws += 1
        return {n: float(self.draws) for n in names}

    def give_feedback(self, rho):
        self.feedback.append(rho)


def make(use_replay=True, buffer_max=5000, score=1.0, seed=0):
    fn = lambda r, v, nv, g, l, st: (score() if callable(score) else score, st)
    sampler = FakeSampler()
    cur = PLRCurriculum(sampler, ("a",), fn, 0.99, 0.95, use_replay=use_replay,
                        buffer_max=buffer_max, rng=np.random.default_rng(seed))
    return cur, sampler


def run_episode(cur):
    params, idx, mode = cur.pick_task()
    cur.report_episode(idx, mode, [1.0, 1.0], [0.0, 0.0], 0.0)
    return idx, mode


def test_constants_match_sipacl():
    from acl_bench.curriculum import plr
    assert (plr._LP_EMA_BETA, plr._RANK_ALPHA, plr._NEW_TASK_LP) == (BETA, ALPHA, PLACEHOLDER)
    from acl_bench.ppo import PPOConfig
    assert PPOConfig().buffer_max == 5000 and PPOConfig().replay_prob == 0.5


@pytest.mark.parametrize("seed", range(5))
def test_rank_probabilities_match_including_ties_and_placeholders(seed):
    rng = np.random.default_rng(seed)
    lp = rng.integers(0, 4, size=25).astype(float)          # many ties
    lp[rng.integers(0, 25, size=3)] = PLACEHOLDER            # unscored new slots
    cur, _ = make()
    from acl_bench.curriculum.plr import _Slot
    cur.slots = [_Slot(params={}, lp=float(x)) for x in lp]
    np.testing.assert_allclose(cur._replay_probs(), _ref_replay_probs(lp))


def test_ema_matches_sipacl_including_first_visit_skip():
    raws = [3.0, 1.0, 4.0, 1.5, 9.0]
    seq = iter(raws)
    cur, _ = make(score=lambda: next(seq))
    idx, _ = run_episode(cur)                                # NEW: placeholder replaced by raw
    ref = _ref_compute_lp(PLACEHOLDER, raws[0])
    assert cur.slots[idx].lp == ref == raws[0]
    for raw in raws[1:]:                                     # revisit the same slot
        cur.report_episode(idx, REPLAY, [1.0], [0.0], 0.0)
        ref = _ref_compute_lp(ref, raw)
        assert cur.slots[idx].lp == pytest.approx(ref)


@pytest.mark.parametrize("seed", range(5))
def test_pvl_equals_sipacl_lp_delta_under_its_terminal_convention(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 60))
    rewards, values = rng.normal(size=n).tolist(), rng.normal(size=n).tolist()
    mine, _ = pvl_gae(rewards, values, 0.0, 0.99, 0.95, {})
    assert mine == pytest.approx(_ref_lp_delta(rewards, values, 0.99, 0.95))


def test_pvl_differs_from_sipacl_only_by_truncation_bootstrap():
    """Documented deviation: a truncated episode bootstraps from the critic."""
    rewards, values = [1.0] * 10, [5.0] * 10
    terminal, _ = pvl_gae(rewards, values, 0.0, 0.99, 0.95, {})
    truncated, _ = pvl_gae(rewards, values, 5.0, 0.99, 0.95, {})
    assert truncated != pytest.approx(terminal)


def test_new_slot_is_placeholder_until_scored():
    cur, _ = make()
    _, idx, mode = cur.pick_task()
    assert mode == NEW and cur.slots[idx].lp == PLACEHOLDER
    cur.report_episode(idx, mode, [1.0], [0.0], 0.0)
    assert cur.slots[idx].lp != PLACEHOLDER


def test_fifo_eviction_drops_oldest_first():
    cur, _ = make(use_replay=False, buffer_max=3)
    for _ in range(5):
        run_episode(cur)
    assert [s.params["a"] for s in cur.slots] == [3.0, 4.0, 5.0]


def test_acl_off_never_replays():
    cur, sampler = make(use_replay=False)
    modes = [run_episode(cur)[1] for _ in range(300)]
    assert set(modes) == {NEW} and sampler.draws == 300


def test_acl_on_replays_about_half_the_time():
    cur, _ = make(use_replay=True, seed=1)
    modes = [run_episode(cur)[1] for _ in range(4000)]
    assert 0.45 < modes.count(REPLAY) / len(modes) < 0.55


def test_sampler_feedback_only_for_new_draws():
    cur, sampler = make(use_replay=True, seed=2)
    modes = [run_episode(cur)[1] for _ in range(500)]
    assert len(sampler.feedback) == modes.count(NEW) == sampler.draws


def test_replay_favours_higher_scored_tasks():
    scores = iter([1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 100000)
    cur, _ = make(score=lambda: next(scores), use_replay=False)
    for _ in range(5):
        run_episode(cur)                                     # slots scored 1..5
    cur.replay_prob = 1.0
    counts = np.zeros(5)
    for _ in range(4000):
        _, idx, mode = cur.pick_task()
        counts[idx] += 1
        cur.slots[idx].lp = float(idx + 1)                   # keep ranking fixed
    assert list(np.argsort(-counts)) == [4, 3, 2, 1, 0]
