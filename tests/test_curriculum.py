"""The curriculum library's math: importance weights undo the proposal's bias, the SIR
density matches its sampling, and the replay posterior is the Beta expectation."""
import numpy as np

from acl_bench.curriculum.core import Curriculum, Episode, TaskSpace
from acl_bench.curriculum.proposers import SIR, Replay, Uniform


class FakeModel:
    """p(task) = the first normalized coordinate: learnability peaks mid-box."""
    ready, version = True, 1

    def __init__(self, space):
        self.space = space

    def p(self, params):
        return self.space.unit(np.atleast_2d(params))[:, 0]

    def learnability(self, params):
        p = self.p(params)
        return p * (1 - p)

    def update(self, ep):
        pass


def _space(seed=0):
    return TaskSpace(np.array([[0.0, 2.0], [-1.0, 1.0]]), np.random.default_rng(seed))


def test_importance_weights_recover_uniform_mean():
    space = _space()
    model = FakeModel(space)
    rng = np.random.default_rng(1)
    props = {"uniform": Uniform(space), "sir": SIR(space, model, rng, alpha=1.0, n_candidates=256)}
    cur = Curriculum(space, props, {"uniform": 0.25, "sir": 0.75}, [model], rng, is_power=1.0, max_weight=1e9)
    f = lambda x: space.unit(x)[0] ** 3                  # any function of the task
    xs, ws = zip(*[(p, w) for p, _, w in (cur.propose(0) for _ in range(40000))])
    fx, ws = np.array([f(x) for x in xs]), np.array(ws)
    assert abs(np.mean(fx) - 0.25) > 0.02                # the proposal is biased toward mid-box...
    assert abs(np.sum(ws * fx) / np.sum(ws) - 0.25) < 0.01   # ...and the weights undo it (E_u[u^3] = 1/4)


def test_sir_density_matches_its_samples():
    space = _space()
    model = FakeModel(space)
    sir = SIR(space, model, np.random.default_rng(2), alpha=2.0, n_candidates=512)
    xs = np.array([space.unit(sir.propose(0).params)[0] for _ in range(20000)])
    hist, edges = np.histogram(xs, bins=10, range=(0, 1), density=True)
    mids = (edges[:-1] + edges[1:]) / 2
    dens = np.array([sir.density(np.array([space.lo[0] + m * space.span[0], 0.0])) for m in mids])
    assert np.max(np.abs(hist - dens)) < 0.1 * dens.max()


def test_replay_posterior_is_beta_expectation():
    space = _space()
    rep = Replay(space, FakeModel(space), np.random.default_rng(3), size=4, prior_strength=2.0, min_fill=1)
    x = np.array([0.5, 0.0])                               # model prior p0 = 0.25
    rep.report(0, Episode(x, True, 1.0, 10, "uniform"))
    a, b = 2 * 0.25 + 1, 2 * 0.75 + 0
    assert np.isclose(rep.scores()[0], a * b / ((a + b) * (a + b + 1)))
