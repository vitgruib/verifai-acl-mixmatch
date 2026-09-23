"""Is a CartPole question winnable at all? Sound proofs both ways, with the grader's
own physics (evaluator.step_physics, deterministic, so a question is a fixed initial
state of a known discrete-time system with two actions).

- WINNABLE (certificate): a beam search over action sequences finds one that keeps the
  pole up for as long as the grader requires (the idea of "some model survives X steps",
  taken to X = the full episode, with a planner that knows the physics as the model).
  The found sequence is replayed through `step_physics` and must survive, so a
  "winnable" label never rests on the heuristic. Beam selection uses each question's LQR
  cost-to-go (discrete Riccati solution of the linearization about upright) as the
  score: the standard value-function heuristic for underactuated balancing (cf. LQR
  trees, Tedrake 2010).
- IMPOSSIBLE (proof), by either of two sound arguments:
  1. exhaustive search: the beam never had to drop a live branch, and every branch
     died, so every action sequence dies;
  2. interval reachability (the over-approximation behind reachability tools such as
     CORA, Althoff et al.): propagate a box containing every state reachable under ANY
     force in [-f, f] (a superset of the two discrete pushes), intersecting it with the
     safe set each step; if the box becomes empty, every trajectory has terminated.
     This proves doom long after the exhaustive search would have to truncate.
  3. split interval: branch exactly over the first few pushes, then (2) from each branch.
- UNKNOWN: neither. A heuristic failure, not evidence either way.

The first-step test used before (lost on step 1 whichever way the agent pushes) is the
special case of (1) and (2) at depth 1; every question it caught is also proven here.

Measured on the 9,993 search questions that 6+ of 10 reference agents fail
(docs/cartpole_suite.md): every question some agent passes is certified winnable (as it
must be), and of the 8,415 that all ten fail, ~87% are proven impossible, ~3% are
certified winnable, and ~11% stay unknown. A wider beam (1024) and deeper splitting
(depth 10, ~45 min) each resolve only a few percent more of the unknowns.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from acl_bench.suite.evaluator import (G, TAU, THETA_THRESHOLD, X_THRESHOLD, _FORCE, _LENGTH, _MASSCART,
                                       _MASSPOLE, step_physics, terminated)
from acl_bench.suite.sets import MAX_STEPS

WINNABLE, IMPOSSIBLE, UNKNOWN = "winnable", "impossible", "unknown"
# rollout_steps records a death at step t as `steps = t` and passes a question when
# steps == max_steps, so dying exactly at the last step still passes: survive to max_steps - 1.
MUST_SURVIVE = MAX_STEPS - 1


# ---------------------------------------------------------------- LQR heuristic
def lqr_cost_matrices(params: np.ndarray) -> np.ndarray:
    """(N, 4, 4) discrete-time LQR cost-to-go P for each question's linearization about
    upright, with the Euler step the grader uses. Only a ranking heuristic."""
    Q = np.diag([1 / X_THRESHOLD ** 2, 0.1, 1 / THETA_THRESHOLD ** 2, 0.1])
    out = np.empty((len(params), 4, 4))
    for i, p in enumerate(params):
        length, mp, mc, f = p[_LENGTH], p[_MASSPOLE], p[_MASSCART], p[_FORCE]
        M = mp + mc
        D = length * (4.0 / 3.0 - mp / M)
        Ac = np.array([[0, 1, 0, 0], [0, 0, -mp * length * G / (M * D), 0], [0, 0, 0, 1], [0, 0, G / D, 0]])
        Bc = np.array([[0], [1 / M + mp * length / (M * M * D)], [0], [-1 / (M * D)]])
        out[i] = solve_discrete_are(np.eye(4) + TAU * Ac, TAU * Bc, Q, np.array([[1 / f ** 2]]))
    return out


# ---------------------------------------------------------------- beam search
def beam_search(params: np.ndarray, s0: np.ndarray, beam: int = 128, horizon: int = MUST_SURVIVE,
                rng: np.random.Generator | None = None):
    """Returns (survived, exhaustive_death_step, actions). `survived[i]`: some branch was
    alive after `horizon` steps. `exhaustive_death_step[i]`: step at which every branch
    had died *without the beam ever dropping a live branch* (a proof), else 0.
    `actions[i]` (horizon,) int8: the surviving sequence (1 = push right), -1 if none."""
    rng = rng or np.random.default_rng(0)
    n = len(params)
    P = lqr_cost_matrices(params)
    force = params[:, _FORCE]
    state = np.zeros((n, beam, 4))
    state[:, 0] = s0
    valid = np.zeros((n, beam), dtype=bool)
    valid[:, 0] = True
    truncated = np.zeros(n, dtype=bool)
    death = np.zeros(n, dtype=np.int64)
    parents = np.empty((horizon, n, beam), dtype=np.int16)
    rep_params = np.repeat(params, 2 * beam, axis=0)
    signs = np.tile([-1.0, 1.0], beam)                              # child c = 2 * slot + action
    for t in range(1, horizon + 1):
        parent_state = np.repeat(state, 2, axis=1).reshape(-1, 4)
        f = (np.repeat(force, 2 * beam) * np.tile(signs, n))
        child = step_physics(parent_state, f, rep_params).reshape(n, 2 * beam, 4)
        alive = np.repeat(valid, 2, axis=1) & ~terminated(child.reshape(-1, 4)).reshape(n, 2 * beam)
        n_alive = alive.sum(axis=1)
        newly_dead = (n_alive == 0) & (death == 0)
        death[newly_dead] = t
        truncated |= n_alive > beam
        cost = np.einsum("nci,nij,ncj->nc", child, P, child)
        cost *= 1.0 + 1e-6 * rng.random(cost.shape)                 # break exact ties between twins
        cost[~alive] = np.inf
        keep = np.argpartition(cost, beam - 1, axis=1)[:, :beam]
        parents[t - 1] = keep
        state = np.take_along_axis(child, keep[:, :, None], axis=1)
        valid = np.take_along_axis(alive, keep, axis=1)
    survived = valid.any(axis=1)
    exhaustive_death = np.where(~survived & ~truncated, death, 0)

    actions = np.full((n, horizon), -1, dtype=np.int8)
    for i in np.flatnonzero(survived):
        slot = int(np.flatnonzero(valid[i])[0])
        for t in range(horizon - 1, -1, -1):
            c = int(parents[t, i, slot])
            actions[i, t], slot = c % 2, c // 2
    return survived, exhaustive_death, actions


def replay_survives(params: np.ndarray, s0: np.ndarray, actions: np.ndarray, horizon: int = MUST_SURVIVE) -> np.ndarray:
    """Independent check of a certificate: play the action sequence through the grader's
    physics and confirm no termination in `horizon` steps."""
    state = np.array(s0, dtype=np.float64)
    ok = (actions[:, 0] >= 0)
    for t in range(horizon):
        f = np.where(actions[:, t] == 1, params[:, _FORCE], -params[:, _FORCE])
        state = step_physics(state, f, params)
        ok &= ~terminated(state)
    return ok


# ---------------------------------------------------------------- interval reachability
def _mul(al, ah, bl, bh):
    c = np.stack([al * bl, al * bh, ah * bl, ah * bh])
    return c.min(axis=0), c.max(axis=0)


def _sin(lo, hi):                     # monotone on [-pi/2, pi/2]; boxes are clipped far inside that
    return np.sin(lo), np.sin(hi)


def _cos(lo, hi):                     # even, decreasing in |theta| on [0, pi/2]
    far = np.maximum(np.abs(lo), np.abs(hi))
    near = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    return np.cos(far), np.cos(near)


def _sq(lo, hi):
    a, b = lo * lo, hi * hi
    low = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(a, b))
    return low, np.maximum(a, b)


def interval_step(lo: np.ndarray, hi: np.ndarray, params: np.ndarray, pad: float = 1e-12):
    """One grader step applied to a box, for any force in [-f, f]: returns a box that
    contains the successor of every state in [lo, hi]. Mirrors step_physics term by term."""
    length, mp = params[:, _LENGTH], params[:, _MASSPOLE]
    M = mp + params[:, _MASSCART]
    pl = mp * length
    f = params[:, _FORCE]
    (xl, vl, tl, wl), (xh, vh, th, wh) = lo.T, hi.T
    sl, sh = _sin(tl, th)
    cl, ch = _cos(tl, th)
    w2l, w2h = _sq(wl, wh)
    a, b = _mul(w2l, w2h, sl, sh)
    templ, temph = (-f + pl * a) / M, (f + pl * b) / M
    ctl, cth = _mul(cl, ch, templ, temph)
    numl, numh = G * sl - cth, G * sh - ctl
    c2l, c2h = _sq(cl, ch)
    denl, denh = length * (4 / 3 - mp * c2h / M), length * (4 / 3 - mp * c2l / M)   # both > 0
    accl, acch = _mul(numl, numh, 1 / denh, 1 / denl)
    tcl, tch = _mul(accl, acch, cl, ch)
    xal, xah = templ - pl * tch / M, temph - pl * tcl / M
    new_lo = np.stack([xl + TAU * vl, vl + TAU * xal, tl + TAU * wl, wl + TAU * accl], axis=1)
    new_hi = np.stack([xh + TAU * vh, vh + TAU * xah, th + TAU * wh, wh + TAU * acch], axis=1)
    return new_lo - pad, new_hi + pad                        # pad: absorb float rounding, stay sound


def interval_doom_step(params: np.ndarray, s0: np.ndarray, horizon: int = MUST_SURVIVE,
                       give_up_width: float = 100.0) -> np.ndarray:
    """Step by which every trajectory has provably terminated (0 = not proven)."""
    lo, hi = np.array(s0, dtype=np.float64), np.array(s0, dtype=np.float64)
    safe_lo = np.array([-X_THRESHOLD, -np.inf, -THETA_THRESHOLD, -np.inf])
    doom = np.zeros(len(params), dtype=np.int64)
    live = np.ones(len(params), dtype=bool)
    for t in range(1, horizon + 1):
        with np.errstate(over="ignore", invalid="ignore"):   # blown-up boxes go inf/nan: never "empty", so no false proof
            lo, hi = interval_step(lo, hi, params)
        lo, hi = np.maximum(lo, safe_lo), np.minimum(hi, -safe_lo)    # survivors are inside the safe set
        empty = (lo > hi).any(axis=1)
        doom[live & empty] = t
        live &= ~empty
        # a box that has blown up (inf/nan, or spinning faster than any real pole could)
        # can never become empty again: stop proving it (stays 0 = not proven)
        blown = ~np.isfinite(lo).all(axis=1) | ~np.isfinite(hi).all(axis=1) | ((hi - lo)[:, 3] > give_up_width)
        live &= ~blown
        lo[~live], hi[~live] = 0.0, 0.0                               # keep dead rows finite
        if not live.any():
            break
    return doom


def split_interval_doom_step(params: np.ndarray, s0: np.ndarray, depth: int,
                             horizon: int = MUST_SURVIVE) -> np.ndarray:
    """Tighter interval proof: branch exactly over the first `depth` pushes (2^depth point
    states, dead branches dropped), then run `interval_doom_step` from every survivor.
    Proven (step = the latest branch's doom step) only if every branch is proven. Boxes
    that start later start as points, so the over-approximation has less time to grow."""
    n = len(params)
    state = np.array(s0, dtype=np.float64)[:, None, :]                 # (n, branches, 4)
    alive = np.ones((n, 1), dtype=bool)
    died_by = np.zeros(n, dtype=np.int64)
    for t in range(1, depth + 1):
        k = state.shape[1]
        f = params[:, _FORCE][:, None] * np.tile([-1.0, 1.0], k)[None, :]
        rep = np.repeat(params, 2 * k, axis=0)
        state = step_physics(np.repeat(state, 2, axis=1).reshape(-1, 4), f.reshape(-1), rep).reshape(n, 2 * k, 4)
        alive = np.repeat(alive, 2, axis=1) & ~terminated(state.reshape(-1, 4)).reshape(n, 2 * k)
        died_by[(~alive.any(axis=1)) & (died_by == 0)] = t
    doom = died_by.copy()
    rest = np.flatnonzero(died_by == 0)
    if len(rest):
        q, b = np.nonzero(alive[rest])
        branch_doom = interval_doom_step(params[rest][q], state[rest][q, b], horizon - depth)
        proven = np.ones(len(rest), dtype=bool)
        latest = np.zeros(len(rest), dtype=np.int64)
        np.logical_and.at(proven, q, branch_doom > 0)
        np.maximum.at(latest, q, branch_doom + depth)
        doom[rest] = np.where(proven, latest, 0)
    return doom


# ---------------------------------------------------------------- combined
def classify(params: np.ndarray, s0: np.ndarray, beam: int = 128, chunk: int = 500, seed: int = 0,
             split_depths: tuple[int, ...] = (6,)) -> dict:
    """Per-question status (WINNABLE / IMPOSSIBLE / UNKNOWN) and the evidence behind it.
    Cheapest first: plain interval proof, then beam search (certificate or exhaustive
    proof), then split interval proofs of growing depth on whatever is still unknown."""
    n = len(params)
    status = np.full(n, UNKNOWN, dtype=object)
    proof = np.full(n, "", dtype=object)
    doom = interval_doom_step(params, s0)
    status[doom > 0], proof[doom > 0] = IMPOSSIBLE, "interval"
    rng = np.random.default_rng(seed)
    todo = np.flatnonzero(doom == 0)
    for start in range(0, len(todo), chunk):
        idx = todo[start:start + chunk]
        survived, exhaustive, actions = beam_search(params[idx], s0[idx], beam=beam, rng=rng)
        verified = survived & replay_survives(params[idx], s0[idx], actions)
        status[idx[verified]], proof[idx[verified]] = WINNABLE, "certificate"
        ex = exhaustive > 0
        status[idx[ex]], proof[idx[ex]] = IMPOSSIBLE, "exhaustive"
    for depth in split_depths:
        idx = np.flatnonzero(status == UNKNOWN)
        if not len(idx):
            break
        d = split_interval_doom_step(params[idx], s0[idx], depth)
        status[idx[d > 0]], proof[idx[d > 0]] = IMPOSSIBLE, f"split-interval (depth {depth})"
        doom[idx[d > 0]] = d[d > 0]
    return {"status": status, "proof": proof, "interval_doom_step": doom}
