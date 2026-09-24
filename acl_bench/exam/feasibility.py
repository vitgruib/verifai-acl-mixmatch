"""Is a CartPole question winnable at all? Sound proofs both ways, with the grader's
own physics (acl_bench.envs.cartpole.step_physics, deterministic, so a question is a fixed initial
state of a known discrete-time system with two actions).

- WINNABLE (certificate, acl_bench.exam.certify): a beam search over push sequences,
  ranked by each question's LQR cost-to-go, finds one that keeps the pole up for the
  whole episode; it is replayed through the grader's physics and must survive.
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
(docs/exam.md): every question some agent passes is certified winnable (as it
must be), and of the 8,415 that all ten fail, ~87% are proven impossible, ~3% are
certified winnable, and ~11% stay unknown. A wider beam (1024) and deeper splitting
(depth 10, ~45 min) each resolve only a few percent more of the unknowns.
"""
from __future__ import annotations

import numpy as np

from acl_bench.envs import cartpole
from acl_bench.envs.cartpole import (G, TAU, THETA_THRESHOLD, X_THRESHOLD, _FORCE, _LENGTH, _MASSCART, _MASSPOLE,
                                     step_physics, terminated)
from acl_bench.exam.certify import beam_search, horizon, replay_passes

WINNABLE, IMPOSSIBLE, UNKNOWN = "winnable", "impossible", "unknown"
MUST_SURVIVE = horizon(cartpole)


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
        survived, exhaustive, actions = beam_search(cartpole, params[idx], s0[idx], beam=beam, rng=rng)
        verified = survived & replay_passes(cartpole, params[idx], s0[idx], actions)
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
