"""Falsification spaces over DCD CarRacing Bezier tracks (CarRacingBezierAdversarial levels:
control points in the 333.3 x 333.3 playfield plus a start angle; DCD's level string is
str(tuple(points + [start_alpha])), see atlas.carracing.policy.level_str).

A space maps a point of a VerifAI box (`bounds`, a dict name -> (lo, hi)) to a level:
(points [(x, y), ...], start_alpha or None).

  dr      the training generator: 12 control points uniform in the playfield
          (bezier.get_random_points(n=12, scale=playfield)), start at track index 0. DR's
          mindst resampling is not applied (it rarely rejects and its check is a no-op in effect).
  sketch  the adversary's level space (minimax/PAIRED step_adversary, ACCEL's mutate_level):
          n_ctrl in 4..12 points on the 10 x 10 sketch grid ((i + 1) * playfield / 10), the
          first n_ctrl of 12 proposed cells, duplicates dropped, plus a start angle around
          the control-point centroid (the env starts at the closest track index).

The env ccw-sorts the control points around their centroid before fitting the curve, so
point order never matters and every track is star-shaped. Descriptors are computed from the
same curve the env builds (bezier.get_bezier_curve(rad=0.2, edgy=0.2, numpoints=40), zero
segments dropped), in pure numpy so the training context can featurize thousands of levels.
"""
from __future__ import annotations

import os
import sys

import numpy as np

PLAYFIELD = 2000 / 6.0      # car_racing_bezier.PLAYFIELD
SKETCH = 10                 # CarRacingBezierAdversarial.sketch_dim
N_CTRL = 12
TRACK_WIDTH = 40 / 6.0      # half-width of the road
LEN_UNIT = 25.0             # track_len descriptor unit (world units)
SHARP_R = 15.0              # radius below which a bend counts as sharp (car's min ~7.7)

SPACES = {
    "dr": {f"p{i}_{c}": (0.0, PLAYFIELD) for i in range(N_CTRL) for c in "xy"},
    "sketch": {"n_ctrl": (4, N_CTRL + 1),
               **{f"p{i}_{c}": (0, SKETCH) for i in range(N_CTRL) for c in "xy"},
               "start_alpha": (0.0, 2 * np.pi)},
}


def bezier():
    """DCD's envs.box2d.bezier (DCD_DIR on sys.path, as in atlas.carracing.policy)."""
    d = os.environ.get("DCD_DIR")
    if d and d not in sys.path:
        sys.path.insert(0, d)
    from envs.box2d import bezier as b
    return b


def _i(v, hi):
    return int(min(max(np.floor(v), 0), hi - 1))


def build(space: str, p: dict):
    """Point -> (points, start_alpha, invalid_reason or None)."""
    if space == "dr":
        return [(p[f"p{i}_x"], p[f"p{i}_y"]) for i in range(N_CTRL)], None, None
    if space == "sketch":
        cells = []
        for i in range(_i(p["n_ctrl"], N_CTRL + 1)):
            c = (_i(p[f"p{i}_x"], SKETCH), _i(p[f"p{i}_y"], SKETCH))
            if c not in cells:
                cells.append(c)
        r = PLAYFIELD / SKETCH
        pts = [((i + 1) * r, (j + 1) * r) for i, j in cells]
        alpha = float(np.clip(p["start_alpha"], 0.0, 2 * np.pi))
        return pts, alpha, ("too_few_points" if len(pts) < 4 else None)
    raise ValueError(f"unknown space {space!r}; choose from {tuple(SPACES)}")


def track(points, start_alpha=None):
    """The env's centre line: (xy (n, 2) in raw playfield coords, start index). Mirrors
    CarRacingBezier._create_track_bezier and CarRacingBezierAdversarial._closest_track_index."""
    x, y, _ = bezier().get_bezier_curve(a=np.array(points, float), rad=0.2, edgy=0.2, numpoints=40)
    xy = np.stack([x, y], 1)
    keep = np.any(np.diff(xy, axis=0) != 0, axis=1)
    xy = xy[:-1][keep]
    if start_alpha is None:
        return xy, 0
    u = np.mean(np.array(points, float), axis=0)
    a = np.arctan2(xy[:, 1] - u[1], xy[:, 0] - u[0]) % (2 * np.pi)
    return xy, int(np.argmin(np.abs(a - start_alpha)))


def _radius(xy, half=10.0):
    """Radius of curvature at each point: arc 2*half / heading change across it. A window of
    a road width or so ignores the sub-unit kinks the Bezier fit leaves at control points."""
    d = np.roll(xy, -1, 0) - xy
    seg = np.linalg.norm(d, axis=1)
    head = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    L = seg.sum()
    arc = np.r_[0, np.cumsum(seg)[:-1]]
    ax, hx = _periodic(arc, head, L)
    lo = np.interp(arc - half, ax, hx)
    hi = np.interp(arc + half, ax, hx)
    return 2 * half / np.maximum(np.abs(hi - lo), 1e-6), seg


def _wrapd(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _periodic(arc, head, L):
    """Unwrapped heading extended one lap either side, for windows that cross the seam."""
    wind = head[-1] - head[0] + _wrapd(head[0] - head[-1])      # heading gained per lap
    return np.r_[arc - L, arc, arc + L], np.r_[head - wind, head, head + wind]


def descriptors(points, start_alpha=None) -> dict:
    """Interpretable track features (ints, binned //4 by the Recorder)."""
    xy, s = track(points, start_alpha)
    rad, seg = _radius(xy)
    arc = np.r_[0, np.cumsum(seg)[:-1]]
    L = float(seg.sum())
    sharp = rad < SHARP_R
    # self-overlap: points within a road width of a part of the track > 3 road widths away
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    da = np.abs(arc[:, None] - arc[None])
    da = np.minimum(da, L - da)
    near = ((d < 2 * TRACK_WIDTH) & (da > 6 * TRACK_WIDTH)).any(1)
    # the car drives towards decreasing track index; first 50 units after the start
    ahead = (arc[s] - arc) % L < 50
    return {"n_ctrl": len(points), "track_len": int(L // LEN_UNIT),
            "min_radius": int(min(rad.min(), 99)),
            "sharp_turns": int((sharp & ~np.roll(sharp, 1)).sum()) if not sharp.all() else 1,
            "overlap_pct": int(100 * near.mean()),
            "start_radius": int(min(rad[ahead].min(), 99))}


DESC_KEYS = ("n_ctrl", "track_len", "min_radius", "sharp_turns", "overlap_pct", "start_radius")
