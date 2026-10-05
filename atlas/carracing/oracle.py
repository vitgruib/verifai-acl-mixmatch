"""Solvability oracle for DCD CarRacing Bezier tracks.

A level is SOLVABLE when a scripted driver completes a lap *in the real env* (the same
CarRacingBezierAdversarial, Box2D car, tile sensors and 1000-env-step limit the policy gets).
The driver sees the track centre line (privileged), so this certifies that a lap is physically
possible within the limit, not that it is easy from pixels.

Driver: pure pursuit along the centre line in the direction the car faces at the start (the
env spawns the car facing decreasing track index), with a speed limit from the curvature
ahead: v(j) = sqrt(A_LAT / kappa_j), and v* = min_j sqrt(v(j)^2 + 2 A_BRK d_j) over the next
HORIZON units, so the car can brake down to every upcoming corner speed in time; yaw-rate
damping and no gas at large steer (the car is rear-wheel drive and spins at full lock).

Control rate: each action is held for `repeat` env steps. The driver is tried at the policy's
own repeat first, then at half of it, down to 1 (the env's native 50 Hz); the first lap wins
and its repeat is reported as `repeat` (witness: one action per held step). A reactive driver
is much weaker with long holds than a learned policy (it laps few tracks at repeat 8, about
half of the DR tracks at repeat 1), so most certificates come from finer control than the
policy has; filter on `repeat` for counterexamples certified at the policy's own rate.
info['finish'] (every tile visited) within the limit -> SOLVABLE. Anything else -> UNKNOWN; the
falsifier treats UNKNOWN as invalid, so counterexamples are only counted on tracks the env
itself showed to be lappable. Rendering is stubbed on the oracle's own env (physics does not
depend on it). The run is deterministic (env.seed(0) before reset_to_level).
"""
from __future__ import annotations

import numpy as np

from atlas.carracing.policy import MAX_ENV_STEPS, make_env

SOLVABLE, UNSOLVABLE, UNKNOWN = "solvable", "unsolvable", "unknown"
A_LAT = 150.0      # planned lateral acceleration (world units / s^2)
A_BRK = 25.0       # planned braking deceleration
V_MAX = 150.0
HORIZON = 120.0    # look this far ahead (world units) for corners
K_STEER = 2.5
K_YAW = 0.4        # yaw-rate damping (steer per rad/s)
GAS_CUT = 0.7      # no gas above this |steer| (rear-wheel drive oversteers)
FPS = 50.0
PRED = 0.5         # fraction of the action hold to look ahead
LD = 0.35          # pure-pursuit lookahead (s)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Oracle:
    name = "pure_pursuit_in_env"

    def __init__(self, flags: dict | None = None, repeat: int = 8):
        self.base, _ = make_env(flags)
        self.base.render = lambda *a, **k: None
        self.repeat = int(repeat)
        self.repeats = sorted({self.repeat >> i for i in range(8) if self.repeat >> i} | {1},
                              reverse=True)                    # e.g. 8, 4, 2, 1
        self.max_steps = MAX_ENV_STEPS

    def _route(self):
        b = self.base
        pts = np.array([t[2:4] for t in b.track]) - [b.x_offset, b.y_offset]
        n = len(pts)
        p0 = np.array(b.car.hull.position)
        fwd = np.array(b.car.hull.GetWorldVector((0, 1)))
        s = int(np.argmin(np.linalg.norm(pts - p0, axis=1)))
        step = 1 if np.dot(pts[(s + 3) % n] - pts[s], fwd) > 0 else -1
        idx = (s + step * np.arange(n + 40)) % n     # a lap plus some slack
        r = pts[idx]
        seg = np.linalg.norm(np.diff(r, axis=0), axis=1)
        dist = np.concatenate([[0], np.cumsum(seg)])
        head = np.arctan2(*np.diff(r, axis=0).T[::-1])
        dh = np.abs(_wrap(np.diff(head)))
        # curvature over a 3-segment window, robust to tiny Bezier segments
        kap = np.zeros(len(r))
        w = 3
        for i in range(len(dh)):
            lo, hi = max(0, i - w), min(len(dh), i + w + 1)
            kap[i + 1] = dh[lo:hi].sum() / max(seg[lo:hi + 1].sum(), 1e-3)
        return r, dist, np.sqrt(A_LAT / np.maximum(kap, 1e-6))

    def _act(self, r, dist, vcap, k, rep):
        car = self.base.car.hull
        # act on the state predicted half-way through the hold (latency compensation)
        lag = PRED * rep / FPS
        vel = np.array(car.linearVelocity)
        p = np.array(car.position) + vel * lag
        f0 = np.array(car.GetWorldVector((0, 1)))
        a0 = np.arctan2(f0[1], f0[0]) + car.angularVelocity * lag
        fwd = np.array([np.cos(a0), np.sin(a0)])
        v = float(np.hypot(*vel))
        win = slice(k, min(k + 40, len(r)))
        k = k + int(np.argmin(np.linalg.norm(r[win] - p, axis=1)))
        ld = max(6.0, LD * v)
        j = min(int(np.searchsorted(dist, dist[k] + ld)), len(r) - 1)
        d = r[j] - p
        err = _wrap(np.arctan2(d[1], d[0]) - np.arctan2(fwd[1], fwd[0]))
        ahead = (dist >= dist[k]) & (dist <= dist[k] + HORIZON)
        vt = min(V_MAX, float(np.min(np.sqrt(vcap[ahead] ** 2 + 2 * A_BRK * (dist[ahead] - dist[k])))))
        steer = float(np.clip(-K_STEER * err + K_YAW * car.angularVelocity, -1, 1))
        gas = 1.0 if v < vt and (abs(steer) < GAS_CUT or v < 15) else 0.0
        brake = 0.6 if v > vt + 3 else 0.0
        return np.array([steer, gas, brake], np.float32), k

    def _drive(self, level: str, rep: int) -> dict:
        b = self.base
        b.seed(0)
        b.reset_to_level(level)
        r, dist, vcap = self._route()
        k, steps, acts, fin = 0, 0, [], False
        while steps < self.max_steps and not fin:
            a, k = self._act(r, dist, vcap, k, rep)
            acts.append(a)
            for _ in range(rep):
                _, _, done, info = b.step(a)
                steps += 1
                if done or steps >= self.max_steps:
                    break
            fin = bool(info.get("finish"))
            if done and not fin:
                break                                  # left the playfield
        out = {"repeat": rep, "steps": steps,
               "tiles": round(b.tile_visited_count / len(b.track), 4),
               "track_len": round(float(dist[len(b.track)]), 1)}
        if fin:
            out.update(verdict=SOLVABLE, witness=np.round(acts, 3).tolist())
        else:
            out.update(verdict=UNKNOWN, reason="driver left playfield" if steps < self.max_steps
                       else "driver timed out")
        return out

    def check(self, level: str) -> dict:
        """Try the driver at each repeat in self.repeats; first lap wins. On failure returns
        the finest-repeat run with `tiles_by_repeat` for every attempt."""
        tiles = {}
        for rep in self.repeats:
            out = self._drive(level, rep)
            tiles[rep] = out["tiles"]
            if out["verdict"] == SOLVABLE:
                break
        out["tiles_by_repeat"] = tiles
        return out
