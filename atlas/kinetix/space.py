"""Falsification spaces over Kinetix S levels (FLAIROx/Kinetix, ICLR 2025): perturbations of the
10 hand-designed S evaluation levels (kinetix/levels/s/h0..h9) that the trained agents are
evaluated on.

Arena (every S level): floor top y = 0.4, ceiling y = 5.0, walls x = 0 and x = 5; polygons 0-3
are those static walls. Dynamic shapes are polygon 4 and circles 0-1 (shape index 5-6 in joint
and thruster indices). Shape roles: 1 green, 2 blue, 3 red; solved = green touches blue.

  pos   level index, plus a rigid (dx, dy) offset of the green ("agent") component and of the
        blue ("goal") component. A component is the set of dynamic shapes linked by active
        joints; thrusters and joints of a moved shape move with it. A static green/blue shape
        (e.g. the green floor of h5) is not moved, so that offset has no effect.
  phys  level index, plus scales of gravity, friction (dynamic shapes), motor power and
        thruster power, each 0.5-1.5.

An offset that would push a component outside the arena is clipped to the arena (at most TOL
past a wall), so every pos point builds (without clipping ~64% of uniform points left the arena);
the applied offsets are returned by build. A point that makes shapes overlap is caught by Checker ("overlap": a noop step
from reset sends a shape flying, compared with the unperturbed level, since h9 "explode" starts
overlapped on purpose). Solvability is left to the oracle (atlas.kinetix.oracle).
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

N_LEVELS = 10
X_LO, X_HI, Y_LO, Y_HI = 0.0, 5.0, 0.4, 5.0
TOL = 0.1         # some levels already poke past the arena slightly (h0's goal: 0.02 into the ceiling)
_LV = {"level": (0, N_LEVELS)}
SPACES = {
    "pos": {**_LV, "agent_dx": (-1.5, 1.5), "agent_dy": (-1.0, 1.5),
            "goal_dx": (-1.5, 1.5), "goal_dy": (-1.0, 1.5)},
    "phys": {**_LV, "gravity": (0.5, 1.5), "friction": (0.5, 1.5),
             "motor": (0.5, 1.5), "thrust": (0.5, 1.5)},
}
DESC_KEYS = ("level", "dist_ag", "dy_ag", "n_dyn", "n_joint", "n_thr", "grav", "fric", "motor", "thrust")


def level_index(p: dict) -> int:
    return int(min(max(np.floor(p["level"]), 0), N_LEVELS - 1))


def _np(level):
    return jax.tree.map(np.array, level)


def _shapes(lv):
    """Per shape index (polygons then circles): active, dynamic, role, position."""
    P, C = lv.polygon, lv.circle
    act = np.concatenate([P.active, C.active]).astype(bool)
    dyn = np.concatenate([P.inverse_mass, C.inverse_mass]) > 0
    role = np.concatenate([lv.polygon_shape_roles, lv.circle_shape_roles])
    pos = np.concatenate([P.position, C.position])
    return act, dyn, role, pos


def components(lv):
    """Union-find over active dynamic shapes joined by active joints -> component id per shape."""
    act, dyn, _, _ = _shapes(lv)
    parent = list(range(len(act)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    J = lv.joint
    for a, b, on in zip(J.a_index, J.b_index, J.active):
        if on and act[a] and act[b] and dyn[a] and dyn[b]:
            parent[find(int(a))] = find(int(b))
    return np.array([find(i) for i in range(len(act))])


def _extent(lv, i):
    """World AABB (xlo, xhi, ylo, yhi) of shape i."""
    n_p = len(lv.polygon.active)
    if i >= n_p:
        c, r = lv.circle.position[i - n_p], lv.circle.radius[i - n_p]
        return c[0] - r, c[0] + r, c[1] - r, c[1] + r
    P = lv.polygon
    th = P.rotation[i]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    v = P.vertices[i][:P.n_vertices[i]] @ R.T + P.position[i]
    return v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()


def _translate(lv, members, d):
    n_p = len(lv.polygon.active)
    for i in members:
        if i < n_p:
            lv.polygon.position[i] += d
        else:
            lv.circle.position[i - n_p] += d
    m = set(int(i) for i in members)
    for j in range(len(lv.joint.active)):
        if int(lv.joint.a_index[j]) in m:
            lv.joint.global_position[j] += d
    for t in range(len(lv.thruster.active)):
        if int(lv.thruster.object_index[t]) in m:
            lv.thruster.global_position[t] += d


def _move_role(lv, role_id, d, skip=()):
    """Translate the component(s) holding dynamic shapes of role_id by d, clipped so their joint
    AABB stays within TOL of the arena -> (roots, applied offset)."""
    act, dyn, role, _ = _shapes(lv)
    comp = components(lv)
    roots = {comp[i] for i in np.flatnonzero(act & dyn & (role == role_id))} - set(skip)
    members = [i for i in range(len(act)) if act[i] and dyn[i] and comp[i] in roots]
    if not members:
        return roots, (0.0, 0.0)
    ext = np.array([_extent(lv, i) for i in members])
    xlo, xhi, ylo, yhi = ext[:, 0].min(), ext[:, 1].max(), ext[:, 2].min(), ext[:, 3].max()
    dx = float(np.clip(d[0], min(0.0, X_LO - TOL - xlo), max(0.0, X_HI + TOL - xhi)))
    dy = float(np.clip(d[1], min(0.0, Y_LO - TOL - ylo), max(0.0, Y_HI + TOL - yhi)))
    if dx or dy:
        _translate(lv, members, np.array([dx, dy], np.float32))
    return roots, (dx, dy)


def build(space: str, p: dict, eval_levels):
    """Point -> (level EnvState (jax), applied: dict of effective params)."""
    lv = _np(jax.tree.map(lambda x: x[level_index(p)], eval_levels))
    applied = {}
    if space == "pos":
        roots, (adx, ady) = _move_role(lv, 1, (p["agent_dx"], p["agent_dy"]))
        _, (gdx, gdy) = _move_role(lv, 2, (p["goal_dx"], p["goal_dy"]), skip=roots)
        applied = {"agent_dx": adx, "agent_dy": ady, "goal_dx": gdx, "goal_dy": gdy}
    elif space == "phys":
        lv = lv.replace(gravity=lv.gravity * np.float32(p["gravity"]))
        dynp, dync = lv.polygon.inverse_mass > 0, lv.circle.inverse_mass > 0
        lv = lv.replace(
            polygon=lv.polygon.replace(friction=np.where(dynp, lv.polygon.friction * p["friction"], lv.polygon.friction).astype(np.float32)),
            circle=lv.circle.replace(friction=np.where(dync, lv.circle.friction * p["friction"], lv.circle.friction).astype(np.float32)),
            joint=lv.joint.replace(motor_power=(lv.joint.motor_power * p["motor"]).astype(np.float32)),
            thruster=lv.thruster.replace(power=(lv.thruster.power * p["thrust"]).astype(np.float32)))
    else:
        raise ValueError(f"unknown space {space!r}; choose from {tuple(SPACES)}")
    return jax.tree.map(jnp.asarray, lv), applied


def descriptors(level, level_idx=None) -> dict:
    """Interpretable level features (ints, binned //4 by the Recorder): distances in tenths of
    the 5-unit arena, physics in twentieths (grav/fric/motor; a bin = 0.2), thrust in 1/40."""
    lv = _np(level)
    act, dyn, role, pos = _shapes(lv)
    g, b = np.flatnonzero(act & (role == 1)), np.flatnonzero(act & (role == 2))
    dist = dy = None
    if len(g) and len(b):
        dd = pos[b][None] - pos[g][:, None]
        k = np.unravel_index(np.argmin(np.linalg.norm(dd, axis=-1)), dd.shape[:2])
        dist, dy = int(round(10 * np.linalg.norm(dd[k]))), int(round(10 * dd[k][1]))
    J, T = lv.joint, lv.thruster
    jon, ton = J.active.astype(bool), T.active.astype(bool)
    fr = np.concatenate([lv.polygon.friction, lv.circle.friction])[act & dyn]
    mean = lambda x, s: int(round(s * float(x.mean()))) if x.size else 0  # noqa: E731
    return {"level": None if level_idx is None else 4 * int(level_idx), "dist_ag": dist, "dy_ag": dy,
            "n_dyn": int((act & dyn).sum()), "n_joint": int(jon.sum()), "n_thr": int(ton.sum()),
            "grav": int(round(20 * float(np.linalg.norm(lv.gravity)) / 9.81)), "fric": mean(fr, 20),
            "motor": mean(J.motor_power[jon & J.motor_on.astype(bool)], 20), "thrust": mean(T.power[ton], 40)}


def to_str(base_name: str, space: str, p: dict) -> str:
    """Replayable description: rebuild with build(space, params, eval_levels)."""
    return json.dumps({"base": base_name, "space": space, "params": p})


class Checker:
    """Overlap check: after reset and one noop step, no shape may move faster than
    max(vmax, ratio x the unperturbed level's speed). Unperturbed max speeds after one step:
    0.33 (free fall) on 8 levels, 0.62 on h2, 5.6 on h9 (overlapped by design)."""

    def __init__(self, env, env_params, eval_levels, vmax: float = 2.0, ratio: float = 2.0):
        self.vmax, self.ratio = vmax, ratio
        n_act = len(env.action_type.number_of_dims_per_distribution)

        def speed(level):
            _, s = env.reset(jax.random.PRNGKey(0), env_params, level)
            _, s, _, _, _ = env.step(jax.random.PRNGKey(0), s, jnp.zeros(n_act, jnp.int32), env_params)
            v = jnp.concatenate([s.polygon.velocity * s.polygon.active[:, None],
                                 s.circle.velocity * s.circle.active[:, None]])
            return jnp.nan_to_num(jnp.linalg.norm(v, axis=-1), nan=1e9).max()
        self._speed = jax.jit(speed)
        n = len(jax.tree.leaves(eval_levels)[0])
        self.base = [self.speed(jax.tree.map(lambda x: x[i], eval_levels)) for i in range(n)]

    def speed(self, level) -> float:
        return float(self._speed(level))

    def limit(self, level_idx: int) -> float:
        return max(self.vmax, self.ratio * self.base[level_idx])

    def __call__(self, level, level_idx: int):
        return None if self.speed(level) <= self.limit(level_idx) else "overlap"
