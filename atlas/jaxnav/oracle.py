"""Solvability oracle for single-agent JaxNav levels (cell-centred start and goal on an 11x11 grid).

1. BFS over 4-connected free cells. No path from the start cell to the goal cell -> UNSOLVABLE
   (the agent is a 0.5 m square and cannot leave the grid cells' free space except through
   shared edges, so a 4-disconnected goal is unreachable).
2. Otherwise drive a scripted controller *in the real env* (env.step_env, same dynamics, collision
   check and 500-step limit the policy gets): follow the BFS path through its corner cells,
   braking to each corner, turning on the spot, then driving on. Corridors are 1 m wide and the
   agent's rotation radius is ~0.35 m, so staying on cell centre lines never touches a wall.
   goal_reached without a collision -> SOLVABLE, with the action sequence as witness.
   Anything else -> UNKNOWN (e.g. a winding path that does not fit in 500 steps at this
   controller's speed); the falsifier treats UNKNOWN as invalid, so a counterexample is only
   ever counted on a level that the env itself showed to be solvable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from atlas.jaxnav.space import bfs_path, corners

SOLVABLE, UNSOLVABLE, UNKNOWN = "solvable", "unsolvable", "unknown"
MAX_WP = 64          # padded waypoint list length (a 9x9 interior has < 64 path corners)
A_BRAKE = 0.9        # planned deceleration (env limit is 1.0 m/s^2)
W_BRAKE = 0.9        # planned angular deceleration (env limit is 1.0 rad/s^2)
TURN_TOL = 0.08      # rad: rotate on the spot until heading error is below this
WP_TOL = 0.08        # m: corner reached


def _wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


class Oracle:
    name = "bfs+scripted_controller_in_env"

    def __init__(self, env):
        self.env = env
        self.agent = env.agents[0]
        self.max_steps = int(env.max_steps)
        self.dt = float(env.dt)
        self._drive = jax.jit(self._drive_impl)

    def _drive_impl(self, inst, wps, n_wp):
        env, agent, dt = self.env, self.agent, self.dt
        _, state = env.set_env_instance(inst)

        def step(c, _):
            state, i = c
            pos, theta, vel = state.pos[0], state.theta[0], state.vel[0]
            last = i >= n_wp - 1
            d = jnp.linalg.norm(wps[i] - pos)
            # corner reached (never advance past the goal; the env ends the episode there)
            i = jnp.where((d < WP_TOL) & ~last, i + 1, i)
            last = i >= n_wp - 1
            tgt = wps[i]
            d = jnp.linalg.norm(tgt - pos)
            e = _wrap(jnp.arctan2(tgt[1] - pos[1], tgt[0] - pos[0]) - theta)
            # angular speed profile that can stop at e = 0 under the acceleration limit
            w = jnp.sign(e) * jnp.minimum(env.max_w, jnp.sqrt(2 * W_BRAKE * jnp.abs(e)))
            w = jnp.where(jnp.abs(e) < 0.02, e / dt * 0.5, w)
            v_stop = jnp.sqrt(2 * A_BRAKE * jnp.maximum(d - 0.02, 0.0))
            v = jnp.where(last, env.max_v, jnp.minimum(env.max_v, v_stop))
            v = jnp.where(jnp.abs(e) > TURN_TOL, 0.0, v)
            act = jnp.stack([v, w]).astype(jnp.float32)
            _, state2, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, {agent: act})
            return (state2, i), (act, state2.pos[0], state2.ep_done)

        (state, _), (acts, pos, done) = jax.lax.scan(step, (state, jnp.int32(0)), None,
                                                     self.max_steps)
        return state.goal_reached[0], state.move_term[0], acts, pos, done

    def check(self, inst, start_cell, goal_cell) -> dict:
        """inst: EnvInstance (agent and goal at the centres of start_cell / goal_cell)."""
        grid = np.asarray(inst.map_data)
        path = bfs_path(grid, start_cell, goal_cell)
        if path is None:
            return {"verdict": UNSOLVABLE, "reason": "goal not 4-connected to start"}
        cs = corners(path)
        wps = np.zeros((MAX_WP, 2), np.float32)
        wps[:len(cs)] = np.asarray(cs, np.float32) + 0.5
        wps[len(cs):] = wps[len(cs) - 1]
        goal, crash, acts, pos, done = jax.device_get(
            self._drive(inst, jnp.asarray(wps), jnp.int32(len(cs))))
        steps = int(np.argmax(done)) + 1 if done.any() else self.max_steps
        out = {"path_len": len(path) - 1, "n_corners": len(cs) - 1, "steps": steps}
        if goal and not crash:
            out.update(verdict=SOLVABLE, witness=np.asarray(acts[:steps]).round(3).tolist())
        else:
            out.update(verdict=UNKNOWN, reason="controller crashed" if crash else "controller timed out")
        return out
