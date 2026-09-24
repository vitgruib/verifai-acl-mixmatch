"""Certificates that a question is winnable: an action sequence that passes it, found by
a planner that knows the physics, then replayed through the grader's physics.

The planner is a beam search over action sequences (every child of every kept branch
is stepped; the `beam` best by the environment's `planning_cost` survive). For a
"survive" goal a branch dies when it terminates and a question is won when some branch
is alive after the horizon; for a "reach" goal a question is won as soon as some branch
terminates. The found sequence is replayed (`replay_passes`), so a "winnable" label
never rests on the heuristic. Failing to find one proves nothing.

The horizon matches the grader: a survive question passes when `steps == max_steps`
(dying on the last step still passes, so surviving max_steps - 1 steps is enough); a
reach question passes when it reaches the goal on or before max_steps.
"""
from __future__ import annotations

import numpy as np


def horizon(env) -> int:
    return env.MAX_EPISODE_STEPS - 1 if env.GOAL == "survive" else env.MAX_EPISODE_STEPS


def beam_search(env, params: np.ndarray, s0: np.ndarray, beam: int = 128, rng: np.random.Generator | None = None):
    """Returns (won, exhaustive_death_step, actions). `won[i]`: a passing sequence was
    found. `exhaustive_death_step[i]` (survive goals): the step at which every branch had
    died *without the beam ever dropping a live branch*, a proof of impossibility, else 0.
    `actions[i]` (horizon,) int8: the winning sequence, -1 where unused or none."""
    rng = rng or np.random.default_rng(0)
    A, H, n = env.ACTION_DIM, horizon(env), len(params)
    reach = env.GOAL == "reach"
    cost_fn = env.planning_cost(params)
    state = np.repeat(np.asarray(s0, dtype=np.float64)[:, None, :], beam, axis=1)
    dim = state.shape[2]
    valid = np.zeros((n, beam), dtype=bool)
    valid[:, 0] = True
    truncated = np.zeros(n, dtype=bool)
    death = np.zeros(n, dtype=np.int64)
    won_at = np.zeros(n, dtype=np.int64)                          # reach: step of the first win
    won_child = np.zeros(n, dtype=np.int64)
    parents = np.empty((H, n, beam), dtype=np.int16)
    rep_params = np.repeat(params, A * beam, axis=0)
    actions_of_child = np.tile(np.arange(A), n * beam)            # child c = A * slot + action
    for t in range(1, H + 1):
        parent_state = np.repeat(state, A, axis=1).reshape(-1, dim)
        child = env.step(parent_state, actions_of_child, rep_params).reshape(n, A * beam, dim)
        ended = np.repeat(valid, A, axis=1) & env.terminated(child.reshape(-1, dim)).reshape(n, A * beam)
        if reach:
            first = ended.any(axis=1) & (won_at == 0)
            won_at[first] = t
            won_child[first] = ended[first].argmax(axis=1)
        alive = np.repeat(valid, A, axis=1) & ~ended
        n_alive = alive.sum(axis=1)
        newly_dead = (n_alive == 0) & (death == 0)
        death[newly_dead] = t
        truncated |= n_alive > beam
        cost = cost_fn(child)
        cost = cost * (1.0 + 1e-6 * rng.random(cost.shape))          # break exact ties between twins
        cost[~alive] = np.inf
        keep = np.argpartition(cost, beam - 1, axis=1)[:, :beam]
        parents[t - 1] = keep
        state = np.take_along_axis(child, keep[:, :, None], axis=1)
        valid = np.take_along_axis(alive, keep, axis=1)
        if reach and (won_at > 0).all():
            break

    actions = np.full((n, H), -1, dtype=np.int8)
    if reach:
        won = won_at > 0
        for i in np.flatnonzero(won):
            t, c = int(won_at[i]), int(won_child[i])
            actions[i, t - 1], slot = c % A, c // A
            for u in range(t - 1, 0, -1):
                c = int(parents[u - 1, i, slot])
                actions[i, u - 1], slot = c % A, c // A
        return won, np.zeros(n, dtype=np.int64), actions
    won = valid.any(axis=1)
    for i in np.flatnonzero(won):
        slot = int(np.flatnonzero(valid[i])[0])
        for t in range(H - 1, -1, -1):
            c = int(parents[t, i, slot])
            actions[i, t], slot = c % A, c // A
    return won, np.where(~won & ~truncated, death, 0), actions


def replay_passes(env, params: np.ndarray, s0: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Independent check of a certificate: play the sequence through the grader's physics."""
    state = np.array(s0, dtype=np.float64)
    ok = actions[:, 0] >= 0
    if env.GOAL == "reach":
        reached = np.zeros(len(params), dtype=bool)
        for t in range(actions.shape[1]):
            state = np.where(reached[:, None], state, env.step(state, np.maximum(actions[:, t], 0), params))
            reached |= (actions[:, t] >= 0) & env.terminated(state)
        return ok & reached
    for t in range(actions.shape[1]):
        state = env.step(state, actions[:, t], params)
        ok &= ~env.terminated(state)
    return ok


def certify(env, params: np.ndarray, s0: np.ndarray, beam: int = 128, chunk: int = 200, seed: int = 0) -> np.ndarray:
    """True where a replayed certificate proves the question winnable."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(params), dtype=bool)
    for start in range(0, len(params), chunk):
        sl = slice(start, start + chunk)
        won, _, actions = beam_search(env, params[sl], s0[sl], beam=beam, rng=rng)
        out[sl] = won & replay_passes(env, params[sl], s0[sl], actions)
    return out
