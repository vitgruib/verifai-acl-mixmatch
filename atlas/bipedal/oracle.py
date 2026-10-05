"""Solvability oracle for DCD BipedalWalker levels.

A level is SOLVABLE when some action sequence reaches the end of the terrain (info['finish'])
*in the real env* (the same BipedalWalkerFull, Box2D walker and 2000-step limit the policy
gets) and the sequence replays to a finish from reset_to_level. Levels are deterministic
(terrain and initial push come from the level's seed), so a witness replays exactly.

Witnesses come from up to two sources, tried in order:

  scripted   a backtracking search over a hand-written gait controller (a port of gym's
             BipedalWalkerHeuristics with hip gain / speed / knee variants, see `ctrl`), in
             100-step segments: when a segment falls the search backs up and tries the next
             variant. It certifies flat and mildly rough levels (probes: flat and roughness 2
             in 15k-200k env steps), and nothing with stumps, stairs, pits or roughness >= 4,
             so it only runs on levels without active obstacles and roughness <= SCRIPT_ROUGH.
  reference  rollouts of other trained checkpoints (--ref-ckpt): any finished attempt is a
             witness (its action sequence, float32, replayed to verify).

Anything else -> UNKNOWN. The falsifier additionally self-certifies a level when the policy
under test finishes it in one of its own attempts (source 'self'; the finishing attempt is the
witness). A level the policy never finishes can therefore only be a counterexample when the
scripted walker or a reference checkpoint certified it: 'hard' counterexamples (solve rate 0)
need an independent witness. Rendering is stubbed on every env.
"""
from __future__ import annotations

import numpy as np

from atlas.bipedal import space
from atlas.bipedal.policy import MAX_STEPS, make_env

SOLVABLE, UNSOLVABLE, UNKNOWN = "solvable", "unsolvable", "unknown"
SCRIPT_ROUGH = 3.0        # scripted walker only runs at or below this roughness, no obstacles
SCRIPT_BUDGET = 200_000   # env steps (about 9 s)
SEG = 100
VARS = [(sp, k, hg) for hg in (1.5, 0.9, 2.5) for sp in (0.29, 0.15) for k in (0.1, 0.2)]


def ctrl(s, c, th):
    """Gym's BipedalWalkerHeuristics state machine (stay on one leg, put other down, push off),
    with variant thresholds th = (speed, max supporting knee angle, hip gain). c = (state,
    moving leg, supporting knee angle). s is the 24-d observation."""
    sp, ska_max, hg = th
    st, ml, ska = c
    sl = 1 - ml
    mb, sb = 4 + 5 * ml, 4 + 5 * sl
    ht, kt = [None, None], [None, None]
    hd, kd = [0.0, 0.0], [0.0, 0.0]
    if st == 1:
        ht[ml], kt[ml] = 1.1, -0.6
        ska += 0.03
        if s[2] > sp:
            ska += 0.03
        ska = min(ska, ska_max)
        kt[sl] = ska
        if s[sb] < 0.10:
            st = 2
    if st == 2:
        ht[ml], kt[ml], kt[sl] = 0.1, ska_max, ska
        if s[mb + 4]:
            st = 3
            ska = min(s[mb + 2], ska_max)
    if st == 3:
        kt[ml], kt[sl] = ska, 1.0
        if s[sb + 2] > 0.88 or s[2] > 1.2 * sp:
            st = 1
            ml = 1 - ml
    for i, b in ((0, 4), (1, 9)):
        if ht[i] is not None:
            hd[i] = hg * (ht[i] - s[b]) - 0.25 * s[b + 1]
        if kt[i] is not None:
            kd[i] = 4.0 * (kt[i] - s[b + 2]) - 0.25 * s[b + 3]
        hd[i] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        kd[i] -= 15.0 * s[3]
    a = np.clip(0.5 * np.array([hd[0], kd[0], hd[1], kd[1]]), -1, 1)
    return a.astype(np.float32), (st, ml, ska)


class Oracle:
    name = "layered_witness"

    def __init__(self, refs=(), ref_attempts: int = 4, script_budget: int = SCRIPT_BUDGET):
        self.env = make_env()
        self.refs = list(refs)                 # atlas.bipedal.policy.Policy objects
        self.ref_attempts = int(ref_attempts)
        self.script_budget = int(script_budget)
        self.max_steps = MAX_STEPS

    def scriptable(self, level) -> bool:
        return level[0] <= SCRIPT_ROUGH and not any(space.active(level).values())

    def search(self, level):
        """Backtracking search over gait variants. -> (actions or None, env steps used, best x)."""
        env, used, best = self.env, 0, 0.0
        stack = [([], (1, 0, 0.1), 0)]        # (action prefix, controller state, next variant)
        while stack and used < self.script_budget:
            acts, c, vi = stack[-1]
            if vi >= len(VARS):
                stack.pop()
                continue
            stack[-1] = (acts, c, vi + 1)
            s = env.reset_to_level(list(level))
            for a in acts:
                s, *_ = env.step(a)
            used += len(acts)
            a, cc = (np.zeros(4, np.float32), c) if not acts else ctrl(s, c, VARS[vi])
            new, ok = list(acts), True
            for _ in range(SEG):
                s, _, done, info = env.step(a)
                new.append(a)
                used += 1
                best = max(best, env.hull.position.x)
                if done:
                    if info.get("finish"):
                        return new, used, best
                    ok = False
                    break
                if len(new) >= self.max_steps:
                    ok = False
                    break
                a, cc = ctrl(s, cc, VARS[vi])
            if ok:
                stack.append((new, cc, 0))
        return None, used, best

    def replay(self, level, actions) -> bool:
        """True when the action sequence finishes the level from reset_to_level."""
        env = self.env
        env.reset_to_level(list(level))
        for t, a in enumerate(actions):
            if t >= self.max_steps:
                return False
            _, _, done, info = env.step(np.asarray(a, np.float32))
            if done:
                return bool(info.get("finish"))
        return False

    def check(self, level, seed: int = 0) -> dict:
        out = {"verdict": UNKNOWN, "source": None, "scriptable": self.scriptable(level)}
        if out["scriptable"]:
            acts, used, best = self.search(level)
            out.update(script_steps=used, script_best_x=round(best, 2))
            if acts is not None and self.replay(level, acts):
                out.update(verdict=SOLVABLE, source="scripted", witness=np.array(acts, np.float32))
                return out
        for j, ref in enumerate(self.refs):
            r = ref.run(level, self.ref_attempts, seed + 7919 * (j + 1), keep_actions=True)
            for i in np.flatnonzero(r["finished"]):
                if self.replay(level, r["actions"][i]):
                    out.update(verdict=SOLVABLE, source=f"ref{j}", ref=ref.ckpt_dir,
                               witness=r["actions"][i])
                    return out
            out[f"ref{j}_progress"] = round(float(r["progress"].max()), 4)
        return out
