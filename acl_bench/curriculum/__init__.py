"""A mix-and-match curriculum library (docs/library.md): proposers (uniform, SIR over a
pass-rate model, a PLR-style replay buffer, ACCEL-style mutation, a VerifAI sampler through
Scenic) mixed with fixed weights, sharing estimators, with optional importance correction.

    from acl_bench.curriculum import build
    cur = build("sir_a1", env, bounds, n_slots, seed)
    params, source, weight = cur.propose(k)
    ...
    cur.report(k, Episode(params, success, ret, length, source))

Arms are specs in ARMS: {"mix": {proposer: weight}, proposer options..., "is_power": x}.
"""
from __future__ import annotations

import numpy as np

from acl_bench.curriculum.core import Curriculum, Episode, TaskSpace
from acl_bench.curriculum.estimators import PassModel
from acl_bench.curriculum.proposers import SIR, Mutate, Replay, Uniform, VerifAISurrogate

# Stage A, batch 1 (docs/library_log.md): one hypothesis per arm.
ARMS: dict[str, dict] = {
    # soft frontier sampling from the pass model (SFL's signal, no scouting)
    "sir_a1": {"mix": {"uniform": 0.5, "sir": 0.5}, "alpha": 1.0},
    # sharper: close to the old best-of-64 argmax picker (fmodel), but stochastic
    "sir_a4": {"mix": {"uniform": 0.5, "sir": 0.5}, "alpha": 4.0},
    # variance-optimal, unbiased for DR's objective: q ~ sqrt(p(1-p)), full importance correction
    "sir_is": {"mix": {"uniform": 0.25, "sir": 0.75}, "alpha": 0.5, "is_power": 1.0},
    # frontier-biased but half-corrected: between SFL-style bias and DR's objective
    "sir_is05": {"mix": {"uniform": 0.25, "sir": 0.75}, "alpha": 1.0, "is_power": 0.5},
    # PLR buffer ranked by posterior learnability (pass-model prior + the task's own visits)
    "replay_post": {"mix": {"uniform": 0.5, "replay": 0.5}},
    # ACCEL-style: edits of tasks the posterior buffer would replay
    "mutate_post": {"mix": {"uniform": 0.5, "replay": 0.25, "mutate": 0.25}},
    # VerifAI's cross-entropy sampler searching the pass model (4 search steps per task)
    "verifai_surr": {"mix": {"uniform": 0.5, "verifai": 0.5}, "sampler": "ce", "inner": 4},
}


def build(name: str, env, bounds: np.ndarray, n_slots: int, seed: int) -> Curriculum:
    spec = ARMS[name]
    rng = np.random.default_rng([seed, 11])
    space = TaskSpace(bounds, np.random.default_rng([seed, 12]))
    model = PassModel(space, seed)
    props = {}
    for p in spec["mix"]:
        if p == "uniform":
            props[p] = Uniform(space)
        elif p == "sir":
            props[p] = SIR(space, model, rng, alpha=spec.get("alpha", 1.0), n_candidates=spec.get("candidates", 64))
        elif p == "replay":
            props[p] = Replay(space, model, rng, alpha=spec.get("replay_alpha", 4.0))
        elif p == "mutate":
            props[p] = Mutate(space, props["replay"], rng, sigma=spec.get("sigma", 0.05))
        elif p == "verifai":
            props[p] = VerifAISurrogate(env, space, model, n_slots, seed, spec.get("sampler", "ce"),
                                        spec.get("inner", 4))
        else:
            raise ValueError(p)
    if "mutate" in props and "replay" not in spec["mix"]:
        raise ValueError("mutate needs a replay buffer in the mix")
    return Curriculum(space, props, spec["mix"], [model], rng, is_power=spec.get("is_power", 0.0))


__all__ = ["ARMS", "Curriculum", "Episode", "TaskSpace", "build"]
