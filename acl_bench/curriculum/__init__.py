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
    # ---- batch 2 (model-specific): the gradient signal regressed over the task space
    # variance-optimal unbiased sampling with the measured gradient size (no binary-outcome assumption)
    "grad_is": {"mix": {"uniform": 0.25, "sir": 0.75}, "signal": "norm", "alpha": 1.0, "is_power": 1.0},
    # tasks whose update aligns with the uniform tasks' gradient (improves DR's objective)
    "align": {"mix": {"uniform": 0.5, "sir": 0.5}, "signal": "align", "alpha": 1.0},
    # ---- batch 3 (docs/library_log.md): variations on the stage-A survivors
    # unbiased, sharper proposal: does the random-suite gain grow past sqrt(p(1-p))?
    "sir_is_a1": {"mix": {"uniform": 0.25, "sir": 0.75}, "alpha": 1.0, "is_power": 1.0},
    # unbiased, gentler: more uniform share, smaller importance weights
    "sir_is_u50": {"mix": {"uniform": 0.5, "sir": 0.5}, "alpha": 0.5, "is_power": 1.0},
    # fmodel's emphasis as loss weights over uniform tasks (reweighting, not resampling)
    "rw_a4": {"mix": {"uniform": 1.0}, "reweight_alpha": 4.0},
    # sharp sampling from a bootstrap ensemble (+ disagreement bonus)
    "ens_a4": {"mix": {"uniform": 0.5, "sir": 0.5}, "alpha": 4.0, "model": "ensemble", "bonus": 1.0},
    # sharp sampling by model-based learning progress
    "prog_a4": {"mix": {"uniform": 0.5, "sir": 0.5}, "alpha": 4.0, "model": "progress"},
    # ---- batch 4: outcome-variance learnability (VarModel: predicted std of the return
    # under the current policy, 512-episode window) instead of the pass model's p(1-p)
    "var_a2": {"mix": {"uniform": 0.5, "sir": 0.5}, "alpha": 2.0, "model": "var"},
    "var_a4": {"mix": {"uniform": 0.5, "sir": 0.5}, "alpha": 4.0, "model": "var"},
    "var_a8": {"mix": {"uniform": 0.5, "sir": 0.5}, "alpha": 8.0, "model": "var"},
    # unbiased: q ~ predicted std (the variance-optimal proposal), full importance correction
    "var_is": {"mix": {"uniform": 0.25, "sir": 0.75}, "alpha": 1.0, "model": "var", "is_power": 1.0},
}


def build(name: str, env, bounds: np.ndarray, n_slots: int, seed: int) -> Curriculum:
    spec = ARMS[name]
    rng = np.random.default_rng([seed, 11])
    space = TaskSpace(bounds, np.random.default_rng([seed, 12]))
    kind = spec.get("model", "pass")
    if kind == "progress":
        from acl_bench.curriculum.estimators import ProgressModel
        model = ProgressModel(space, seed)
    elif kind == "var":
        from acl_bench.curriculum.estimators import VarModel
        model = VarModel(space, seed, window=spec.get("window", 512))
    elif kind == "ensemble":
        from acl_bench.curriculum.estimators import EnsembleModel
        model = EnsembleModel(space, seed, bonus=spec.get("bonus", 1.0))
    else:
        model = PassModel(space, seed)
    estimators = [model]
    if spec.get("signal"):
        from acl_bench.curriculum.estimators import GradSignal
        model = GradSignal(space, seed, n_slots, spec["signal"])
        estimators = [model]
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
    rw = SIR(space, model, rng, alpha=spec["reweight_alpha"]) if "reweight_alpha" in spec else None
    return Curriculum(space, props, spec["mix"], estimators, rng, is_power=spec.get("is_power", 0.0),
                      reweighter=rw)


__all__ = ["ARMS", "Curriculum", "Episode", "TaskSpace", "build"]
