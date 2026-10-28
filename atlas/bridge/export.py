"""The bridge network's flax params (the patch's ActorCriticMLP) -> an npz in the state-dict layout
of acl_bench.ppo.Agent. Plain numpy, so it runs in .venv-jax (no torch) and .venv (no JAX)."""
from __future__ import annotations

import numpy as np

LAYERS = ("actor_0", "actor_2", "actor_4", "critic_0", "critic_2", "critic_4")


def to_state_dict(params) -> dict[str, np.ndarray]:
    """flax Dense kernels are (in, out), torch Linear weights (out, in)."""
    p = params["params"]
    out = {}
    for layer in LAYERS:
        branch, i = layer.split("_")
        out[f"{branch}.{i}.weight"] = np.asarray(p[layer]["kernel"], np.float32).T.copy()
        out[f"{branch}.{i}.bias"] = np.asarray(p[layer]["bias"], np.float32)
    return out


def save(path, params) -> None:
    np.savez(path, **to_state_dict(params))
