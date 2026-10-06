"""Load a Kinetix (FLAIROx/Kinetix) checkpoint and run it on arbitrary levels, headless.

A training run (cluster/2_kinetix/train.sbatch) leaves in its run dir:
  .hydra/config.yaml                                     the Hydra config (hydra.run.dir = run dir)
  <run_name>-<hash>-<steps>/full_model.pbz2              {step, params, opt_state, extra}
The config is rebuilt with Kinetix's own normalise_config, the env with make_kinetix_env (no
auto-reset, so an episode ends at its first done) and the network with make_network_from_config.
Never calls a renderer: this module opens no windows.

Episodes: an attempt is solved when it ends (done) with GoalR (a green shape touched a blue one).
Rollouts record the actions taken, so a solved attempt is itself a replayable witness
(atlas.kinetix.oracle.replay).
"""
from __future__ import annotations

import glob
import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.serialization import to_state_dict

from kinetix.environment.env import make_kinetix_env
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.util.config import generate_params_from_config, normalise_config
from kinetix.util.learning import rms_normalise
from kinetix.util.saving import load_evaluation_levels, load_params


def find_model(ckpt_dir: str) -> str:
    """Latest full_model.pbz2 under a run dir (largest step suffix, then newest)."""
    if ckpt_dir.endswith(".pbz2"):
        return ckpt_dir
    fs = glob.glob(os.path.join(ckpt_dir, "**", "full_model.pbz2"), recursive=True)
    if not fs:
        raise FileNotFoundError(f"no full_model.pbz2 under {ckpt_dir}")

    def key(f):
        m = re.search(r"-([0-9.]+)B?$", os.path.basename(os.path.dirname(f)))
        return (float(m.group(1)) if m else -1.0, os.path.getmtime(f))
    return max(fs, key=key)


def find_config(ckpt_dir: str) -> str:
    for d in (ckpt_dir, os.path.dirname(ckpt_dir), os.path.dirname(os.path.dirname(ckpt_dir))):
        f = os.path.join(d, ".hydra", "config.yaml")
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"no .hydra/config.yaml at or above {ckpt_dir}")


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw.setdefault("misc", {})["use_wandb"] = False
    return normalise_config(raw, "atlas", save_config=False)


def make_env(config, static_env_params, env_params):
    return make_kinetix_env(config["action_type"], config["observation_type"], None, env_params,
                            static_env_params, auto_reset=False)


class Policy:
    def __init__(self, ckpt_dir: str, config_path: str | None = None):
        self.ckpt_dir = ckpt_dir
        self.model_path = find_model(ckpt_dir)
        self.config = load_config(config_path or find_config(ckpt_dir))
        self.env_params, self.static_env_params = generate_params_from_config(self.config)
        self.config["env_params"] = to_state_dict(self.env_params)            # as plr.py / sfl.py
        self.config["static_env_params"] = to_state_dict(self.static_env_params)
        self.env = make_env(self.config, self.static_env_params, self.env_params)
        self.max_steps = int(self.env_params.max_timesteps)
        d = load_params(self.model_path)
        self.params, self.step = d["params"], int(np.asarray(d["step"]))
        extra = d.get("extra") or {}
        self.rms = extra.get("rms") if self.config.get("rms_norm") else None
        self.network = make_network_from_config(self.env, self.env_params, self.config)
        self.eval_names = list(self.config["eval_levels"])
        self.eval_levels, eval_static = load_evaluation_levels(self.eval_names)
        assert eval_static == self.static_env_params, (eval_static, self.static_env_params)
        self._run = jax.jit(self._rollouts, static_argnums=(2,))

    def eval_level(self, i: int):
        return jax.tree.map(lambda x: x[i], self.eval_levels)

    def _rollouts(self, level, key, n: int):
        env, ep = self.env, self.env_params
        k_reset, k_run = jax.random.split(key)
        obs, state = jax.vmap(env.reset, (0, None, None))(jax.random.split(k_reset, n), ep, level)
        h = ScannedRNN.initialize_carry(n)

        def step(c, _):
            key, h, obs, state, done, alive = c
            key, ka, ks = jax.random.split(key, 3)
            o = rms_normalise(self.rms, obs, flatten=True) if self.rms is not None else obs
            h, pi, _ = self.network.apply(self.params, h, jax.tree.map(lambda x: x[None], (o, done)))
            act = pi.sample(seed=ka).squeeze(0)
            obs, state, r, d, info = jax.vmap(env.step, (0, 0, 0, None))(jax.random.split(ks, n), state, act, ep)
            solved = alive & d & info["GoalR"]
            return (key, h, obs, state, d, alive & ~d), (act, r * alive, alive, solved)

        init = (k_run, h, obs, state, jnp.zeros(n, bool), jnp.ones(n, bool))
        _, (acts, rews, alive, solved) = jax.lax.scan(step, init, None, self.max_steps)
        return acts, rews.sum(0), alive.sum(0), solved.any(0)

    def run(self, level, attempts: int, seed: int):
        """-> returns (A,), lengths (A,), solved (A,) bool, actions (T, A, ...)."""
        acts, rets, lens, solved = self._run(level, jax.random.PRNGKey(seed), attempts)
        return np.asarray(rets), np.asarray(lens), np.asarray(solved), np.asarray(acts)
