"""Load an SFL-repo JaxNav checkpoint (DR, PLR, minimax, SFL: all the same ActorCriticRNN) and
roll it out on arbitrary single-agent levels with the repo's own env and network.

Needs the SFL repo installed in the venv (`pip install -e $SFL_DIR`, cluster/setup/setup_sfl.sh).
Checkpoints are bare params (model.safetensors), so the env and network config is rebuilt from
the repo's yamls: env/jaxnav.yaml with num_agents=1 (as jaxnav-{sfl,minimax,...}.yaml set it)
and learning/ippo-jaxnav.yaml. Episodes follow the repo's evaluation: stochastic actions
(pi.sample), RNN carry and done flags as in training, one episode per attempt, no auto-reset.
Success is the env's goal_reached (what the repo's GoalR / solved-rate counts).
"""
from __future__ import annotations

import os

os.environ.setdefault("WANDB_MODE", "disabled")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from jaxmarl.environments.jaxnav.jaxnav_env import EnvInstance, JaxNav  # noqa: E402
import sfl.train.common.network as _network  # noqa: E402
from sfl.train.common.network import ActorCriticRNN, ScannedRNN  # noqa: E402
from sfl.train.train_utils.common import load_params  # noqa: E402

_PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(_network.__file__))))  # .../sfl
SFL_DIR = os.path.dirname(os.path.abspath(_PKG))
CFG_DIR = os.path.join(_PKG, "train", "config")


def env_params() -> dict:
    with open(os.path.join(CFG_DIR, "env", "jaxnav.yaml")) as f:
        return yaml.safe_load(f)["env_params"]


def learning_config() -> dict:
    with open(os.path.join(CFG_DIR, "learning", "ippo-jaxnav.yaml")) as f:
        return yaml.safe_load(f)


def make_env() -> JaxNav:
    return JaxNav(num_agents=1, **env_params())


def instance(map_data, start, theta, goal, rew_lambda=0.5) -> EnvInstance:
    """One-agent EnvInstance from an (11, 11) 0/1 grid (indexed [y, x]), world-coordinate start
    and goal (x, y) and heading theta (radians)."""
    return EnvInstance(agent_pos=jnp.asarray([start], jnp.float32),
                       agent_theta=jnp.asarray([theta], jnp.float32),
                       goal_pos=jnp.asarray([goal], jnp.float32),
                       map_data=jnp.asarray(map_data, jnp.int32),
                       rew_lambda=jnp.float32(rew_lambda))


class Policy:
    def __init__(self, ckpt_dir: str, step: int = -1):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.step = step   # SFL saves final params only; kept for the header
        self.env = make_env()
        self.max_steps = int(self.env.max_steps)
        cfg = learning_config()
        self.hidden = int(cfg["HIDDEN_SIZE"])
        self.network = ActorCriticRNN(self.env.agent_action_space().shape[0], config=cfg)
        self.params = load_params(os.path.join(self.ckpt_dir, "model.safetensors"))
        self.config = {"format": "sfl_safetensors", "num_agents": 1, "env_params": env_params(),
                       "learning": cfg, "seed": _seed_from_path(self.ckpt_dir),
                       "sfl_commit": _git(SFL_DIR)}
        self.buffer = None
        self._rollout = jax.jit(jax.vmap(self._episode, in_axes=(None, 0)))

    def _episode(self, inst: EnvInstance, key):
        env, agent = self.env, self.env.agents[0]
        obs, state = env.set_env_instance(inst)
        h = ScannedRNN.initialize_carry(1, self.hidden)

        def step(c, _):
            state, obs, h, done, ret, length, key = c
            key, k_act, k_env = jax.random.split(key, 3)
            x = (obs[agent][None, None, :], done[None, None])
            h, pi, _, _ = self.network.apply(self.params, h, x)
            act = pi.sample(seed=k_act)[0, 0]
            obs2, state2, rew, dones, _ = env.step_env(k_env, state, {agent: act})
            live = ~state.ep_done
            ret = ret + jnp.where(live, rew[agent], 0.0)
            length = length + live.astype(jnp.int32)
            return (state2, obs2, h, dones[agent], ret, length, key), state2.pos[0]

        c0 = (state, obs, h, jnp.bool_(False), jnp.float32(0), jnp.int32(0), key)
        (state, _, _, _, ret, length, _), pos = jax.lax.scan(step, c0, None, self.max_steps)
        return ret, length, state.goal_reached[0], state.move_term[0], pos

    def run(self, inst: EnvInstance, attempts: int, seed: int):
        """-> returns (A,), lengths (A,), success (A,) bool, crashed (A,) bool, pos (T, A, 2)."""
        keys = jax.random.split(jax.random.PRNGKey(seed), attempts)
        ret, length, succ, crash, pos = jax.device_get(self._rollout(inst, keys))
        return (np.asarray(ret), np.asarray(length), np.asarray(succ, bool),
                np.asarray(crash, bool), np.asarray(pos).transpose(1, 0, 2))


def _seed_from_path(p: str):
    b = os.path.basename(p.rstrip("/"))
    return int(b[1:]) if b[:1] == "s" and b[1:].isdigit() else None


def _git(path):
    from atlas.record import git_commit
    return git_commit(path)
