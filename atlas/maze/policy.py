"""Load a JaxUED maze checkpoint (DR, PLR, Robust PLR, ACCEL, PAIRED student) and roll it
out on arbitrary levels, using JaxUED's own network and `evaluate_rnn` so behaviour matches
the paper code exactly. Needs JAX and a JaxUED checkout: set JAXUED_DIR (default
third_party/jaxued, as cluster/setup_jaxued.sh creates it)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAXUED_DIR = os.environ.get("JAXUED_DIR", os.path.join(ROOT, "third_party", "jaxued"))
sys.path[:0] = [os.path.join(JAXUED_DIR, "src"), os.path.join(JAXUED_DIR, "examples")]
os.environ.setdefault("WANDB_MODE", "disabled")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
from jaxued.environments import Maze  # noqa: E402
from jaxued.environments.maze import Level  # noqa: E402
from maze_plr import ActorCritic, evaluate_rnn  # noqa: E402


class Policy:
    def __init__(self, ckpt_dir: str, step: int = -1):
        """ckpt_dir = checkpoints/<run_name>/<seed> as written by JaxUED's training scripts."""
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        with open(os.path.join(self.ckpt_dir, "config.json")) as f:
            self.config = json.load(f)
        mgr = ocp.CheckpointManager(os.path.join(self.ckpt_dir, "models"),
                                    item_handlers=ocp.StandardCheckpointHandler())
        self.step = mgr.latest_step() if step == -1 else step
        ckpt = mgr.restore(self.step)
        self.params = ckpt["params"]
        self.buffer = _buffer(ckpt.get("sampler"))
        self.env = Maze(max_height=13, max_width=13, agent_view_size=self.config["agent_view_size"],
                        normalize_obs=True)
        self.env_params = self.env.default_params
        net = ActorCritic(self.env.action_space(self.env_params).n)

        class _TS:  # evaluate_rnn only needs .apply_fn and .params
            apply_fn, params = staticmethod(net.apply), self.params
        self._ts = _TS

        @jax.jit
        def _rollout(rng, level):
            k = jax.tree_util.tree_flatten(level)[0][0].shape[0]
            rng_reset, rng_run = jax.random.split(rng)
            obs, state = jax.vmap(self.env.reset_to_level, (0, 0, None))(
                jax.random.split(rng_reset, k), level, self.env_params)
            states, rewards, lengths = evaluate_rnn(
                rng_run, self.env, self.env_params, _TS, ActorCritic.initialize_carry((k,)),
                obs, state, self.env_params.max_steps_in_episode)
            mask = jnp.arange(self.env_params.max_steps_in_episode)[..., None] < lengths
            return (rewards * mask).sum(0), lengths, states.agent_pos
        self._rollout = _rollout

    def run(self, walls, agent, agent_dir, goal, attempts: int, seed: int):
        """K stochastic attempts on one level -> (returns (K,), lengths (K,), agent_pos (T, K, 2))."""
        lvl = Level(wall_map=jnp.asarray(walls, bool), goal_pos=jnp.asarray(goal, jnp.uint32),
                    agent_pos=jnp.asarray(agent, jnp.uint32), agent_dir=jnp.asarray(agent_dir, jnp.uint8),
                    width=13, height=13)
        batch = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.asarray(x)[None], attempts, 0), lvl)
        r, l, pos = self._rollout(jax.random.PRNGKey(seed), batch)
        return np.asarray(r), np.asarray(l), np.asarray(pos)


def _buffer(sampler):
    """The PLR/ACCEL level buffer saved in the checkpoint (None for DR/PAIRED)."""
    if sampler is None:
        return None
    n = int(np.asarray(sampler["size"]))
    if n == 0:
        return None
    lv = sampler["levels"]
    return {"walls": np.asarray(lv["wall_map"][:n], bool), "agent": np.asarray(lv["agent_pos"][:n]),
            "goal": np.asarray(lv["goal_pos"][:n]), "scores": np.asarray(sampler["scores"][:n]),
            "timestamps": np.asarray(sampler["timestamps"][:n])}


def replay_actions(env, env_params, walls, agent, agent_dir, goal, actions: str):
    """Replay an oracle certificate (atlas.maze.oracle) in the real JaxUED env.
    -> (total reward, steps until done or len(actions)). Reward > 0 means the goal was reached."""
    from atlas.maze.oracle import ACTIONS
    lvl = Level(wall_map=jnp.asarray(walls, bool), goal_pos=jnp.asarray(goal, jnp.uint32),
                agent_pos=jnp.asarray(agent, jnp.uint32), agent_dir=jnp.asarray(agent_dir, jnp.uint8),
                width=13, height=13)
    rng = jax.random.PRNGKey(0)
    _, state = env.reset_to_level(rng, lvl, env_params)
    total, steps = 0.0, 0
    for a in actions:
        _, state, r, done, _ = env.step(rng, state, ACTIONS[a], env_params)
        total, steps = total + float(r), steps + 1
        if bool(done):
            break
    return total, steps
