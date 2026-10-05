"""Load a DCD BipedalWalker checkpoint (DR, PLR, Robust PLR, ACCEL, minimax, SFL: all the same
BipedalWalkerStudentPolicy MLP) and roll it out on arbitrary levels with the repo's own env.

Needs the DCD repo (cluster/setup/setup_dcd.sh) on sys.path: set DCD_DIR, or run from inside it.
Nothing is rendered (render is stubbed on every env). Episodes follow the repo's evaluation
(eval.py): a bare BipedalWalkerFull reset_to_level([8 params, seed]), raw 24-d observations,
stochastic Gaussian actions from actor_critic.act fed unprocessed (the env clips them to
[-1, 1]), and the registered 2000-step limit. Success is reaching the end of the terrain
(info['finish']); falling (game_over) or walking off the left edge ends the episode.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("ACL27_SOFT_RENDER", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
if os.environ.get("DCD_DIR") and os.environ["DCD_DIR"] not in sys.path:
    sys.path.insert(0, os.environ["DCD_DIR"])

import numpy as np  # noqa: E402
import torch  # noqa: E402
import envs.bipedalwalker  # noqa: E402
from envs.bipedalwalker.adversarial import BipedalWalkerFull  # noqa: E402
from models.walker_models import BipedalWalkerStudentPolicy  # noqa: E402

MAX_STEPS = 2000          # registered max_episode_steps
POS_EVERY = 10            # hull position trace subsampling
DCD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(envs.bipedalwalker.__file__)))


def make_env():
    env = BipedalWalkerFull()
    env.render = lambda *a, **k: None
    return env


def finish_x(env) -> float:
    """x the hull must pass for info['finish'] (BipedalWalkerCustom.step)."""
    from envs.bipedalwalker import walker_env as w
    return float((w.TERRAIN_LENGTH - w.TERRAIN_GRASS) * w.TERRAIN_STEP)


class Policy:
    def __init__(self, ckpt_dir: str, step: int = -1, model_tar: str = "model"):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.step = step   # DCD keeps the latest model.tar only; kept for the header
        with open(os.path.join(self.ckpt_dir, "meta.json")) as fh:
            self.flags = json.load(fh)["args"]
        if int(self.flags.get("frame_stack", 1)) != 1:
            raise ValueError("frame_stack > 1 not supported")
        env = make_env()
        self.net = BipedalWalkerStudentPolicy(obs_shape=env.observation_space.shape,
                                              action_space=env.action_space, recurrent=False)
        ck = torch.load(os.path.join(self.ckpt_dir, f"{model_tar}.tar"), map_location="cpu",
                        weights_only=False)
        if "runner_state_dict" in ck:
            ck = ck["runner_state_dict"]["agent_state_dict"]["agent"]
        self.net.load_state_dict(ck)
        self.net.eval()
        self.config = {"format": "dcd_model_tar", "args": self.flags,
                       "seed": self.flags.get("seed"), "dcd_commit": _git(DCD_DIR)}
        self._envs = [env]
        self.finish_x = finish_x(env)

    def _env(self, i):
        while len(self._envs) <= i:
            self._envs.append(make_env())
        return self._envs[i]

    @torch.no_grad()
    def run(self, level, attempts: int, seed: int, keep_actions: bool = False):
        """-> dict: returns (A,), lengths (A,), finished (A,) bool, fell (A,) bool (game over or
        off the left edge), progress (A,) max hull x / finish x, pos (T//10, A, 2) hull (x, y)
        every POS_EVERY steps, and with keep_actions a list of (len_i, 4) float32 actions."""
        torch.manual_seed(seed)
        A, T = attempts, MAX_STEPS
        obs = [self._env(i).reset_to_level(list(level)) for i in range(A)]
        alive = np.ones(A, bool)
        ret, length = np.zeros(A), np.zeros(A, int)
        fin, fell, maxx = np.zeros(A, bool), np.zeros(A, bool), np.zeros(A)
        pos = np.full((T // POS_EVERY, A, 2), np.nan, np.float32)
        acts = [[] for _ in range(A)]
        for t in range(T):
            x = torch.as_tensor(np.stack(obs), dtype=torch.float32)
            _, a, _, _ = self.net.act(x, None, None, deterministic=False)
            a = a.reshape(A, -1).numpy().astype(np.float32)
            for i in np.flatnonzero(alive):
                env = self._envs[i]
                obs[i], r, done, info = env.step(a[i])
                if keep_actions:
                    acts[i].append(a[i])
                ret[i] += r
                length[i] += 1
                hx, hy = env.hull.position
                maxx[i] = max(maxx[i], hx)
                if t % POS_EVERY == 0:
                    pos[t // POS_EVERY, i] = (hx, hy)
                if done:
                    alive[i] = False
                    fin[i] = bool(info.get("finish"))
                    fell[i] = not fin[i]
            if not alive.any():
                break
        out = {"returns": ret, "lengths": length, "finished": fin, "fell": fell,
               "progress": np.clip(maxx / self.finish_x, 0, 1), "pos": pos}
        if keep_actions:
            out["actions"] = [np.array(x, np.float32).reshape(-1, 4) for x in acts]
        return out


def _git(path):
    from atlas.record import git_commit
    return git_commit(path)
