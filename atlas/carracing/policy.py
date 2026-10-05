"""Load a DCD CarRacing checkpoint (DR, PLR, Robust PLR, ACCEL, minimax, SFL: all the same
CarRacingNetwork) and roll it out on arbitrary Bezier tracks with the repo's own env and wrapper.

Needs the DCD repo (cluster/setup/setup_dcd.sh) on sys.path: set DCD_DIR, or run from inside it.
Rendering must be headless: set ACL27_SOFT_RENDER=1 (patched numpy renderer, no window, no GL);
this module sets it if unset. Episodes follow the repo's evaluation (eval.py Evaluator.make_env):
CarRacingWrapper with the checkpoint's grayscale / frame_stack / action-repeat / crop flags,
reward_shaping False, channels-first obs, stochastic Beta actions (actor_critic.act), and the
registered 1000-env-step limit (= 1000 / action_repeat agent steps, the TimeLimit DCD trains with).
Success is a completed lap (every tile visited; info['finish'], the env's own done condition).
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
import envs.box2d  # noqa: E402,F401  (registers CarRacing-Bezier-Adversarial-v0)
from envs.registration import make as gym_make  # noqa: E402
from envs.wrappers import CarRacingWrapper  # noqa: E402
from models.car_racing_models import CarRacingNetwork  # noqa: E402

ENV_NAME = "CarRacing-Bezier-Adversarial-v0"
MAX_ENV_STEPS = 1000     # registered max_episode_steps
DCD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(envs.box2d.__file__)))


def make_env(flags: dict | None = None, seed: int = 0):
    """Bare CarRacingBezierAdversarial (no gym TimeLimit: the step limit is counted here) and the
    eval wrapper around it."""
    f = flags or {}
    base = gym_make(ENV_NAME, n_control_points=int(f.get("num_control_points", 12))).unwrapped
    base.seed(seed)
    env = CarRacingWrapper(base, grayscale=bool(f.get("grayscale", False)), reward_shaping=False,
                           num_action_repeat=int(f.get("num_action_repeat", 8)),
                           nstack=int(f.get("frame_stack", 4)), crop=bool(f.get("crop_frame", False)),
                           eval_=True)
    return base, env


def level_str(points, start_alpha=None) -> str:
    """DCD's level encoding (CarRacingBezierAdversarial.level / reset_to_level)."""
    return str(tuple([(float(x), float(y)) for x, y in points] + [start_alpha]))


class Policy:
    def __init__(self, ckpt_dir: str, step: int = -1, model_tar: str = "model"):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.step = step   # DCD keeps the latest model.tar only; kept for the header
        with open(os.path.join(self.ckpt_dir, "meta.json")) as fh:
            self.flags = json.load(fh)["args"]
        self.repeat = int(self.flags.get("num_action_repeat", 8))
        self.max_steps = MAX_ENV_STEPS // self.repeat
        self.base, self.env = make_env(self.flags)
        h, w, c = self.env.observation_space.shape
        self.net = CarRacingNetwork((c, h, w), self.base.action_space, hidden_size=100,
                                    crop=bool(self.flags.get("crop_frame", False)))
        ck = torch.load(os.path.join(self.ckpt_dir, f"{model_tar}.tar"), map_location="cpu",
                        weights_only=False)
        if "runner_state_dict" in ck:
            ck = ck["runner_state_dict"]["agent_state_dict"]["agent"]
        self.net.load_state_dict(ck)
        self.net.eval()
        self.config = {"format": "dcd_model_tar", "args": self.flags,
                       "seed": self.flags.get("seed"), "dcd_commit": _git(DCD_DIR)}
        self._envs = [(self.base, self.env)]

    def _env(self, i):
        while len(self._envs) <= i:
            self._envs.append(make_env(self.flags))
        return self._envs[i]

    @torch.no_grad()
    def run(self, level: str, attempts: int, seed: int):
        """-> returns (A,), lengths (A,) in agent steps, finished (A,) bool, offtrack (A,) bool
        (left the playfield), tiles (A,) fraction of tiles visited, pos (T, A, 2) car position."""
        torch.manual_seed(seed)
        A, T = attempts, self.max_steps
        obs, alive = [], np.ones(A, bool)
        for i in range(A):
            base, env = self._env(i)
            base.seed(seed + i)
            obs.append(env.reset_to_level(level))
        ret, length = np.zeros(A), np.zeros(A, int)
        fin, off = np.zeros(A, bool), np.zeros(A, bool)
        pos = np.full((T, A, 2), np.nan, np.float32)
        for t in range(T):
            x = torch.as_tensor(np.stack(obs), dtype=torch.float32).permute(0, 3, 1, 2)
            _, a, _, _ = self.net.act(x, None, None)
            a = self.net.process_action(a.reshape(A, -1).numpy())
            for i in np.flatnonzero(alive):
                base, env = self._envs[i]
                obs[i], r, done, info = env.step(a[i])
                ret[i] += r
                length[i] += 1
                pos[t, i] = base.car.hull.position
                if done:
                    alive[i] = False
                    fin[i] = bool(info.get("finish"))
                    off[i] = not fin[i]
            if not alive.any():
                break
        tiles = np.array([b.tile_visited_count / len(b.track) for b, _ in self._envs[:A]])
        return ret, length, fin, off, tiles, pos


def _git(path):
    from atlas.record import git_commit
    return git_commit(path)
