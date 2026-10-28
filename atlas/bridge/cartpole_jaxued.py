"""Train one CartPole bridge run with JaxUED's own PLR script (examples/gymnax/gymnax_plr.py, with
cluster/patches/jaxued_cartpole_bridge.patch) at ACL27's settings, then export the agent for the
PyTorch grader. Runs in .venv-jax (no torch):

    python -m atlas.bridge.cartpole_jaxued --arm PLR --seed 1 --steps 614400 --out DIR

writes DIR/params.npz (acl_bench.ppo.Agent's state-dict layout, atlas/bridge/export.py) and
DIR/config.json (the arm, seed, steps, every flag passed to JaxUED, wall time and the final
level-buffer stats). The settings mirror acl_bench/plr/fast.py FastConfig (except the rollout
length, below) and acl_bench/suite/arms.py; tests/test_bridge.py checks they still agree.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from atlas.bridge import export

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAXUED_DIR = os.environ.get("JAXUED_DIR", os.path.join(ROOT, "third_party", "jaxued"))
SCRIPT = os.path.join(JAXUED_DIR, "examples", "gymnax", "gymnax_plr.py")

# acl_bench/plr/fast.py FastConfig, except the rollout: 8 envs x 512 steps (JaxUED's gymnax length;
# 4096 samples per update) rather than our 128, because JaxUED's loop restarts every env each update
# and scores a level -inf unless an episode ends inside the rollout; CartPole episodes run 500 steps
PPO_FLAGS = {"num_train_envs": 8, "num_steps": 512, "num_minibatches": 4, "epoch_ppo": 4, "lr": 3e-4,
             "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2, "max_grad_norm": 0.5,
             "entropy_coeff": 0.01, "critic_coeff": 0.5}
# acl_bench/suite/arms.py _PLR; our sampler replays once buffer // 10 levels are stored
PLR_FLAGS = {"score_function": "MaxMC", "level_buffer_capacity": 1000, "temperature": 0.3,
             "staleness_coeff": 0.3, "minimum_fill_ratio": 0.1, "prioritization": "rank"}
ARM_FLAGS = {"DR": {"replay_prob": 0.0, "exploratory_grad_updates": True},
             "PLR": {"replay_prob": 0.8, "exploratory_grad_updates": True},
             "RPLR": {"replay_prob": 0.8, "exploratory_grad_updates": False}}
# our network and PPO details (the patch's options), one seed per process, no orbax checkpoints
FIXED_FLAGS = {"env": "cartpole_acl27", "use_rnn": False, "acl27_mlp": True, "clip_value": False,
               "anneal_lr": False, "shuffle_samples": True, "num_repeats": 1,
               "checkpoint_save_interval": 0, "eval_num_attempts": 1}
N_LOGS = 10            # JaxUED evaluates and logs this many times per run
UPDATE = PPO_FLAGS["num_train_envs"] * PPO_FLAGS["num_steps"]


def flags(arm: str, seed: int, steps: int, params_out: str) -> dict:
    return {**FIXED_FLAGS, **PPO_FLAGS, **PLR_FLAGS, **ARM_FLAGS[arm], "seed": seed,
            "num_env_steps": steps, "eval_freq": steps // UPDATE // N_LOGS,
            "run_name": f"cartpole_bridge_{arm}_{seed}", "params_out": params_out}


def argv(f: dict) -> list[str]:
    out = []
    for k, v in f.items():
        if isinstance(v, bool):
            out.append(f"--{k}" if v else f"--no-{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_FLAGS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(args)
    if a.steps <= 0 or a.steps % (UPDATE * N_LOGS):
        raise SystemExit(f"--steps must be a positive multiple of {UPDATE * N_LOGS}, got {a.steps}")
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.abspath(os.path.join(a.out, "jaxued_final.npz"))
    f = flags(a.arm, a.seed, a.steps, raw)
    env = {**os.environ, "WANDB_MODE": "disabled", "WANDB_SILENT": "true"}
    t0 = time.time()
    subprocess.run([sys.executable, SCRIPT, *argv(f)], cwd=a.out, env=env, check=True)
    wall = time.time() - t0

    with np.load(raw) as z:
        flat = dict(z)
    params = {}
    for key, value in flat.items():
        if key.startswith("params/"):
            _, layer, name = key.split("/")
            params.setdefault(layer, {})[name] = value
    export.save(os.path.join(a.out, "params.npz"), {"params": params})
    scores = flat["sampler/scores"]
    config = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "wall_time_sec": wall, "jaxued_flags": f,
              "jaxued_dir": JAXUED_DIR,
              "buffer": {"size": int(flat["sampler/size"]), "finite_scores": int(np.isfinite(scores).sum()),
                         "episode_count": int(flat["sampler/episode_count"])},
              "num_dr_updates": int(flat["num_dr_updates"]), "num_replay_updates": int(flat["num_replay_updates"])}
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)
    os.remove(raw)
    print(json.dumps({k: config[k] for k in ("arm", "seed", "steps", "wall_time_sec", "buffer",
                                             "num_dr_updates", "num_replay_updates")}))


if __name__ == "__main__":
    main()
