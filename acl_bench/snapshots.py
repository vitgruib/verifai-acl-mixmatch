"""Save agent snapshots and grade them later: training as a black box.

`acl_bench.study.run --snapshots DIR` writes one file per run, `DIR/<arm>/<replicate>.npz`,
holding the agent's weights at every check-in (about 37 KB each). Any exam, including
sections built after training, can then be graded against the saved agents with
`python -m acl_bench.study.regrade`, without retraining.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from acl_bench.ppo import Agent


def save_run(path: str, snapshots: dict[int, dict[str, np.ndarray]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{f"{step}|{name}": arr
                                 for step, state in snapshots.items() for name, arr in state.items()})


def load_run(path: str) -> dict[int, Agent]:
    """Rebuild the agent at every saved check-in, ordered by step."""
    import torch
    z = np.load(path)
    states: dict[int, dict] = {}
    for key in z.files:
        step, name = key.split("|", 1)
        states.setdefault(int(step), {})[name] = torch.as_tensor(z[key])
    agents = {}
    for step in sorted(states):
        agent = Agent()
        agent.load_state_dict(states[step])
        agents[step] = agent
    return agents


def find_runs(snapshot_dir: str) -> list[tuple[str, int, str]]:
    """(arm, replicate, path) for every saved run."""
    runs = []
    for path in sorted(glob.glob(os.path.join(snapshot_dir, "*", "*.npz"))):
        stem = os.path.basename(path)[:-4]
        if stem.isdigit():                       # runs are <arm>/<replicate>.npz; ignore anything else
            runs.append((os.path.basename(os.path.dirname(path)), int(stem), path))
    return runs
