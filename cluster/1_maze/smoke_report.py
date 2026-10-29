"""After the Maze smoke test (local mode): pick JOBS_PER_GPU and estimate when the full run ends.

    python cluster/1_maze/smoke_report.py <smoke runs dir>

prints the report on stderr and the chosen JOBS_PER_GPU on stdout. JOBS_PER_GPU: the environment
variable if set, else as many smoke-sized jobs as fit in 90% of the smallest GPU's memory with 25%
headroom over the largest peak the smoke jobs reached (gpu_mem.csv, sampled while they ran one per
GPU with XLA_PYTHON_CLIENT_PREALLOCATE=false), at most 4; 1 without a GPU. Speeds: JaxUED's from
the times of the smoke runs' checkpoints (DR stands in for ref; RPLR for PLR and ACCEL; minimax,
which also trains its antagonist, is taken as twice DR); SFL's from the time between its saves after
each smoke round, a fifth of a full round (submit.sh). Exam and grading scale the smoke's
times. The run sizes come from the same variables train.sbatch reads.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from atlas.maze.grade import SFL_RECIPE

MAX_JOBS = 4
TASK_SPEED = {"ref": ("dr", 1), "dr": ("dr", 1), "minimax": ("dr", 2), "plr": ("rplr", 1),
              "rplr": ("rplr", 1), "accel": ("rplr", 1)}
FULL_EXAM_LEVELS = 300 + 14 + 3000   # random, custom, VerifAI cap


def gpus() -> list[int]:
    """Total memory (MiB) of each GPU nvidia-smi lists."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [int(x) for x in out.split()]


def peak_mib(path: Path) -> int | None:
    """Largest rise of any GPU's memory.used over its lowest sample while the smoke jobs ran."""
    used = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[1].isdigit():
                used.setdefault(parts[0], []).append(int(parts[1]))
    return max((max(u) - min(u) for u in used.values()), default=None)


def jaxued_sec_per_update(runs: Path, algo: str) -> float:
    """Seconds per update between the first and last checkpoint of the smoke run."""
    (d,) = glob.glob(str(runs / "maze_train" / "checkpoints" / f"maze_{algo}_n*" / "0"))
    cfg = json.loads((Path(d) / "config.json").read_text())
    steps = sorted(int(p.name) for p in (Path(d) / "models").iterdir() if p.name.isdigit())
    t = {s: (Path(d) / "models" / str(s)).stat().st_mtime for s in (steps[0], steps[-1])}
    return (t[steps[-1]] - t[steps[0]]) / ((steps[-1] - steps[0]) * cfg["eval_freq"])


def sfl_full_sec(runs: Path, full_steps: int) -> tuple[float, str]:
    """-> (seconds for a full SFL run, note). A smoke round is timed between two of SFL's saves."""
    d = runs / "sfl" / "maze_sfl" / "s0"
    cfg = json.loads((d / "config.json").read_text())
    saves = [int(line.split()[0]) for line in (d / "train.log").read_text().splitlines()
             if "Parameters of saved in" in line]
    gaps = [b - a for a, b in itertools.pairwise(saves)][:-1]   # the last save directly follows the one before
    if not gaps:
        raise SystemExit(f"{d}/train.log has too few timed saves to time a round")
    round_sec = sorted(gaps)[len(gaps) // 2]
    r = SFL_RECIPE
    rounds = full_steps // r["num_steps"] // r["num_envs"] // r["eval_freq"]
    scale_train = r["eval_freq"] * r["num_steps"] * r["num_envs"] / (cfg["eval_freq"] * cfg["num_steps"] * cfg["num_envs"])
    scale_scout = (r["num_batches"] * r["batch_size"] * r["rollout_steps"]
                   / (cfg["num_batches"] * cfg["batch_size"] * cfg["rollout_steps"]))
    note = "" if abs(scale_train / scale_scout - 1) < 0.01 else " (smoke shape differs from the full one: rough)"
    return rounds * round_sec * max(scale_train, scale_scout), note


def say(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


def hours(sec: float) -> str:
    return f"{sec / 3600:.1f} h"


def clock(sec: float) -> str:
    return time.strftime("%a %d %b %H:%M", time.localtime(time.time() + sec))


def main(smoke: str):
    runs = Path(smoke)
    mem = gpus()
    n_gpu = len(mem)
    peak = peak_mib(runs / "gpu_mem.csv")
    say(f"GPUs: {n_gpu}" + (f" ({min(mem)} MiB the smallest)" if mem else " (training runs on the CPU)"))
    if os.environ.get("JOBS_PER_GPU"):
        k, why = int(os.environ["JOBS_PER_GPU"]), "set by JOBS_PER_GPU"
    elif not n_gpu or not peak:
        k, why = 1, "no GPU memory measured"
    else:
        k = max(1, min(MAX_JOBS, int(0.9 * min(mem) / (1.25 * peak))))
        why = f"smoke peak {peak} MiB per job; override with JOBS_PER_GPU=<n>"
    say(f"JOBS_PER_GPU={k} ({why})")

    n_up, n_seeds = int(os.environ.get("NUM_UPDATES", "30000")), int(os.environ.get("NSEEDS", "10"))
    algos = os.environ.get("ALGOS", "sfl ref dr plr rplr accel minimax").split()
    spu = {a: jaxued_sec_per_update(runs, a) for a in ("dr", "rplr")}
    sfl_sec, note = sfl_full_sec(runs, int(os.environ.get("SFL_STEPS", "245760000")))
    say(f"smoke speed: DR {spu['dr']:.3f} s/update, RPLR {spu['rplr']:.3f} s/update")
    per_run = {a: (sfl_sec if a == "sfl" else spu[TASK_SPEED[a][0]] * TASK_SPEED[a][1] * n_up) for a in algos}
    say("estimated time per training run, one at a time: "
        + ", ".join(f"{a} {hours(s)}" for a, s in per_run.items()) + note)
    tasks = [per_run[a] for a in algos for _ in range(10 if a == "ref" else n_seeds)]
    timing = dict(line.split() for line in (runs / "timing.txt").read_text().splitlines())
    exam_sec = float(timing["exam_sec"]) * int(os.environ.get("EXAM_ITERS", "3000")) / float(timing["exam_iters"])
    with open(runs / "exam" / "manifest.json") as f:
        smoke_levels = sum(s["n"] for s in json.load(f)["sets"].values())
    n_scored = sum(n_seeds for a in algos if a != "ref")
    grade_sec = float(timing["grade_sec"]) / float(timing["grade_runs"]) * n_scored * FULL_EXAM_LEVELS / smoke_levels
    slots = max(n_gpu, 1)
    fast = max(sum(tasks) / (slots * k), max(tasks)) + exam_sec + grade_sec
    slow = max(sum(tasks) / slots, max(tasks)) + exam_sec + grade_sec
    say(f"{len(tasks)} training runs ({hours(sum(tasks))} of GPU time), then the exam (~{hours(exam_sec)}) "
        f"and grading (~{hours(grade_sec)}); the CartPole bridge (CPU) comes after and is not included.")
    if k > 1:
        say(f"expected finish: {clock(fast)} if {k} jobs share a GPU at full speed, "
            f"{clock(slow)} if sharing gains nothing; the log updates the estimate as runs finish")
    else:
        say(f"expected finish: {clock(fast)}; the log updates the estimate as runs finish")
    print(k)


if __name__ == "__main__":
    main(sys.argv[1])
