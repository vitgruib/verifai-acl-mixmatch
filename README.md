# VerifAI × UED: Maze run

Needs Linux with NVIDIA GPU(s) (CUDA 12 driver), `python3.11`, `git`, internet, and ~30 GB of disk.

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
cd atlas
```

## Install

```bash
bash cluster/setup/clone.sh        # JaxUED and SFL at pinned commits, patched, into third_party/
bash cluster/setup/setup_jax.sh    # .venv-jax: JaxUED training, the exam, grading
bash cluster/setup/setup_sfl.sh    # .venv-sfl: SFL (needs an older jax than JaxUED)
bash cluster/setup/setup_torch.sh  # .venv-torch: grades the CartPole check
```

Optional quick test (~30 min, prints an estimate of the full run's time on your GPUs):

```bash
CUDA_VISIBLE_DEVICES=0 LOCAL=1 bash cluster/submit.sh maze-smoke
```

## Training

Six methods (SFL, DR, PLR, Robust PLR, ACCEL, minimax) × seeds 0-9, plus 10 reference DR agents.
Split by seed band, one terminal each. Pick the GPU with `CUDA_VISIBLE_DEVICES`:

| Terminal | Command |
|----------|---------|
| 1 | `CUDA_VISIBLE_DEVICES=0 bash run/train.sh 0 4` |
| 2 | `CUDA_VISIBLE_DEVICES=1 bash run/train.sh 5 9` |
| 3 | `CUDA_VISIBLE_DEVICES=2 bash run/ref.sh` |
| 4 | `bash run/bridge.sh` (CPU only) |

- Use any split you like, e.g. `run/train.sh 0 1`, `2 3`, ... across five GPUs. Don't give two
  terminals the same seed.
- To put two terminals on one GPU, add `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` to both (JAX takes 75%
  of the GPU's memory by default).
- Rerunning a command skips finished runs and restarts unfinished ones, so it's safe after a crash
  or reboot.
- `run/ref.sh` copies in reference agents 100-104 (trained on my side) from GitHub when they are
  there, and trains them otherwise.

## After training

```bash
CUDA_VISIBLE_DEVICES=0 bash run/exam.sh    # after run/ref.sh: builds the exam from mazes the reference agents fail
CUDA_VISIBLE_DEVICES=0 bash run/grade.sh   # after run/train.sh and run/exam.sh: grades the 60 runs
bash run/pack.sh                           # -> maze_results.tgz, send this back
```

Logs are in `runs/slurm/`. W&B is forced offline. Everything stays inside `atlas/`, apart from `/tmp`
and, on some GPUs, a CUDA cache in `~/.nv`. Don't `git pull` while it's running.

All of the above in one command, on SLURM (or `LOCAL=1` on one machine): `bash cluster/submit.sh maze`.

## Where to look

- `run/*.sh`: a few lines each. They call the job scripts below.
- `cluster/1_maze/train.sbatch`: the exact training command and flags for each method.
- `cluster/1_maze/exam.sbatch`, `grade.sbatch`, `bridge.sbatch`: the exam, grading, CartPole check.
- `cluster/env.sh`: paths and pinned commits. Any variable there can be overridden.
- `docs/library_log.md`, "Batch 22": the plan and statistics, registered before any run.
- `docs/maze_gpu_audit.md`: what was checked in the upstream code, and the settings used.

Rough cost: 65-210 GPU-hours in total (SFL runs take longest). It hasn't run on a GPU yet.
