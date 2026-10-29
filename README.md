# VerifAI × UED: Maze run

Needs Linux with NVIDIA GPU(s) (CUDA 12 driver), `python3.11`, `git`, internet, and ~30 GB of disk.

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
cd atlas
bash run/install.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/train.sh     # days; run it inside tmux
CUDA_VISIBLE_DEVICES=0 bash run/finish.sh          # exam + grading, writes maze_results.tgz
```

Send back `maze_results.tgz`.

- `train.sh` runs one job at a time on each GPU you list. `JOBS_PER_GPU=2` runs two on each, and
  each gets 40% of the GPU's memory.
- Ctrl-C stops it. Run the same command again to resume: finished runs are kept, unfinished ones
  restart.
- Optional, before training: `CUDA_VISIBLE_DEVICES=0 bash run/test.sh` (~30 min) runs tiny versions
  of everything. It prints the memory per job, a suggested `JOBS_PER_GPU` and the expected time.
- Logs: one per run in `runs/slurm/`.
- W&B is forced offline. Everything stays inside `atlas/`, apart from `/tmp` and, on some GPUs, a
  CUDA cache in `~/.nv`. Don't `git pull` while it's running.

**Several machines:** give each one part of the work, for example machine A
`bash run/train.sh 0-4 ref` and machine B `bash run/train.sh 5-9 bridge`. On a shared disk, use one
checkout and you're done. Otherwise copy each machine's `runs/` into one checkout
(`rsync -a B:atlas/runs/ runs/`) and run `finish.sh` there.

## What it trains

- Six curriculum methods (SFL, DR, PLR, Robust PLR, ACCEL, minimax), seeds 0-9, at the paper
  budget (~246M steps per run).
- 10 reference DR agents. The exam is built from mazes they fail. Agents 100-104 are trained on
  my side and copied from GitHub when the run gets to them.
- A CartPole check (CPU) comparing JaxUED's PLR code with ours.

The exam has 300 random mazes, the VerifAI mazes, and 14 hand-picked ones. Rough cost: 65-210
GPU-hours in total (SFL runs take longest). It hasn't run on a GPU yet.

## Where to look

- `run/*.sh`: what each command does, in a few lines.
- `cluster/1_maze/train.sbatch`: the exact training command and flags for each method.
- `cluster/1_maze/exam.sbatch`, `grade.sbatch`, `bridge.sbatch`: the exam, grading, CartPole check.
- `cluster/env.sh`: paths and pinned commits. Any variable there can be overridden.
- `docs/library_log.md`, "Batch 22": the plan and statistics, registered before any run.
- `docs/maze_gpu_audit.md`: what was checked in the upstream code, and the settings used.
