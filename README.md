# VerifAI × UED: Maze run

## What you need

- Linux with NVIDIA GPU(s) and a CUDA 12 driver (check with `nvidia-smi`)
- `python3.11` and `git`, with internet access
- About 30 GB of free disk

## How to run

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
cd atlas
bash cluster/setup/clone.sh        # JaxUED and SFL at pinned commits, patched, into third_party/
bash cluster/setup/setup_jax.sh    # .venv-jax: JaxUED training, the exam, grading
bash cluster/setup/setup_sfl.sh    # .venv-sfl: SFL (needs an older jax)
bash cluster/setup/setup_torch.sh  # .venv-torch: grades the CartPole check
CUDA_VISIBLE_DEVICES=0,1 LOCAL=1 nohup bash cluster/submit.sh maze > run_maze.log 2>&1 &
```

`submit.sh maze` runs a ~30 min smoke test, stops if it fails, and otherwise runs everything below
and packs `maze_results.tgz`. Send that file back.

- **GPUs:** it uses exactly the ones in `CUDA_VISIBLE_DEVICES`. After the smoke test it picks how
  many jobs share each GPU from their memory use; set `JOBS_PER_GPU=<n>` to choose yourself. Each
  job sees only its own GPU. With several per GPU, each gets `0.8 / JOBS_PER_GPU` of its memory.
- **Resuming:** run the same command again. Finished runs are kept and unfinished ones restart.
- **SLURM:** drop `LOCAL=1` and it submits array jobs instead. Extra sbatch flags go in
  `SBATCH_ARGS`.
- `bash cluster/run_maze.sh` does the setup and the `submit.sh` line above in the background, with
  a `stop` command (`bash cluster/run_maze.sh stop`). It needs `CUDA_VISIBLE_DEVICES` too.
- Jobs run under `nice -n 10`. W&B is forced offline. Everything stays inside `atlas/`, apart from
  `/tmp` and, on some GPUs, a CUDA cache in `~/.nv`.
- Don't `git pull` while it's running. It reads the repo's scripts as it goes.

Where to look:

- `cluster/env.sh`: paths, pinned commits, the task list. Any variable there can be overridden.
- `cluster/submit.sh`: the job order and the local scheduler.
- `cluster/1_maze/train.sbatch`: the exact training command and flags for each method.
- `cluster/1_maze/exam.sbatch`, `grade.sbatch`: building the exam and grading.
- `runs/slurm/*.out`: one log per job.
- `docs/library_log.md`, "Batch 22": the plan and statistics, registered before any run.
- `docs/maze_gpu_audit.md`: what was checked in the upstream code, and the settings used.

## What it does

1. Trains six curriculum methods (DR, PLR, Robust PLR, ACCEL, minimax, SFL) on Maze, 10 seeds
   each, at the paper budget (~246M steps per run).
2. Trains 10 extra DR agents and builds an exam from mazes they fail. Five of these agents
   (seeds 100-104) are trained on my side and fetched from this repo when the run gets to them.
   If they aren't there yet, it trains them.
3. Grades the 60 agents on the exam: 300 random mazes, the VerifAI mazes, and 14 hand-picked
   ones.
4. Runs a CartPole check comparing JaxUED's PLR code with ours.

It hasn't run on a GPU yet. My rough estimate is 65-210 GPU-hours (SFL runs take longest), so
3-9 days on one GPU, or less with several. After the smoke test, the log prints its own estimate
for your machine.
