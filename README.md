# VerifAI × UED: Maze run

## What you need

- Linux with NVIDIA GPU(s) and a CUDA 12 driver (check with `nvidia-smi`)
- `python3.11` and `git`, with internet access
- About 30 GB of free disk

## How to run

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
bash atlas/cluster/run_maze.sh
```

It runs in the background. Leave the machine on with sleep turned off.

- Progress: `tail -f atlas/run_maze.log`
- Use only some GPUs: `CUDA_VISIBLE_DEVICES=0,1 bash atlas/cluster/run_maze.sh`
- After a restart, run the same command again. It picks up where it stopped.
- Don't `git pull` while it's running.

When the log's last line starts with `done` (or `FAILED`), send back `atlas/maze_results.tgz`.

## What it does

1. Installs the papers' code (JaxUED, SFL) at pinned commits, in Python environments inside
   `atlas/`.
2. Runs a short test. It stops if the test fails.
3. Trains six curriculum methods (DR, PLR, Robust PLR, ACCEL, minimax, SFL) on Maze, 10 seeds
   each, at the paper budget (~246M steps per run).
4. Trains 10 extra DR agents and builds an exam from mazes they fail. Five of these agents come
   from GitHub if they are already there.
5. Grades the 60 agents on the exam: 300 random mazes, the VerifAI mazes, and 14 hand-picked
   ones.
6. Runs a CartPole check comparing JaxUED's PLR code with ours.

It hasn't run on a GPU yet. My rough estimate is 65-210 GPU-hours (SFL runs take longest), so
3-9 days on one GPU, or less with several. After the short test, the log prints its own
estimate for your machine.
