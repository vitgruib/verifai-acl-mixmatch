# VerifAI × UED

This runs the Maze part of the generalization suite. Six curriculum (UED) methods (DR, PLR, Robust
PLR, ACCEL, minimax and SFL) are trained on the Maze benchmark with each paper's own code, 10 seeds
each, at the paper budget (about 246M environment steps per run). 10 extra DR agents are trained
only to build the exam; 5 of them are trained on our side, and the run downloads them from GitHub
when it gets to them (or trains them itself if they are not there). Each agent then takes a three-part exam: 300 random mazes, mazes found with
VerifAI that the extra DR agents fail, and 14 hand-picked mazes (the standard named test mazes plus a
few extremes). Last, a CartPole check runs the papers' PLR code (JaxUED) on our CartPole setup, to
compare it with our own implementation, which the CartPole, FrozenLake and LunarLander results use.

## What you need

- Linux with one or more NVIDIA GPUs, and a driver recent enough for CUDA 12 (`nvidia-smi` shows
  the CUDA version in its top right corner).
- `python3.11`, `git`, and internet access for the install.
- About 30 GB of free disk, mostly for the Python environments with their CUDA libraries. Each run
  keeps its last two checkpoints, about 6 GB in all (most of it minimax, 184 MB per checkpoint),
  plus about 3 GB of training logs. This is estimated from a small test run.

To see your GPUs:

```bash
nvidia-smi -L
```

The script uses every GPU listed there. To use only some of them, name them when you start it (and
again when you rerun it), for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash atlas/cluster/run_maze.sh
```

## Running it

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
bash atlas/cluster/run_maze.sh
```

What the script does:

1. Installs everything inside the `atlas` folder: the papers' code at pinned commits with our
   patches, and three Python environments.
2. Runs a short test (DR, Robust PLR, ACCEL, minimax and SFL for a few updates, a small exam,
   grading). If it fails, the script stops there and packs the logs.
3. Trains the Maze runs, several at a time on each GPU: 6 methods x 10 seeds, then the 10 extra
   DR agents (5 trained here, 5 downloaded), 65 to 70 runs in all.
4. Builds the VerifAI part of the exam from the 10 extra DR agents. The other two parts are already
   in the repo.
5. Grades the 60 scored agents on the exam.
6. The CartPole check: trains 60 CartPole runs (DR, PLR and Robust PLR, 20 seeds each) with
   JaxUED's PLR code on the CPU, grades them with our grader and compares them with our runs.
7. Packs the results into `atlas/maze_results.tgz`.

## How long it takes

None of this has run on a GPU yet, so this is a guess: 1 to 3 GPU-hours per training run, 65 to
210 GPU-hours in all. The 10 SFL runs will probably take several times longer than the others,
since SFL also plays about 1.85 billion steps per run to pick its training mazes, so the real total
may be higher. Assuming several jobs sharing one GPU are no faster than one job at a time:

- 1 GPU: 3 to 9 days
- 2 GPUs: 1.5 to 4.5 days
- 4 GPUs: 17 to 53 hours

Building the exam and grading come on top of that. The script runs up to 4 jobs per GPU when they
fit in memory, which may make it faster. After the short test, the log prints how many jobs it
runs per GPU and its own estimate of the finish time, measured on your machine. Trust that one
over the numbers above.

## While it runs

It runs in the background, so you can close the terminal. Just leave the computer on, with
automatic suspend (sleep) turned off. To watch progress:

```bash
tail -f atlas/run_maze.log
```

Every finished job adds a line with an estimate of when the current stage ends. Each job's own log
is in `atlas/runs/slurm/`.

If the computer restarts, run the same command again. It picks up where it stopped: finished
training runs, the exam and the graded agents are kept, and only the runs that were cut off start
over. Running the command while it is still going just tells you so. Please don't `git pull` in
`atlas` while it runs.

## What to send back

When the last line of the log starts with `done`, please send me `atlas/maze_results.tgz` (probably
a couple of GB, mostly the minimax checkpoints). If the run ends with `FAILED` instead, send the
same file; it has the logs.
