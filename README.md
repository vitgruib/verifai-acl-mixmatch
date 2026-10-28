# VerifAI × UED

This trains six curriculum (UED) methods (DR, PLR, Robust PLR, ACCEL, minimax and SFL) on the
Maze benchmark, 10 seeds each, using each paper's own code. Then it uses VerifAI to search for
solvable mazes that each trained agent fails.

## Running it

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
bash atlas/cluster/run_maze.sh
```

When it's done, please send me `atlas/maze_results.tgz`.

You need Linux with an NVIDIA GPU, `python3.11` and `git`. On a SLURM cluster, run it from the
login node and add `-p <partition> -A <account>` if your cluster needs them.

The script installs everything (about 15 minutes), runs a short test, and only starts the full
run if the test passes.

## How long it takes

- **SLURM cluster:** a few hours once the jobs start, 14 hours at most.
- **One computer with a GPU:** about 4 days (with N GPUs, about 4/N days). It runs in the
  background, so just leave the computer on. `tail -f atlas/run_maze.log` shows progress and an
  estimated finish time.
