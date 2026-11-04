# VerifAI × UED: Maze run

Requires Linux, NVIDIA GPUs (CUDA 12 driver), `python3.11`, `git` and ~30 GB of disk.

## Setup

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
cd atlas
bash run/install.sh       # clones the pinned JaxUED/SFL and builds the Python envs
```

## Run

On a SLURM cluster, run it from the login node. It submits everything as dependent jobs (1 GPU each, up to 48 h):

```bash
SBATCH_ARGS="--account=<acct> --partition=<gpu-partition>" bash run/maze.sh
```

On a single machine, list the GPUs to use and run it in tmux (it takes days):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/maze.sh
```

Either way, rerunning the same command resumes. Please send back `maze_results.tgz` when it finishes.

It trains six curriculum methods over 10 seeds plus 10 reference agents, about 65–210 GPU-hours in total.
