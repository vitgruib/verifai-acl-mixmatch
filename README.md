# VerifAI × UED: Maze run

Requires Linux, NVIDIA GPUs (CUDA 12 driver), `python3.11`, `git` and ~30 GB of disk.

## Setup

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
cd atlas
bash run/install.sh       # clones the pinned JaxUED/SFL and builds the Python envs
```

## Run

List the GPUs to use and run it in tmux (it takes days):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/maze.sh
```

Rerunning the same command resumes. Please send back `maze_results.tgz` when it finishes.

It trains six curriculum methods over 10 seeds plus 10 reference agents, about 65–210 GPU-hours in total.
