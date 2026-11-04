# VerifAI × UED: Maze run

Requires Linux, NVIDIA GPUs (CUDA 12 driver), `python3.11`, `git` and ~30 GB of disk.

## Setup

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
cd atlas
bash run/install.sh       # clones the pinned JaxUED/SFL and builds the Python envs
```

## Run

One terminal per GPU (in tmux; it takes days). Each trains a band of seeds:

```bash
CUDA_VISIBLE_DEVICES=0 bash run/train.sh 0-2            # terminal 1
CUDA_VISIBLE_DEVICES=1 bash run/train.sh 3-5            # terminal 2
CUDA_VISIBLE_DEVICES=2 bash run/train.sh 6-7 ref        # terminal 3, plus the reference agents
CUDA_VISIBLE_DEVICES=3 bash run/train.sh 8-9 bridge     # terminal 4, plus a CartPole check on the CPU
```

When all four say done, in any terminal:

```bash
CUDA_VISIBLE_DEVICES=0 bash run/finish.sh               # exam and grading, writes maze_results.tgz
```

Rerunning a command resumes it. Please send back `maze_results.tgz`.

It trains six curriculum methods over 10 seeds plus 10 reference agents, about 65–210 GPU-hours in total.
