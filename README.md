# VerifAI × UED: Maze run

Needs Linux with NVIDIA GPU(s) (CUDA 12 driver), `python3.11`, `git` and ~30 GB of disk.

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
cd atlas
bash run/install.sh                               # installs the code and Python environments (once)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/maze.sh     # runs the experiment on GPUs 0-3 (list the ones you want to use); takes days, run it in tmux
```

Send back `maze_results.tgz`. If it stops, rerun the same command and it resumes.

It trains six curriculum methods over 10 seeds plus 10 reference agents, about 65–210 GPU-hours in total.
