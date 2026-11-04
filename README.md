# VerifAI × UED: Maze run

Needs Linux with NVIDIA GPU(s) (CUDA 12 driver), `python3.11`, `git` and ~30 GB of disk.

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
cd atlas
bash run/install.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/train.sh     # the GPUs to use; takes days, run it in tmux
CUDA_VISIBLE_DEVICES=0 bash run/finish.sh          # writes maze_results.tgz
```

Send back `maze_results.tgz`. If training stops, rerun the same command and it resumes.

It trains six curriculum methods over 10 seeds plus 10 reference agents, about 65–210 GPU-hours in total.
