# VerifAI × UED: Maze run

Needs Linux with NVIDIA GPU(s) (CUDA 12 driver), `python3.11`, `git` and ~30 GB of disk.

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
cd atlas
bash run/install.sh
CUDA_VISIBLE_DEVICES=0 bash run/test.sh            # ~30 min check; prints a suggested JOBS_PER_GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/train.sh     # days, run it in tmux; rerun the same command to resume
CUDA_VISIBLE_DEVICES=0 bash run/finish.sh          # writes maze_results.tgz
```

Send back `maze_results.tgz`.

**Several machines:** split the seeds, e.g. machine A `CUDA_VISIBLE_DEVICES=0,1 bash run/train.sh 0-4 ref`
and machine B `CUDA_VISIBLE_DEVICES=0,1 bash run/train.sh 5-9 bridge`. Then copy `runs/` into one
checkout (`rsync -a B:atlas/runs/ runs/`; not needed on a shared disk) and run `finish.sh` there.

It trains six curriculum methods over 10 seeds plus 10 reference agents, about 65–210 GPU-hours in total.
