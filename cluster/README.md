# Maze run

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas
bash atlas/cluster/run_maze.sh
```

When it finishes, send me `atlas/maze_results.tgz`.

Needs Linux with an NVIDIA GPU, `python3.11` and `git`. On a SLURM cluster, run it on the login
node, adding `-p <partition> -A <account>` if the cluster needs them.

| where | time |
|---|---|
| one computer, one GPU | about 4 days; with N GPUs, about 4/N days |
| SLURM cluster | a few hours once the jobs start; at most 14 h |

On one computer it runs in the background (keep the computer on). Progress and an updated ETA:
`tail -f atlas/run_maze.log`.
