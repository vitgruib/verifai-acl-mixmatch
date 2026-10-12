# Maze run

On the login node:

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas && cd atlas
SBATCH_ARGS="-p <gpu partition> -A <account>" bash cluster/run_maze.sh
```

When `squeue -u $USER` is empty (about a day later):

```bash
bash cluster/collect.sh
```

Send me the `.tgz` file it prints.
