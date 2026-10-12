# Maze runs on the cluster

**Goal.** Curriculum methods for RL (PLR, Robust PLR, ACCEL, SFL and others) claim to make agents
robust. We train each method with its paper's own code. We then use VerifAI falsification to search
for maze levels the trained agent fails, counting only levels proven solvable (a BFS path). This
shows where each method still fails, and why: did its curriculum ever train on those levels, and
how did it rate them?

**Cost.** 60 GPU jobs + 60 CPU jobs. The hard ceiling (SLURM time limits) is 720 GPU-h + 240 CPU-h,
likely far less. About 16 h if the jobs run in parallel.

**Needs.** SLURM with a GPU partition (CUDA 12), `python3.11`, `git`, and internet on the login node.

## What to run (from the repo root, on the login node)

1. **Set up, once (~15 min).** This downloads the papers' code at fixed versions and builds two
   Python environments.
   ```bash
   git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas && cd atlas
   bash cluster/setup/clone.sh && bash cluster/setup/setup_jax.sh && bash cluster/setup/setup_sfl.sh
   export SBATCH_ARGS="-p <gpu partition> -A <account>"   # whatever your cluster needs
   ```

2. **Smoke test (~30 min, 2 GPU + 2 CPU jobs).** It trains two methods for 50 updates and runs
   a 50-level search, to check that everything works before the big run.
   ```bash
   bash cluster/submit.sh maze-smoke
   ```
   Go on if `sacct -u $USER -S today` shows every job `COMPLETED`.

3. **Full Maze run (60 GPU jobs, then 60 CPU jobs).**
   ```bash
   bash cluster/submit.sh 1
   ```
   - **Training:** DR, PLR, Robust PLR, ACCEL, minimax and SFL, 10 seeds each, at the papers'
     settings (30k updates). One GPU, ≤ 12 h per job.
   - **Falsification** starts automatically when training finishes. Each agent is searched for
     failing levels with a cross-entropy sampler (finds failures) and a random sampler (an unbiased
     baseline for failure rates). 4 CPUs, ≤ 1 h per job.

4. **Send the results back.**
   ```bash
   bash cluster/collect.sh      # prints the path of one .tgz file; send us that file
   ```

All jobs are headless, write only inside the repo, and use no network. If a training job fails or
times out, see step 7 of [DETAILS.md](DETAILS.md).

---

This code was written with an AI coding agent (Claude Code) under our direction.
[DETAILS.md](DETAILS.md) covers the motivation, our changes to the papers' code, the later Kinetix
and JaxNav runs, and the recovery steps.
