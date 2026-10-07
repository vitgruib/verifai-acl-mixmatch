# UED failure atlas: cluster run

## Summary

**Question.** Unsupervised environment design (UED) methods such as PLR, Robust PLR, ACCEL and
SFL train an RL agent on a self-generated curriculum of levels. Each method's paper reports that
this improves robustness. We want to know **on which levels these agents still fail, and why**.

**What this run does.** It trains each UED method with its paper's own code, on three standard
UED benchmarks (Maze, Kinetix, JaxNav), with 10 seeds each. It then audits every trained agent with
**VerifAI falsification**: a cross-entropy sampler searches the level space for levels the agent
fails. A failure only counts if it comes with a certificate that the level is solvable:
- Maze and JaxNav: an exact BFS path.
- Kinetix: a replayed action sequence that solves the level.

**What comes back.** One tarball with the *failure atlas*: for every (environment, algorithm,
seed), the certified failures with their witnesses and level descriptors, plus training eval
curves, configs, logs and final checkpoints. The analysis is done afterwards, off the cluster.

**Cost.** 170 one-GPU training jobs and 170 CPU-only falsification jobs.
- Hard ceiling: **1,920 GPU-h + 2,000 CPU-h**. SLURM kills any job at its time limit.
- Likely well under half the GPU figure.
- Wall clock ≈ 28 h if the queue runs everything in parallel.

## Pipeline and compute

```
setup (login node, once) ─► smoke test (~1–3 h) ─► per environment: train array ──afterok──► falsify array ─► collect
```

| stage | per task | tasks | worst case (every task hits its limit) | expected |
|---|---|---|---|---|
| smoke test | ≤ 0.5 h (Kinetix speed probe: 1 h / 3 h) | 13 | 7 GPU-h + 10 CPU-h | under 1 h; probe ≤ 3 h |
| **1 Maze** train | 1 GPU, 12 h | 6 algos × 10 seeds = 60 | 720 GPU-h | a few hours each |
| 1 Maze falsify | 4 CPU, 1 h | 60 | 240 CPU-h | ~10 min each |
| **2 Kinetix** train | 1 GPU, 6 h; SFL 24 h (`SFL_TIME`) | 5 × 10 = 50 (40 + 10 SFL) | 40 × 6 + 10 × 24 = 480 GPU-h | measured by the smoke probe |
| 2 Kinetix falsify | 4 CPU, 4 h | 50 | 800 CPU-h | ~1–1.5 h each |
| **3 JaxNav** train | 1 GPU, 12 h | 6 × 10 = 60 | 720 GPU-h | a few hours each |
| 3 JaxNav falsify | 4 CPU, 4 h | 60 | 960 CPU-h | ~1.8 h each |
| **total** | | 170 GPU + 170 CPU | **1,920 GPU-h + 2,000 CPU-h** | |

**Wall clock.** With unlimited parallelism the slowest chain is Kinetix SFL: 24 h of training plus
4 h of falsification, 28 h. Everything else finishes within 16 h (12 h + 4 h).

**Source of the estimates.**
- **Falsification:** timed locally per level.
- **GPU training:** not measured, since we have no local GPU. The smoke test's Kinetix probe
  measures it before the full run (step 3).

## How to run

Run every command from the repository root on the login node.

**0. Requirements.**
- SLURM with a GPU partition (one GPU per training task; CUDA 12 driver).
- `python3.11` and `git`.
- Internet access from the login node, for cloning and pip.
- ~20 GB of disk for runs (`RUNS` can point at scratch).

**1. Get the code and build the environments (once, ~15 min).**

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas && cd atlas
bash cluster/setup/clone.sh          # upstream repos at pinned commits + our patches
bash cluster/setup/setup_jax.sh      # .venv-jax      (Part 1 Maze)
bash cluster/setup/setup_kinetix.sh  # .venv-kinetix  (Part 2 Kinetix)
bash cluster/setup/setup_sfl.sh      # .venv-sfl      (Maze SFL cell, Part 3 JaxNav)
```

- Each setup script ends with an import check that prints `ok <jax version> [devices]`.
  On a login node without a GPU the device list shows CPU; that is fine.
- If Python 3.11 has another name, prefix `PYBIN=/path/to/python3.11`.
- If jobs need `module load cuda` (or similar), put it in `~/.bashrc`.

**2. Cluster settings.** Export these in the shell you submit from, or in `~/.bashrc`:

```bash
export SBATCH_ARGS="-p <gpu partition> -A <account>"   # anything your cluster requires
export RUNS=/scratch/$USER/atlas                       # optional; default ./runs
DRY_RUN=1 bash cluster/submit.sh all                   # prints the sbatch lines, submits nothing
```

The dry run should show:
- three train arrays (Maze 0-59, Kinetix 0-39, JaxNav 0-59);
- a fourth Kinetix SFL array (0-9, `--time=24:00:00`);
- one falsify array per part, each depending on its training.

**3. Smoke test.**

```bash
bash cluster/submit.sh smoke
squeue -u $USER                                         # wait until empty (≤ 3 h)
sacct -u $USER -S today --format=JobName%30,State,ExitCode,Elapsed
ls $RUNS/smoke/*_falsify/*/*/*/summary.json             # falsification outputs
grep full_run_h runs/slurm/atlas-2-kinetix_*.out        # Kinetix GPU speed probe
```

The smoke test runs a tiny train → falsify chain for every codebase. It also runs a full-width
Kinetix speed probe: DR and SFL, each for 1/8 to 1/12 of a real run. Each probe prints
`full_run_h`, its wall time scaled to a full 201M-step run (slightly pessimistic, because it
includes JIT compilation).

Go ahead only if all of these hold:
- every job is `COMPLETED` with exit code `0:0`;
- `summary.json` files exist for Maze, Kinetix and JaxNav;
- the **DR probe** says `full_run_h` ≲ 5. Otherwise raise `#SBATCH --time` in
  `cluster/2_kinetix/train.sbatch`.
- the **SFL probe** gives the SFL limit. Set it to ~1.5× that probe's `full_run_h`, e.g.
  `export SFL_TIME=30:00:00` (default 24 h).

**4. Full run.**

```bash
bash cluster/submit.sh all      # or one part at a time: bash cluster/submit.sh 1 | 2 | 3
```

- **Logs:** `runs/slurm/<job>_<jobid>_<task>.out` (always under the repository, even when `RUNS`
  is elsewhere).
- **Task index:** the first line of each log names the task's algorithm and seed. Task `i` is
  algorithm `ALGOS[i / 10]`, seed `i % 10`; the order is in each `.sbatch` header.
- **Falsification output:** `$RUNS/<env>_falsify/<algo>/s<seed>/<space>_ce_r<fseed>/summary.json`.

**5. If a training task fails or times out.** Falsification waits on training with `afterok`, so
a single failed training task leaves that part's whole falsify array pending forever (reason
`DependencyNeverSatisfied`). To recover:

```bash
scancel <falsify jobid>
sbatch $SBATCH_ARGS --array=<failed ids, e.g. 3,17> cluster/<part>/train.sbatch
sbatch $SBATCH_ARGS --array=0-<N-1> --dependency=afterok:<new train jobid> cluster/<part>/falsify.sbatch
```

- `<part>` is `1_maze`, `2_kinetix` or `3_jaxnav`; `N` is 60, 50 or 60.
- A rerun restarts training from scratch. For a timeout, add a larger `--time=` to the rerun.
- **Kinetix SFL:** its array numbers seeds 0-9, so rerun with
  `ALGOS=sfl sbatch $SBATCH_ARGS --export=ALL --time=$SFL_TIME --array=<seed> cluster/2_kinetix/train.sbatch`.
- **Kinetix falsify:** the falsify array still uses the full 0-49 numbering.

**6. Collect and send back.**

```bash
bash cluster/collect.sh     # -> $RUNS/atlas_collect_<date>.tgz
```

Send that one file back. It holds:
- the falsifier records;
- the eval CSVs, configs and SLURM logs;
- only the **final** checkpoint of each run.

## What runs

| | Maze 13×13 | Kinetix S (symbolic) | JaxNav, single agent |
|---|---|---|---|
| source paper / code | PAIRED, Robust PLR, ACCEL; [JaxUED](https://github.com/DramaCow/jaxued) | Kinetix, ICLR 2025; [FLAIROx/Kinetix](https://github.com/FLAIROx/Kinetix) | SFL; [sampling-for-learnability](https://github.com/amacrutherford/sampling-for-learnability) |
| algorithms | DR, PLR, Robust PLR, ACCEL, minimax, SFL | DR, PLR, Robust PLR, ACCEL, SFL | DR, PLR, Robust PLR, ACCEL, minimax, SFL |
| training budget | 30k PPO updates (paper setting) | 201M env steps (paper: 1B+) | paper's single-agent sweep config |
| falsification spaces | `dr` (random walls), `seg` (wall segments) | `pos` (move agent/goal), `phys` (gravity, friction, motor, thrust ×0.5–1.5) on the 10 S eval levels | `dr`, `seg` |
| levels per search | 5000 | 1000 | 5000 |
| solvability certificate | BFS shortest path | open-loop action sequence, replayed (physics is deterministic) | BFS + scripted controller |

**Shared falsification settings.**
- The spec is "the agent solves the level in at least 5 of 10 attempts".
- Each trained agent gets 2 spaces × the cross-entropy sampler × 3 falsifier seeds.
- Levels that cannot be certified solvable are recorded as invalid, never as failures.
- Record format: [docs/failure_records.md](../docs/failure_records.md).

**Why the cross-entropy sampler.** On a Maze pilot it beat VerifAI's bandit sampler, with
random, Halton and simulated annealing as the other baselines. Relative to random sampling:

| | `dr` space: ce | `dr` space: bandit | `seg` space: ce | `seg` space: bandit |
|---|---|---|---|---|
| distinct failure modes | 1.13× | 1.06× | 1.07× | 0.87× |
| counterexample rate | 4.5× | 2.5× | 5.9× | 5.2× |

Its cost is more samples spent on unsolvable levels (21% vs bandit 12% in `dr`). Caveat: the
pilot was Maze only, with agents trained to 10% of budget. `SAMPLERS="ce mab"` re-adds the bandit.

**Budgets were cut deliberately.**
- Kinetix trains for 201M rather than 1B+ steps; Maze and JaxNav use the papers' settings.
- Kinetix SFL keeps the paper's learnability scoring (16 batches × 16384 levels × 512 steps
  every 48 updates). That is ~1.07B scoring steps, ~5× its training steps as in the paper,
  hence the separate 24 h array.
- Dropped for cost: PAIRED (three networks), and CarRacing and BipedalWalker (pixel rendering
  and 300M–2B step budgets were ~2/3 of the GPU hours). Kinetix replaces them as the
  general, physics-based benchmark.

## Validation status (as of 2026-10-07)

Every smoke-test line, and every training and falsification path, was run locally: headless,
CPU only, at tiny budgets, through the actual `.sbatch` scripts.
- **Maze:** Robust PLR and SFL train → checkpoint → falsify. Both wrote certified counterexamples.
- **Kinetix:** all 5 algorithms train and checkpoint. Every checkpoint loads in the falsifier,
  and replay-verified counterexamples were found in both spaces. This includes SFL at the
  paper's learnability settings.
- **JaxNav:** SFL train → falsify. Our minimax port passes unit tests: action masking, geometry
  against the env's collision map, and flood fill against a Python BFS on 300 levels. It trains
  and saves both networks.
- **Submission:** `DRY_RUN=1 submit.sh all` and `smoke` produce the arrays and dependencies above.

**Falsification cost** (measured locally, one CPU per process, untrained policies):

| part | s / level | per task | limit |
|---|---|---|---|
| Maze | ~0.04 | ~10 min | 1 h |
| Kinetix | ~1.2–1.8 | ~1–1.5 h | 4 h |
| JaxNav | ~0.4 | ~1.8 h | 4 h |

**Not yet verified (needs the cluster):**
- GPU training speed: the Maze and JaxNav 12 h limits are generous guesses; the Kinetix probe
  measures its own.
- The CUDA venv builds.
- Falsification rates on fully trained agents.

## Appendix

**Files.**

| path | contents |
|---|---|
| `submit.sh` | submits `smoke`, `1`, `2`, `3` or `all` (train array, then falsify array with `afterok`) |
| `collect.sh` | packs results into one tarball |
| `env.sh` | paths, venv locations, pinned upstream commits; override any variable before running |
| `1_maze/`, `2_kinetix/`, `3_jaxnav/` | `train.sbatch` and `falsify.sbatch` per environment; each header documents its settings |
| `setup/` | `clone.sh` plus one venv script per codebase |
| `patches/` | our changes to upstream code, applied by `clone.sh` |
| `tests/` | headless unit tests for the ported minimax cell |

**Python environments.** There are three, because the codebases pin incompatible JAX versions:

| venv | JAX | used by |
|---|---|---|
| `.venv-jax` | 0.10.2 | Maze training (JaxUED) and Maze falsification |
| `.venv-kinetix` | 0.9.0 | Kinetix training and falsification |
| `.venv-sfl` | 0.4.30 / flax 0.8.5 | Maze SFL cell, JaxNav training and falsification |

The SFL repo is 2024-era code. If the CUDA driver is too old for `jax-cuda12-plugin==0.4.30`,
upstream's container `nvcr.io/nvidia/jax:23.10-py3` also works.

**Upstream commits.**
- JaxUED `0f8f128`
- Kinetix `80ee9c2`
- sampling-for-learnability `8b14e2e`

**Our changes to paper code.**
- `jaxued_gif.patch`: logs videos as GIFs, so no ffmpeg is needed.
- `jaxued_minimax.patch`: adds `maze_paired.py --minimax`, where the adversary reward is the
  negative student return. JaxUED ships no minimax baseline.
- `kinetix_acl27.patch`: upstream bug fixes needed on JAX 0.9, with no behaviour change:
  - the removed `tree_map` alias;
  - SFL's nested `shard_map`;
  - a boolean mask in SFL's buffer update;
  - ACCEL's `create_empty_env` call.
- Kinetix algorithm configs:
  - Kinetix's `ued=accel` config crashes on a missing key, so ACCEL runs as `ued=plr
    use_accel=true num_edits=5`, the same algorithm.
  - DR is `replay_prob=0`.
  - Robust PLR is `exploratory_grad_updates=false`.
- `sfl_minimax.patch`: minimax for single-agent JaxNav, the one cell no paper ships code for.
  - A PPO adversary places goal, start and 48 walls on a 9×9 grid. Its reward is the negative
    protagonist return.
  - The protagonist, its update and the evaluation are the SFL repo's `jaxnav_plr` code,
    unchanged.
- JaxNav sweep fixes:
  - Upstream's ACCEL sweep has a typo (`use.USE_ACCEL`), so we pass `ued.USE_ACCEL=True`.
  - Non-SFL runs use the single-agent eval set, since the default 4-agent set crashes with
    one agent.

**Settings you can change** (environment variables read by `submit.sh` and the `.sbatch` files):

| variable | what it changes |
|---|---|
| `NSEEDS` | seeds per algorithm (default 10) |
| `ALGOS` | run a subset, e.g. `ALGOS="dr rplr" bash cluster/submit.sh 1` |
| `SFL_TIME` | Kinetix SFL time limit |
| `BUDGET`, `FSEEDS`, `SAMPLERS`, `SPACES` | falsification size and samplers |
| `RUNS`, `SBATCH_ARGS` | output location and extra sbatch flags |
| `DRY_RUN=1` | print instead of submit |

Every job is headless (`MPLBACKEND=Agg`, `SDL_VIDEODRIVER=dummy`; no renderer is called), and
wandb runs offline (`wandb sync` afterwards for dashboards).
