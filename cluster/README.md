# UED failure atlas: cluster runbook

This trains five or six unsupervised-environment-design (UED) algorithms per environment on
three standard UED benchmarks, using the papers' own code. It then searches each trained agent for levels it fails
on (VerifAI falsification). Everything here is a SLURM array job. Results come back as one
tarball.

## Pipeline at a glance

Each environment runs the same two-stage chain. Falsification waits on training through a SLURM
`afterok` dependency.

```
setup (login node, once) ─► smoke (~1 h) ─► for each part 1-3:  train array ──afterok──► falsify array ─► collect
```

| stage | purpose | resources per task | worst case (every task hits its limit) | expected |
|---|---|---|---|---|
| setup | Clone the upstream repos at pinned commits, apply our patches, build 3 venvs | login node | — (pip installs; not timed on the cluster) | minutes |
| smoke | Run tiny train → falsify chains for every codebase on 1 seed, plus a Kinetix GPU speed probe; catches broken installs and configs before the real run | 0.5 h (probe 1 h) | 5 GPU-h + 20 CPU-h; ~1 h wall | well under 1 h |
| 1 train (Maze) | Train 6 UED algos × 5 seeds at the papers' settings: the agents to be audited | 1 GPU, 12 h | 30 × 12 = 360 GPU-h | a few hours each (unmeasured on GPU) |
| 1 falsify | VerifAI searches each Maze agent for solvable levels it fails (BFS-certified) | 8 CPU, 1 h | 30 × 1 h × 8 = 240 CPU-h | ~10 min each |
| 2 train (Kinetix) | Train 5 UED algos × 5 seeds on Kinetix S, 201M steps: a physics-based, more general benchmark | 1 GPU, 6 h | 25 × 6 = 150 GPU-h | set by the probe's steps/s |
| 2 falsify | VerifAI on moved-object / changed-physics versions of the 10 S eval levels (replay-certified witness) | 8 CPU, 4 h | 25 × 4 h × 8 = 800 CPU-h | ~1–1.5 h each |
| 3 train (JaxNav) | Train 6 UED algos × 5 seeds on single-agent robot navigation (SFL paper) | 1 GPU, 12 h | 30 × 12 = 360 GPU-h | a few hours each (unmeasured on GPU) |
| 3 falsify | VerifAI on JaxNav levels (BFS + scripted-controller certified) | 8 CPU, 4 h | 30 × 4 h × 8 = 960 CPU-h | ~1.8 h each |
| collect | Pack failure records, eval CSVs, configs, logs and final checkpoints into one tarball | login node | — | seconds |

**Suite worst case: 870 GPU-h + ~2,000 CPU-h** (85 GPU tasks, 85 CPU tasks). If the queue runs
every array task at once, wall clock is bounded by the slowest chain: JaxNav, 12 h + 4 h =
16 h. A realistic total is well under half the GPU figure. Nothing is ever charged beyond the
limits: SLURM kills a job at its `--time`.

The output of the whole pipeline is the **failure atlas**: for each (environment, algorithm,
seed), the levels the trained agent fails even though a solution provably exists, each with its
witness solution and descriptors (record format: [docs/failure_records.md](../docs/failure_records.md)).

## How to run

**0. Requirements.** SLURM with a GPU partition (1 GPU per training task, CUDA 12 driver),
`python3.11` and `git`, and internet on the login node for the clone and pip steps.

**1. Get the code and set up, once, on the login node.**

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas && cd atlas
bash cluster/setup/clone.sh          # upstream repos at pinned commits + our patches
bash cluster/setup/setup_jax.sh      # .venv-jax      (Part 1)
bash cluster/setup/setup_kinetix.sh  # .venv-kinetix  (Part 2)
bash cluster/setup/setup_sfl.sh      # .venv-sfl      (Part 1 SFL cell, Part 3)
```

Each setup script ends with an import check that prints `ok ...`. If `python3.11` has a
different name, pass it as `PYBIN=/path/to/python3.11`. If the cluster needs `module load cuda`,
put that in `~/.bashrc` (see Setup notes).

**2. Point it at your cluster.** These variables are read by every later command, so export
them in the same shell (or in `~/.bashrc`):

```bash
export SBATCH_ARGS="-p <gpu partition> -A <account>"   # whatever your cluster needs
export RUNS=/scratch/$USER/atlas                       # optional; default is ./runs
```

Check the submission without sending anything: `DRY_RUN=1 bash cluster/submit.sh all` prints
every `sbatch` line (30 / 25 / 30 tasks for Parts 1 / 2 / 3, each followed by a falsify array).

**3. Smoke test (about 1 h).**

```bash
bash cluster/submit.sh smoke
squeue -u $USER                                    # wait until empty
sacct -u $USER -S today --format=JobName%30,State,ExitCode,Elapsed
grep "steps/s" runs/slurm/atlas-2-kinetix_*.out    # probe throughput
ls $RUNS/smoke/*_falsify/*/*/*/summary.json        # falsify outputs
```

Go on only if: every job is `COMPLETED` with exit `0:0`; `summary.json` files exist for Maze,
Kinetix and JaxNav; and the Kinetix probe (the full-width DR line) shows at least ~20,000
steps/s. If the probe is slower, the 6 h Kinetix limit is too short. Raise `#SBATCH --time`
in `cluster/2_kinetix/train.sbatch` by the same factor, or lower `STEPS`.

**4. Full run.**

```bash
bash cluster/submit.sh all      # or one part at a time: bash cluster/submit.sh 1 | 2 | 3
```

Logs go to `runs/slurm/<job>_<jobid>_<task>.out`. Each line names the task's algorithm and
seed (task `i` = algorithm `i / 5`, seed `i % 5`). A training log ends with
`done ... steps/s=` (Kinetix) or the final evaluation. Falsification writes
`$RUNS/<env>_falsify/<algo>/s<seed>/<space>_<sampler>_r<seed>/summary.json`.

**5. If a task fails or times out.** `afterok` means a failed training task leaves that part's
whole falsify array pending forever (`DependencyNeverSatisfied`). To recover:

```bash
scancel <falsify jobid>                                          # the stuck array
sbatch $SBATCH_ARGS --array=<i> cluster/<part>/train.sbatch      # rerun failed task(s), e.g. --array=3,17
sbatch $SBATCH_ARGS --array=0-<N-1> --dependency=afterok:<new train jobid> cluster/<part>/falsify.sbatch
```

A rerun restarts training from scratch. Timed-out tasks need a longer `--time` passed on that
`sbatch` line.

**6. Bring results home.**

```bash
bash cluster/collect.sh   # -> $RUNS/atlas_collect_<date>.tgz; send this one file back
```

It holds the falsifier records, eval CSVs, configs, SLURM logs and only the final checkpoints
(not the intermediate checkpoints or full run directories).

## The three parts (one per environment)

| part | environment (paper) | code | algorithms | array tasks | per task (sbatch limit) |
|---|---|---|---|---|---|
| **1** | Maze, 13×13 (PAIRED / Robust PLR / ACCEL papers) | JaxUED; SFL repo for the SFL cell | DR, PLR, Robust PLR, ACCEL, minimax, SFL | 30 train + 30 falsify | train 1 GPU, 12 h; falsify 8 CPU (no GPU), 1 h |
| **2** | Kinetix S, symbolic entity obs (Kinetix paper, ICLR 2025; 201M of the paper's 1B+ steps) | Kinetix repo (JAX) | DR, PLR, Robust PLR, ACCEL, SFL (no minimax in Kinetix) | 25 train + 25 falsify | train 1 GPU, 6 h; falsify 8 CPU (no GPU), 4 h |
| **3** | JaxNav, single agent (SFL paper) | SFL repo (JAX) | DR, minimax, PLR, Robust PLR, ACCEL, SFL | 30 train + 30 falsify | train 1 GPU, 12 h; falsify 8 CPU (no GPU), 4 h |

Each array task is one (algorithm, seed), with 5 seeds per algorithm (`NSEEDS=10` doubles
it). Task `i` is algorithm `ALGOS[i / 5]` and seed `i % 5`. Run a subset with e.g. `ALGOS="dr rplr" bash cluster/submit.sh 1`.

| directory | contents |
|---|---|
| `1_maze/` | `train.sbatch` (JaxUED and SFL training), `falsify.sbatch` (VerifAI falsification) |
| `2_kinetix/` | `train.sbatch` (Kinetix's `experiments/plr.py` for DR/PLR/Robust PLR/ACCEL, `experiments/sfl.py`), `falsify.sbatch` (CPU) |
| `3_jaxnav/` | `train.sbatch` (SFL repo's `jaxnav_plr` / `jaxnav_sfl`, our `jaxnav_minimax`), `falsify.sbatch` |
| `setup/` | `clone.sh` (upstream repos at pinned commits), one environment script per codebase |
| `patches/` | our patches to JaxUED, Kinetix and the SFL repo, applied by `clone.sh` (below) |
| `tests/` | headless unit and env tests for the ported cells (see `tests/README.md`) |
| `submit.sh`, `collect.sh` | submit a part; pack results |

## Setup notes

There are three Python environments, one per codebase, because their dependency pins are
incompatible:

| script | env | used by |
|---|---|---|
| `setup_jax.sh` | `.venv-jax`: py3.11, current `jax[cuda12]`, JaxUED, VerifAI, Scenic | Part 1 (JaxUED algorithms, falsifier) |
| `setup_kinetix.sh` | `.venv-kinetix`: py3.11, `jax[cuda12]` 0.9.0, flax 0.12.6, Kinetix, VerifAI/Scenic (pins of the local test) | Part 2 (train and falsify) |
| `setup_sfl.sh` | `.venv-sfl`: py3.11, **jax 0.4.30 / flax 0.8.5**, VerifAI/Scenic | Part 1 SFL cell, Part 3 |

- The SFL code is 2024-era. It uses `jax.tree_map` and older flax internals, both removed in
  current JAX. `setup_sfl.sh` pins the versions that ran in our local test. On GPU these need
  `jax-cuda12-plugin==0.4.30`, which the pinned `jax[cuda12]==0.4.30` pulls in. If your CUDA
  driver is too old for it, upstream's container `nvcr.io/nvidia/jax:23.10-py3` also works.
- If your cluster needs `module load cuda` or similar, put it in `~/.bashrc` or at the top of
  each `.sbatch`.
- wandb runs offline. Run `wandb sync` afterwards if you want the dashboards.
- Nothing renders. Every job is headless (`MPLBACKEND=Agg`, `SDL_VIDEODRIVER=dummy`); Kinetix's
  pygame renderer is never called.

## What each part does

**Part 1, Maze.** Training uses JaxUED's own example scripts at their paper settings (30k
updates). Falsification starts after training finishes (SLURM dependency). For each
checkpoint it runs 2 search spaces × 2 VerifAI samplers (`ce`, `mab`: the two best in a local
prelim, compared head to head) × 3 falsifier seeds, at 5000 levels each. The spec is "solves the level in at
least 5 of 10 attempts". An exact BFS oracle certifies every level solvable before it counts as
a failure, so each counterexample comes with a shortest solution. The record format is in
[docs/failure_records.md](../docs/failure_records.md).

**Parts 2 and 3** follow the same train → falsify pattern (spec: solves in at least 5 of 10
attempts; samplers `ce` and `mab`, compared head to head; 3 falsifier seeds). A failure counts only once a
solvability witness exists:
- Kinetix (`atlas/kinetix`; 1000 levels per space). Levels are perturbations of the 10 hand-designed S
  evaluation levels the agents are evaluated on. `pos` moves the agent (green) and goal (blue)
  components rigidly, clipped to the arena; `phys` scales gravity, friction, motor and thruster
  power by 0.5–1.5. Levels where shapes overlap are rejected. The physics is deterministic, so a
  witness is an action sequence that solves the level when replayed open loop: from the policy's own
  solved attempts, or else from random and cross-entropy open-loop search. Every witness is replayed
  before the level counts. Levels with no witness are recorded as invalid, never as failures.
- JaxNav (`dr`, `seg` spaces; 5000 levels): BFS plus a scripted controller.

## The algorithm × environment matrix

| | Maze | Kinetix | JaxNav |
|---|---|---|---|
| DR | JaxUED | Kinetix | SFL repo |
| minimax | JaxUED + patch | — (not in Kinetix) | **port** (ours) |
| PLR | JaxUED | Kinetix | SFL repo |
| Robust PLR | JaxUED | Kinetix | SFL repo |
| ACCEL | JaxUED | Kinetix | SFL repo |
| SFL | SFL repo | Kinetix | SFL repo |

17 cells run. A **port** cell is one no paper ships code for, so we wrote it on top of the
environment's own codebase and kept it as close to the paper's algorithm as we could (only
minimax on JaxNav). Our changes to paper code:

- `patches/jaxued_gif.patch`: logs videos as GIFs, so no ffmpeg is needed.
- `patches/jaxued_minimax.patch`: adds `maze_paired.py --minimax`, where the adversary maximises
  the student's negative return. It reuses the PAIRED script, so the unused antagonist still
  trains, and minimax is the slowest Maze job.
- `patches/kinetix_acl27.patch` (Kinetix): bug fixes only, needed on current JAX (0.9):
  `jax.tree.map` for the removed `jax.tree_util.tree_map` alias, SFL's nested `shard_map` (run
  on one device), a boolean mask in SFL's buffer update, and the `create_empty_env` signature used
  by ACCEL's mutator. Kinetix's `ued=accel` config lacks a key and crashes, so ACCEL runs as the
  `ued=plr` config with `use_accel=true num_edits=5`, which is the same algorithm. DR is
  `replay_prob=0`; Robust PLR is `exploratory_grad_updates=false`.
- `patches/sfl_minimax.patch` (SFL repo): **minimax on JaxNav**, `sfl/train/jaxnav_minimax.py`
  and its config. A feed-forward PPO adversary edits a 9×9 inner grid cell by cell: goal, then
  start, then 48 wall placements. Its reward is the negative mean protagonist return on the
  level, given at the last edit. No solvability is enforced, as in minimax; we log the
  solvable fraction (4-connected flood fill). The protagonist, its PPO update and the
  evaluation are the SFL repo's `jaxnav_plr` code, imported unchanged.
- Kinetix budget: 384 PPO updates × 2048 envs × 256 steps = 201M steps (the paper trains 1B+),
  evaluated on the 10 S levels every 32 updates. SFL rescores its buffer every 32 updates with
  2 batches (paper 16), so its learnability rollouts cost about as much as training.
- Part 3 uses the paper's single-agent JaxNav sweep (`env.num_agents=1 env.test_set=single`).
  The upstream ACCEL sweep has a typo (`use.USE_ACCEL`), so we pass `ued.USE_ACCEL=True`.
  `jaxnav_plr`'s default evaluation set has 4 agents and crashes with one, so DR/PLR/Robust
  PLR/ACCEL/minimax pass the single-agent set `sampled_tc_100e_1a.pkl` (the one `jaxnav_sfl` uses).

PAIRED was dropped on 2026-10-04: it trains three networks and was too slow. CarRacing and
BipedalWalker (DCD, PyTorch) were dropped on 2026-10-06 for Kinetix: pixel rendering under Xvfb
and 300M+ step budgets made them ~2/3 of the suite's GPU hours.

## FAQ

**Why Kinetix?** It is a standard, recent JAX UED benchmark (FLAIROx, the SFL group) with
DR, PLR, Robust PLR, ACCEL and SFL built in, symbolic observations (no rendering), and a shared
physics engine across very different tasks, so failures are about the curriculum rather than one
environment's quirks. A run costs a fraction of CarRacing or Bipedal in DCD.

**Why does SFL have its own setup?** The SFL repo needs older JAX (above). SFL itself is
no longer a separate part: its Maze cell runs in Part 1, and Part 3 is the JaxNav environment.

## Smoke test

`bash cluster/submit.sh smoke` (under 1 h) runs, on one seed with tiny budgets, a train → falsify
chain per codebase:

- Maze: Robust PLR (JaxUED) and SFL (SFL repo) training, then falsification of both;
- JaxNav: SFL training then falsification, plus minimax training;
- Kinetix: tiny ACCEL and SFL training (32 envs, 131k steps), then falsification of both, plus a
  throughput probe (DR and SFL at full width, 16.8M steps, under `runs/probe`). Its log ends with
  `done ... steps/s=`; the 6 h Kinetix limit needs at least ~20k steps/s (DR). If it is slower,
  lower `STEPS` or raise the limit before `submit.sh 2`.

All jobs should exit 0 and write `summary.json` files under `runs/smoke/*_falsify/`. Every
smoke line was run locally (CPU, headless) before shipping (see Status).

## Status (2026-10-06)

- **Local CPU validity run passed.** Run on 2026-10-04 at smoke budgets, which are not results:
  - Maze Robust PLR: JaxUED train → checkpoint → falsify, which wrote `summary.json` (44/47 counterexamples).
  - Maze SFL: SFL repo train → `model.safetensors` → falsify (46/47).
  - JaxNav SFL: train → `model.safetensors`.
  - The atlas unit tests pass (8/8).
- **Kinetix, run locally and headless on 2026-10-06** (CPU, tiny budgets, `kinetix_acl27.patch`
  applied to the pinned clone): all 5 algorithms train and checkpoint; each checkpoint loads in
  `atlas.kinetix.falsify`, and falsification found replay-verified counterexamples on all 4 checked
  (untrained policies, ~16–18 s for 10 levels). The smoke chain was also run through
  `train.sbatch` and `falsify.sbatch` themselves.
- **Minimax on JaxNav** (our port): unit tests pass (action masking, cell→world geometry checked
  against the env's own collision map, the JAX flood fill against a Python BFS on 300 levels),
  and it trains, evaluates and saves `model.safetensors` + `adversary.safetensors`. Learning check
  (32 levels, 512-step rollouts, 150 updates): protagonist return on adversary levels drops from
  -5.7 to -6.7, then recovers to -5.8 as the protagonist adapts; levels stay hard (about 44%
  solvable, about 34 walls). Short tests need `learning.NUM_STEPS` >= the env's 500 max steps.
- `DRY_RUN=1 submit.sh all` produces 30 / 25 / 30 train tasks for Parts 1 / 2 / 3, each with a
  falsify array that waits on it (`afterok`).
- **Falsification cost, measured locally per level on one CPU** (oracle included; untrained
  policies, so trained ones may differ):

  | part | s / level | full task (3 falsifier seeds; processes in parallel) | limit |
  |---|---|---|---|
  | 1 Maze | ~0.04 | ~10 min (5000 levels, 2 spaces) | 1 h |
  | 2 Kinetix | ~1.2–1.8 | ~1–1.5 h (1000 levels, 2 spaces) | 4 h |
  | 3 JaxNav | ~0.4 | ~1.8 h (5000 levels, 2 spaces) | 4 h |

- **Samplers.** Only cross-entropy (`ce`) and the bandit (`mab`) run, head to head, as asked.
  On the Maze preliminary, relative to random sampling: counterexample rate ce 4.5× / mab 2.5×
  (`dr` space) and ce 5.9× / mab 5.2× (`seg` space); distinct failure modes about the same
  (0.9–1.1×). The cost: adaptive samplers spend 2–3× more of the budget on unsolvable
  (invalid) levels (dr: ce 21%, mab 12%, random 7%). Sampler compute itself is negligible;
  time goes to simulation and the solvability oracle.
- **Training wall-clock has not been measured on GPU.** All three parts are JAX on one GPU.
  Kinetix: the 6 h limit assumes ≥ ~20k steps/s (SFL spends about as much again on scoring);
  the smoke probe measures it. Maze and JaxNav are expected to take a few hours; their 12 h
  limit is a guess.
- **Total compute (NSEEDS=5, 85 train tasks):** Kinetix at most 6 × 25 = 150 GPU-h, Maze and
  JaxNav at most 12 × 30 each, so at most ~870 GPU-hours at the limits and likely well under
  half that. Falsification is CPU only (~85 tasks × 1–4 h × 8 CPUs).
- Not testable locally: GPU timing, falsify rates with trained checkpoints, and the cluster-side
  venv builds.
