# UED failure atlas: cluster runbook

This trains six unsupervised-environment-design (UED) algorithms on the environments from their
own papers, using the papers' own code. It then searches each trained agent for levels it fails
on (VerifAI falsification). Everything here is a SLURM array job. Results come back as one
tarball.

## TL;DR

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas && cd atlas
bash cluster/setup/clone.sh && bash cluster/setup/setup_jax.sh \
  && bash cluster/setup/setup_dcd.sh && bash cluster/setup/setup_sfl.sh   # once, login node
bash cluster/submit.sh smoke     # ~15 min of GPU; check runs/slurm/*.out before going on
bash cluster/submit.sh all       # or one part at a time: 1, 2, 3, 4
bash cluster/collect.sh          # when done: runs/atlas_collect_<date>.tgz -> send back
```

Use `DRY_RUN=1 bash cluster/submit.sh all` to print every `sbatch` line without submitting.
Pass partition or account flags through `SBATCH_ARGS`, e.g. `SBATCH_ARGS="-p gpu -A mylab"`.
Paths can be overridden too, e.g. `RUNS=/scratch/$USER/atlas` (all paths are in
[`env.sh`](env.sh)).

## The four parts (one per environment)

| part | environment (paper) | code | algorithms | array tasks | per task (sbatch limit) |
|---|---|---|---|---|---|
| **1** | Maze, 13×13 (PAIRED / Robust PLR / ACCEL papers) | JaxUED; SFL repo for the SFL cell | DR, PLR, Robust PLR, ACCEL, minimax, SFL | 60 train + 60 falsify | 1 GPU; 24 h train, 8 h falsify |
| **2** | CarRacing Bezier → F1 tracks (Robust PLR paper) | DCD (PyTorch) | DR, minimax, PLR, Robust PLR, ACCEL, SFL | 60 | 1 GPU, 16 CPU, 48 h, **Xvfb** |
| **3** | BipedalWalker (ACCEL paper) | DCD (PyTorch) | DR, minimax, PLR, Robust PLR, ACCEL, SFL | 60 | 1 GPU, 16 CPU, 72 h (resubmit on timeout) |
| **4** | JaxNav, single agent (SFL paper) | SFL repo (JAX) | DR, minimax, PLR, Robust PLR, ACCEL, SFL | 60 | 1 GPU, 24 h |

Each array task is one (algorithm, seed), with 10 seeds per algorithm. Task `i` is algorithm
`ALGOS[i / 10]` and seed `i % 10`. Run a subset with e.g. `ALGOS="dr rplr" bash cluster/submit.sh 1`
(or `CONFIGS=...` for Parts 2 and 3).

| directory | contents |
|---|---|
| `1_maze/` | `train.sbatch` (JaxUED and SFL training), `falsify.sbatch` (VerifAI falsification) |
| `2_carracing/`, `3_bipedal/` | `train.sbatch` runs one line of a DCD command file per task (made by `dcd_cmds.sh`) |
| `4_jaxnav/` | `train.sbatch` (SFL repo's `jaxnav_plr` / `jaxnav_sfl`, our `jaxnav_minimax`) |
| `setup/` | `clone.sh` (upstream repos at pinned commits), one environment script per codebase |
| `dcd_configs/` | DCD grid configs that upstream lacks (below) |
| `patches/` | our patches to JaxUED, DCD and the SFL repo, applied by `clone.sh` (below) |
| `tests/` | headless unit and env tests for the ported cells (see `tests/README.md`) |
| `submit.sh`, `collect.sh` | submit a part; pack results |

## Setup notes

There are three Python environments, one per codebase, because their dependency pins are
incompatible:

| script | env | used by |
|---|---|---|
| `setup_jax.sh` | `.venv-jax`: py3.11, current `jax[cuda12]`, JaxUED, VerifAI, Scenic | Part 1 (JaxUED algorithms, falsifier) |
| `setup_dcd.sh` | conda env `dcd`: py3.8, DCD's original requirements | Parts 2, 3 |
| `setup_sfl.sh` | `.venv-sfl`: py3.11, **jax 0.4.30 / flax 0.8.5** | Part 1 SFL cell, Part 4 |

- The SFL code is 2024-era. It uses `jax.tree_map` and older flax internals, both removed in
  current JAX. `setup_sfl.sh` pins the versions that ran in our local test. On GPU these need
  `jax-cuda12-plugin==0.4.30`, which the pinned `jax[cuda12]==0.4.30` pulls in. If your CUDA
  driver is too old for it, upstream's container `nvcr.io/nvidia/jax:23.10-py3` also works.
- If your cluster needs `module load cuda` or similar, put it in `~/.bashrc` or at the top of
  each `.sbatch`.
- wandb runs offline. Run `wandb sync` afterwards if you want the dashboards.
- **CarRacing renders with pyglet.** On the cluster it runs under `xvfb-run` (the Part 2 sbatch
  refuses any command without it), which is DCD's own renderer and the default. If Xvfb is
  unavailable, `export ACL27_SOFT_RENDER=1` switches to our pure-numpy renderer
  (`envs/box2d/soft_render.py`, from `dcd_acl27.patch`): no window, no OpenGL. It draws the
  same 96×96 observation scene but is not pixel-identical to pyglet, so don't mix the two
  within one comparison. All our local CarRacing tests used it. Never run CarRacing on a
  desktop without it.

## What each part does

**Part 1, Maze.** Training uses JaxUED's own example scripts at their paper settings (30k
updates). Falsification starts after training finishes (SLURM dependency). For each
checkpoint it runs 2 search spaces × 3 VerifAI samplers (`random`, `ce`, `mab`, chosen in a
local prelim) × 3 falsifier seeds, at 5000 levels each. The spec is "solves the level in at
least 5 of 10 attempts". An exact BFS oracle certifies every level solvable before it counts as
a failure, so each counterexample comes with a shortest solution. The record format is in
[docs/failure_records.md](../docs/failure_records.md).

**Parts 2–4** train only for now. Their falsifiers (CarRacing track control points, the
Bipedal terrain vector, JaxNav) are written on our side against the returned checkpoints.
DCD uses its paper configs (seeds 88–97). Afterwards, DCD's own benchmarks can be run with
`python -m eval --xpid <xpid> --benchmark f1` (CarRacing) or `--benchmark bipedal`.

## The algorithm × environment matrix

| | Maze | CarRacing | BipedalWalker | JaxNav |
|---|---|---|---|---|
| DR | JaxUED | DCD | DCD | SFL repo |
| minimax | JaxUED + patch | DCD + derived config | DCD | **port** (ours) |
| PLR | JaxUED | DCD | DCD + derived config | SFL repo |
| Robust PLR | JaxUED | DCD | DCD | SFL repo |
| ACCEL | JaxUED | **port** (ours) | DCD | SFL repo |
| SFL | SFL repo | **port** (ours) | **port** (ours) | SFL repo |

All 24 cells run. A **port** cell is one no paper ships code for, so we wrote it on top of the
environment's own codebase and kept it as close to the paper's algorithm as we could. All
four ports were unit-tested and run end to end locally at smoke budgets (see Status). Our
changes to paper code:

- `patches/jaxued_gif.patch`: logs videos as GIFs, so no ffmpeg is needed.
- `patches/jaxued_minimax.patch`: adds `maze_paired.py --minimax`, where the adversary maximises
  the student's negative return. It reuses the PAIRED script, so the unused antagonist still
  trains, and minimax is the slowest Maze job.
- `patches/dcd_acl27.patch` (DCD):
  - **ACCEL on CarRacing**: `mutate_level` for Bezier tracks. Each edit moves one control point
    to a free neighbouring cell of the adversary's sketch grid, or adds or removes a point
    (4 to 12 points). Edited tracks stay in the adversary's level space. Config
    `cr_accel.json` is the ACCEL paper's Bipedal recipe (Robust PLR + 3 random edits on
    "easy" base levels) on CarRacing's Robust PLR settings.
  - **SFL in DCD** (`envs/runners/sfl.py`): a port of the SFL repo's teacher. Every N updates
    it scores M random levels by learnability p(1−p) of the current policy (p = success rate
    over the episodes it completes on that level) and keeps the top K. Each training process
    then gets a buffer level with probability ρ = 0.5 and a fresh random level otherwise.
    Success is `info['finish']`: lap completed (CarRacing) or end of course reached (Bipedal).
    Scoring rollouts are extra simulation, logged as `sfl_eval_steps_total` and not counted in
    `num_env_steps`. Configs: `cr_sfl.json`, `bipedal_sfl.json`.
  - Fixes needed to run 2021 DCD today: numpy aliases (`np.bool`, `np.float`, `fromstring`),
    `torch.load(weights_only=False)`, Box2D crashes on degenerate track tiles, and CarRacing
    level replay (random tracks used to serialise as empty tracks). Plus the optional soft
    renderer above.
- `patches/sfl_minimax.patch` (SFL repo): **minimax on JaxNav**, `sfl/train/jaxnav_minimax.py`
  and its config. A feed-forward PPO adversary edits a 9×9 inner grid cell by cell: goal, then
  start, then 48 wall placements. Its reward is the negative mean protagonist return on the
  level, given at the last edit. No solvability is enforced, as in minimax; we log the
  solvable fraction (4-connected flood fill). The protagonist, its PPO update and the
  evaluation are the SFL repo's `jaxnav_plr` code, imported unchanged.
- `dcd_configs/car_racing/cr_minimax.json`: DCD's `cr_paired.json` with `ued_algo=minimax`,
  `adv_normalize_returns=false` (the two keys separating DCD's own bipedal minimax/PAIRED configs).
- `dcd_configs/bipedal/bipedal_plr.json`: `bipedal_robust_plr.json` with exploratory gradient
  updates on.
- Part 4 uses the paper's single-agent JaxNav sweep (`env.num_agents=1 env.test_set=single`).
  The upstream ACCEL sweep has a typo (`use.USE_ACCEL`), so we pass `ued.USE_ACCEL=True`.
  `jaxnav_plr`'s default evaluation set has 4 agents and crashes with one, so DR/PLR/Robust
  PLR/ACCEL/minimax pass the single-agent set `sampled_tc_100e_1a.pkl` (the one `jaxnav_sfl` uses).

PAIRED was dropped on 2026-10-04: it trains three networks and was too slow.

## FAQ

**Why both JaxUED and DCD?** We always use the code the environment's paper released.
CarRacing (Robust PLR paper) and BipedalWalker (ACCEL paper) exist only in DCD (PyTorch).
For the Maze, JaxUED is the JAX reimplementation by the same group. It reproduces DCD's maze
results at roughly 100× the speed, so we use it there. JaxNav exists only in the SFL repo.

**Why was ACCEL on CarRacing special?** ACCEL improves levels by *mutating* them, so it needs
a level mutator. DCD has mutators for mazes and Bipedal terrain but none for Bezier race
tracks, and the ACCEL paper never ran CarRacing. We wrote one (above), so the cell is now a
port, as are SFL on CarRacing and Bipedal (there was no PyTorch SFL) and minimax on JaxNav
(there was no adversary generator for JaxNav).

**Why does SFL have its own setup?** The SFL repo needs older JAX (above). SFL itself is
no longer a separate part: its Maze cell runs in Part 1, and Part 4 is the JaxNav environment.

## Smoke test

`bash cluster/submit.sh smoke` runs, on one seed with tiny budgets:

- Robust PLR (JaxUED) and SFL (SFL repo) training on Maze;
- falsification of both checkpoints;
- SFL and minimax training on JaxNav.

All five jobs should exit 0. The falsify job should write
`runs/smoke/maze_falsify/*/s0/dr_random_r0/summary.json`. The same pipeline was run locally on
CPU before shipping (see Status).

## Bring results home

```bash
bash cluster/collect.sh   # -> runs/atlas_collect_<date>.tgz: falsifier records, eval CSVs,
                          #    configs, slurm logs, final checkpoints only
```

## Status (2026-10-04)

- **Local CPU validity run passed.** Run on 2026-10-04 at smoke budgets, which are not results:
  - Maze Robust PLR: JaxUED train → checkpoint → falsify, which wrote `summary.json` (44/47 counterexamples).
  - Maze SFL: SFL repo train → `model.safetensors` → falsify (46/47).
  - JaxNav SFL: train → `model.safetensors`.
  - The atlas unit tests pass (8/8).
- **DCD and the ports, run locally and headless** (CPU, `ACL27_SOFT_RENDER=1`, smoke budgets,
  `dcd_acl27.patch` and `sfl_minimax.patch` applied to the pinned clones):
  - CarRacing Robust PLR and Bipedal ACCEL (paper code): train, test on the F1 tracks and
    Bipedal test envs, checkpoint.
  - ACCEL on CarRacing: trains, and replayed levels carry edits (mean 3.4 edits per level).
  - SFL on CarRacing and Bipedal: unit tests (learnability, top-K buffer, ρ mixing, state
    round trip) pass, `info['finish']` reaches the runner through DCD's vec-env stack, and
    both train with SFL evaluations logged.
  - Minimax on JaxNav: unit tests pass (action masking, cell→world geometry checked against
    the env's own collision map, the JAX flood fill against a Python BFS on 300 levels), and
    it trains, evaluates and saves `model.safetensors` + `adversary.safetensors`.
    Learning check (32 levels, 512-step rollouts, 150 updates): protagonist return on adversary
    levels drops from -5.7 to -6.7, then recovers to -5.8 as the protagonist adapts; levels stay
    hard (about 44% solvable, about 34 walls). The adversary's signal has the right sign but is
    weak at this budget. Short tests need `learning.NUM_STEPS` >= the env's 500 max steps, or
    levels finish no episode and the adversary is scored on partial returns.
  - Both patches apply cleanly to fresh pinned clones, and `clone.sh` is idempotent.
- `DRY_RUN=1 submit.sh all` produces 60 tasks per part.
- Falsifiers for Parts 2–4 are not written yet; they need these checkpoints.
- Training wall-clock at full budget has not been measured on GPU. The limits above are
  generous guesses, except Bipedal: we measured about 63 h at 8.8k steps/s. SFL's level scoring adds
  extra simulation (about +31% on Bipedal, so ~83 h; about +96% on CarRacing), so the SFL tasks
  will likely exceed 72 h / 48 h. SLURM does not requeue on timeout. Either resubmit the timed-out
  array index with the same command file (`sbatch --array=<idx> cluster/3_bipedal/train.sbatch
  runs/.../cmds.txt`); DCD resumes from `model.tar`, saved every 100 updates, including the SFL
  buffer. Or set the limit up front: `SBATCH_TIMELIMIT=120:00:00 bash cluster/submit.sh 3`
  (`96:00:00` for part 2).
