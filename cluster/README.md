# UED failure atlas: cluster runbook

This trains six unsupervised-environment-design (UED) algorithms on the environments from their
own papers, using the papers' own code. It then searches each trained agent for levels it fails
on (VerifAI falsification). Everything here is a SLURM array job. Results come back as one
tarball.

## TL;DR

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas && cd atlas
bash cluster/setup/clone.sh && bash cluster/setup/setup_jax.sh \
  && bash cluster/setup/setup_dcd.sh && bash cluster/setup/setup_dcd_falsify.sh \
  && bash cluster/setup/setup_sfl.sh                                     # once, login node
bash cluster/submit.sh smoke     # <1 h, all 4 parts; check runs/slurm/*.out before going on
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
| **1** | Maze, 13×13 (PAIRED / Robust PLR / ACCEL papers) | JaxUED; SFL repo for the SFL cell | DR, PLR, Robust PLR, ACCEL, minimax, SFL | 30 train + 30 falsify | train 1 GPU, 12 h; falsify 8 CPU (no GPU), 1 h |
| **2** | CarRacing Bezier → F1 tracks (Robust PLR paper) | DCD (PyTorch) | DR, minimax, PLR, Robust PLR, ACCEL, SFL | 30 train + 30 falsify | train 1 GPU, 16 CPU, **Xvfb**, 2 chained 24 h segments; falsify 2 CPU, 18 h |
| **3** | BipedalWalker (ACCEL paper; 300M of the paper's 2B steps) | DCD (PyTorch) | DR, minimax, PLR, Robust PLR, ACCEL, SFL | 30 train + 30 falsify | train 1 GPU, 16 CPU, 18 h; falsify 2 CPU, 8 h |
| **4** | JaxNav, single agent (SFL paper) | SFL repo (JAX) | DR, minimax, PLR, Robust PLR, ACCEL, SFL | 30 train + 30 falsify | train 1 GPU, 12 h; falsify 8 CPU (no GPU), 4 h |

Each array task is one (algorithm, seed), with 5 seeds per algorithm (`NSEEDS=10` doubles
it). Task `i` is algorithm `ALGOS[i / 5]` and seed `i % 5`. Run a subset with e.g. `ALGOS="dr rplr" bash cluster/submit.sh 1`
(or `CONFIGS=...` for Parts 2 and 3).

| directory | contents |
|---|---|
| `1_maze/` | `train.sbatch` (JaxUED and SFL training), `falsify.sbatch` (VerifAI falsification) |
| `2_carracing/`, `3_bipedal/` | `train.sbatch` runs one line of a DCD command file per task (made by `dcd_cmds.sh`); `falsify.sbatch` (CPU; finds the checkpoint with `dcd_ckpt.sh`) |
| `4_jaxnav/` | `train.sbatch` (SFL repo's `jaxnav_plr` / `jaxnav_sfl`, our `jaxnav_minimax`), `falsify.sbatch` |
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
| `setup_dcd_falsify.sh` | `.venv-dcdf`: py3.11, torch + gym 0.15.7 + VerifAI/Scenic (exact local freeze) | Parts 2, 3 falsify |
| `setup_sfl.sh` | `.venv-sfl`: py3.11, **jax 0.4.30 / flax 0.8.5**, VerifAI/Scenic | Part 1 SFL cell, Part 4 |

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
checkpoint it runs 2 search spaces × 2 VerifAI samplers (`ce`, `mab`: the two best in a local
prelim, compared head to head) × 3 falsifier seeds, at 5000 levels each. The spec is "solves the level in at
least 5 of 10 attempts". An exact BFS oracle certifies every level solvable before it counts as
a failure, so each counterexample comes with a shortest solution. The record format is in
[docs/failure_records.md](../docs/failure_records.md).

**Parts 2–4** follow the same train → falsify pattern (spec: solves in at least 5 of 10
attempts; samplers `ce` and `mab`, compared head to head; 3 falsifier seeds). A failure counts only once a
solvability witness exists:
- CarRacing (`dr`, `sketch` track spaces; 500 levels): an in-env pure-pursuit controller,
  tried at finer action repeats.
- Bipedal (`dr` terrain vector; 1000 levels): the scripted backtracking walker, the other
  algorithms' checkpoints of the same seed (`--ref-ckpt`), or the policy itself. Without
  reference checkpoints the hard counterexamples are limited to near-flat terrain.
- JaxNav (`dr`, `seg` spaces; 5000 levels): BFS plus a scripted controller.

The DCD falsifiers run on CPU in their own py3.11 venv (`.venv-dcdf`; DCD's training env is
py3.8), with one process per sampler. DCD uses its paper configs (seeds 88–97), so DCD falsifier
records sit under `s<88 + trial>`. Afterwards, DCD's own benchmarks can be run with
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

`bash cluster/submit.sh smoke` (under 1 h) runs, on one seed with tiny budgets, a train → falsify
chain per codebase:

- Maze: Robust PLR (JaxUED) and SFL (SFL repo) training, then falsification of both;
- JaxNav: SFL training then falsification, plus minimax training;
- CarRacing and Bipedal: the DCD SFL port (`cr_sfl`, `bipedal_sfl`) training, then falsification.

All jobs should exit 0 and write `summary.json` files under `runs/smoke/*_falsify/`. Every
smoke line was run locally (CPU, headless) before shipping (see Status).

## Bring results home

```bash
bash cluster/collect.sh   # -> runs/atlas_collect_<date>.tgz: falsifier records, eval CSVs,
                          #    configs, slurm logs, final checkpoints only
```

## Status (2026-10-05)

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
- `DRY_RUN=1 submit.sh all` produces 30 train tasks per part, and for Parts 1–4 a falsify array
  that waits on the train array (`afterok`).
- **Parts 2–4 falsify, run locally end to end on 2026-10-05** (headless, smoke budgets, fake
  `RUNS` with linked checkpoints). Every `falsify.sbatch` exited 0, and `collect.sh` packed the
  10 resulting `summary.json` files.
  - Bipedal: found one reference checkpoint and skipped a missing one.
  - CarRacing: both spaces.
  - JaxNav: `JAX_PLATFORMS=cpu`, found counterexamples in both spaces.
  Untested on the cluster: `.venv-dcdf`, and the VerifAI install in `.venv-sfl`.
- **Resume, tested locally on 2026-10-05.** A DCD run killed mid-training resumes from its
  `model.tar` (same update counter, SFL buffer restored); a finished run resumes into an empty
  loop and exits. `submit.sh` therefore chains `SEGMENTS=2` 24 h train arrays for Part 2, whose GPU speed is a guess
  (`--dependency=afterany`, so the second segment starts even if the first timed out), and
  falsify waits on the last segment (`afterok`). Part 3 runs one segment (`SEGMENTS=2 bash cluster/submit.sh 3` if it times out).
- **Falsification cost, measured locally per level on one CPU** (oracle included; untrained
  policies, so trained ones may differ):

  | part | s / level | full task (3 falsifier seeds; samplers in parallel) | limit |
  |---|---|---|---|
  | 1 Maze | ~0.04 | ~10 min (5000 levels, 2 spaces) | 1 h |
  | 2 CarRacing | ~13 | ~11 h (500 levels, 2 spaces) | 18 h |
  | 3 Bipedal | ≤5 | ≤4 h (1000 levels, dr space only) | 8 h |
  | 4 JaxNav | ~0.4 | ~1.8 h (5000 levels, 2 spaces) | 4 h |

- **Samplers.** Only cross-entropy (`ce`) and the bandit (`mab`) run, head to head, as asked.
  On the Maze preliminary, relative to random sampling: counterexample rate ce 4.5× / mab 2.5×
  (`dr` space) and ce 5.9× / mab 5.2× (`seg` space); distinct failure modes about the same
  (0.9–1.1×). The cost: adaptive samplers spend 2–3× more of the budget on unsolvable
  (invalid) levels (dr: ce 21%, mab 12%, random 7%). Sampler compute itself is negligible;
  time goes to simulation and the solvability oracle.
- **Bipedal counterexamples are relative.** A Bipedal level counts as a counterexample only if
  some reference checkpoint (or the policy itself on another attempt) finishes it, because
  there is no exact solver. With no reference checkpoints every level is "invalid".
- **Training wall-clock** has not been measured on GPU. Estimates: CarRacing DR ~11–13 h if 16
  processes reach 120–140 steps/s (70/s measured on 8 local processes), SFL ~2× (~25 h);
  Bipedal at 8.8k steps/s (measured): the paper's 2B steps would take ~63 h (SFL ~83 h), so we
  train 300M steps, ~10 h (SFL ~13 h). No DCD schedule depends on the total, so this is the paper
  run stopped early; `DCD_ARGS=--num_env_steps=2000000000` restores it. Maze and JaxNav (JAX on
  GPU) are expected to take a few hours; their 12 h limit is a guess.
- **Total compute (NSEEDS=5, 120 train tasks):** CarRacing ~12 GPU-h × 25 + ~25 × 5, Bipedal
  ~10 × 25 + ~13 × 5, Maze and JaxNav at most 12 × 30 each, so about 1,000–1,500 GPU-hours
  at the limits and likely under 1,000. Falsification is CPU only. Limits are about 1.5–2× the
  estimates.
- Not testable locally: the cluster's conda DCD env (`dcd_task.sh`), Xvfb rendering speed,
  GPU timing, falsify rates with trained checkpoints, and the cluster-side venv builds.
