# Maze GPU Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command runs the suite's Maze part (papers' JAX code, paper budget, 10 seeds, three-section exam) and the CartPole bridge check on NVIDIA GPUs, and packs one results tarball.

**Architecture:** Extend `cluster/` (pinned JaxUED and SFL repos, SLURM or local-GPU launcher) and `atlas/maze/` (JaxUED checkpoint loader, BFS oracle, Scenic spaces). New: a Maze exam builder and grader that write the classic suite's `scores.csv` format, so `acl_bench.suite.analyze` reads Maze unchanged in kind; a CartPole bridge that trains with JaxUED's gymnax PLR and grades with our own PyTorch grader.

**Tech Stack:** JAX 0.10.2 / flax / orbax (`.venv-jax`, `cluster/setup/setup_jax.sh`), JaxUED @ 0f8f128, SFL repo @ 8b14e2e, VerifAI + Scenic, PyTorch (`.venv`) for the bridge grading and the analysis, bash + SLURM.

**Spec:** `docs/superpowers/specs/2026-10-28-maze-gpu-suite-design.md` (extends `docs/superpowers/specs/2026-10-28-ued-generalization-suite-design.md`).

## Global Constraints

- Maze methods and budget: DR `maze_dr.py`, PLR `maze_plr.py --exploratory_grad_updates`, RPLR `maze_plr.py`, ACCEL `maze_plr.py --use_accel`, minimax `maze_paired.py --minimax`, all at `--num_updates 30000` (32 envs x 256 steps = 245,760,000 env steps); SFL `minigrid_sfl.py` at 245,760,000 training steps, scouting not charged (user decision).
- One training box for every Maze method: 13x13 interior, at most 25 walls, 5x5 view, 250 steps. SFL runs at 25 walls, not its default 60.
- Seeds: scored Maze runs 0-9 per method (60 runs); reference agents DR seeds 100-109 (10 runs, never scored).
- Maze exam: random 300 BFS-solvable levels from the 25-wall generator; VerifAI = cross-entropy search over the 25-wall `dr` space, 3,000 iterations per reference agent, keep a level if >= 3 of 10 reference agents fail it, BFS-solvable, capped at 3,000 levels (seeded uniform subset; full pool kept); custom = SixteenRooms, SixteenRooms2, Labyrinth, LabyrinthFlipped, Labyrinth2, StandardMaze, StandardMaze2, StandardMaze3, plus empty, longest start-to-goal path, goal in a dead end, single winding corridor, 30 walls, 50 walls. Every level played 10 times with the agent's stochastic policy; pass = solved >= 5 of 10. Frozen with sha256 digests before any scored grading. Exam seed 20261028.
- `results/suite/maze/scores.csv` columns exactly as the classic suite: arm, seed, section, check, n_questions, pass_rate, play_rate, steps, scouted_steps, budget, n_checks, wall_time_sec. Arms DR, PLR, RPLR, ACCEL, SFL, MINIMAX.
- Bridge: JaxUED `examples/gymnax/gymnax_plr.py` on CartPole with our box (length 0.25-1.5, masspole 0.05-0.5, masscart 0.5-2.0, force_mag 4-16, init_range 0.05-0.3), our PPO except the rollout (8 envs x 512 steps = 4,096 samples/update, JaxUED's rollout length: its loop resets levels every update, so a 128-step rollout never finishes a 500-step episode; ruling in the ledger), 4 minibatches, 4 epochs, lr 3e-4, Adam eps 1e-5, gamma 0.99, lambda 0.95, clip 0.2, max grad norm 0.5, the 64-64 tanh MLP of `acl_bench/ppo.py`; entropy and value coefficients copied from `acl_bench/plr/fast.py`), our replay (replay 0.8, MaxMC, temperature 0.3, staleness 0.3, buffer 1000), 614,400 steps; arms DR (replay 0), PLR, RPLR; 20 seeds; no ACCEL. Graded by our PyTorch grader on `frozen_sets/suite/cartpole`. Reading: passes if every arm's mean is within 0.10 of batch 21's on VerifAI and random, and DR > PLR and DR > RPLR on VerifAI.
- The papers' code changes only through documented patches in `cluster/patches/`.
- Target machine (user, 2026-10-28): the professor's computer, NVIDIA GPU(s), no SLURM. The README covers only that case; the existing SLURM path stays in the code, untested and unmentioned. With one GPU the 70 Maze runs take days, so local mode runs several jobs per GPU (`JOBS_PER_GPU`, default chosen from the smoke test's measured GPU memory and speed, `XLA_PYTHON_CLIENT_MEM_FRACTION` set per job) and the README states the expected time for 1, 2 and 4 GPUs.
- No GPU here: every path is verified on the Mac CPU (`ALLOW_CPU=1`, JAX CPU build) at tiny budgets. Never claim GPU behaviour was tested.
- While the LunarLander pipeline runs (`results/suite/pipeline_ll.sh`), do not edit `acl_bench/plr/`, `acl_bench/suite/run.py`, `exam.py`, `arms.py`, `acl_bench/envs/`, or `acl_bench/exam/` (spawned workers import them). `acl_bench/suite/analyze.py` is safe.

## Review Focus

1. A checkpoint from SFL (safetensors, final params) and one from JaxUED (orbax) both load and grade; a missing or partial checkpoint (job killed at the time limit) is reported per run, not silently skipped or graded as zeros.
2. A rerun after a crash resumes: finished training runs, a built exam and graded rows are not redone or duplicated, and a mismatched exam digest refuses to grade.
3. The VerifAI section is built only after all 10 reference agents exist; with fewer it stops with a clear message (the "3 of 10" rule needs exactly 10).
4. The two maze encodings (atlas 13x13 interior vs acl_bench 15x15 with border; (x, y) vs [row, col]) round-trip: a named maze graded through the exam is the same maze JaxUED's prefab loader returns.
5. On a machine with N local GPUs, N training jobs run at once and each sees one GPU; with no GPU and no `ALLOW_CPU=1` the script refuses.

---

### Task 1: Audit the reused code against the papers before building on it

Pre-existing code the results rest on gets its own gate, before any costly step.

**Files:**
- Create: `docs/maze_gpu_audit.md`
- Possibly create/modify: `cluster/patches/*.patch` (documented fixes only)

- [ ] **Step 1:** Read `atlas/maze/policy.py`, `oracle.py`, `space.py`, `falsify.py`, `cluster/1_maze/train.sbatch`, `cluster/submit.sh`, `cluster/run_maze.sh`, and in the pinned checkouts `third_party/jaxued/examples/maze_plr.py`, `maze_dr.py`, `maze_paired.py`, `third_party/sfl/sfl/train/minigrid_sfl.py` and its config.
- [ ] **Step 2:** Check and record, each with file:line evidence and a verdict (OK / defect / deviation):
  - JaxUED PLR/RPLR/ACCEL settings match the Robust PLR and ACCEL papers' Maze settings (`docs/ued_atlas.md` section 1) and the suite spec table; ACCEL's mutations are capped at 25 walls.
  - `maze_plr.py --exploratory_grad_updates` really is PLR (gradients on new levels) and the default is Robust PLR.
  - SFL: how to set 25 walls and 245,760,000 steps through its config; that its scouting N/K/T/rho match the spec (N=5000, K=1000, T=50, rho=0.5); how its scouting steps can be counted for reporting.
  - `oracle.solve` uses JaxUED's movement rule (turn left/right, forward; goal entered by a forward move) and the 250-step limit.
  - `policy.run` plays the agent's stochastic policy with the LSTM state reset per episode, on exactly the level given.
  - minimax patch: adversary reward is the negative student return; student network matches.
- [ ] **Step 3:** Any defect: fix it as a patch under `cluster/patches/` (applied by `cluster/setup/clone.sh`), with a test or a CPU reproduction showing before/after.
- [ ] **Step 4:** Commit `docs/maze_gpu_audit.md` and any patches.

### Task 2: Maze exam sections: random and custom

**Files:**
- Create: `atlas/maze/suite.py`
- Test: `tests/test_maze_suite.py`

**Interfaces:**
- Produces: a level is `(walls: (13,13) bool, agent: (x,y), agent_dir: int, goal: (x,y))`, the same tuple `space.build` returns. `MazeSet` = arrays `walls (N,13,13) bool, agent (N,2), agent_dir (N,), goal (N,2)` plus `labels: list[str]` (custom only) and `ref_fail_frac (N,)` (verifai only). `save_set(path, name, mazeset)` / `load_sets(dir) -> dict[str, MazeSet]` write `<name>.npz` and a `manifest.json` with each set's sha256 over its arrays (same scheme as `acl_bench/exam/sets.py`); loading checks the digests and raises on mismatch.
- `random_section(n=300, seed=20261028) -> MazeSet`: 25-wall levels drawn as JaxUED's own 25-wall level generator draws them, BFS-solvable only (`oracle.solve`).
- `custom_section() -> MazeSet` with `labels` in the order of the Global Constraints list.

- [ ] **Step 1: Write the failing tests**
  - `test_random_section_is_solvable_and_in_box`: 300 levels, every one `oracle.solve(...)["status"] == oracle.SOLVABLE`, at most 25 walls each; same seed gives identical arrays.
  - `test_custom_labels_and_solvable`: 14 labels in the stated order; every level solvable; `empty` has 0 walls; `walls30` has 30 and `walls50` 50 walls; `longest_path` has a BFS path length at least as long as every other custom level's; `dead_end_goal`'s goal cell has exactly one free neighbour.
  - `test_named_mazes_match_jaxued_prefabs` (skipped without JAX): each named maze's walls, start and goal equal `Level.load_prefabs([name])`.
  - `test_digest_mismatch_refuses`: altering a saved array makes `load_sets` raise.
- [ ] **Step 2:** Run `.venv-jax/bin/python -m pytest tests/test_maze_suite.py -v`; expected FAIL (module missing).
- [ ] **Step 3:** Implement `atlas/maze/suite.py`. Named mazes come from JaxUED's prefabs (the papers' own held-out mazes); extremes are constructed in numpy and BFS-checked.
- [ ] **Step 4:** Tests pass.
- [ ] **Step 5:** Commit.

### Task 3: Maze exam: VerifAI section and the freeze

**Files:**
- Create: `atlas/maze/exam.py`
- Test: `tests/test_maze_exam.py`

**Interfaces:**
- Consumes: `atlas.maze.policy.Policy(ckpt_dir)`, `Policy.run(walls, agent, dir, goal, attempts, seed) -> (returns, lengths, pos)`; `falsify.make_scenario`; `space.build`; Task 2's `MazeSet`, `save_set`, `random_section`, `custom_section`.
- Produces: `solve_rates(policy, mazeset, plays=10, seed) -> (N,) float`; `verifai_section(policies: list[Policy], iters=3000, seed=20261028, min_fail=3) -> (capped MazeSet, full MazeSet)`; CLI `python -m atlas.maze.exam --ref-dir <checkpoints root> --out frozen_sets/suite/maze [--iters N]` that loads exactly the 10 REF runs, builds and freezes all three sections, and writes the full VerifAI pool to `frozen_sets/suite_pool/maze`.
- The search uses a 25-wall `dr` space (`n_walls` bounds (0, 26)); each agent's search has its own seeded RNG; a search's levels are pooled, deduplicated, regraded by all 10 agents, then filtered (>= 3 fail, solvable) and capped at 3,000 with `np.random.default_rng(SeedSequence([seed, 2]))`.

- [ ] **Step 1: Write the failing tests** (use a fake policy object with a `run` method, no JAX):
  - `test_keeps_levels_three_of_ten_fail`: 10 fake agents whose solve rate depends on wall count so that exactly the >= 3-fail levels are known; the section holds exactly those, and `ref_fail_frac` equals failures/10.
  - `test_cap_is_seeded_subset`: more than 3,000 kept -> 3,000, a subset of the full pool, same on rerun.
  - `test_refuses_without_ten_reference_agents`: CLI with 9 REF runs exits with a message naming 10.
- [ ] **Step 2:** Run, expected FAIL.
- [ ] **Step 3:** Implement. A rerun with the exam already frozen and the REF checkpoints unchanged does nothing (Review Focus 2).
- [ ] **Step 4:** Tests pass; then a CPU check with 2 tiny real JaxUED checkpoints from Task 6's smoke settings is deferred to Task 7.
- [ ] **Step 5:** Commit.

### Task 4: Maze grader and analysis support

**Files:**
- Create: `atlas/maze/grade.py`
- Modify: `acl_bench/suite/analyze.py` (seeds per env, MINIMAX arm)
- Test: `tests/test_maze_grade.py`, `tests/test_suite_analyze.py`

**Interfaces:**
- Consumes: `load_sets`, `solve_rates`, `Policy`.
- Produces: CLI `python -m atlas.maze.grade --runs <runs root> --exam frozen_sets/suite/maze --out results/suite/maze/scores.csv`: one row per (arm, seed, section) for every scored run's final checkpoint, `check` = `steps` = `budget` = 245,760,000, `n_checks` 1, `scouted_steps` 0 except SFL (its counted scouting steps, Task 1), `pass_rate` = share of questions with solve rate >= 0.5, `play_rate` = mean solve rate. Appends only missing rows; a run whose checkpoint is missing or short of the budget is listed on stderr and the command exits non-zero after grading the rest.
- `analyze.py`: required seeds per env (`maze`: 10, others 20) and arm MINIMAX accepted; everything else unchanged.

- [ ] **Step 1: Write the failing tests:** `test_rows_match_suite_columns` (fake policies, 2 runs -> 6 rows, exact column list); `test_resume_appends_only_missing`; `test_missing_checkpoint_reported_nonzero_exit`; in `test_suite_analyze.py`, `test_maze_requires_ten_seeds_and_accepts_minimax`.
- [ ] **Step 2:** Run, expected FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Tests pass (`.venv-jax` for grade, `.venv` for analyze).
- [ ] **Step 5:** Commit.

### Task 5: CartPole bridge

**Files:**
- Create: `atlas/bridge/cartpole_jaxued.py` (training wrapper), `atlas/bridge/export.py` (JAX params -> npz in our torch `Agent` state-dict layout), `atlas/bridge/grade_cartpole.py` (runs in `.venv`: loads npz into `acl_bench.ppo.Agent`, grades on `frozen_sets/suite/cartpole` with `acl_bench.suite.run.section_scores`, writes `results/suite/cartpole_jaxued/scores.csv` in suite columns), `atlas/bridge/compare.py` (the pre-registered reading -> `results/suite/analysis/bridge_cartpole.md`)
- Possibly create: `cluster/patches/jaxued_cartpole_bridge.patch` (start-spread field, our box, our PPO/network flags)
- Test: `tests/test_bridge.py`

- [ ] **Step 1: Write the failing tests:**
  - `test_physics_matches_ours` (skips without JAX): for 50 random levels in our box and 200 random states/actions, JaxUED's CartPole step equals `acl_bench.envs.cartpole.step` within 1e-5, and termination agrees.
  - `test_export_roundtrip`: a JAX MLP with known params, exported and loaded into `acl_bench.ppo.Agent`, gives the same action logits on 100 observations (within 1e-5).
  - `test_compare_reading`: hand-built score tables give PASS (all within 0.10, order kept) and FAIL (one arm off by 0.15).
- [ ] **Step 2:** Run, expected FAIL.
- [ ] **Step 3:** Implement. The training wrapper runs JaxUED's gymnax PLR code path, not a reimplementation; differences from upstream go in the patch and are listed in its header.
- [ ] **Step 4:** Tests pass; a 1-seed, 20k-step CPU run per arm trains, exports and grades end to end.
- [ ] **Step 5:** Commit.

### Task 6: Cluster pipeline for the suite

**Files:**
- Modify: `cluster/1_maze/train.sbatch`, `cluster/submit.sh`, `cluster/run_maze.sh`, `cluster/collect.sh`, `cluster/collect.sbatch`, `cluster/env.sh`
- Create: `cluster/1_maze/exam.sbatch`, `cluster/1_maze/grade.sbatch`, `cluster/1_maze/bridge.sbatch`
- Delete: `cluster/1_maze/falsify.sbatch` (superseded by the exam)

- [ ] **Step 1:** Training array: 6 scored methods x seeds 0-9 plus REF = DR seeds 100-109 (70 tasks); SFL with 25 walls and 245,760,000 steps; one GPU per task; time limits per method.
- [ ] **Step 2:** Stages with dependencies: smoke test (DR, RPLR, SFL, 1 seed, 50 updates; then exam build with `--iters 20` on 10 copies of the smoke DR checkpoint, and grading) -> training array and bridge array (60 tasks, CPU or GPU) -> exam (after the 10 REF tasks) -> grade (after training and exam) -> bridge grading and compare -> collect.
- [ ] **Step 3:** Local mode keeps the existing round-robin over every GPU `nvidia-smi` lists; adds `JOBS_PER_GPU` concurrent jobs per GPU (see Global Constraints); prints the GPU count and an estimated finish time from the smoke test's measured speed.
- [ ] **Step 4:** `collect` packs `results/suite/maze/`, `results/suite/cartpole_jaxued/`, `frozen_sets/suite/maze/`, `frozen_sets/suite_pool/maze/`, final checkpoints, configs and logs into `maze_results.tgz`.
- [ ] **Step 5:** Commit.

### Task 7: End-to-end CPU run on the Mac

- [ ] **Step 1:** `ALLOW_CPU=1 NUM_UPDATES=50 ... bash cluster/run_maze.sh` (tiny-budget switches added in Task 6 for this) runs every stage without SLURM, on CPU, to `maze_results.tgz`.
- [ ] **Step 2:** Unpack the tarball in the scratchpad; `python -m acl_bench.suite.analyze --envs maze --allow-partial` reads its `scores.csv` and prints the tables; the bridge comparison file exists.
- [ ] **Step 3:** Rerun the same command: nothing is retrained, rebuilt or regraded (Review Focus 2).
- [ ] **Step 4:** Record the run (commands, times, outputs) in `docs/maze_gpu_audit.md`; commit.

### Task 8: README, pre-registration, docs

**Files:**
- Modify: `README.md` (replace), `docs/library_log.md` (batch 22), `docs/cluster_details.md` (one line at the top saying it describes the earlier atlas run and pointing to the README)

- [ ] **Step 1:** README, plain and short, in the current README's voice (no mention of how the code was written): what it runs (Maze suite + CartPole bridge check, one paragraph); how to see your GPUs (`nvidia-smi -L`; `sinfo -o "%P %G"`); the one command; how long on 1, 2 and 4 GPUs, that it runs in the background and survives closing the terminal, how to watch progress and how to restart after a reboot (rerun the same command; it resumes); what to send back. No SLURM instructions. A numbered "what the script does" list that includes the bridge check as its own step.
- [ ] **Step 2:** Batch 22 pre-registration in `docs/library_log.md`: the Maze screen (one-sided Mann-Whitney per method vs DR on VerifAI, 10 seeds, Holm over Maze's 5 tests; descriptive with the classic tasks), SFL full training and the user's follow-up rule, the bridge reading.
- [ ] **Step 3:** Commit.
