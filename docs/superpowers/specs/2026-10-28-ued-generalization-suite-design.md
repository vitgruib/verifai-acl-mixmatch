# UED generalization suite: design

Date: 2026-10-28. Status: approved by the user 2026-10-28 (2-day total compute limit).

## Intent

Compare how well curriculum (UED) methods generalize, on one PyTorch codebase that runs on the
user's Mac (CPU, no GPU, no cluster). Every method trains on the same tasks with the same agent and
budget; only the rule that picks training levels differs. Every trained agent takes the same
three-section exam per task.

**H1 (the question):** for each method, how do its scores compare across the three exam sections
(random, VerifAI, custom)? No failure-type binning or search-efficiency metric: just the section
scores side by side.

**The VerifAI section is the screen (user's decision).** It is built from DR's failures, so DR is
at a disadvantage there by design. That is the point: it acts as a falsifier in its own right.
- **Primary test, pre-registered:** on each classic task, does any UED method score above DR on
  the VerifAI section? One-sided comparison per method over the 20 seeds, Holm-corrected across
  methods and tasks.
- **If some method improves:** follow-up testing on the random and custom sections, designed then.
- **If nothing improves, despite DR's disadvantage:** the line of work is dropped early.
- Maze, with 3 seeds, is reported descriptively on the same rule.

## Goals

- Methods: DR, PLR, Robust PLR, ACCEL, SFL on every task; minimax on Maze only.
- Tasks: Maze, CartPole, LunarLander, FrozenLake.
- Exam per task: random, VerifAI-found, custom (standard + extremes, including beyond-range cases).
- Method implementations reproduce the papers' algorithms and settings, in PyTorch, for
  consistency across tasks.

## Non-goals

- VerifAI in training (no VerifAI-picker methods for now). VerifAI is used only to build the exam.
- JAX, and running the papers' own code (JaxUED, DCD, SFL repo). The cluster package in `cluster/`
  stays as it is but is not part of this suite.
- PAIRED, REPAIRED (too slow on CPU; dropped earlier).
- Solvability proofs for continuous tasks. The grids get a cheap path check (below).
- Acrobot, Pendulum, MountainCar, PointNav, LavaGap, Hopper (considered and removed: redundant
  with CartPole/Maze, no natural pass/fail, or too small a layout space).

## Anti-goals

- A bloated suite: every task and exam section must answer something the others don't.
- Results dominated by impossible exam cases that no agent can pass.
- Presenting the VerifAI section as neutral: it is built from DR's failures (below), so DR is the
  baseline there, not a competitor.
- Short training that makes methods indistinguishable. Batch 19 (`docs/library_log.md`) found the
  methods do not separate at 10% of the Maze budget.

## Constraints

- Hardware: the user's Mac (Apple M5, 10 cores, 24 GB). CPU only. Runs follow the existing safety
  guard (`acl_bench/study/safety.py`: pause on battery, heat, low memory or disk; launch with
  `--allow-battery` per the user's standing preference).
- **Total compute: 2 days of the Mac, wall clock (user's limit)**, covering calibration, timing,
  reference agents, exam building and scored runs. Split (my call): about 12 hours for the classic
  tasks, about 36 hours for Maze. The classic estimate comes from the CartPole study (261 runs in
  4.7 hours, about 1 minute per run with runs in parallel); LunarLander is unmeasured and is timed
  first. If a measured cost breaks the split, the builder reports it to the user before cutting
  seeds or budget.
- Fair compute: equal total environment steps per method, counting every step a method spends,
  including SFL's scouting rollouts and ACCEL's edit evaluations.
- Seeds: 20 per method on CartPole, LunarLander, FrozenLake; 3 per method on Maze. Plus separate
  reference DR agents for building the VerifAI exam (10 per classic task, 5 on Maze), never scored.
- Pre-registration: the main comparisons and their pass/fail rules go into `docs/library_log.md`
  as a new batch before any scored run.

## Approaches considered

| | approach | trade-off | decision |
|---|---|---|---|
| A | classic tasks in `acl_bench` (PyTorch), Maze on the papers' JaxUED code | paper-exact Maze, two codebases | rejected: user wants no JAX |
| B | everything rebuilt in JAX on JaxUED | fast, one codebase; days of rewrite; no JAX LunarLander | rejected: no JAX |
| C | DCD (original PyTorch paper code) for Maze | paper code, but 250M steps with an LSTM on CPU is likely days per agent; SFL still a port | rejected |
| **D** | **everything in `acl_bench` (PyTorch); the methods reimplemented faithfully from the papers** | one codebase, consistent across tasks; Maze results are "the same benchmark, reimplemented", not a reproduction with paper code | **chosen by the user** ("reproduce the papers' code without JAX, for consistency") |

## Methods (training)

Reimplement each from its paper, matching the paper's algorithm and settings where the paper
defines them (the Maze column of `docs/ued_atlas.md` section 1, "Paper-exact settings", taken from
the DCD configs; SFL from its paper). Reuse `acl_bench/plr/` (level sampler, PLR, Robust PLR, ACCEL,
SFL already exist there for the classic tasks) where it matches the paper; fix it where it doesn't.

**One Maze protocol (my call):** JaxUED's 25-wall maze and its PLR/ACCEL defaults
(`third_party/jaxued/examples/maze_plr.py`), which reproduce Robust PLR and ACCEL on the 25-wall
box. All methods share that one training box; ACCEL's edits are capped at the box's wall limit,
so no method trains outside the box the exam is built on. This is not ACCEL's own 60-wall,
empty-room setup.

| method | rule for training levels | key settings (Maze, JaxUED defaults) |
|---|---|---|
| DR | uniform random levels from the training box | none |
| PLR | random levels + replay of high-score levels | replay 0.8, buffer 4000, score MaxMC, rank temperature 0.3, staleness 0.3, gradients on new levels |
| Robust PLR | as PLR, but gradient updates only on replayed levels | as PLR |
| ACCEL | Robust PLR + small edits to replayed levels | as Robust PLR, 5 tile flips per edit, capped at 25 walls |
| SFL | sample N levels, roll out, train on the most learnable p(1-p) | paper defaults N=5000, K=1000, T=50, rho=0.5 |
| minimax (Maze only) | an RL adversary places walls to minimize the agent's return | DCD's adversary network and PPO settings |

On the classic tasks the papers give no settings. Use the Maze settings scaled to the smaller budget
(buffer size and SFL's N, K in proportion to the number of training levels seen); record each
scaling in the pre-registration.

**Maze agent and budget.** The papers use an LSTM agent (256 units, conv encoder) and about 250M
steps. Maze gets whatever is left of the 2-day total (see Constraints), about 36 hours of the Mac.
Its 23 runs (18 scored + 5 reference) run in parallel waves, so the per-run budget is set by timing
the PyTorch Maze under that concurrency: the paper budget if it fits, otherwise what fits. Any
shortfall against the paper is reported with the results: batch 19 found the methods don't
separate at 10% of the budget, so a large shortfall weakens the Maze results.

**Maze reference agents (my call):** 5 instead of 10, keeping a question if at least 2 of the 5 fail
it (40%, as with 4 of 10).

**Order (my call):** the classic tasks are built and run first, then Maze. Their screen result comes
first, and the user can decide whether Maze is still worth its share of the budget.

**Fidelity check (my call): a soft check, not a gate.** After the first DR and Robust PLR Maze
runs, compare their named-maze solve rates with the published ones and report the comparison to
the user before the remaining Maze runs. A large gap is a finding to discuss, not an automatic stop.

**Classic-task budgets (my call):** CartPole keeps the training length calibrated in the earlier
study (`results/calibration/`). LunarLander and FrozenLake get a length calibrated the same way:
the point where DR's random-section score stops improving, measured before any scored run.

## Tasks and parameter boxes

"Training box" = the range DR samples during training. Defaults are the library's standard values.

| task | setting | training box | default | pass |
|---|---|---|---|---|
| CartPole | pole half-length | 0.25-1.5 | 0.5 | survive 500 steps |
| | pole mass | 0.05-0.5 | 0.1 | |
| | cart mass | 0.5-2.0 | 1.0 | |
| | push force | 4-16 | 10 | |
| | start spread | 0.05-0.3 | 0.05 | |
| LunarLander | gravity | -12 to -2 | -10 | episode return >= 200 |
| | wind power | 0-20 | 0 | |
| | turbulence power | 0-2 | 0 | |
| FrozenLake (random maps) | map size | 4-10 | 8 | reach the goal |
| | hole density | 0.1-0.3 | 0.2 | |
| | slip probability | 0-0.67 | 0.67 (Gymnasium's slippery ice) | |
| Maze | walls | up to 25 on a 13x13 interior; agent sees 5x5 | 25 | reach the goal within 250 steps |

Maze levels are encoded as the wall grid plus start and goal (not as a random seed), so ACCEL's
edits and VerifAI's search act on the layout. `acl_bench/envs/maze.py` already holds the layout in
the state and has the named mazes and a BFS check; its seed-based level encoding has to change.

CartPole keeps its existing environment and Scenic box (`acl_bench/envs/cartpole.*`). LunarLander
and FrozenLake are new environments with Scenic boxes in the same style.

**FrozenLake observation (my call):** an egocentric local window, like the Maze's (5x5 around the
agent, hole and goal channels), so one network plays every map size, including the beyond-range
sizes 11 and 15.

## The exam

Three sections per task, the same for every scored agent. Stochastic tasks (FrozenLake, LunarLander
with wind or turbulence) play each question 10 times; the score is the pass rate.

**1. Random.** Questions drawn uniformly from the training box. This is the in-distribution score.
DR trains on exactly this distribution, so report it as such, not as overall generalization.

**2. VerifAI.** Built once per task, then frozen:
- Train 10 reference DR agents (separate seeds, never scored).
- Run VerifAI's cross-entropy sampler against each reference agent, the same budget per agent, over
  the training box.
- Keep a question if at least 3 of the 10 reference agents fail it. On stochastic tasks an agent
  "fails" a question when its pass rate over 10 plays is below 0.5 (my call).
- Grids: drop any question the path check finds unsolvable. FrozenLake also drops any question
  whose optimal success probability (exact, by value iteration) is below 0.5 (my call), so heavy
  slip can't fill the section with near-impossible maps. The same filter applies to its custom cases.
- Freeze and checksum with `acl_bench/exam/sets.py`.

This section asks whether a method fixes the places where plain random training fails.

**3. Custom.**
- Continuous tasks: the standard case (all defaults), then for each setting, the others held at
  default, the setting at six values: min edge, max edge, slightly beyond each edge (10% of the
  range width past it), way beyond each edge (50% of the range width past it). A beyond value that
  is physically invalid or rejected by the library is skipped (for example pole mass below 0,
  LunarLander gravity outside Gymnasium's -12 to 0). No corner combinations.
  CartPole: 1 + 5 x 6 = 31 cases at most.
- Maze: the 8 named mazes (SixteenRooms, SixteenRooms2, Labyrinth, LabyrinthFlipped, Labyrinth2,
  StandardMaze, StandardMaze2, StandardMaze3), plus extremes: empty maze; longest possible
  start-to-goal path; goal in a dead end; a single winding corridor; 30 walls (slightly beyond); 50
  walls (way beyond).
- FrozenLake: Gymnasium's classic 4x4 and 8x8 maps, plus per-setting edges and beyond values as for
  continuous tasks (map size 11 and 15; hole density 0.32 and 0.4 on the high side, 0.08 and 0.0 on
  the low side; slip 0.73 and 0.9), plus a single-safe-path map.
- Grids: every custom case passes the path check.
- Report each custom case's score next to the best method's score on that case, and flag any case
  that no scored agent passes (it may be impossible).

## Results

Per task: a table of methods x exam sections (mean pass rate over seeds, with spread), plus the
custom section broken out per case. The screen decision comes from the VerifAI column; H1 is read
from the whole table.

## Left to the builder

File layout, module names, the test plan, how runs are parallelized across the 10 cores, the
cross-entropy sampler's search budget per reference agent, PPO settings for the classic tasks
(start from the existing `acl_bench` PPO), and any other choice not fixed above, recorded in the
pre-registration where it affects results.
