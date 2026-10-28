# Maze GPU suite and CartPole bridge check: design

Date: 2026-10-28. Status: draft for the user's approval. Extends
`docs/superpowers/specs/2026-10-28-ued-generalization-suite-design.md` (the suite spec); everything
not stated here follows it.

## Intent

Run the suite's Maze part on NVIDIA GPUs with the papers' own JAX code, at the paper budget, instead
of the PyTorch reimplementation on the Mac (which cannot reach the paper budget). Add one check that
the PyTorch reimplementation behind the classic-task results behaves like the papers' code.

User decisions (2026-10-28):
- JAX is used for Maze only. CartPole, FrozenLake and LunarLander stay on the Mac in PyTorch
  (no JAX LunarLander or FrozenLake exists; the classic runs are already cheap).
- Maze: paper budget, 10 seeds per method.
- Add the CartPole bridge check, and list it in the run instructions.
- The README becomes the instructions for running this on GPUs. The user doesn't yet know how many
  GPUs they have, so the pipeline detects them.

## Goals

- One command runs the Maze suite and the bridge check on a SLURM cluster or on a machine with one or
  more NVIDIA GPUs, and packs one results tarball.
- The Maze exam follows the suite spec's recipe: random, VerifAI (from 10 reference DR agents'
  failures, kept if at least 3 of 10 fail), custom (named mazes and extremes).
- Results land in the same `scores.csv` format, so `acl_bench.suite.analyze` produces the screen
  and tables for Maze exactly as for the classic tasks.

## Non-goals

- Kinetix and JaxNav (the old atlas's later stages stay in `cluster/` but aren't run).
- The old per-agent falsification atlas and its H2 test (batch 20). Superseded by this suite.
- JAX versions of LunarLander or FrozenLake.

## Anti-goals

- Changing the papers' algorithms beyond documented bug fixes and the shared-box settings below.
- Pushing anything to the user's GPU machine that hasn't run end to end on the Mac at a tiny budget.
  The pipeline has never run on a GPU; the Mac CPU smoke test is the only verification available
  here, so the first GPU run starts with an automatic smoke test before the full run.

## Approaches considered

| | approach | trade-off | decision |
|---|---|---|---|
| A | extend `cluster/` (JaxUED, SFL repo pinned and patched; `atlas/maze` already plays JaxUED checkpoints and BFS-certifies levels) | reuses tested pieces; needs an exam builder and grader for JAX agents | **chosen (my call)** |
| B | port the PyTorch suite to the GPU | our code is CPU-bound on environment stepping; small speedup | rejected |
| C | new pipeline from scratch | throws away tested setup and patches | rejected |

## Maze part

**Methods and training (papers' code, pinned commits in `cluster/env.sh`).**

| method | code | budget |
|---|---|---|
| DR | JaxUED `maze_dr.py` | 30,000 updates x 32 envs x 256 steps = 245.76M steps |
| PLR | JaxUED `maze_plr.py --exploratory_grad_updates` | same |
| Robust PLR | JaxUED `maze_plr.py` | same |
| ACCEL | JaxUED `maze_plr.py --use_accel` | same |
| minimax | JaxUED `maze_paired.py --minimax` (existing patch) | same |
| SFL | SFL repo `minigrid_sfl.py` | 245.76M training steps (paper: 3e8) |

- Seeds 0-9 per method (60 runs). Reference agents: DR, seeds 100-109 (10 runs, never scored).
- **One training box (suite spec):** 13x13 interior, up to 25 walls, 5x5 view, 250 steps. SFL's repo
  defaults to 60 walls; it is run at 25 (my call: the suite spec puts every method in one box).
- **SFL compute (my call, check this):** SFL's scouting rollouts are not charged to its budget, as
  in its paper. On the classic tasks they were charged (SFL then trained on half the steps, which
  confounded its result). Here SFL trains for the same 245.76M steps as the others, and its extra
  scouting steps are reported next to the result. Charging them would mean patching the paper's code.

**Exam (frozen before any scored grading, same recipe as the classic tasks).**
- Random: 300 levels from JaxUED's 25-wall level generator, BFS-solvable only.
- VerifAI: the existing cross-entropy search (`atlas/maze/falsify.py`, Scenic space `dr`, the
  25-wall training box) against each reference agent, 3,000 levels per agent. Keep a level if at
  least 3 of the 10 reference agents fail it; BFS-solvable only; capped at 3,000 levels
  (amendment 1's rule).
- Custom: the 8 named mazes of the suite spec, plus empty maze, longest start-to-goal path, goal in a
  dead end, single winding corridor, 30 walls (slightly beyond), 50 walls (way beyond). BFS-solvable.
- Every level is played 10 times with the agent's own (stochastic) policy, as JaxUED evaluates;
  passing means solving it at least 5 of 10 times. Frozen with sha256 digests like the classic exams.

**Grading.** Final checkpoint of each scored run, graded on all three sections, written to
`results/suite/maze/scores.csv` in the classic tasks' columns. Runs on the GPU when present.

**Order on the cluster.** Smoke test (1 seed of RPLR, SFL and DR, 50 updates, plus exam build and
grading on those) -> the 70 training runs in parallel -> exam build (needs the 10 reference agents)
-> grading -> tarball.

## CartPole bridge check

**Question:** does JaxUED's PLR and Robust PLR code, run on our CartPole with our settings, score
like our PyTorch PLR and Robust PLR on our frozen CartPole exam?

- JaxUED's own `examples/gymnax/gymnax_plr.py` and its CartPole environment, with:
  - our training box (pole half-length 0.25-1.5, pole mass 0.05-0.5, cart mass 0.5-2.0, push force
    4-16, start spread 0.05-0.3). Start spread is not in JaxUED's CartPole level, so it is added;
  - our PPO settings and network (from `acl_bench/plr/fast.py`), our replay settings (replay 0.8,
    MaxMC, rank temperature 0.3, staleness 0.3, buffer 1000), our budget (614,400 steps);
  - arms DR (replay 0), PLR, Robust PLR, 20 seeds each.
- ACCEL is left out: that script's ACCEL is broken upstream (`mutate_level` is never defined), so
  running it would test our fix, not the papers' code (my call).
- Grading: the trained JAX weights are exported into our PyTorch network and graded by our own
  grader on `frozen_sets/suite/cartpole`, so the only difference is the training code.
- Before training, a physics check: our CartPole and JaxUED's step the same states to the same
  next states (within float tolerance) for a sample of levels.

**Pre-registered reading (my call).** Per arm, compare JaxUED's 20 seeds with our 20 (batch 21) on
the VerifAI and random sections. The reimplementation passes if every arm's mean is within 0.10 of
ours and the order DR > PLR, RPLR on VerifAI holds. If it fails, the classic-task results are
flagged and rechecked before they're reported. Cost: a few GPU-hours.

## README

Replaces the current README, which describes the old atlas run. Same plain style, for the person
running it:
1. What it runs (one paragraph): the Maze suite and the CartPole bridge check.
2. How to see your GPUs (`nvidia-smi -L`; on a cluster, `sinfo -o "%P %G"`).
3. One command to run it; SLURM is detected, otherwise every local GPU is used.
4. How long it takes, by number of GPUs, and how to watch progress.
5. What to send back.

## Constraints

- No GPU here. Everything is verified on the Mac CPU (JAX CPU build) at tiny budgets, end to end,
  including the tarball. GPU timing is unmeasured, so the time estimates in the README come from
  the smoke test on the user's machine, printed by the script.
- Compute ceiling: about 70 runs x 1-3 GPU-hours, plus SFL scouting, grading and the bridge check:
  roughly 100-250 GPU-hours. On SLURM, every job has a time limit.

## Left to the builder

File layout under `cluster/` and `atlas/`, the exact conversion between the two maze encodings,
the extremes' construction, job time limits, and how the grader batches levels on the GPU.
