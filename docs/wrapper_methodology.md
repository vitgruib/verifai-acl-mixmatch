# Discovering a cheap curriculum wrapper: methodology and pipeline

**Aim.** Find a method that wraps around VerifAI's samplers or any RL algorithm and
improves the trained agent for little work. "Little work" means: black-box (it sees
only *task parameters in, episode outcome out*, never the learner's internals), at most
20% extra simulation charged to its budget, and no worse than plain domain
randomization (DR) on any environment. The PLR direction (replaying chosen tasks) is the
starting point; other directions and combinations are in scope.

This document is the plan. `docs/plr.md` is the evidence it builds on; `docs/ablation.md`
and `docs/exam.md` define the existing study and exams.

## 1. What we already know (docs/plr.md)

1. **Critic-based scores pick the wrong tasks.** PVL tracks how *easy* a task is
   (Spearman with return: +0.99 Acrobot, +0.70 Pendulum, +0.50 CartPole); L1 on a
   continuous cost chases hopeless tasks. The SFL paper reports the same from other
   domains: regret approximations "correlate with success rate", and PVL's variance
   "causes already-solved maps to be prioritised" (Rutherford et al. 2024). No fix to
   the critic, the noise or the replay rate turned PVL into a confirmed gain (12 rounds,
   3 failed 100-seed confirmations; best +0.06 on CartPole's hard suite).
2. **An outcome-based signal works.** Learnability p(1-p) (SFL) is the one confirmed win:
   CartPole hard suite +0.105 (Holm p = 0.002, 100 seeds), MountainCar random +0.06; it
   costs Acrobot 0.05, and it uses uncharged scouting rollouts.
3. **Most environments leave no room.** Only CartPole passes the oracle test (below).
4. **Strong curricula always hurt somewhere** (replay 0.9, PLR-perp, ACCEL, the oracle
   on Acrobot/Pendulum): keep uniform sampling in the mix.
5. **24-seed screens mislead.** Three leads of +0.09 to +0.145 fell to +0.02 to +0.06 at
   100 seeds: selecting the best of many screens selects luck.

## 2. Benchmarks

Two tiers. The fast tier is where the search happens; the standard tier is where a
finding has to hold to count, using the literature's own protocols.

### Fast tier (this repo's batched harness, seconds to minutes per run)

| environment | reward | goal | hard suite | oracle headroom |
|---|---|---|---|---|
| CartPole | +1 per step | survive (pass/fail) | VerifAI | **yes** (+0.20) |
| Acrobot | -1 per step | reach | VerifAI | no (oracle hurts) |
| MountainCar | -1 per step | reach | VerifAI | no (hard suite unreachable) |
| Pendulum | continuous cost | balance (thresholded) | VerifAI | no (oracle hurts) |
| **Maze** (to build) | sparse, time-discounted | reach the goal (pass/fail) | held-out mazes + VerifAI | to test |
| **Continuous-goal task** (to build) | shaped | return threshold | VerifAI | to test |

Additions:
- **Maze (grid navigation).** The standard testbed of the UED literature (PAIRED, PLR,
  Robust PLR, ACCEL, SFL): a 13x13 MiniGrid-style maze with random walls, goal and start,
  partial observation, episodes capped at 250 steps (minimax, JaxUED). Evaluation is the
  **zero-shot solved rate on the named held-out mazes** (SixteenRooms, Labyrinth,
  StandardMaze and the others of the DCD / minimax benchmark), 10 seeds in those papers.
  Built as a batched numpy environment like the others, with the held-out layouts copied
  from the reference implementations. It adds what the classic-control tasks lack:
  combinatorial task structure, where a task is a layout rather than a few scalars.
- **A continuous-goal task** with a return threshold, to complement Pendulum. Candidate:
  a batched 2D point-mass / reacher with parameterized obstacles and goal regions, graded
  by return and by "mastered" = return above a threshold (TeachMyAgent's convention).

Each new environment gets the same exam as the others (random suite, VerifAI suite built
by falsifying reference agents, winnability certified) and must pass the oracle test to
join the discovery set.

### Standard tier (the literature's protocols; for final checks only)

| benchmark | why | protocol we adopt |
|---|---|---|
| **TeachMyAgent Stump Tracks** (Romac et al. 2021) | *the* benchmark for teachers over continuous task parameters; continuous control, shaped reward | 2D task space (stump height, spacing); % of 100 fixed test tasks "mastered" (episodic return > 230), evaluated every 500k steps; 32 seeds; Welch's t-test; 20M steps. Compared teachers: Random, ADR, ALP-GMM, Covar-GMM, RIAC, GoalGAN, Setter-Solver, SPDL |
| **MiniGrid mazes** (DCD / minimax / JaxUED) | the benchmark for PLR / ACCEL / SFL | zero-shot solved rate on the held-out mazes; 10 seeds; ~30k PPO updates in minimax |
| **SFL's CVaR evaluation** (Rutherford et al. 2024) | measures robustness directly | sample 10,000 random solvable levels, 10 rollouts each, re-evaluate on the worst alpha% |

Stump Tracks needs a continuous-action PPO and 20M steps a run, so it is expensive: a
confirmed winner goes there last, as the external check that the result is not an
artifact of our own benchmarks. Our VerifAI suites play the role of SFL's CVaR set:
questions found by searching for failures, proven solvable.

### Reporting (rliable, Agarwal et al. 2021)

Alongside the per-metric Welch tests with Holm correction: interquartile mean (IQM) across
runs with stratified-bootstrap confidence intervals, and performance profiles (the share
of runs above each score). IQM is less swayed by the collapsed and lucky runs that
CartPole produces.

## 3. The oracle test (does an environment have room?)

The **oracle** is a deliberately cheating curriculum, used only as a diagnostic: half of
the training episodes are drawn from the hard suite's own tasks, the rest uniformly. It
knows exactly where the exam is hard, which no real method can. If even the oracle does
not beat DR on the hard suite, no choice of training tasks can, and a wrapper has
nothing to show there.

- **Run:** `oracle50` vs `DR`, 24 seeds, on each environment's hard suite.
- **Pass:** the oracle beats DR on hard-suite final success at p < 0.05.
- **Use:** environments that pass form the **discovery set** (where gains are measured).
  Those that fail form the **no-harm set** (a method must not lose there, but cannot be
  expected to gain).
- **Variants** when an oracle fails, to tell "unlearnable" from "wrong dose": 20% and 80%
  oracle mixes, and an oracle over the frontier (tasks the reference agents pass
  30-70% of the time) instead of the hardest ones.

## 4. Offline signal screen (before any training run)

The cheapest step, and the one that would have saved most of rounds 1-11. A candidate
signal is scored against ground truth on saved agents, without training anything.

1. Take saved checkpoints (the study's snapshots: early, middle and late in training, for
   several seeds).
2. Draw 2,000 random tasks. For each, compute **ground truth**: its pass rate p over 16
   rollouts of the stochastic policy, so learnability p(1-p), and the planner's best result
   (acl_bench.exam.certify), so regret.
3. Compute the **candidate signal** for the same tasks from what it would see during
   training (for example one episode's PVL, or a pass rate from 2 rollouts).
4. Report: Spearman with learnability, **precision@10%** (the share of the signal's top
   10% that are frontier tasks, 0.1 < p < 0.9), Spearman with the success rate (the
   easiness bias), and whether the top tasks lean toward the hard suite's region.
5. **Advance** only signals with precision@10% at least half of the true-learnability
   ceiling and less easiness bias than PVL.

This also sizes a signal's cost: how many rollouts per task it needs to reach a given
precision.

## 5. The design space

A wrapper is a choice along each axis. Vary one axis at a time from a fixed base.

| axis | options (source) |
|---|---|
| **signal** | learnability p(1-p) (SFL); learning progress (ALP-GMM, Portelas et al. 2019); VerifAI's falsification rho; return vs a threshold (TeachMyAgent mastery); PVL / L1 / MaxMC (PLR, Robust PLR, ACCEL: controls, known to fail) |
| **where the signal comes from** | training episodes only (free); scouting rollouts (SFL; must be charged) |
| **selector** | replay buffer (PLR); a VerifAI sampler over the box (ce, mab, sa); a Gaussian-mixture teacher (ALP-GMM); level edits (ACCEL, genetic operators) |
| **diversity** | none; per-parameter bucket caps and categories (the `nk_buffer` in Kv139/ACL-experiments); staleness (PLR) |
| **mixing** | share of uniform draws kept (0.25, 0.5, 0.75) |
| **cost** | extra rollouts per update, charged to the budget |

**First candidates**, in order:
1. **Learnability-steered VerifAI sampler.** ce or mab sampler, fed rho = -p(1-p) with p
   from ordinary training episodes (running pass rate per sampler bucket), 50% uniform.
   Wraps VerifAI directly, black-box, no extra simulation.
2. **Learnability replay buffer.** A PLR buffer ranked by running pass-rate learnability
   from training episodes (no PVL), 50% uniform, with bucket caps for diversity.
3. **Budget-fair SFL.** SFL's scouting, with its rollouts counted in the budget and
   fewer of them (for example 200 tasks x 4 rollouts), to see how much of SFL's gain
   survives at equal cost.
4. **ALP-GMM** as the literature's standard teacher over continuous parameters (it is
   TeachMyAgent's strongest baseline family), wrapped the same way.

## 6. Staged tests

| stage | runs | decides | rule (declared before running) |
|---|---|---|---|
| **S: screen** | 24 seeds per arm, fast tier, discovery set | which candidates advance | at most 3 advance: those with hard-suite final gain > 0 at p < 0.1, and no loss on the random suite beyond -0.03 |
| **C: confirm** | 100 fresh seeds per arm | whether it works | 4-6 declared metrics, Holm-corrected; the gain must survive |
| **X: no harm** | 100 seeds on every no-harm environment | whether it is safe to wrap by default | no metric worse than DR by more than a declared margin (non-inferiority) |
| **P: port** | the study pipeline (acl_bench.study) | that the result is not a harness artifact | the confirmed metric holds in the study's own run / analyze tools |
| **G: generality** | a second RL algorithm (DQN, or SAC with continuous actions) | that it wraps "any RL algorithm" | the same confirmed metric holds |
| **T: standard tier** | Stump Tracks, MiniGrid held-out mazes | that it holds on the literature's benchmarks | the benchmark's own protocol and statistics |

Everything is budget-fair: every simulated step a wrapper takes, including scouting and
evaluation it uses to choose tasks, counts against its training budget.

## 7. Pipeline

| step | command / module | status |
|---|---|---|
| train and grade fast | `python -m acl_bench.plr.screen --env E --configs ... --seeds ...` | exists |
| summarize, screen or confirm | `python -m acl_bench.plr.summarize CSV [--continuous] [--holm ...]` | exists; add IQM and bootstrap CIs |
| oracle test | `screen --configs DR oracle50` | exists; add the 20% / 80% / frontier variants |
| offline signal screen | `python -m acl_bench.plr.signals --env E --snapshots ...` | to build |
| new environments | `acl_bench/envs/maze.py`, `acl_bench/envs/<continuous>.py` + exams | to build |
| wrapper candidates | `acl_bench/plr/levels.py` (buffer, signals), `acl_bench/sampling.py` (VerifAI feedback) | partly exists |
| port | new arms in `acl_bench/study/arms.py`, `acl_bench/acl.py` | to build |
| second algorithm | `acl_bench/plr/fast_dqn.py` or a continuous-action PPO | to build |
| standard tier | TeachMyAgent Stump Tracks, a MiniGrid maze runner | to build |

**Order of work:**
1. Offline signal screen on existing CartPole snapshots (fast; tests candidates 1-2's
   signal before any training).
2. Oracle variants on the four existing environments (fills in the discovery set).
3. Maze environment and exam; its oracle test.
4. Stage S for candidates 1-4 on the discovery set.
5. Stages C, X, P for whatever advances; then G and T.

## References

- Jiang, Grefenstette, Rocktaschel. Prioritized Level Replay. ICML 2021.
- Jiang et al. Replay-Guided Adversarial Environment Design (Robust PLR). NeurIPS 2021.
- Parker-Holder et al. Evolving Curricula with Regret-Based Environment Design (ACCEL). ICML 2022.
- Rutherford et al. No Regrets: Investigating and Improving Regret Approximations for
  Curriculum Discovery (SFL). NeurIPS 2024. arXiv:2408.15099.
- Romac et al. TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL.
  ICML 2021. arXiv:2103.09815.
- Portelas et al. Teacher Algorithms for Curriculum Learning of Deep RL in Continuously
  Parameterized Environments (ALP-GMM). CoRL 2019. arXiv:1910.07224.
- Jiang et al. minimax: Efficient Baselines for Autocurricula in JAX. 2023. arXiv:2311.12716.
- Coward et al. JaxUED: A simple and useable UED library in Jax. 2024.
- Agarwal et al. Deep Reinforcement Learning at the Edge of the Statistical Precipice
  (rliable). NeurIPS 2021. arXiv:2108.13264.
- Kv139/ACL-experiments (branch joelle): ScenicGym ACL on MetaDrive / CARLA / Webots;
  PVL replay of saved scenes and a diversity-capped `nk_buffer` with genetic operators.
