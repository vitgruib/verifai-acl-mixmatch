# CartPole test suite (design)

Status: **partly built.** Implemented and tested: the solvability oracle
(`acl_bench/oracle.py`), the batched evaluator (`acl_bench/suite/evaluator.py`), the
frozen sets E0-E5 (`frozen_sets/cartpole_v1/`), the arm runner
(`acl_bench/suite/run_arms.py`) and the paired analysis (`acl_bench/suite/paired.py`).
Not yet built: the Stage 0 calibration and, because they need its reference agents,
the sets E6, E7 and POOL. The numbers under "Motivating probe" come from one policy
and one seed; they justify building the suite, they are not results.

## What the suite measures

The grid trains agents under different combinations of (sampler, ACL, scoring
function). The suite answers one question about each combination: **does the
trained agent get better, and does it get better faster, especially on the hard
CartPole tasks?** Samplers only ever choose training tasks; every agent is judged
on the same frozen, sampler-independent evaluation sets.

## Why CartPole

- **Fast**: ~5.4 s per 60k training steps, so a 400k-step run is ~36 s
  (single process, excluding evaluation). Seeds are cheap, which matters (see
  "Paired design").
- **It has real failures to fix**: 400k-step policies failed (mean return < 195) on
  15% and 2% of sampled tasks in two sensitivity runs (seeds 1 and 2), so the
  amount of failure depends heavily on the seed.
- **It has an oracle**: CartPole's physics linearizes cleanly, so an LQR controller
  can label which tasks are actually solvable. Acrobot has no such simple oracle
  and (in earlier runs) no outright failures.

## Three kinds of failure, and why they must be separated

A policy failing on a task can mean three different things. A curriculum should
only be credited or blamed for the first:

| class | definition | fixable? |
|---|---|---|
| **fixable** | the oracle balances the pole from this exact start; the policy does not | yes: what ACL and adaptive sampling should reduce |
| **infeasible** | the episode is guaranteed to end on step 1 whatever the policy does (`abs(theta0 + tau*theta_dot0) > 12 deg` or the cart analogue; a pole starting *past* 12 degrees can be rescued by velocity pointing back) | no: wasted training episodes |
| **unresolved** | feasible start, oracle also fails | unknown: the oracle is sufficient, not necessary, for solvability |

`acl_bench/oracle.py` implements the labels: discrete LQR on the linearized dynamics,
run bang-bang in the real environment from the exact initial state.
`survival_steps == 500` proves solvable; failure only means "not known solvable".

### Motivating probe (one 400k-step policy, seed 1, 1000 Halton-sampled (task, start) pairs)

| | share of pairs |
|---|---|
| guaranteed one-step failure (infeasible) | 5.0% |
| feasible, oracle fails (unresolved) | 0.8% |
| oracle solves | 94.2% |

The policy's 167 failures split into 50 infeasible, 8 unresolved and **109
fixable** (an 11.6% fixable failure rate). The policy never succeeded where the
oracle failed (0 of 8), so the oracle is not obviously too conservative.

Fixable failure rate by quartile of each parameter (Q1 = lowest values):

| parameter | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| `force_mag` | **37.1%** | 5.9% | 2.9% | 0.9% |
| `masscart` | 1.7% | 6.8% | 15.8% | **22.3%** |
| `length` | 10.9% | 10.1% | 10.2% | 15.2% |
| `masspole` | 7.6% | 8.8% | 15.4% | 14.8% |
| `init_range` | 10.0% | 10.8% | 12.9% | 12.9% |

Weak push (force Q1) with a heavy cart (mass Q4): **70%** fixable failure (39/56).
This is the clearest fixable edge case found so far. It also revises an earlier
sensitivity finding: `init_range` looked like the top parameter for raw return, but on
oracle-solvable pairs it barely moves failures; its raw-return effect was mostly the
~5% of starts that are unrecoverable.

## Task parameters

A task is `tau = (length, masspole, masscart, force_mag, init_range)`; an *episode* is
`(tau, s0)` where `s0` is the exact initial state. Bounds are those in
`acl_bench/envs/param_cartpole.py`:

| parameter | bounds | bottom quarter | top quarter | role |
|---|---|---|---|---|
| `force_mag` | 4-16 N | [4, 7] | [13, 16] | actuator authority; dominant fixable-failure driver |
| `masscart` | 0.5-2.0 kg | [0.5, 0.875] | [1.625, 2.0] | inertia the push must move; interacts with `force_mag` |
| `length` | 0.25-1.5 m (half-length) | [0.25, 0.5625] | [1.1875, 1.5] | pole time constant |
| `masspole` | 0.05-0.5 kg | [0.05, 0.1625] | [0.3875, 0.5] | weak effect |
| `init_range` | 0.05-0.3 | [0.05, 0.1125] | [0.2375, 0.3] | start perturbation; drives *infeasible* starts more than policy weakness |

Derived difficulty coordinate for reporting:
`authority = force_mag / ((masspole + masscart) * g)` (range ~0.19-2.22).

Training-time factors are the grid's (sampler x ACL x scoring function, 58 cells).
PPO hyperparameters stay fixed. The **step budget `B` is set in Stage 0**, not assumed.

## Evaluation protocol

- **One rollout per `(tau, s0)` pair, deterministic policy (argmax action).** This
  removes rollout noise, so a pair's outcome is reproducible for a given agent and the
  only noise left is which pairs were sampled (binomial) and training-seed variance.
- **Success** = survives 500 steps. **Learnable pair** = feasible start and
  oracle-solvable. Headline metrics use learnable pairs only; raw all-pairs numbers
  are reported alongside.
- **Frozen, paired sets.** Every set is a fixed list of pairs generated once from a
  seed and stored with its oracle labels, so all agents face identical pairs.

## Evaluation sets

Sets differ in *which part of the task space* they probe. Unless stated, parameters
not named are drawn uniformly over the full box, and `s0` is drawn uniformly within
`init_range`. Sizes are pairs before the learnable filter; the standard error of a
success rate from `n` pairs is 0.017 (n=300, p=0.9), 0.024 (150), 0.030 (100) and
about 0.05 (100, p=0.5).

| set | contents | question | pre-declared prediction |
|---|---|---|---|
| **E0 Uniform** | 300 pairs uniform over the box | average-case generalization | none |
| **E1 Weak actuation** | `force_mag` in [4, 7], 150 pairs | the largest fixable failure region | adaptive samplers and failure-seeking scores over-visit it, raising E1 |
| **E1b Weak push + heavy cart** | `force_mag` in [4, 7] and `masscart` in [1.625, 2.0], 100 pairs | the worst corner found (70% failure in the probe) | same, most strongly |
| **E2 Heavy cart** | `masscart` in [1.625, 2.0], 100 pairs | second-largest region | as E1 |
| **E3 Slow / heavy pole** | `length` in [1.1875, 1.5], and separately `masspole` in [0.3875, 0.5], 100 pairs each | weak effects in the probe | little difference; a sanity slice |
| **E4 Large recoverable disturbance** | `init_range` in [0.2375, 0.3] with *feasible* starts only, 100 pairs | recovery from big-but-recoverable perturbations, without infeasible starts contaminating it | little difference |
| **E5 Corners** | all 32 vertices of the parameter box x 10 starts | extremes uniform training rarely visits | space-filling `halton` covers corners better than `random` |
| **E6 Hard-but-solvable** | pairs from a 5000-pair pool that the oracle solves and at least half of a reference population fails; target 150-250 pairs | difficult tasks isolated by measured difficulty, not by region | ACL replay of failures raises E6 |
| **E7 Easy control** | pairs every reference agent solves, 100 pairs | regression check: a method must not buy hard-task gains by degrading easy tasks | no loss expected |

The predictions are hypotheses to test, not findings; the suite is useful partly
because a method can be *wrong* about them.

**Reference population** (for E6/E7): 10 baseline-arm agents (`random`, ACL off)
trained in Stage 0 on seeds not used by any evaluated agent, so difficulty is not
defined by the agents being compared. If the pool yields fewer than 150 hard pairs,
enlarge the pool. E6 could still favor methods that differ from the baseline; that
is a stated limitation.

**Difficulty-stratified reporting.** Alongside the named sets, report success in bins
of the pool's reference failure fraction (0-10%, 10-50%, 50-90%, 90%+), so
performance is visible as a function of measured difficulty.

## Metrics

**Primary (two, fixed in advance):**
1. `AUC_E0`: mean E0 success over all checkpoints from step 0 to `B` (sample
   efficiency in one number).
2. `success_E6`: learnable-pair success on the full hard set, averaged over the final
   three checkpoints to reduce evaluation noise.

**Performance** (final three checkpoints, per set): success rate on E1, E1b, E2, E3,
E4, E5, E7; `S_edge` = macro-average success over E1-E5 (a min over slices would be
biased downward by noise); mean return on E0 as a secondary check.

**Convergence speed** (every set is evaluated at every checkpoint, every 20,480 steps, step 0 included):
- steps to reach 50 / 80 / 95% of the *baseline arm's* plateau success (thresholds
  are relative so they are reachable), right-censored if never reached, compared
  with survival methods rather than by dropping runs;
- **plateau step** = the earliest checkpoint from which the curve stays within 0.02
  of its final level (a formal criterion, replacing eyeballing);
- instability = SD of E0 success over the last 5 checkpoints;
- the E6 curve and its AUC, as a secondary read on how fast hard tasks are learned.

**Mechanism diagnostics** (why a method behaved as it did, not whether):
training-time task mix (share of training episodes that were infeasible, easy, or
hard-but-solvable, labeled by the oracle and reference population; this requires
logging each training episode's `s0`); wasted-episode fraction (episodes ending in
at most 2 steps); rank correlation between each scoring function's score and true
difficulty over the replay buffer (does it actually rank hard tasks higher?); replay
share by difficulty bin; sampler concentration and coverage of the E1 region.

**Cost:** environment steps, wall-clock, sampler overhead.

## Paired design, and what the pilot showed

**The design.** Every arm is trained on the same seeds. A seed fixes, for each arm, the
network's starting weights, the order of practice minibatches, the stream of episode
starting states, the action-noise stream and the task-sampler stream, so two arms on
the same seed start from identical conditions. Comparisons are made seed by seed (the
difference `arm A - arm B` on each seed), with a paired bootstrap confidence interval.
The hope was that shared luck makes paired outcomes move together, so the difference
is far less noisy than an unpaired comparison.

**What the pilot measured** (30 seeds x 4 arms `N`, `A`, `S_sa`, `B_sa`, 204,800 steps
each, `results/pilot_pairing.csv`; the budget is provisional):

| paired comparison | correlation across seeds (AUC_E0) | share of variance pairing removed |
|---|---|---|
| A vs N | 0.29 | 28% |
| S_sa vs N | 0.09 | 9% |
| B_sa vs N | -0.12 | none (slightly worse) |
| B_sa vs S_sa | -0.09 | none |
| B_sa vs A | 0.06 | 6% |

**Pairing helps very little.** The correlation between two arms is 1.0 at step 0
(identical starting brains) and falls to about 0 by the first checkpoint, 20,480 steps
in, and does not recover. The shared seed is not broken (tests confirm it fixes the
initial weights and the stream of brand-new tasks); training is simply chaotic, so runs
decorrelate almost at once.

**Why the noise is so large.** Outcomes are close to two-valued. For the plain arm `N`
at 204,800 steps, 50% of seeds were below 0.2 success and 33% above 0.8, with few in
between, and early progress did not predict late progress (correlation -0.24 between 61k
and 205k steps). Most of the run-to-run variation is *when* a run takes off, not small
fluctuations around a common curve. Consequently the whole-curve area `AUC_E0` has a
seed-to-seed SD of about 0.10-0.13, versus about 0.30 for the score at the final
checkpoint.

**Seeds needed** (per arm, paired, two-sided alpha = 0.0125 for 4 contrasts, 80% power,
`n = (2.50 + 0.84)^2 (sd/D)^2`, using the pilot's measured SD of the paired difference):

| metric (SD of the difference) | detect D = 0.10 | D = 0.05 | D = 0.03 |
|---|---|---|---|
| `AUC_E0` (~0.15) | 26 seeds | 101 | 279 |
| final `E0` success at 205k steps (~0.42) | 197 | 787 | 2,186 |

With 30 seeds the smallest detectable difference in `AUC_E0` is about 0.09. The pilot's
own comparisons (best: `A` vs `N`, +0.038 AUC, 95% interval -0.011 to +0.087) are
therefore inconclusive, which is what these numbers predict; **no conclusion about the
methods can be drawn from the pilot.**

**What follows for the design:**
1. **Keep the shared seeds** (free, and they make runs reproducible and comparable) **but
   plan power as if unpaired.** Do not count on pairing.
2. **Prefer whole-curve metrics.** `AUC_E0` is ~3x less noisy than any single-checkpoint
   score, so it stays primary. Final-checkpoint scores are only trustworthy at a budget
   where nearly all baseline runs have already taken off; Stage 0 must check this,
   because at 205k steps half of them had not. Add a **take-off time** metric (steps
   until E0 success first exceeds 50%, right-censored if never), analysed with survival
   methods, since it matches the two-valued structure.
3. **Buy power with seeds, on the primary arms only.** Measured throughput: 120 runs of
   204,800 steps took 757 s on 9 workers, about 6.3 s of wall-clock per run, so about
   12 s per 400k-step run (an extrapolation). Eight primary arms x 100 seeds is 800
   runs, about 2.7 h; all 39 arms x 30 seeds is 1,170 runs, about 4 h, and can only see
   differences of about 0.09 AUC, so hyperparameter ablations are a coarse sensitivity
   screen, not fine measurements.
4. **Try to reduce the noise itself** (untested): a training setup that is less chaotic
   (lower learning rate, longer rollouts) would shrink every arm's variance at once.
   Stage 0 should test this because it is cheaper than seeds.

**A-priori contrasts** on the two primary metrics: C1 ACL only vs. neither; C2 adaptive
sampler only vs. neither; C3 both vs. neither; C4 best scoring function vs. `neg_return`
within ACL-on. Holm correction across them. C1 uses `pvl_gae` (SIPACL's default). **All
three adaptive samplers are run at full seed depth** rather than picking one by
screening (see Stages). Report effect sizes with bootstrap confidence intervals over
seeds, not only p-values.

## Stages

- **Stage 0, calibrate and freeze.** Train the baseline arm `N` for >= 30 seeds to 1M
  steps with checkpoints. Set the budget `B` where nearly all runs have taken off (the
  plateau criterion); measure the SD of `AUC_E0`, of the final score and of take-off
  time at that budget; test whether a less chaotic training setup (lower learning rate,
  longer rollouts) reduces them, and fix the setup for all arms if it does. Train 10
  more seeds as the reference population, then build and freeze E6, E7 and POOL.
- **Stage 1, the primary arms at full depth.** All eight primary arms (`N`, `A`,
  `S_ce`, `S_mab`, `S_sa`, `B_ce`, `B_mab`, `B_sa`) at the seed count Stage 0's
  measured SD calls for. There is **no small-seed screening stage**: with the spread
  measured, 3-5 seeds could detect only differences of about 0.2-0.3 AUC, so a
  screening pass would rank arms by noise.
- **Stage 2, sensitivity.** The hyperparameter ablations (docs/ablation.md) at a lower
  seed count, reported as effect sizes with intervals, not as tests.

## Build order

1. `acl_bench/oracle.py`: done, tested.
2. Batched evaluator: done, tested against the real environment, ~87x faster (the whole
   1,270-pair suite in ~0.13 s, so every set is evaluated at every checkpoint).
3. Frozen sets E0-E5 with oracle labels and checksums: done. E6, E7 and POOL are built
   by `build_pool_sets` (tested) once Stage 0 supplies reference agents.
4. Arm runner (shared seeds, checkpointed evaluation) and paired analysis: done.
   Still to add: logging each training episode's starting state and task, for the
   mechanism diagnostics, and the take-off-time metric.
5. Stage 0 calibration, then Stage 1.

## Known limits

- Everything here is about CartPole; conclusions may not transfer.
- The oracle is sufficient, not necessary, for solvability.
- The probe used one policy; the failure-region ranking may shift with the policy
  (two earlier CartPole policies had different binding parameters).
- E6/E7 difficulty is defined by baseline-arm agents.
- Deterministic evaluation can hide differences a stochastic policy would show.
- The pilot used a provisional 205k-step budget, mid-climb for most runs; the noise at
  the plateau may be smaller. Stage 0 measures it.
