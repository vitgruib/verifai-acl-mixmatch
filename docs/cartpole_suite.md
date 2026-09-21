# CartPole test suite (design)

Status: **specification, not yet implemented.** Only `acl_bench/oracle.py`
exists (with tests). The numbers under "Motivating probe" come from one policy and
one seed; they justify building the suite, they are not results.

## What the suite measures

The grid trains agents under different combinations of (sampler, ACL, scoring
function). The suite answers one question about each combination: **does the
trained agent get better, and does it get better faster, especially on the hard
CartPole tasks?** Samplers only ever choose training tasks; every agent is judged
on the same frozen, sampler-independent evaluation sets.

## Why CartPole

- **Fast**: ~5.4 s per 60k training steps, so a 400k-step run is ~36 s
  (single process, excluding evaluation). Seeds are cheap, which matters (see
  "Statistics").
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
| **E0 Uniform** | 300 pairs uniform over the box; **E0-lite** is a fixed 100-pair subset used at every learning-curve checkpoint | average-case generalization | none |
| **E1 Weak actuation** | `force_mag` in [4, 7], 150 pairs | the largest fixable failure region | adaptive samplers and failure-seeking scores over-visit it, raising E1 |
| **E1b Weak push + heavy cart** | `force_mag` in [4, 7] and `masscart` in [1.625, 2.0], 100 pairs | the worst corner found (70% failure in the probe) | same, most strongly |
| **E2 Heavy cart** | `masscart` in [1.625, 2.0], 100 pairs | second-largest region | as E1 |
| **E3 Slow / heavy pole** | `length` in [1.1875, 1.5], and separately `masspole` in [0.3875, 0.5], 100 pairs each | weak effects in the probe | little difference; a sanity slice |
| **E4 Large recoverable disturbance** | `init_range` in [0.2375, 0.3] with *feasible* starts only, 100 pairs | recovery from big-but-recoverable perturbations, without infeasible starts contaminating it | little difference |
| **E5 Corners** | all 32 vertices of the parameter box x 10 starts | extremes uniform training rarely visits | space-filling `halton` covers corners better than `random` |
| **E6 Hard-but-solvable** | pairs from a 5000-pair pool that the oracle solves and at least half of a reference population fails; target 150-250 pairs, **E6-lite** = fixed 60-pair subset | difficult tasks isolated by measured difficulty, not by region | ACL replay of failures raises E6 |
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
1. `AUC_E0`: mean E0-lite success over all checkpoints from step 0 to `B` (sample
   efficiency in one number).
2. `success_E6`: learnable-pair success on the full hard set, averaged over the final
   three checkpoints to reduce evaluation noise.

**Performance** (final three checkpoints, per set): success rate on E1, E1b, E2, E3,
E4, E5, E7; `S_edge` = macro-average success over E1-E5 (a min over slices would be
biased downward by noise); mean return on E0 as a secondary check.

**Convergence speed** (checkpoints every 20k steps on E0-lite):
- steps to reach 50 / 80 / 95% of the *baseline arm's* plateau success (thresholds
  are relative so they are reachable), right-censored if never reached, compared
  with survival methods rather than by dropping runs;
- **plateau step** = the earliest checkpoint from which the curve stays within 0.02
  of its final level (a formal criterion, replacing eyeballing);
- instability = SD of E0-lite success over the last 5 checkpoints;
- E6-lite curve (every 40k steps) and its AUC, as a secondary read on how fast hard
  tasks are learned.

**Mechanism diagnostics** (why a method behaved as it did, not whether):
training-time task mix (share of training episodes that were infeasible, easy, or
hard-but-solvable, labeled by the oracle and reference population; this requires
logging each training episode's `s0`); wasted-episode fraction (episodes ending in
at most 2 steps); rank correlation between each scoring function's score and true
difficulty over the replay buffer (does it actually rank hard tasks higher?); replay
share by difficulty bin; sampler concentration and coverage of the E1 region.

**Cost:** environment steps, wall-clock, sampler overhead.

## Statistics

- **The unit of replication is the training seed.** Pairs are shared across agents
  (paired), so seeds are the remaining source of variance, and it is large: two
  earlier CartPole policies differed by mean return 251 vs 465.
- **Seeds needed** (per arm, two-sided alpha = 0.0125 for 4 contrasts, 80% power,
  `n = 2 (2.50 + 0.84)^2 (s/D)^2`) for seed-to-seed SD `s` in success rate and a true
  difference `D`:

  | | D = 0.03 | D = 0.05 | D = 0.10 |
  |---|---|---|---|
  | s = 0.05 | 62 | 23 | 6 |
  | s = 0.10 | 248 | 90 | 23 |
  | s = 0.15 | 558 | 201 | 51 |

  `s` is unknown until Stage 0 measures it. If it is near 0.10, detecting a
  5-point difference needs ~90 seeds per arm, and 10 seeds could only detect
  ~15 points. CartPole is cheap enough to buy power with seeds (50 seeds x 4 arms
  x 36 s is ~2 h of training, single process), but not to run the whole grid at that
  depth; the full grid can only be screened.
- **A-priori contrasts** on the two primary metrics: C1 ACL only vs. neither; C2
  adaptive sampler only vs. neither; C3 both vs. neither; C4 best scoring function vs.
  `neg_return` within ACL-on. Holm correction across them. C1 uses `pvl_gae` (SIPACL's
  default); the adaptive sampler and scoring function for C2-C4 are chosen by
  screening and confirmed on fresh seeds.
- **Report** effect sizes with bootstrap confidence intervals over seeds, not only
  p-values.

## Stages

- **Stage 0, calibrate and freeze.** Train the baseline arm for >= 20 seeds to 1M
  steps with checkpoints: set `B` by the plateau criterion, measure `s` for both
  primary metrics (fixing the confirmation seed count), and define the relative
  convergence thresholds. Train 10 more seeds as the reference population. Generate
  E0-E7 and freeze them with the oracle labels and a checksum; verify each set has
  enough learnable pairs.
- **Stage 1, screening.** All 58 cells x 3-5 seeds at budget `B`, ranked by the
  primary metrics. Used to choose candidates, not to make claims.
- **Stage 2, confirmation.** The four primary arms plus the screened winners, with
  the seed count from Stage 0, on seeds disjoint from screening so a selected cell is
  not tested on the data that selected it.

## Build order

1. `acl_bench/oracle.py` (done, tested).
2. **A batched evaluator**: step many pairs in lockstep with one batched policy
   forward pass. Required, not optional: the sets total ~1,570 pairs x up to 500
   steps, and E0-lite alone across 20 checkpoints is ~1M policy steps, more than
   twice the ~400k of training.
3. Set generator and freezer (pairs, oracle labels, checksum).
4. Checkpointed training runner that logs each training episode's `s0` and task.
5. Metric computation, then the Stage 0 calibration.

## Known limits

- Everything here is about CartPole; conclusions may not transfer.
- The oracle is sufficient, not necessary, for solvability.
- The probe used one policy; the failure-region ranking may shift with the policy
  (two earlier CartPole policies had different binding parameters).
- E6/E7 difficulty is defined by baseline-arm agents.
- Deterministic evaluation can hide differences a stochastic policy would show.
