# CartPole test suite (design)

Status: **built, and the main run is ready to start or running.** Built and tested: the
whole exam (E0-E7, locked), the grader, the scorecard code, the comparison code, the
runner with safeguards and snapshot saving, and the calibration run. Not built: a
take-off-time score and the "why" diagnostics.

## The suite at a glance (plain English)

**Training is a black box.** The testing suite does not care how an agent is trained.
All it ever receives is this:

> *"Here is the agent that method **M** produced on run **R**, photographed at regular
> check-ins: untrained, then after 20,480 steps, 40,960 steps, and so on."*

(A **method** is one combination of the components being compared. A **run** is one
complete training, with its own starting luck. A **snapshot** is the agent frozen at
a check-in.)

The suite turns snapshots into a verdict in five chunks:

```
   TRAINING (black box)
   method + run  ->  [ snapshot, snapshot, snapshot, ... ]
                                   |
                                   v
                          +----------------+
      1. THE EXAM  ------>| 2. THE GRADER  |   plays every question,
      fixed, locked       +-------+--------+   pass / fail each
                                  |
                                  v
                          +----------------+
                          | 3. THE         |   a few numbers per run:
                          |    SCORECARD   |   how well, how fast, how steady
                          +-------+--------+
                                  |
                                  v
                          +----------------+
                          | 4. COMPARISON  |   many independent runs per method;
                          |    RULES       |   real difference, or luck?
                          +-------+--------+
                                  v
                               VERDICT

      5. THE PLAN  decides which methods, and how many runs, feed the top
```

| chunk | one-sentence job | built? |
|---|---|---|
| **1. The exam** | A fixed set of game setups, grouped into sections that each probe one weakness. | all sections E0-E7 built and locked |
| **2. The grader** | Plays a snapshot through every question, quickly and the same way every time. | built and tested |
| **3. The scorecard** | Reduces the grades to a few numbers per run. | mostly built; take-off time and diagnostics not yet |
| **4. Comparison rules** | Decides whether a difference between methods is real or luck. | built |
| **5. The plan** | Lists which methods are compared and in what order to run them. | methods, runner and calibration built |

### Chunk 1: the exam

Each **question** is one game setup: a particular set of physics (how strong the push
is, how heavy the cart, how long the pole) and one specific starting position for the
pole. The questions are generated once, saved, and locked with a fingerprint, so every
snapshot from every method faces the *same* questions, and training never sees them.

**Impossible questions were removed once, before the exam was locked.** A few random
questions cannot be won by anyone (for example the pole already starts past saving), and
blaming a method for those would be unfair. So a hand-built expert controller was used a
single time to keep only questions it could win, and was then discarded. About 9% of the
questions were dropped (100 impossible, 13 where the expert lost but a win was not
ruled out). Nothing in the running suite depends on that expert.

Questions are grouped into **sections**, and each asks its own question:

| section | questions | what its questions have in common | what it tells us |
|---|---|---|---|
| **General** (E0) | 282 | random settings across the whole range | how good is the agent overall? |
| **Weak push** (E1) | 138 | the push is in the weakest quarter of its range | the biggest weak spot found so far |
| **Weak push + heavy cart** (E1b) | 93 | weakest pushes *and* heaviest carts | the worst corner found (in an early test a typical trained agent failed about 70% of the winnable ones) |
| **Heavy cart** (E2) | 95 | heaviest quarter of carts | the second weak spot |
| **Long pole** (E3a) / **heavy pole** (E3b) | 96 / 92 | slowest-moving poles | mostly a sanity check; weak effects in early tests |
| **Big shove at the start** (E4) | 99 | the pole starts far off-center but recoverably | recovery from a large disturbance |
| **Extremes** (E5) | 272 | all 32 "corners" of the settings box (every corner keeps at least 3 questions) | the far edges that random practice rarely visits |
| **Hard by measurement** (E6) | 248 | questions that 5 to 9 of the 10 reference agents fail | the hard cases, found by evidence instead of by guess |
| **Easy** (E7) | 100 | questions every reference agent passes | a safety check: did a method get better at hard questions by getting worse at easy ones? |

Every section also carries a written **prediction** (which method should do well on it),
so the suite can catch a method being wrong about itself.

### Chunk 2: the grader

The grader takes one snapshot and plays every question: the agent wins a question if it
keeps the pole up for 500 steps. Two properties matter. It uses **no dice** (the agent
always takes its single best action), so grading the same snapshot twice gives the same
answer. And it is **fast**: all ~1,170 questions in about a tenth of a second, roughly 87
times faster than playing them one at a time, and checked to give identical answers to
the slow, trusted way. That speed is why every section can be graded at every check-in.

### Chunk 3: the scorecard

The grades become a small number of scores per run:

| score | plain meaning |
|---|---|
| **Learning-curve score** (`AUC_E0`; primary) | how well and how *quickly* it learned overall: the average General-exam score across all check-ins |
| **Hard-section score** (`success_E6`; primary) | the score on the hard questions at the end |
| **Section scores** | the end score in each section, to see *where* a method helps |
| **Take-off time** (not built) | how many steps until it first passes half of the General exam |
| **Steadiness** | how much the score bounces near the end |

A separate set of **"why" checks** (not built) looks inside training, for example how
many practice questions were wasted on ones that end in a step or two.

### Chunk 4: the comparison rules

To say method A beats method B, the suite compares many training runs of each, and:

- trains every method on **its own independent seeds**, so the two groups of runs are
  independent and are compared as groups (an earlier plan to match runs seed by seed was
  tried and removed; see below);
- decides the **comparisons in advance** (does the review pile help? does the smart
  picker help? do both?), so it cannot go hunting for a lucky one afterwards;
- reports a **range of uncertainty** for every difference, not just a yes or no.

The hard part is noise, and the reference section below explains how much of it there is.

### Chunk 5: the plan

```
   Stage 0  CALIBRATE   How long must training run? How noisy is it?
              |
              v
   Stage 1  MAIN        The 8 main methods, 100 runs each; every setting
                        held at a standard value (an existence test)

   Stage 2  (parked)    Turning individual settings up and down
```

The methods and their settings are in docs/ablation.md.

---

# Reference details

## Why CartPole

- **Fast**: ~5.4 s per 60k training steps, so a 400k-step run is ~36 s (single process,
  excluding evaluation).
- **It has real failures to fix**: 400k-step policies failed (mean return < 195) on 15%
  and 2% of sampled tasks in two sensitivity runs, so the amount of failure depends
  heavily on the seed.
- **Its physics is simple enough** that a hand-built controller could decide which
  questions are winnable, which is what let the exam be cleaned once.

## Task parameters

A task is `(length, masspole, masscart, force_mag, init_range)`; a question is a task
plus the exact initial state `s0`. Bounds are those in `acl_bench/envs/param_cartpole.py`:

| parameter | bounds | bottom quarter | top quarter | role |
|---|---|---|---|---|
| `force_mag` | 4-16 N | [4, 7] | [13, 16] | actuator authority; dominant driver of fixable failures |
| `masscart` | 0.5-2.0 kg | [0.5, 0.875] | [1.625, 2.0] | inertia the push must move; interacts with `force_mag` |
| `length` | 0.25-1.5 m (half-length) | [0.25, 0.5625] | [1.1875, 1.5] | pole time constant |
| `masspole` | 0.05-0.5 kg | [0.05, 0.1625] | [0.3875, 0.5] | weak effect |
| `init_range` | 0.05-0.3 | [0.05, 0.1125] | [0.2375, 0.3] | start perturbation; drives *impossible* starts more than policy weakness |

## How the exam was made, and where the sections came from

The question sets are in `frozen_sets/cartpole_v2/`, with a manifest of fingerprints and
counts. They were made by generating each section from a fixed seed, then keeping only
the questions a hand-built expert (an LQR controller, run bang-bang in the real
environment from the exact start) could win. The manifest records which commit's expert
was used (`fe37d00`), so it can be recovered from history; it is not in the current tree.

The sections came from an early probe with that expert (one 400k-step policy, one seed,
1,000 random questions), **before** the expert was removed. It found:

| | share of questions |
|---|---|
| guaranteed to end on the first step (impossible) | 5.0% |
| feasible start, expert also loses (unclear) | 0.8% |
| expert wins | 94.2% |

The policy's 167 failures split into 50 impossible, 8 unclear and **109 fixable** (an
11.6% fixable failure rate). Fixable failure rate by quartile of each parameter
(Q1 = lowest values):

| parameter | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| `force_mag` | **37.1%** | 5.9% | 2.9% | 0.9% |
| `masscart` | 1.7% | 6.8% | 15.8% | **22.3%** |
| `length` | 10.9% | 10.1% | 10.2% | 15.2% |
| `masspole` | 7.6% | 8.8% | 15.4% | 14.8% |
| `init_range` | 10.0% | 10.8% | 12.9% | 12.9% |

Weak push (force Q1) with a heavy cart (mass Q4): **70%** fixable failure (39/56). This
is one policy and one seed, so it motivates the sections; it is not a result. It also
corrected an earlier reading: `init_range` had looked like the most important
parameter for raw return, but on winnable questions it barely moves failures; its
raw-return effect was mostly the impossible starts.

### v3: edge sections found by search instead of assumed

E1-E5 above were hand-set parameter quartiles taken from that one-policy probe. v3
(`frozen_sets/cartpole_v3/`) replaces them with sections found by actually searching for
failures: each VerifAI sampler (ce, mab, sa) was pointed at the ten `REF` agents instead
of a training run and kept every question it visited (`acl_bench/suite/find_edge_cases.py`,
3,000 tasks x 5 starts per sampler, raw pools in `frozen_sets/cartpole_v3_search/`). A
question counts as hard-but-winnable when 50-99% of the reference agents fail it, the same
rule as E6.

| sampler | questions drawn | hard, winnable | nobody passes |
|---|---|---|---|
| ce | 15,000 | 1,302 (8.7%) | 5,323 |
| mab | 15,000 | 524 (3.5%) | 2,111 |
| sa | 15,000 | 47 (0.3%) | 981 |

For comparison, passive random sampling found hard questions 0.31% of the time (E6's
pool), so ce is about 28x more efficient; sa is no better than random here.

The 1,873 hard questions were clustered (k-means, 5 clusters, normalized parameter space;
`acl_bench/suite/build_edge_sections.py`) and each cluster became one locked section of up
to 100 questions. E0, E6 and E7 carry over from v2 unchanged. Rebuilding from the saved
search reproduces the manifest fingerprints exactly.

| section | found | kept | mean fail fraction | dominant parameters |
|---|---|---|---|---|
| E1v | 520 | 100 | 0.72 | weak push, long pole |
| E2v | 742 | 100 | 0.74 | long pole, wide start range |
| E3v | 12 | 12 | 0.53 | short pole, light cart |
| E4v | 47 | 47 | 0.65 | wide start range, weak push |
| E5v | 552 | 100 | 0.74 | weak push, heavy cart |

What this shows: the probe's "weak push" and "weak push + heavy cart" regions are real
(E1v, E5v), but **a long pole** is a weak spot the quartile sections never isolated (E1v,
E2v). Caveats:

- **E3v (12 questions) is too small to score on its own** (standard error ~0.14);
  E4v (47) is marginal (~0.07). Use them only inside a macro-average, or drop them.
- The clusters overlap in meaning (three involve weak push, two long pole); they are
  regions of one failure landscape, not independent skills.
- "Hard" is judged by the same ten reference agents the search used, the same selection
  bias as E6, and adaptive samplers revisit neighborhoods, so questions within a section
  can be near-duplicates.
- The Stage 1 snapshots have not yet been graded on v3.

## Evaluation protocol

- **One rollout per question, deterministic policy (argmax action).** A question's
  outcome is reproducible for a given snapshot; the only noise left is which questions
  were chosen (binomial) and run-to-run training variance.
- **Success** = keeps the pole up for 500 steps.
- **Frozen sets.** Every section is a fixed list of questions, so all methods face
  identical questions.
- Standard error of a success rate from `n` questions: about 0.017 (n = 300, p = 0.9),
  0.024 (150), 0.030 (100), and about 0.05 (100, p = 0.5).

## Section predictions

| section | pre-declared prediction (a hypothesis, not a finding) |
|---|---|
| E0 General | none |
| E1 Weak push, E1b, E2 Heavy cart | adaptive samplers and failure-seeking scores over-visit these regions, raising the scores |
| E3 Long / heavy pole | little difference; a sanity slice |
| E4 Big shove | little difference |
| E5 Extremes | space-filling `halton` covers corners better than `random` |
| E6 Hard | ACL replay of failures raises it |
| E7 Easy | no loss expected |

**E6 and E7 without a referee.** Both come from a pool of 80,000 random questions and a
**reference population** of 10 baseline agents (`random` picker, review pile off,
trained separately from anything compared). Each reference run is represented by its
best of its last 5 check-ins on the General exam, because runs sometimes dip briefly
after reaching their plateau (one reference run ended at 0.25 on its final check-in
while the others sat at 0.85-1.0).
- **E6** = questions that at least half the reference agents fail **and at least one
  passes**. The "at least one passes" condition is the evidence a question is winnable,
  since there is no expert to ask; it excludes impossible questions and also the very
  hardest winnable ones that nobody passes. It is rare: 248 of 80,000 questions (0.31%),
  because converged agents are near-perfect on winnable questions. 5,000 questions
  gave only 18, which is why the pool is large.
- **E7** = questions every reference agent passes (72,450 of 80,000 qualified; the first
  100 are kept).
- **POOL** is a stored 10,000-question sample of the pool, with each question's reference
  failure fraction, kept in `frozen_sets/cartpole_v2_pool/`. It is graded from saved
  snapshots after a run, in bins of reference failure fraction (0-10%, 10-50%, 50-90%,
  90%+), not at every check-in (a large pool would more than double each run's time).
  6.5% of the pool is passed by nobody; that mixes impossible questions with the hardest
  possible ones, so the 90%+ bin is read that way.

E6 could still favor methods that differ from the baseline; that is a stated limitation.

## Metrics

**Primary (two, fixed in advance):**
1. `AUC_E0`: mean E0 success over all checkpoints from step 0 to `B` (sample efficiency
   in one number).
2. `success_E6`: success on the hard set, averaged over the final three checkpoints to
   reduce evaluation noise.

**Performance** (final three checkpoints, per section): success on E1, E1b, E2, E3a,
E3b, E4, E5, E7; `S_edge` = macro-average success over the edge sections; mean steps on
E0 as a secondary check.

**Convergence speed** (every section is evaluated at every checkpoint, every 20,480
steps, step 0 included):
- steps to reach 50 / 80 / 95% of the *baseline's* plateau success (relative, so the
  thresholds are reachable), right-censored if never reached, compared with survival
  methods rather than by dropping runs;
- **plateau step**: the earliest checkpoint from which the curve stays within 0.02 of its
  final level;
- steadiness: SD of E0 success over the last 5 checkpoints;
- the E6 curve and its AUC.

**"Why" diagnostics** (not built): training-time task mix (this requires logging each
training episode's start), wasted-episode fraction, rank correlation between each
scoring function's score and true difficulty, replay share by difficulty, sampler
concentration.

**Cost:** environment steps, wall-clock, sampler overhead.

## Comparing methods: independent runs, and why pairing was removed

**The design.** Each method is trained on its own independent seeds (`run_arms` derives
a separate seed for every (method, replicate) by hashing the method name and replicate
number). Two methods are compared as two independent groups: a difference of means, a
bootstrap interval that resamples each group separately, and Welch's t-test
(`acl_bench/suite/compare.py`).

**Why not match runs seed by seed.** That was the original plan, on the hope that runs
sharing a seed move together. A pilot (30 seeds x 4 methods, 204,800 steps,
`results/pilot_pairing.csv`) showed they do not:

| comparison | correlation across seeds (`AUC_E0`) | variance pairing removed |
|---|---|---|
| A vs N | 0.29 | 28% |
| S_sa vs N | 0.09 | 9% |
| B_sa vs N | -0.12 | none |
| B_sa vs S_sa | -0.09 | none |
| B_sa vs A | 0.06 | 6% |

Two runs on the same seed start identical (correlation 1.0 at step 0) but are unrelated
by the first check-in, 20,480 steps later, and never recover. Training is chaotic, so
early differences snowball. Pairing removed 0-28% of the variance and made some
comparisons slightly worse, so it was dropped.

**Why the noise is so large.** Outcomes are close to two-valued. For the plain method at
204,800 steps, 50% of seeds were below 0.2 success and 33% above 0.8, with few in
between, and early progress did not predict late progress (correlation -0.24 between 61k
and 205k steps). Most of the run-to-run variation is *when* a run takes off. The
whole-curve score `AUC_E0` has a seed-to-seed SD of about 0.10-0.13, versus about 0.30
for a score at one checkpoint.

**The pilot's numbers came from a 205k-step snapshot, mid-climb.** With the training
length fixed by calibration (614,400 steps, see Stages), the noise is much smaller:

| training length | `AUC_E0` SD (20 plain runs) | final-score SD | smallest `AUC_E0` difference detectable at 100 runs |
|---|---|---|---|
| 204,800 (pilot) | 0.130 | 0.279 | 0.062 |
| 409,600 | 0.088 | 0.113 | 0.042 |
| **614,400 (chosen)** | **0.075** | **0.042** | **0.035** |
| 1,228,800 | 0.035 | 0.105 | 0.017 |

(Longer training also dilutes `AUC_E0`, because it averages in more plateau, so
differences in units of `AUC_E0` are not comparable across lengths.)

**Runs needed per method** at 614,400 steps (independent groups, two-sided alpha =
0.0125 for 4 comparisons, 80% power, `n = (2.50 + 0.84)^2 (sd_a^2 + sd_b^2) / D^2`),
assuming every method has the plain method's SD, which has been measured only for the
plain method:

| metric | detect D = 0.10 | D = 0.05 | D = 0.03 |
|---|---|---|---|
| `AUC_E0` (SD 0.075 per method) | 13 | 51 | 140 |
| final score (SD 0.042 per method) | 4 | 16 | 44 |

The final score is close to its ceiling (mean 0.96), so a 0.05 difference in it is hard
to have at all; `AUC_E0` is the better-behaved primary. With 100 runs per method the
smallest detectable `AUC_E0` difference is about 0.035. The pilot's own comparisons
(205k steps, best: `A` vs `N`, +0.038 AUC) were inconclusive, as its noise predicted:
**no conclusion about the methods can be drawn from the pilot.**

**What follows:**
1. **Prefer whole-curve scores.** `AUC_E0` is ~3x less noisy than any single-checkpoint
   score. Final-checkpoint scores are only trustworthy at a budget where nearly all
   baseline runs have already taken off; at 205k steps half had not. Stage 0 must check.
   Add a **take-off time** score analysed with survival methods.
2. **Buy power with runs, on the main methods only.** Measured throughput: 120 runs of
   204,800 steps took 757 s on 9 workers (~6.3 s wall-clock per run), so about 12 s per
   400k-step run (extrapolated). Eight main methods x 100 runs is ~2.7 h.
3. **Shrinking the learning rate did not calm training; it slowed it.**
   The plain method, 20 runs per rate, 400k steps (`results/lr_*.csv`):

   | learning rate | runs that took off (final score > 0.5) | median take-off step | `AUC_E0` | SD of final score |
   |---|---|---|---|---|
   | 3e-4 (current) | 85% | 164k | 0.43 | 0.29 |
   | 1e-4 | 40% | 266k | 0.21 | 0.29 |
   | 3e-5 | 0% | never | 0.12 | 0.04 |

   The drop in `AUC_E0` is clear (1e-4 vs 3e-4: -0.22, 95% interval -0.27 to -0.16). The
   small spread at 3e-5 only means nobody learned, so all runs sit near zero, and the
   runs that did take off at 1e-4 were as scattered as at 3e-4. The rate stays at 3e-4.
   A second test at three times the length (1.2M steps, 20 runs each,
   `results/lr_long_*.csv`) settled it: 1e-4 was still worse (80% of runs took off
   against 100%; `AUC_E0` 0.37 against 0.76) and not steadier (final-score SD 0.36
   against 0.105).

**A-priori comparisons** on the two primary scores: C1 ACL only vs. neither; C2 adaptive
sampler only vs. neither; C3 both vs. neither; C4 best scoring function vs. `neg_return`
within ACL-on. Holm correction across them. C1 uses `pvl_gae` (SIPACL's default). All
three adaptive samplers are run at full depth rather than picking one by screening.

## Stages

- **Stage 0, calibrate: done.** 20 plain runs were trained to 1.2M steps (`results/lr_long_3e-4.csv`;
  20 rather than the 30 originally planned). Every run had taken off by 307k steps, and
  the mean General-exam success plateaus at about 0.9-0.97 from around 490k steps. The
  training length is **614,400 steps** (all runs at their plateau; noise as tabulated
  above), and the learning rate stays at 3e-4. One thing to know: 15 of the 20 runs
  dipped below 0.5 at some check-in after 512k steps (one as low as 0.01), so agents
  sometimes collapse temporarily; final scores therefore average the last three
  check-ins. Ten reference agents (`REF`, 614,400 steps, ~95 s each with 6 workers)
  then built and locked E6, E7 and the POOL sample.
- **Stage 1, the eight main methods** (`N`, `A`, `S_ce`, `S_mab`, `S_sa`, `B_ce`, `B_mab`,
  `B_sa`), **100 runs each** (800 runs) of 614,400 steps, every setting fixed at a standard value; agent
  snapshots are saved so grading can be redone with sections built later. There is **no small-run
  screening stage**: with the noise measured, 3-5 runs could detect only differences of
  about 0.2-0.3 AUC, so a screen would rank methods by noise.
- **Stage 2, parked.** Turning individual settings up and down (docs/ablation.md). Held
  for now: every setting stays at its standard value.

## Build order

1. Batched grader: done, tested against the real environment, ~87x faster.
2. Locked exam sections E0-E7 (winnable questions only): done.
3. Runner (independent seeds, checkpointed grading, optional snapshot saving) with a
   re-grader, and the comparison code: done. Still to add: logging each training
   episode's start and task, and the take-off-time score.
4. Stage 0 calibration and E6 / E7 / POOL: done. Stage 1: next.
5. v3 edge sections (E1v-E5v) found by falsification search: done. Re-grading the Stage 1
   snapshots on them: next.

## Known limits

- Everything here is about CartPole; conclusions may not transfer.
- The exam was cleaned once by an expert that can miss winnable questions (in the early
  probe it never lost a question the policy won, but that was 8 questions).
- The early probe used one policy; the failure-region ranking may shift with the policy.
- E6 excludes the hardest winnable questions (nobody passes) because winnability is
  judged from reference agents.
- Deterministic grading can hide differences a stochastic policy would show.
- The pilot used a provisional 205k-step budget, mid-climb for most runs; noise at the
  plateau may be smaller. Stage 0 measures it.
