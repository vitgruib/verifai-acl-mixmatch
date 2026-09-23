# The CartPole exam

The exam is how every trained agent is scored. Training is a black box: all the exam
ever receives is an agent snapshot, *"the agent method M produced on run R, at check-in
S"*. It plays a fixed, locked set of questions and reports the share it passes.

A **question** is one game setup: a task (five physical parameters) plus one exact
starting state. The agent passes if it keeps the pole up for 500 steps. Grading is
deterministic (the agent always takes its single best action), so grading the same
snapshot twice gives the same answer, and batched (`acl_bench/exam/grader.py`), about
87 times faster than one environment at a time and checked step by step against the
real environment (`tests/test_grader.py`).

## Sections

`frozen_sets/cartpole/`, each section locked with a SHA-256 fingerprint in the manifest.

| section | questions | what they have in common | how it was made |
|---|---|---|---|
| **E0 General** | 282 | random tasks across the whole box | fixed seeds; cleaned once by an LQR expert controller (since removed); all 282 are also proven winnable |
| **E1v-E4v Edge** | 100, 100, 100, 50 | clusters of the hard questions the search found (below) | search, then proof, then k-means |
| **E6 Hard** | 250 | 6 or more of the 10 reference agents fail it, including questions all 10 fail | search, then proof; drawn uniformly from the hard questions not used by an edge section |
| **E7 Easy** | 100 | every reference agent passes | the random sampler's pool, proven winnable |

**Every question in every section is proven winnable.** No section holds an impossible
question or one whose status is unknown.

The **reference agents** are 10 runs of the plain method (random tasks, no replay) on
seeds independent of every compared method. Each is represented by its best of its last
five check-ins on E0, because agents sometimes dip briefly after reaching their plateau.

### How the searched sections are built

1. **Search** (`acl_bench/exam/search.py`). Each sampler (random, ce, mab, sa) proposes
   3,000 tasks, with 5 random starts each. Every (task, start) pair is played by all 10
   reference agents and saved with the share that fail it (`frozen_sets/search/`). The
   adaptive samplers are steered toward failure: `rho = success rate - 0.5`, so a task
   most agents fail is a VerifAI counterexample.
2. **Prove** (`acl_bench/exam/feasibility.py`). Every question that 6+ agents fail is
   labeled winnable, impossible or unknown, with a proof (next section).
3. **Build** (`acl_bench/exam/build.py`). Proven-winnable hard questions are clustered
   into edge sections (clusters under 30 questions dropped); E6 is drawn from the rest;
   E7 from the random pool's all-pass questions.

| sampler | questions | 5-9 of 10 fail | all 10 fail | 6+ fail and proven winnable |
|---|---|---|---|---|
| random | 15,000 | 66 | 1,027 | 52 (0.35%) |
| **ce** | 15,000 | 1,302 | 5,323 | **1,277 (8.5%)** |
| mab | 15,000 | 524 | 2,111 | 472 (3.1%) |
| sa | 15,000 | 47 | 981 | 33 (0.22%) |

Cross-entropy finds hard, winnable questions about 25 times as often as random sampling.
Simulated annealing does worse than random. Most "all 10 fail" hits from every sampler
are impossible starts, not hard questions.

### Proving a question winnable or impossible

"Every reference agent fails it" is not evidence of impossibility, and "some agent
survives X steps" is evidence but not proof. The grader's physics is deterministic and
known, so both directions can be *proven*:

- **Winnable (certificate).** A beam search over push sequences, ranked by the task's
  LQR cost-to-go (the value-function heuristic used for balancing tasks, cf. LQR-trees,
  Tedrake 2010), finds a sequence that keeps the pole up for the whole episode. The
  sequence is replayed through the grader's physics and must survive. This is "some
  model lasts X steps" with X = the full episode and a physics-aware planner as the model.
- **Impossible (proof).** Either the search was exhaustive (it never had to drop a live
  branch) and every branch died, or **interval reachability** shows it: a box containing
  every state reachable under *any* force in [-f, f] (a superset of the two pushes),
  intersected with the safe set each step, becomes empty. This is the over-approximation
  behind reachability tools such as CORA (Althoff et al.) and viability-kernel methods
  (Aubin; Saint-Pierre). A split variant branches exactly over the first 6 pushes and
  starts a box from each branch, which keeps the boxes tight for longer.
- **Unknown.** Neither.

Of the 11,067 search questions that 6+ agents fail:

| | 6-9 of 10 fail | all 10 fail |
|---|---|---|
| proven winnable | 1,625 | **209** |
| proven impossible | 0 | 8,263 |
| unknown | 0 | 970 |

Soundness checks: no question that any reference agent passes is ever called impossible
(all 1,625 here, a 3,000-question sample, and all of E0), and a unit test confirms the
interval step contains the true next state on 40,000 random state/force pairs. The 209
questions all ten agents fail but a planner wins are the hardest genuinely winnable
questions found. A wider beam (1,024) and deeper splitting (depth 10) each resolved only
a few percent of the unknowns, which are most likely slow-dying impossible starts; they
are left out.

### What the hard questions look like

All four edge clusters sit in **one corner** of the task space: a weak push, a long
pole, a heavy cart and a wide start range. K-means slices that corner rather than
finding four separate weak spots. Mean position of each section in its range
(0 = lowest, 1 = highest):

| section | pole length | pole mass | cart mass | push force | start range |
|---|---|---|---|---|---|
| E1v | 0.95 | 0.70 | 0.89 | 0.10 | 0.90 |
| E2v | 0.84 | 0.69 | 0.89 | 0.09 | 0.90 |
| E3v | 0.90 | 0.70 | 0.71 | 0.09 | 0.70 |
| E4v | 0.84 | **0.21** | 0.73 | 0.15 | 0.79 |
| E6 | 0.89 | 0.70 | 0.84 | 0.09 | 0.83 |
| E7 (easy) | 0.55 | 0.47 | 0.49 | 0.52 | 0.46 |
| E0 (general) | 0.50 | 0.52 | 0.51 | 0.53 | 0.47 |

A weak push is in every hard section. E6 and the edge sections therefore measure much
the same thing; the edge sections are secondary metrics, E6 is primary.

### Limits

- "Hard" is judged by the same ten reference agents the search ran against, so it means
  hard *for agents trained the plain way*. A method that differs from the baseline could
  find these easier or harder for reasons unrelated to skill.
- Adaptive samplers revisit neighborhoods, so questions within a section can be
  near-duplicates.
- Everything here is CartPole; the method (search, prove, cluster) transfers, the
  findings may not.

## Metrics

Every run is graded at the same **10 checkpoints**, one every 61,440 steps up to the
final model at 614,400 (`study.compare.checkpoint_grid`).

- **Learning-curve score** `auc_<section>`: mean success across the 10 checkpoints
  (trapezoid). `AUC_E0` is the pre-declared primary one: how well *and* how quickly it
  learned.
- **Final score** `final_<section>`: mean success over the last three checkpoints
  (491k, 553k, 614k steps). Calibration found 15 of 20 plain runs dipped below 0.5 at
  some check-in after 512k steps, so a single final snapshot is noisy.
  `final_E6` is the other pre-declared primary metric.
- `S_edge` / `auc_edge`: the macro-average over edge sections.

## Comparing methods

Each method is trained on its own independent seeds (the seed hashes the method name
and replicate number), and two methods are compared as independent groups: difference
of means, a bootstrap interval that resamples each group separately, and Welch's t-test
(`acl_bench/study/compare.py`).

**Why not match runs seed by seed.** A pilot (30 seeds x 4 methods, 204,800 steps,
`results/calibration/pilot_pairing.csv`) found runs that share a seed are identical at
step 0 but unrelated by the first check-in. Pairing removed 0-28% of the variance and
made some comparisons slightly worse, so it was dropped.

| comparison | correlation across seeds (`AUC_E0`) | variance pairing removed |
|---|---|---|
| A vs N | 0.29 | 28% |
| S_sa vs N | 0.09 | 9% |
| B_sa vs N | -0.12 | none |
| B_sa vs S_sa | -0.09 | none |
| B_sa vs A | 0.06 | 6% |

## Calibration (Stage 0)

20 plain runs trained to 1.2M steps (`results/calibration/lr_long_3e-4.csv`): every run
had taken off by 307k steps and mean E0 success plateaus around 490k-610k. The training
length is therefore **614,400 steps**. Noise at that length, and what 100 runs per method
can detect:

| training length | `AUC_E0` SD (20 plain runs) | final-score SD | smallest `AUC_E0` difference detectable at 100 runs |
|---|---|---|---|
| 204,800 (pilot) | 0.130 | 0.279 | 0.062 |
| 409,600 | 0.088 | 0.113 | 0.042 |
| **614,400 (chosen)** | **0.075** | **0.042** | **0.035** |
| 1,228,800 | 0.035 | 0.105 | 0.017 |

Outcomes are close to two-valued: most run-to-run variation is *when* a run takes off,
which is why the whole-curve score is about three times less noisy than any single
checkpoint.

**Learning rate stays at 3e-4.** Lowering it slowed learning without calming it
(20 runs per rate, 400k steps, `results/calibration/lr_*.csv`):

| learning rate | runs that took off | median take-off step | `AUC_E0` | final-score SD |
|---|---|---|---|---|
| 3e-4 | 85% | 164k | 0.43 | 0.29 |
| 1e-4 | 40% | 266k | 0.21 | 0.29 |
| 3e-5 | 0% | never | 0.12 | 0.04 |

At 1.2M steps (`lr_long_*.csv`) 1e-4 was still worse (80% took off against 100%;
`AUC_E0` 0.37 against 0.76) and not steadier.

## History

Earlier versions are in git history. The first exam (`cartpole_v2`) used hand-set
parameter quartiles for its edge sections ("weak push", "heavy cart", ...), taken from
one early probe with the LQR expert, and built E6/E7 from 80,000 random questions graded
by the reference agents (E6 then meant "5-9 of 10 fail", 248 questions, a 0.31% hit
rate). Both were replaced by the search-and-prove pipeline above: the quartile guesses
were partly right (weak push, heavy cart) but missed the long pole, and random sampling
cannot find hard questions efficiently or include the ones no agent passes.
