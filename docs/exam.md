# The CartPole exam

The exam is how every trained agent is scored. Training is a black box: all the exam
ever receives is an agent snapshot, *"the agent method M produced on run R, at check-in
S"*. It plays a fixed, locked set of questions and reports, per suite, the share it
passes and the mean number of steps it keeps the pole up.

A **question** is one game setup: a task (five physical parameters) plus one exact
starting state. The agent passes if it keeps the pole up for 500 steps. Grading is
deterministic (the agent always takes its single best action) and batched
(`acl_bench/exam/grader.py`), checked step by step against the real environment
(`tests/test_grader.py`).

## Two suites

`frozen_sets/cartpole/`, each locked with a SHA-256 fingerprint in the manifest.

| suite | questions | what it is |
|---|---|---|
| **random** | 282 | random tasks and starts across the whole task box, from fixed seeds; cleaned once of impossible questions by an LQR expert controller (since removed); all 282 are also proven winnable |
| **verifai** | 1,782 | every question the VerifAI samplers discovered that 6 or more of the 10 reference agents fail (including questions all 10 fail), proven winnable |

**Every question in both suites is proven winnable.** The random suite measures overall
competence; the VerifAI suite measures competence on the failures falsification finds.

The **reference agents** are 10 runs of the plain method (random tasks, no replay) on
seeds independent of every compared method, each represented by its best of its last
five check-ins on the random suite.

## How the VerifAI suite is built

1. **Falsify** (`acl_bench/exam/search.py`). Each adaptive VerifAI sampler
   (cross-entropy, bandit, simulated annealing) proposes 3,000 tasks, with 5 random
   starts each. All 10 reference agents play every (task, start) pair. The sampler's
   feedback is `rho = success rate - 0.5` over agents and starts, so, following
   VerifAI's convention, a task is a counterexample when rho is below the threshold 0
   (its falsifier counts `rho <= fal_thres`, default 0; the cross-entropy and bandit
   samplers update on `rho < thres`). Every visited pair is saved with the share of
   agents that fail it (`frozen_sets/search/`).
2. **Prove** (`acl_bench/exam/feasibility.py`). Every question 6+ agents fail is labeled
   winnable, impossible or unknown, with a proof (below).
3. **Build** (`acl_bench/exam/build.py`). The proven-winnable ones form the suite.

| sampler | questions tried | 6+ of 10 fail and proven winnable |
|---|---|---|
| cross-entropy | 15,000 | 1,277 (8.5%) |
| bandit | 15,000 | 472 (3.1%) |
| simulated annealing | 15,000 | 33 (0.2%) |

(Random sampling, measured the same way, found 0.35%: see commit `5800fb3`.)

One difference from VerifAI's own notion of "discovered": VerifAI's error table holds
*tasks* with rho <= 0 (failed at least half the time over agents and starts). The suite
is built per *question* (one task, one start) with the stricter 6-of-10 rule, because a
question, not a task, is what the grader plays.

### Proving a question winnable or impossible

The grader's physics is deterministic and known, so both directions can be *proven*:

- **Winnable (certificate).** A beam search over push sequences, ranked by the task's
  LQR cost-to-go, finds a sequence that keeps the pole up for the whole episode; the
  sequence is replayed through the grader's physics and must survive.
- **Impossible (proof).** Either the search was exhaustive and every branch died, or
  interval reachability shows it: a box containing every state reachable under any
  force in [-f, f], intersected with the safe set each step, becomes empty (the
  over-approximation behind reachability tools such as CORA). A split variant branches
  exactly over the first 6 pushes first.
- **Unknown.** Neither; left out.

Of the 9,993 search questions 6+ agents fail: 1,782 proven winnable (204 of them failed
by all ten agents), 7,258 proven impossible, 953 unknown. No question any reference
agent passes is ever called impossible, and a unit test confirms the interval step
contains the true next state on 40,000 random state/force pairs.

## Metrics

Every run is graded at the same **10 checkpoints**, one every 61,440 steps up to the
final model at 614,400 (`study.compare.checkpoint_grid`). Per suite, two measures, each
summarized two ways, so four metrics per suite and eight in all:

| | final (last three checkpoints: 491k, 553k, 614k) | curve (all ten checkpoints) |
|---|---|---|
| **success** (share passed) | `final_<suite>` | `auc_<suite>` |
| **mean steps survived** (0-500) | `final_steps_<suite>` | `auc_steps_<suite>` |

"Final" averages three checkpoints because agents dip briefly at single checkpoints.
"Curve" is the trapezoid average over the learning curve, so it rewards learning fast.
Mean steps survived gives partial credit: an agent that keeps the pole up for 450 steps
on a question it fails scores better than one that drops it at step 20.

## Comparing methods

Each method is trained on its own independent seeds (the seed hashes the method name
and replicate number), and two methods are compared as independent groups: difference
of means, a bootstrap interval that resamples each group separately, and Welch's t-test
(`acl_bench/study/compare.py`). A pilot (`results/calibration/pilot_pairing.csv`) found
runs that share a seed are identical at step 0 but unrelated by the first check-in, so
seed-matching was dropped.

## Calibration

20 plain runs trained to 1.2M steps (`results/calibration/lr_long_3e-4.csv`): every run
had taken off by 307k steps and mean success plateaus around 490k-610k, so the training
length is **614,400 steps**. The learning rate stays at 3e-4: lower rates slowed learning
without making it steadier (`results/calibration/lr_*.csv`).
