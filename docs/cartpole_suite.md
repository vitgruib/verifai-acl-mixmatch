# CartPole test suite (design)

Status: **specification, not yet implemented.** Only `acl_bench/oracle.py`
exists (with tests); the evaluation sets, metrics code and runners below do not.
The numbers in "Motivating probe" are from one policy and one seed and are
evidence that the design is worth building, not results.

## Why CartPole

- **Fast**: ~5.4 s per 60k training steps, so a 400k-step run is ~36 s
  (single process, excluding evaluation).
- **It has failures to falsify**: 400k-step policies failed (mean return < 195) on
  15% and 2% of sampled tasks in two sensitivity runs (seeds 1 and 2), so how many
  failures there are depends heavily on the seed.
- **It has an oracle**: CartPole's physics linearizes cleanly, so an LQR
  controller can label which tasks are actually solvable. Acrobot has no such
  simple oracle and (in earlier sensitivity runs) no outright failures.

## Three kinds of failure, and why they must be separated

A policy failing on a task can mean three different things, and a curriculum
should only be credited or blamed for the first:

| class | definition | fixable? |
|---|---|---|
| **fixable** | the oracle balances the pole from this exact start, the policy does not | yes. This is what ACL and falsification-and-fix should target |
| **infeasible** | the episode is guaranteed to end on step 1 whatever the policy does (`abs(theta0 + tau*theta_dot0) > 12 deg` or the cart analogue; note a pole starting *past* 12 degrees can be rescued by velocity pointing back) | no. Wasted training episodes, and a falsification sampler will chase them |
| **unresolved** | feasible start, oracle also fails | unknown: the oracle is sufficient, not necessary, for solvability |

`acl_bench/oracle.py` implements the labels: discrete LQR on the linearized
dynamics, run bang-bang in the real environment from the exact initial state.
`survival_steps == 500` proves solvable; failure only means "not known solvable".

### Motivating probe (one 400k-step policy, seed 1, 1000 Halton-sampled (task, start) pairs)

| | share of pairs |
|---|---|
| guaranteed one-step failure (infeasible) | 5.0% |
| feasible, oracle fails (unresolved) | 0.8% |
| oracle solves | 94.2% |

The policy's 167 failures split into 50 infeasible, 8 unresolved and **109
fixable**, an 11.6% fixable failure rate. The oracle was never beaten by the
policy (0 of 8 oracle failures), so it is not obviously too conservative.

Fixable failure rate by parameter quartile (Q1 = lowest values):

| parameter | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| `force_mag` | **37.1%** | 5.9% | 2.9% | 0.9% |
| `masscart` | 1.7% | 6.8% | 15.8% | **22.3%** |
| `length` | 10.9% | 10.1% | 10.2% | 15.2% |
| `masspole` | 7.6% | 8.8% | 15.4% | 14.8% |
| `init_range` | 10.0% | 10.8% | 12.9% | 12.9% |

Weak push (force Q1) with a heavy cart (mass Q4): **70%** fixable failure
(39/56). This is the clearest falsifiable edge case found so far, and it is
fixable, not infeasible. It also revises an earlier sensitivity finding:
`init_range` looked like the top parameter for raw return, but on
oracle-solvable pairs it barely moves failures; its raw-return effect was
mostly the ~5% of starts that are unrecoverable.

## Task parameters

The task is `tau = (length, masspole, masscart, force_mag, init_range)`; an
*episode* is `(tau, s0, seed)` where `s0` is the exact initial state. Bounds are
those in `acl_bench/envs/param_cartpole.py`:

| parameter | bounds | role |
|---|---|---|
| `force_mag` | 4-16 N | actuator authority; the dominant fixable-failure driver in the probe |
| `masscart` | 0.5-2.0 kg | inertia the push must move; interacts with `force_mag` |
| `length` | 0.25-1.5 m (half-length) | pole time constant |
| `masspole` | 0.05-0.5 kg | weak effect in the probe |
| `init_range` | 0.05-0.3 | start perturbation size; drives *infeasible* starts more than policy weakness |

Derived difficulty coordinate for slicing and reporting:
`authority = force_mag / ((masspole + masscart) * g)` (probe range 0.19-2.22).

Training-time factors are the grid's: sampler x ACL x scoring function (58
cells). PPO hyperparameters stay fixed (see README). The **step budget `B` is not
fixed yet**: Stage 0 sets it from a convergence calibration, because the earlier
~400k estimate came from code that has since been fixed.

## Evaluation sets

Every set is a frozen list of `(tau, s0)` pairs generated once from a fixed seed
and stored with its oracle labels. All agents are scored on identical pairs
(paired comparison, which removes task-sampling noise; seed-to-seed noise in
earlier runs was large). "Learnable" = feasible start and oracle-solvable; the
headline metrics are computed on learnable pairs only, with raw (all-pairs)
numbers reported alongside.

| set | contents | question it answers |
|---|---|---|
| **E0 Uniform** | ~300 pairs uniform over the box | average-case generalization |
| **E1 Weak actuation** | `force_mag` in [4, 7] (bottom quartile), rest uniform, ~150 pairs; **E1b** adds `masscart` in [1.5, 2.0], ~100 pairs | the largest fixable failure region found |
| **E2 Heavy cart** | `masscart` top quartile, ~100 pairs | second-largest region |
| **E3 Long / heavy pole** | `length` top quartile; `masspole` top quartile, ~100 pairs each | slower dynamics, weaker in the probe |
| **E4 Large recoverable disturbance** | `init_range` in [0.2, 0.3] with *feasible* starts only, ~100 pairs | recovery from big-but-recoverable perturbations, without infeasible starts contaminating it |
| **E5 Corners** | all 32 vertices of the parameter box x 10 starts each | extremes that uniform training rarely visits |
| **E6 Hard-but-solvable** | from a 5000-pair pool, pairs the oracle solves but at least half of a reference population fails, ~200 pairs | difficult tasks isolated by measured difficulty, not by region |
| **E7 Easy control** | pairs every reference agent solves, ~100 pairs | regression check: a curriculum must not degrade easy tasks |

The **reference population** for E6/E7 is baseline-arm agents (`random`, ACL
off) trained in Stage 0 on seeds *not* used for any evaluated agent, so
difficulty is not defined by the agents being compared. E6 could still favor
methods that differ from the baseline; that limitation is stated, not solved.

### Falsification-based evaluation

A frozen final policy is attacked by the same VerifAI samplers, in test mode:

| | |
|---|---|
| **F1 Falsification search** | `ce`, `mab`, `sa` each propose candidate tasks (start drawn per candidate with a fixed seed) for 200 evaluations x 3 searcher seeds, minimizing return; a hit counts only if the pair is learnable |
| F1 metrics | fixable-failure discovery rate (hits / evaluations); evaluations to first hit; number of distinct failure modes (clusters in normalized parameter space) |
| **F2 Fix loop** (phase 2) | inject F1's found failures into training, retrain, re-run F1 with fresh searcher seeds and score on E6 (held out from the injected points) to measure repair, not memorization |

A more robust policy has a *lower* discovery rate. This measures what the
mean held-out return does not: how easy the policy is to break.

## Metrics

**Primary (two, fixed in advance):**
1. `AUC_E0`: area under the learnable-pair success-rate curve on E0 from step 0
   to `B` (sample efficiency in one number).
2. `success_E6`: learnable-pair success rate on the hard set at step `B`.

**Performance** (at step `B`, per set): success rate = fraction of learnable
pairs survived to 500 steps; mean return (secondary); **worst-slice success** =
minimum over E1-E5; **CVaR10** = mean success of the worst 10% of E0 tasks.

**Convergence speed** (checkpoints every ~20k steps on a small fixed E0
subset): steps to 50 / 80 / 90% success (right-censored if never reached, so
compare with survival methods, not by dropping runs); **plateau step** = the
earliest checkpoint from which the curve stays within 0.02 of its final level
(a formal criterion; the earlier plateau estimates were eyeballed); steps to
50% success on E6; final instability = SD of success over the last 5
checkpoints; across-seed SD of final success.

**Mechanism diagnostics** (why a method worked, not whether):
training-time task mix (share of training episodes that were infeasible,
easy, or hard-but-solvable); wasted-episode fraction (episodes ending within 2
steps); rank correlation between each scoring function's score and true
difficulty over the replay buffer (does the score actually rank hard tasks
higher?); replay share by difficulty bin; sampler concentration and coverage of
E1.

**Cost:** environment steps, wall-clock, sampler overhead.

## Protocol

- **Unit of replication is the training seed.** Task pairs are shared across
  agents (paired), so seeds are the only remaining source of variance.
- **A-priori contrasts** on the two primary metrics: C1 ACL only vs. neither;
  C2 adaptive sampler only vs. neither; C3 both vs. neither; C4 best scoring
  function vs. `neg_return` within ACL-on. Holm correction across them.
- **Stages.** *Stage 0*: calibrate `B` from baseline convergence, train the
  reference population, validate and freeze E0-E7. *Stage 1 (screening)*: all
  58 cells, 3 seeds, ranked by primary metrics; not for claims. *Stage 2
  (confirmation)*: baselines plus the top screened cells, 10 seeds, on seeds
  disjoint from screening, so the selected cell is not tested on the data that
  selected it. *Stage 3*: F1 on confirmed agents, then F2.
- **Report** effect sizes with bootstrap confidence intervals over seeds, not
  only p-values.

## Cost, and one engineering requirement

At ~11k steps/s, a 400k-step run is ~36 s, so screening (58 x 3 = 174 runs) is
~1.7 h single-process and confirmation (say 8 arms x 10 seeds) ~48 min. These
are training-only estimates, not measured end to end.

**Evaluation would cost as much as training** if done naively: the sets above
total ~1,570 pairs x up to 500 steps, i.e. up to ~800k policy steps per final agent, and
learning-curve checkpoints multiply that. The evaluator therefore needs to be
**batched** (step many pairs in lockstep with one batched policy forward pass);
that is a hard requirement of building this suite, not an optimization.

## Known limits

- Everything here is about CartPole; the conclusions may not transfer.
- The oracle is sufficient, not necessary, for solvability.
- The probe used one policy; the failure-region ranking may shift with the policy
  (the earlier sensitivity runs showed CartPole's binding parameters differed
  between two seeds).
- E6/E7 difficulty is defined by baseline-arm agents.
