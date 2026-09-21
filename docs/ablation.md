# Ablation design (CartPole)

Status: the arms are defined and runnable (`acl_bench/suite/ablation.py`,
`acl_bench/suite/run_arms.py`); none of the ablations below has been run yet. Value
ranges are my choices, bracketing each default, not tuned.

## Structure

Two levels, both compared **paired by seed** (docs/cartpole_suite.md, "Paired
design"): every arm is trained on the same seeds, and every variant is compared with
a named reference arm seed by seed.

1. **Component ablation** (the 2x2): which components do anything at all.
2. **Hyperparameter ablation**: whether that conclusion depends on how a component is
   configured. One factor at a time against a reference arm, except the two ACL
   selection-pressure parameters, which are crossed.

Not ablated here: the scoring-function axis (its own study; the arms use `pvl_gae`,
SIPACL's score), the environment, and PPO's own hyperparameters (held fixed, below).

## Level 1: the four primary arms

| arm | sampler | ACL | scoring fn | role |
|---|---|---|---|---|
| **N** | `random` | off | none (unused) | neither component: plain domain randomization |
| **A** | `random` | on | `pvl_gae` | ACL only |
| **S_x** | `x` in `ce`, `mab`, `sa` | off | `pvl_gae` (feedback only) | adaptive sampler only |
| **B_x** | `x` | on | `pvl_gae` | both |

Pre-declared contrasts (Holm-corrected across them): C1 `A` vs `N`; C2 `S_x` vs `N`;
C3 `B_x` vs `N`; plus `B_x` vs `S_x` and `B_x` vs `A` to separate the components'
contributions.

## Level 2: hyperparameters

"Source" says where the default comes from: **SIPACL** (`Work/custom/custom_gym.py`),
**VerifAI** (its code defaults), or **mine** (chosen by me, no external basis).

### ACL (reference arm `A`)

| hyperparameter | default (source) | values | what it controls |
|---|---|---|---|
| `replay_prob` | 0.5 (SIPACL) | 0.25, 0.5, 0.75 | share of episodes that replay a buffered task |
| `rank_alpha` | 1.0 (SIPACL) | 0.5, 1.0, 2.0 | replay concentration, `P(i) ~ rank^-alpha`. For a 100-task buffer the top-ranked task gets 5% / 19% / **61%** of replay and the top 10% of tasks get 27% / 57% / 95% |
| `ema_beta` | 0.2 (SIPACL) | 0.1, 0.5 | smoothing of a task's score across visits: low = long memory, high = trust the latest noisy episode |
| `buffer_max` | 5000 (SIPACL) | 100, 500 | FIFO buffer size. A 400k-step run draws hundreds to a few thousand new tasks, so 5000 barely evicts; smaller buffers test whether old tasks matter |

`replay_prob` x `rank_alpha` is a full 3x3 grid (8 non-default cells) because the two
jointly set selection pressure (how often, and how narrowly, replay revisits the top
tasks) and are the most likely to interact. `ema_beta` and `buffer_max` are one-at-a-time.
That is 8 + 2 + 2 = **12 arms**.

### Sampler (reference arms `S_ce`, `S_mab`, `S_sa`: adaptive sampler, ACL off)

| sampler | hyperparameter | default (source) | values | what it controls |
|---|---|---|---|---|
| `ce`, `mab` | `thres` | 0.0 (VerifAI) | -0.33, 0.33 | counterexample cutoff on `rho = -z/3` (z is the running z-score of the score). 0 counts a new task when its score is above the running mean (about 50% of tasks); -0.33 is stricter (z > +1, ~16%); +0.33 is looser (z > -1, ~84%) |
| `ce`, `mab` | `buckets` | 8 (**mine**; VerifAI's is 5) | 4, 16 | resolution of each dimension's bucket distribution |
| `ce` | `alpha` | 0.9 (VerifAI) | 0.5, 0.98 | retention: each counterexample moves a bucket distribution (1 - alpha) toward the sampled bucket. At 0.5 one counterexample halves the original weight; at 0.9 it takes 7 in the same bucket; at 0.98 it takes 35 |
| `sa` | `T` | 1.0 (**mine**) | 0.3, 3.0 | acceptance temperature. With `rho` spanning [-1, 1], a sample worse by 0.5 is accepted 19% / 61% / 85% of the time at T = 0.3 / 1 / 3 |
| `sa` | `decay_rate` | 0.9 (**mine**) | 0.8, 0.95 | proposal width `decay_rate^k` k steps after a re-heat: after 10 steps 0.11 / 0.35 / 0.60 |
| `sa` | `iterations` | 20 (**mine**) | 10, 40 | steps per epoch before re-heating |

That is 6 (`ce`) + 4 (`mab`) + 6 (`sa`) = **16 arms**. Held fixed at VerifAI's code
defaults: SA's cooling multiplier (0.8) and re-heat threshold (0.01); the bandit's UCB
constant. The bandit's `alpha` is stored by VerifAI but unused, so it is not ablated.

### Coupling (reference arms `B_ce`, `B_mab`, `B_sa`)

| variant | replay ranks by | sampler steers by |
|---|---|---|
| shared (default, the `B_x` arms) | `pvl_gae` | `pvl_gae` |
| **decoupled** (`B_x_decoupled`) | `pvl_gae` | `neg_return` |

This tests directly whether the sampler and ACL should share one scoring function.
The decoupled variant is closer to SIPACL, whose Scenic feedback is a separate signal.
3 arms.

### Held fixed everywhere

PPO: learning rate 3e-4 (constant), gamma 0.99, GAE lambda 0.95, clip 0.2, entropy
coefficient 0.01, value coefficient 0.5, gradient clip 0.5, 1024-step rollouts, 4
minibatches, 4 epochs, two 64-unit tanh layers for actor and critic. The `rho`
mapping (running z-score clipped to +-3, divided by 3). Deterministic evaluation on
the frozen sets. Step budget and checkpoint interval: `DEFAULT_BUDGET = 400,000`
(**provisional**, to be replaced by the calibration stage) and every 20,480 steps.

## Totals and cost

**39 arms** (8 primary, 12 ACL, 16 sampler, 3 coupling). Runs = arms x seeds.
Measured throughput (pilot: 120 runs of 204,800 steps in 757 s on 9 workers): about 6.3 s
of wall-clock per run, so roughly 12 s per 400k-step run (extrapolated). All 39 arms x
30 seeds is 1,170 runs, about 4 h.

**What that can and cannot see.** The pilot found run-to-run noise large (the SD of a
paired difference in whole-curve success, `AUC_E0`, is about 0.15) and pairing by seed
barely helps (docs/cartpole_suite.md, "Paired design"). With 30 seeds the smallest
difference detectable is about 0.09 AUC, so these ablations can only reveal *large*
sensitivity. They are a coarse screen: a hyperparameter that swings the result by
0.1 or more will show; a subtle one will not, and its absence should not be read as
"does not matter".

## Reading the results

Hyperparameter ablations are **sensitivity analyses, not confirmatory tests**: report
each variant's paired difference from its reference with a bootstrap confidence
interval, and look for structure (does a primary effect survive across the range, or
flip?), not for isolated p-values. Within a family, Holm-correct if p-values are
quoted. A component effect that holds across the ranges is robust; one that appears
only at a specific setting is fragile and should be reported as such.

## Known limits

- Ranges are my choices, bracketing defaults; no value has been tuned or justified
  beyond the effects tabulated above.
- One-factor-at-a-time misses interactions other than the crossed `replay_prob` x
  `rank_alpha` grid.
- Sampler defaults marked **mine** have no external basis; results for those samplers
  are conditional on them.
- All arms share the fixed PPO settings, so conclusions are conditional on them.
