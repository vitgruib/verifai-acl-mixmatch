# The ablation: does each component exist as an effect?

**Status: done.** Eight methods x 100 runs (800 runs of 614,400 steps), every run graded
on both test suites (docs/exam.md) at 10 checkpoints. Results below; the interactive
report is linked from the README.

## The question

Two components can each be switched on or off:

- **The review pile** (ACL): sometimes replay an already-seen task, favouring the ones
  that taught the most (SIPACL's prioritized replay; docs/sipacl.md).
- **The problem picker** (a VerifAI sampler): choose new tasks by steering toward the
  ones the agent is learning most from (cross-entropy, bandit or simulated annealing),
  instead of at random.

**The feedback function.** Both components are driven by the same task score,
**positive value loss** (`pvl_gae`, SIPACL's learning-potential score): the mean over an
episode of max(GAE advantage, 0), i.e. how much better the episode went than the critic
expected. The pile replays tasks ranked by it; after every newly drawn task the picker
receives `rho = -clip(z, -3, 3) / 3`, where `z` is that score z-scored against the run's
feedback so far, so an above-average learning potential reads as a VerifAI
"counterexample" (docs/sipacl.md). **So the pickers here seek learning potential, not
failure.** VerifAI's samplers were built for falsification (feedback = how badly the
system did); a failure-seeking feedback (e.g. `neg_return`, minus the episode return) is
a different experiment and is not tested here. Every conclusion below is conditional on
PVL feedback.

**Does each help at all, and do they help together?** This is an *existence* test: every
setting is fixed at a standard value, so a result cannot be an artifact of tuning.

## The eight methods

| method | picker | review pile | plain meaning |
|---|---|---|---|
| **N** | random | off | neither: plain random practice |
| **A** | random | on | review pile only |
| **S_ce**, **S_mab**, **S_sa** | cross-entropy / bandit / annealing | off | that picker only |
| **B_ce**, **B_mab**, **B_sa** | cross-entropy / bandit / annealing | on | both |

## The comparisons

| id | comparison | asks |
|---|---|---|
| C1 | A vs N | does the review pile help by itself? |
| C2 | S_x vs N (3 pickers) | does that picker help by itself? |
| C3 | B_x vs N (3 pickers) | do both help together? |
| C4 | B_x vs S_x (3 pickers) | does adding the pile help beyond the picker? |
| C5 | B_x vs A (3 pickers) | does adding the picker help beyond the pile? |

Each is measured on two test suites (docs/exam.md): **random** (282 random questions)
and **verifai** (1,782 questions the VerifAI samplers discovered that the plain-trained
reference agents fail, each proven winnable). Per suite, four metrics:

| | final (last three checkpoints, 491k-614k steps) | curve (all ten checkpoints) |
|---|---|---|
| success (share passed) | `final_<suite>` | `auc_<suite>` |
| mean steps survived (0-500) | `final_steps_<suite>` | `auc_steps_<suite>` |

13 comparisons x 8 metrics = **104 tests, Holm-corrected together.**

## The settings, all fixed at standard values

| setting | value | source |
|---|---|---|
| how often to replay (`replay_prob`) | 0.5 | SIPACL |
| how strongly to favour the best tasks (`rank_alpha`) | 1.0 | SIPACL |
| how long scores remember (`ema_beta`) | 0.2 | SIPACL |
| pile size (`buffer_max`) | 5,000 | SIPACL |
| cross-entropy picker: `alpha` / `thres` / `buckets` | 0.9 / 0 / 5 | VerifAI |
| bandit picker: `thres` / `buckets` | 0 / 5 | VerifAI |
| annealing picker: `T` / `decay_rate` / `iterations` | 1.0 / 0.9 / 20 | **ours** (VerifAI ships none) |
| score to picker feedback | running z-score, clipped to +-3, divided by 3, negated | **ours** |
| PPO | lr 3e-4, gamma 0.99, GAE lambda 0.95, clip 0.2, entropy 0.01; 1,024-step rollouts, 4 minibatches, 4 epochs | standard CleanRL |

The annealing values and the feedback mapping have no outside basis, so results about
annealing are conditional on them.

## Results

| method | random: final success | random: curve success | random: final steps | random: curve steps | verifai: final success | verifai: curve success | verifai: final steps | verifai: curve steps |
|---|---|---|---|---|---|---|---|---|
| N | 0.906 | 0.626 | 479.7 | 412.4 | 0.292 | 0.244 | 205.7 | 209.8 |
| A | 0.913 | 0.650 | 481.5 | 418.3 | 0.298 | 0.255 | 208.4 | 215.2 |
| S_ce | **0.961** | 0.660 | 490.5 | 409.0 | 0.263 | 0.254 | 186.1 | 209.6 |
| S_mab | 0.915 | 0.644 | 482.8 | 413.1 | 0.290 | 0.250 | 206.8 | 213.4 |
| S_sa | 0.876 | 0.624 | 473.1 | 411.1 | 0.280 | 0.228 | 201.8 | 205.2 |
| B_ce | 0.942 | 0.645 | 485.9 | 406.6 | 0.282 | 0.249 | 197.0 | 210.1 |
| B_mab | 0.922 | 0.665 | 482.1 | 417.4 | 0.320 | 0.274 | 220.4 | 220.0 |
| B_sa | 0.887 | 0.632 | 476.4 | 411.6 | 0.305 | 0.260 | 213.4 | 215.5 |

**Finding: the cross-entropy picker improves final performance on random questions.**
Final success on the random suite rises from 0.906 (plain) to 0.961 with the
cross-entropy picker alone: +0.055, 95% CI [+0.026, +0.085], p = 0.0003, Holm-adjusted
p = 0.034 across all 104 tests. The other measures point the same way without passing
correction: final mean steps +10.8 (p = 0.004), and the combined method
(cross-entropy + replay, separate runs) +0.036 final success (p = 0.019).

**Nothing else is detectable.** No other test survives correction (7 of 104 have raw
p < 0.05, about 5 expected by chance). In particular, **no method improved on the
VerifAI-discovered questions**; cross-entropy alone is, if anything, slightly lower there
(final success -0.030, p = 0.30). Each test could reliably detect (80% power, Bonferroni
level) a difference of about:

| | random suite | verifai suite |
|---|---|---|
| final success | 0.08 | 0.12 |
| curve success | 0.07 | 0.07 |
| final mean steps | 18 | 57 |
| curve mean steps | 17 | 28 |

## Known limits

- Conclusions are conditional on the fixed standard settings, on CartPole, and on PVL as
  the feedback function; failure-seeking feedback is untested.
- "Hard" is defined by the plain method's reference agents (docs/exam.md).
- Earlier plans also defined 31 hyperparameter variants and an Acrobot environment;
  they were never run in this study and were removed (see git history).
