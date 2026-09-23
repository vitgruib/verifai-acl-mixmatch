# The ablation: does each component exist as an effect?

**Status: done.** Eight methods x 100 runs (800 runs of 614,400 steps), every run graded
on the final exam (docs/exam.md) at 10 checkpoints. Results below; the interactive
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

## The comparisons, decided in advance

| id | comparison | asks |
|---|---|---|
| C1 | A vs N | does the review pile help by itself? |
| C2 | S_x vs N (3 pickers) | does that picker help by itself? |
| C3 | B_x vs N (3 pickers) | do both help together? |
| C4 | B_x vs S_x (3 pickers) | does adding the pile help beyond the picker? |
| C5 | B_x vs A (3 pickers) | does adding the picker help beyond the pile? |

13 comparisons x 2 primary metrics = 26 tests, Holm-corrected together:
- **`AUC_E0`**: success on the general section averaged over the learning curve (how
  well *and* how fast it learned);
- **`final_E6`**: success on the hard section at the end.

Secondary metrics (easy section E7, the E6 learning curve, every edge section and their
average) are one exploratory family of 156 tests, Benjamini-Hochberg-corrected.

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

| method | `AUC_E0` | final E0 | **final E6 (hard)** | final E7 (easy) | edge average |
|---|---|---|---|---|---|
| N | 0.626 | 0.906 | 0.291 | 0.918 | 0.287 |
| A | 0.650 | 0.913 | 0.298 | 0.925 | 0.291 |
| S_ce | 0.660 | **0.961** | 0.262 | **0.974** | 0.258 |
| S_mab | 0.644 | 0.915 | 0.289 | 0.926 | 0.285 |
| S_sa | 0.624 | 0.876 | 0.280 | 0.888 | 0.276 |
| B_ce | 0.645 | 0.942 | 0.282 | 0.954 | 0.277 |
| B_mab | **0.665** | 0.922 | **0.319** | 0.934 | **0.314** |
| B_sa | 0.632 | 0.887 | 0.305 | 0.898 | 0.300 |

Run-to-run SD per method: `AUC_E0` 0.09-0.13, final E6 0.18-0.23.

**Finding: the cross-entropy picker improves final performance.** Final success on the
general section rises from 0.906 (plain) to 0.961 with the cross-entropy picker alone:
+0.055, 95% CI [+0.026, +0.085], p = 0.0003. Adding final general success to the two
primary metrics (39 tests) and Holm-correcting all of them, it survives (adjusted
p = 0.013); it was not one of the metrics declared in advance. The combined method
(cross-entropy + replay, separate runs) points the same way: +0.036, p = 0.019. The easy
section agrees (+0.056). No other picker, and not replay, shows an effect.

**On the two pre-declared primary metrics, nothing is detectable.** None of the 26 tests
survives Holm correction (smallest adjusted p = 0.28). The largest differences are
both-with-bandit vs plain on `AUC_E0` (+0.039, 95% CI [+0.009, +0.068], p = 0.011) and
cross-entropy alone vs plain on `AUC_E0` (+0.033, [+0.005, +0.061], p = 0.019). With
this noise, each comparison could reliably detect (80% power, Bonferroni level) a true
difference of about **0.06 in `AUC_E0`** and **0.11 in final E6**; smaller effects may
exist and would be missed.

**Secondary family:** 9 of 156 raw p < 0.05 (about 8 expected by chance); nothing
survives Benjamini-Hochberg (smallest q = 0.07).

**What this does and does not show.** At the calibrated budget and standard settings,
the cross-entropy picker raised final performance on ordinary tasks; nothing measurably
changed learning speed or hard-question success. Effects smaller than the
detection limits above are not ruled out, and a component that helps only at other
settings would look absent here.

## Known limits

- Conclusions are conditional on the fixed standard settings, on CartPole, and on PVL as
  the feedback function; failure-seeking feedback is untested.
- "Hard" is defined by the plain method's reference agents (docs/exam.md).
- Earlier plans also defined 31 hyperparameter variants and an Acrobot environment;
  they were never run in this study and were removed (see git history).
