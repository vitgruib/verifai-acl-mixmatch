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

**Does each help at all, and do they help together?** This is an *existence* test: every
setting is fixed at a standard value, so a result cannot be an artifact of tuning.

## The eight methods

| method | picker | review pile | plain meaning |
|---|---|---|---|
| **N** | random | off | neither: plain random practice |
| **A** | random | on | review pile only |
| **S_ce**, **S_mab**, **S_sa** | cross-entropy / bandit / annealing | off | that picker only |
| **B_ce**, **B_mab**, **B_sa** | cross-entropy / bandit / annealing | on | both |

The pile and the pickers score tasks with `pvl_gae`, SIPACL's own score.

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

**No component has a detectable effect on either primary metric.** None of the 26 tests
survives Holm correction (smallest adjusted p = 0.28). The largest differences are
both-with-bandit vs plain on `AUC_E0` (+0.039, 95% CI [+0.009, +0.068], p = 0.011) and
cross-entropy alone vs plain on `AUC_E0` (+0.033, [+0.005, +0.061], p = 0.019). With
this noise, each comparison could reliably detect (80% power, Bonferroni level) a true
difference of about **0.06 in `AUC_E0`** and **0.11 in final E6**; smaller effects may
exist and would be missed.

**Secondary family:** 9 of 156 raw p < 0.05 (about 8 expected by chance); nothing
survives Benjamini-Hochberg (smallest q = 0.07).

**One exploratory lead.** Cross-entropy alone ends up better on *ordinary* questions:
final E0 +0.055 vs plain ([+0.026, +0.085], p = 0.0003, not a pre-declared test) and
final E7 +0.056 (p = 0.0005, q = 0.07), with about half the run-to-run spread (final E0
SD 0.076 vs 0.130). It is *not* better on hard questions (final E6 -0.029, p = 0.33). A
plausible reading is that it steers practice toward the bulk of the task space and makes
the final policy more reliable there, at no gain on the hard corner. It needs a
confirmatory test before it is a finding.

**What this does and does not show.** At the calibrated budget and standard settings,
neither the review pile, nor any picker, nor both together measurably changed learning
speed or hard-question success against plain random practice. Effects smaller than the
detection limits above are not ruled out, and a component that helps only at other
settings would look absent here.

## Known limits

- Conclusions are conditional on the fixed standard settings and on CartPole.
- "Hard" is defined by the plain method's reference agents (docs/exam.md).
- The PPO update treats step-cap truncation as terminal while the task score does not
  (docs/sipacl.md).
- Earlier plans also defined 31 hyperparameter variants and an Acrobot environment;
  they were never run in this study and were removed (see git history).
