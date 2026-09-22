# Ablation design (CartPole): does each component exist as an effect?

Status: **the solo-component comparison is done (N, A, S_ce, S_mab, S_sa; 100 runs
each, 500 runs). The combined "both" arms (B_ce, B_mab, B_sa) were not run this pass**
(pruned to shorten turnaround; a few leftover runs from before pruning remain in
`results/main.csv` but are not analyzed here). Results are in "Results" below.

## The question, in plain English

There are two components we are testing:

- **The review pile** (ACL): sometimes re-do old practice problems, favouring the ones
  that taught the most.
- **The problem picker** (a VerifAI sampler): choose new practice problems, some by
  chance and some by noticing where the agent struggles.

**Does each one help at all, and do they help together?** That is an *existence* test.
It deliberately does not tune anything: every setting is fixed at a standard value, so
a result cannot be an artifact of a lucky setting.

## The eight methods

| method | picker | review pile | plain meaning |
|---|---|---|---|
| **N** | random | off | neither component: plain random practice |
| **A** | random | on | review pile only |
| **S_ce**, **S_mab**, **S_sa** | cross-entropy / bandit / annealing | off | that picker only |
| **B_ce**, **B_mab**, **B_sa** | cross-entropy / bandit / annealing | on | both |

(The pile and the pickers all score problems with `pvl_gae`, SIPACL's own score. Which
scoring function is best is a separate study.)

## The comparisons, decided in advance

| id | comparison | asks |
|---|---|---|
| C1 | A vs N | does the review pile help by itself? |
| C2 | S_x vs N (3 pickers) | does that picker help by itself? |
| C3 | B_x vs N (3 pickers) | do both help together? |
| C4 | B_x vs S_x (3 pickers) | does adding the pile help beyond the picker? |
| C5 | B_x vs A (3 pickers) | does adding the picker help beyond the pile? |

13 comparisons, Holm-corrected together. Methods are trained on independent seeds and
compared as groups (docs/cartpole_suite.md, "Comparing methods").

## The settings, all fixed at standard values

"Source" says where each value comes from: **SIPACL** (`Work/custom/custom_gym.py`),
**VerifAI** (its own code defaults), or **ours** (chosen here with no external basis).

| setting | value | source |
|---|---|---|
| how often to redo old problems (`replay_prob`) | 0.5 | SIPACL |
| how strongly to favour the best old problems (`rank_alpha`) | 1.0 | SIPACL |
| how long scores remember (`ema_beta`) | 0.2 | SIPACL |
| pile size (`buffer_max`) | 5000 | SIPACL |
| cross-entropy picker: `alpha` / `thres` / `buckets` | 0.9 / 0 / 5 | VerifAI |
| bandit picker: `thres` / `buckets` | 0 / 5 | VerifAI |
| annealing picker: `T` / `decay_rate` / `iterations` | 1.0 / 0.9 / 20 | **ours** (VerifAI ships no defaults for annealing) |
| how a score becomes the picker's feedback | running z-score, clipped to +-3, divided by 3 | **ours** |
| PPO learning rate, discount, etc. | 3e-4 (constant), 0.99, ... | standard PPO settings |

The annealing picker's three values and the feedback mapping are the only ones with no
outside basis, so any result about annealing is conditional on them.

## How many runs, and what that can see

**100 runs per method: 8 x 100 = 800 runs.** At the calibrated length of 614,400 steps
the plain method's learning-curve score (`AUC_E0`, 0 to 1) has a run-to-run SD of about
0.075 (measured on 20 runs; other methods assumed similar), so the smallest difference
detectable is about **0.035** at 100 runs (about 0.065 at 30). Measured throughput: 10
runs of 614,400 steps took 3.2 minutes on 6 workers (about 95 s each), so 800 runs is
about **3.5 hours**, run at the lowest CPU priority with health safeguards (pauses if the
Mac is throttling, low on memory, on battery or short on disk; `touch results/STOP` stops
it cleanly).

**How to read "existence".** A component is shown to exist when the uncertainty range on
its difference excludes zero. If it does not, the honest statement is "no effect larger
than about 0.035 was detectable", *not* "no effect".

**Snapshots are saved** (`run_arms --snapshots`, about 1 GB for all 800 runs, not
committed), so the exam can be re-graded later, including the separate reporting pool,
without retraining.

## Parked: the hyperparameter variants

An earlier plan turned settings up and down (31 variants: 12 review-pile, 16 picker,
3 for whether the picker and the pile share one score). Given the noise, most would
have been measured too coarsely to tell anything, and the decision was to hold every
setting fixed for now. They are **defined and tested but not part of the plan**:

```
python -m acl_bench.suite.run_arms --arms family:acl      # 12 review-pile variants
python -m acl_bench.suite.run_arms --arms family:sampler  # 16 picker variants
python -m acl_bench.suite.run_arms --arms family:coupling # 3 shared-vs-separate score
```

If revived, the trimmed plan discussed was about 7 knobs: how often to redo (`replay_prob`)
and how strongly to favour the best (`rank_alpha`), optionally score memory (`ema_beta`),
one "aggressiveness" setting per picker (cross-entropy `alpha`, bandit `thres`,
annealing `T`), and the shared-vs-separate switch, about 15 variants at 100 runs each.
That would lose any interaction between `replay_prob` and `rank_alpha`, which the full
3x3 grid would have kept.

## Results (solo-component comparison, 100 runs each)

Ran C1 and C2 (a/b/c) -- the review pile alone, and each picker alone, each against
plain random practice with the pile off. The combined arms were not run this pass.

| arm | `AUC_E0` mean (SD) | `success_E6` mean (SD) |
|---|---|---|
| N (neither) | 0.590 (0.090) | 0.334 (0.171) |
| A (pile only) | 0.590 (0.075) | 0.337 (0.195) |
| S_ce (cross-entropy only) | 0.604 (0.072) | 0.309 (0.194) |
| S_mab (bandit only) | 0.592 (0.094) | 0.343 (0.186) |
| S_sa (annealing only) | 0.583 (0.087) | 0.337 (0.179) |

Measured noise matched the calibration estimate (SD ~0.07-0.09 per arm on `AUC_E0`,
close to the predicted 0.075), so the planned power held.

**On both primary metrics, no comparison is significant after Holm correction, and none
was significant even before correction.** All four 95% confidence intervals include
zero on `AUC_E0` (largest difference: cross-entropy +0.014, CI [-0.008, +0.037],
p=0.22) and on `success_E6` (largest: bandit +0.009, CI [-0.042, +0.058], p=0.73). With
100 runs per arm the study could detect a difference of about 0.035 in `AUC_E0`; the
largest difference actually observed was 0.014, well under that. **Reading this as "no
effect" rather than "not enough data" is reasonable here**, unlike the earlier pilot,
because the observed effects are small relative to the detection threshold, not merely
non-significant.

One secondary, non-primary result: cross-entropy alone scored higher than plain
practice on the macro-average across the edge sections (E1-E5), `S_edge` +0.028, CI
[+0.005, +0.055], p=0.032 unadjusted. This was not one of the two pre-declared primary
metrics, was not Holm-corrected against the other secondary checks available, and nothing
in the corrected primary analysis supports it, so it is reported as a lead worth
re-testing, not a finding.

**What this does and does not show.** At the calibrated budget and standard settings,
neither the review pile alone nor any single VerifAI picker alone moved the two primary
scores measurably against plain random practice. This does not test the combined arms
(does the pile plus a picker help together, which was the original "mix-and-match"
question) -- that comparison was deliberately deferred and can be resumed with
`--arms B_ce B_mab B_sa --resume` (12-16 runs of it already exist from before pruning).

## Exploratory sweep: every metric, every section, FDR-controlled

The result above uses only the two pre-declared primary metrics. To check whether
anything shows up elsewhere, every metric on every exam section was tested: final
success, area-under-the-learning-curve success, and final mean steps survived, for
all 10 sections (E0, E1, E1b, E2, E3a, E3b, E4, E5, E6, E7), against each of the four
comparisons -- **120 tests**. Because this is exploratory rather than a small
pre-declared set, it is controlled for **false discovery rate** (Benjamini-Hochberg:
bounds the expected share of false positives *among findings called significant*)
rather than Holm's family-wise control (which bounds the chance of *any* false
positive and is the right tool for the two primary metrics above, not for a sweep
this size). Full table: `results/analysis/fdr_sweep.csv`.

**Nothing survives.** 13 of 120 raw p-values are below 0.05 (versus about 6 expected
by chance alone if every one of the 120 null hypotheses were true), all of them in a
single comparison (cross-entropy alone vs. neither); the other three comparisons have
zero. After Benjamini-Hochberg correction, the smallest q-value in the whole sweep is
0.32 -- nothing clears even a lenient FDR of 10%, let alone 5%.

The 13 raw hits are not 13 independent pieces of evidence: they are `final_<set>` and
`final_steps_<set>` for many different, correlated sections (a run that does slightly
better tends to do slightly better on most sections at once), so they are closer to
one weak, unconfirmed signal counted many times than to a broad effect. Combined with
the FDR result, the honest reading is that the earlier secondary lead (cross-entropy
on the edge-section macro-average, `S_edge` p=0.032) does not gain support from
looking further -- it looks like the same noise viewed from another angle, not a
finding that keeps showing up. **The two-primary-metric result stands: no detectable
effect from any solo component at this budget.**

## Known limits

- Conclusions are conditional on the fixed standard settings; a component that helps
  only at a different setting would look absent.
- The plain random picker is the only non-adaptive picker in the plan; quasi-random
  (Halton) sampling is not tested here.
- All methods share the fixed PPO settings.
