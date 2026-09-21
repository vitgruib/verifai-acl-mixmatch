# Ablation design (CartPole): does each component exist as an effect?

Status: the eight methods are defined and runnable (`acl_bench/suite/ablation.py`,
`acl_bench/suite/run_arms.py`). **Nothing has been run yet.** Two things must come
first: the calibration run that fixes how long training lasts, and the locked hard and
easy exam sections (E6, E7), which need reference agents (docs/cartpole_suite.md,
"Stages").

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

**100 runs per method: 8 x 100 = 800 runs.** Results are noisy (a run either "takes off"
or it does not), so the smallest difference detectable in the learning-curve score
(`AUC_E0`, 0 to 1) is about **0.052** at 100 runs (about 0.095 at 30). At the
provisional 400k-step length and an extrapolated 12 s per run that is about 2.7 hours;
if calibration lengthens training, the time grows in proportion (roughly 7 h at 1M
steps).

**How to read "existence".** A component is shown to exist when the uncertainty range on
its difference excludes zero. If it does not, the honest statement is "no effect larger
than about 0.05 was detectable", *not* "no effect".

**Snapshots are saved** (`run_arms --snapshots`), so the exam can be re-graded later
with sections that did not exist at training time, without retraining.

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

## Known limits

- Conclusions are conditional on the fixed standard settings; a component that helps
  only at a different setting would look absent.
- The plain random picker is the only non-adaptive picker in the plan; quasi-random
  (Halton) sampling is not tested here.
- All methods share the fixed PPO settings.
