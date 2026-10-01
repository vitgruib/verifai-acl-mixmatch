# Curriculum library: log of every arm tried

Protocol: `docs/protocol.md`. Design: `docs/library.md`. Each batch is registered here
before it runs; results and verdicts are added after. Nothing is deleted.

## Batch 1 (registered 2026-09-28): the library's first arms, stage A

DR pool: 48 seeds per dev environment (`results/<env>/lib/runs.csv`). Arms: 8 seeds on
CartPole and on Acrobot (`results/<env>/lib/batch1.csv`).

| arm | hypothesis |
|---|---|
| `fmodel` | reference: the pre-library frontier picker (best of 64 uniform tasks by model p(1-p), 50% uniform) |
| `sir_a1` | sampling in proportion to model learnability (soft SFL signal, no scouting) helps |
| `sir_a4` | the same, sharper (alpha 4) |
| `sir_is` | variance-optimal unbiased sampling: q ~ sqrt(p(1-p)) with full importance correction speeds up learning DR's own objective without changing it (no harm by construction) |
| `sir_is05` | half-corrected frontier bias: keeps part of the frontier gain, limits harm |
| `replay_post` | a PLR buffer ranked by posterior learnability (model prior + the task's own visits) |
| `mutate_post` | ACCEL-style edits of the posterior buffer's picks |
| `verifai_surr` | VerifAI's cross-entropy sampler, searching the pass model instead of the simulator |

## Batch 2 (registered 2026-09-28): model-specific gradient signals -- abandoned at the mechanism check

| arm | hypothesis |
|---|---|
| `grad_is` | variance-optimal unbiased sampling with each episode's measured policy-gradient size (actor last layer, closed form), q ~ E‖g(task)‖, full importance correction |
| `align` | sample tasks whose gradient aligns with the uniform tasks' gradient in the same rollout |

**Mechanism check (before any screen):** is the per-episode signal predictable from the
task? Cross-validated R² of the task regressor at the end of short runs (2 seeds each):
CartPole grad size -0.25 / -0.22, alignment -0.62 / -0.71; Acrobot grad size -0.77 / -0.84,
alignment -0.72 / -1.23. On the same check a pure-noise target scores -0.29 and a real
signal +0.18 (the regressor overfits, so the check is conservative). Neither signal can be
predicted from the task: single-episode gradients are dominated by noise and episode
length. **Both abandoned untrained.** Note: batch 1's Acrobot half runs on code that
initializes the pass model without touching torch's global random stream (CartPole's half
did touch it): a different random stream, not a different algorithm.

## Batch 1, stage A results (2026-09-29)

8 seeds per arm against the 48-seed DR pool. `vd`: CartPole hard suite, dev half (primary);
`r`: random suite (guard). Welch p in brackets.

| arm | CartPole vd | CartPole r | Acrobot r | verdict |
|---|---|---|---|---|
| `fmodel` | +0.103 (0.16) | +0.029 (0.63) | -0.011 (0.64) | advance to B |
| `sir_a1` | -0.004 (0.95) | +0.005 (0.94) | -0.013 (0.48) | abandon: primary < 0 |
| `sir_a4` | +0.018 (0.82) | +0.040 (0.39) | -0.021 (0.35) | advance to B |
| `sir_is` | +0.007 (0.94) | **+0.091 (0.02)** | -0.028 (0.28) | advance to B |
| `sir_is05` | -0.042 (0.51) | +0.003 (0.96) | -0.069 (0.08) | abandon: primary < 0 |
| `replay_post` | -0.155 (0.003) | +0.066 (0.17) | -0.024 (0.23) | abandon: primary < 0 |
| `mutate_post` | -0.015 (0.80) | +0.023 (0.69) | -0.114 (0.001) | abandon: primary < 0, Acrobot guard |
| `verifai_surr` | -0.008 (0.91) | +0.020 (0.74) | +0.002 (0.89) | abandon: primary < 0 |

Reading: soft sampling (alpha 1) and the replay buffers do nothing or harm; only the sharp
picker (`fmodel`, and weakly `sir_a4`) moves the hard suite. `sir_is` is the one result
matching its own theory: it is built to learn DR's objective faster, and DR's objective is
the random suite, where it gains +0.091 -- but this was not its declared primary, and it is
one of 16 guard comparisons (uncorrected).

**Registered before stage B's data (secondary hypothesis H-is):** on CartPole seeds 9-24
only (fresh), `sir_is` improves the random suite over DR (one-sided Welch, p < 0.05). If it
holds, `sir_is` goes to stage C with the random suite as its primary, on all envs.

Stage B (`results/lib/stageB1.sh`): the three survivors, CartPole seeds 9-24, MountainCar
and Pendulum seeds 1-8.

## Batch 3 (registered 2026-09-29, before stage B's results): variations on the survivors, stage A

| arm | hypothesis |
|---|---|
| `sir_is_a1` | unbiased with a sharper proposal (alpha 1, full correction): the random-suite gain grows past sqrt(p(1-p)) |
| `sir_is_u50` | unbiased and gentler (alpha 0.5, 50% uniform): smaller weights, same gain with less harm on Acrobot |
| `rw_a4` | fmodel's emphasis applied as loss weights s^4/Z on uniform tasks: the hard-suite gain without narrowing what the agent sees |
| `ens_a4` | alpha-4 sampling from a 4-model bootstrap ensemble plus a disagreement bonus: better frontier estimates early |
| `prog_a4` | alpha-4 sampling by model-based learning progress \|p_now - p_then\| instead of p(1-p) |

Same stage-A rules, seeds 1-8, CartPole and Acrobot (`results/<env>/lib/batch3.csv`).

## Batch 1, stage B results (2026-09-29): all three abandoned

24 CartPole seeds, 8 on MountainCar and Pendulum, against 48-seed DR pools. Guards are the
random suite.

| arm | CartPole vd | CartPole r | Acrobot r | MountainCar r | Pendulum r | verdict |
|---|---|---|---|---|---|---|
| `fmodel` | +0.022 (0.62) | -0.064 | -0.011 | +0.097 | -0.105 | abandon (primary p) |
| `sir_a4` | +0.055 (0.24) | +0.023 | -0.021 | +0.013 | +0.030 | abandon (primary p) |
| `sir_is` | +0.073 (0.20) | +0.027 | -0.028 | -0.042 | +0.020 | abandon (primary p) |

On the fresh seeds 9-24 alone: `fmodel` vd -0.019, r -0.111; `sir_a4` vd +0.074, r +0.014;
`sir_is` vd +0.107, r -0.005. **H-is fails** (fresh-seed random-suite gain -0.005,
one-sided p = 0.54): stage A's +0.091 was noise. Stage A's leads did not replicate in
either direction, so its ranking carries little information at this effect size.

This led to protocol Amendment 1 (`docs/protocol.md`): a 96-seed DR pool and 72-seed
stage B. **Registered re-screen (batch 1R):** `sir_a4` and `sir_is` get stage B under the
amendment, on seeds 25-72 added to their 24 (their stage-B results above are disclosed
here; stage C on fresh seeds decides any claim). `fmodel` stays dropped: negative on fresh
seeds, with a Pendulum guard at -0.105.

## Batch 3, stage A results (2026-09-29)

8 seeds, against the 48-seed DR pools (read before the CartPole pool grew to 96).

| arm | CartPole vd | CartPole r | Acrobot r | verdict |
|---|---|---|---|---|
| `sir_is_a1` | -0.017 (0.87) | +0.092 | +0.004 | abandon: primary < 0 |
| `sir_is_u50` | +0.082 (0.45) | +0.065 | +0.002 | advance to B |
| `rw_a4` | -0.090 (0.29) | +0.054 | -0.154 | abandon: primary < 0, Acrobot guard |
| `ens_a4` | +0.131 (0.14) | -0.004 | -0.014 | advance to B |
| `prog_a4` | +0.140 (0.12) | +0.076 | +0.002 | advance to B |

Loss reweighting (`rw_a4`) harms: weighting without resampling puts heavy weights on few
episodes and hurts Acrobot. Stage B under Amendment 1 (`results/lib/stageB3.sh`): CartPole
seeds 9-72, MountainCar and Pendulum seeds 1-8.

## Batch 1R results (2026-09-29): both abandoned

Amended stage B: 72 CartPole seeds against the 96-seed DR pool.

| arm | CartPole vd | CartPole r | verdict |
|---|---|---|---|
| `sir_a4` | -0.002 (0.96) | -0.012 | abandon |
| `sir_is` | +0.039 (0.22) | -0.005 | abandon |

With 72 seeds, sharp SIR sampling does nothing and the unbiased variant's lead shrinks to
+0.039. Batch 1 is closed. (Queueing note: a `pgrep -f` wait on a script name also matches
any shell whose command line contains that name; later scripts use `pgrep -f "name[.]sh"`
patterns that cannot match themselves.)

## Batch 3, stage B results (2026-09-29): all abandoned

Amended stage B: 72 CartPole seeds vs the 96-seed DR pool; 8 on MountainCar and Pendulum.

| arm | CartPole vd | CartPole r | Acrobot r | MountainCar r | Pendulum r | verdict |
|---|---|---|---|---|---|---|
| `sir_is_u50` | -0.008 (0.81) | -0.001 | +0.002 | -0.008 | -0.026 | abandon |
| `ens_a4` | +0.016 (0.60) | -0.043 | -0.014 | +0.097 | +0.012 | abandon |
| `prog_a4` | +0.010 (0.75) | +0.011 | +0.002 | +0.044 | -0.069 | abandon |

Stage A's +0.13 / +0.14 leads fell to about +0.01.

## Where the library stands (2026-09-29)

13 arms tried, none survives stage B. Every pass-model-driven arm (SIR at any sharpness,
importance-corrected or not, posterior replay, mutation, VerifAI on the model, ensemble,
learning progress, loss reweighting) lands within about +/-0.04 of DR on CartPole's hard
suite at 72 seeds, while the oracle (training on the suite's own tasks, `docs/plr.md`)
gains about +0.19 and scouted SFL about +0.09. Stage A with 8 seeds had no predictive
value: its leads (up to +0.14) did not replicate once. Conclusion: sampling by an
episode-outcome model of p(1-p), in any form tried, does not find the region the hard
suite tests. More variants of the same signal are not worth screening.

## Diagnostic (2026-09-29): why the pass model misses the hard suite

`python -m acl_bench.curriculum.diagnose --seeds 1-6 --checks 5`: plain DR on CartPole with a
passive pass model; at each check-in it is compared with the greedy agent's pass rate from
4 random starts on the hard suite's dev tasks and 400 uniform tasks. End of training (614k):

| | uniform tasks | hard dev tasks |
|---|---|---|
| true pass rate (greedy, random starts) | 0.89 | 0.57 |
| pass model p | 0.19 | 0.02 |
| true p(1-p) | 0.05 | 0.17 |

The pass model is wrong in level, not only rank: its 4,096-episode window holds almost the
whole run (~2,800 episodes), so 79% of its labels are failures, most from early policies,
and "success" (a full 500-step survival of the *stochastic* policy) is rare. It calls the
hard suite impossible: the dev tasks sit at the 9th percentile of its frontier score among
uniform tasks, and its score's rank correlation with true learnability is -0.14.

Candidate estimators fed the same episode stream (mean of 6 seeds, end of training):

| estimator | dev tasks' frontier percentile | top-10% truly frontier (base 0.26) | Spearman vs true p(1-p) |
|---|---|---|---|
| pass model, window 4,096 (all batch 1-3 arms) | 0.09 | 0.21 | -0.14 |
| pass model, window 512 | 0.31 | 0.28 | +0.14 |
| success variance, window 512 | 0.28 | 0.32 | +0.06 |
| **return std, window 512** (`VarModel`) | **0.67** | **0.55** | **+0.39** |
| return std, window 2,048 | 0.21 | 0.30 | +0.03 |

Both parts matter: a continuous outcome (return, which every environment has; for a binary
outcome its variance is p(1-p)) and a short window (the current policy). Early in training
(before ~250k steps) no estimator is informative.

## Batch 4 (registered 2026-09-29): outcome-variance learnability, stage A

`VarModel` (`estimators.py`): heteroscedastic Gaussian fit of the standardized return over
the task space, last 512 episodes, refit every 128; score = predicted std of the return.
Sampled through the same SIR proposer as batch 1.

| arm | hypothesis |
|---|---|
| `var_a2` | sampling in proportion to std^2 (the predicted return variance) moves the hard suite |
| `var_a4` | sharper (std^4) |
| `var_a8` | sharper still (close to argmax of 64) |
| `var_is` | q ~ std (the variance-optimal proposal), full importance correction: DR's objective, faster |

Stage A rules, seeds 1-8, CartPole and Acrobot (`results/<env>/lib/batch4.csv`).

## Batch 4, stage A results (2026-09-29)

8 seeds vs the DR pools (96 CartPole, 48 Acrobot).

| arm | CartPole vd | CartPole r | Acrobot r | verdict |
|---|---|---|---|---|
| `var_a2` | -0.023 (0.75) | -0.005 | +0.003 | abandon: primary < 0 |
| `var_a4` | +0.066 (0.28) | +0.035 | -0.030 | advance to B |
| `var_a8` | -0.021 (0.70) | +0.004 | -0.019 | abandon: primary < 0 |
| `var_is` | +0.045 (0.56) | -0.045 | +0.005 | advance to B |

No monotone trend in alpha; stage B (`results/lib/stageB4.sh`) decides.

## Batch 4, stage B results (2026-09-29)

72 CartPole seeds vs the 96-seed DR pool. The guard runs stopped part-way (battery screen:
MountainCar 4 of 8 seeds, Pendulum none); they are moot because both arms fail the primary.

| arm | CartPole vd (p) | CartPole r | Acrobot r | MountainCar r (4 seeds) | verdict |
|---|---|---|---|---|---|
| `var_a4` | +0.008 (0.79) | +0.003 | -0.030 | +0.097 | abandon |
| `var_is` | -0.000 (0.99) | -0.032 | +0.005 | -0.115 | abandon |

Batch 4 is closed. The better offline signal (VarModel: dev tasks at the 67th percentile vs the
pass model's 9th) did not translate into gains; the 8-seed stage A leads shrank to zero again,
as in batches 1 and 3. Next step is a cheap probe of *why*: does SIR on VarModel actually shift
training mass toward the dev region, and when (the signal is uninformative before ~250k steps)?

## Mass probe: where do the proposers put training mass? (2026-09-29)

`acl_bench/curriculum/massprobe.py`: train DR with passive estimators; at each check-in draw
1000 proposals from SIR configs and report `near_dev` (share within 2x the median
nearest-neighbour distance of a hard-suite dev task, unit coordinates) and `learn` (mean true
p(1-p) of 200 proposals, greedy agent, 4 random starts). CartPole, seeds 1-3, cost ~3 min.

The dev tasks sit in a corner (unit means: length 0.90, masspole 0.69, masscart 0.84,
force 0.10, init_range 0.84); uniform puts 0.3% of its mass near them. At 614k steps:

| score (SIR) | near_dev | learn |
|---|---|---|
| uniform | 0.003 | 0.063 |
| VarModel std, alpha 4-16, 64 or 4096 cand. | 0.003-0.008 | 0.133-0.145 |
| PassModel p(1-p) (window 512) | 0.001-0.002 | 0.077-0.081 |
| exp(-mean) ("low") | 0.004-0.018 | 0.102-0.113 |
| std*exp(-mean) ("lowstd") | 0.003-0.015 | 0.148-0.176 |

Findings: (1) VarModel SIR does what it should, doubling true learnability of proposals, but
the frontier it finds is not the hard suite's corner, which explains why batch 4's better
offline signal gave no primary gain. (2) Tilting toward low predicted return moves mass
toward the corner (~5x uniform) and raises learnability further; sharper alpha raises
learnability but not corner mass. No score reaches more than ~2% corner mass.

## Batch 5 (registered 2026-09-29, before running)

`VarModel(tilt=1)`: score = std * exp(-mean_z), mean_z the predicted standardized return
(clipped to +-4). Env-agnostic: both are in units of the return's own spread.

| arm | hypothesis |
|---|---|
| `var_low` | tilted frontier, SIR alpha 4 over 64 candidates, 50% uniform |
| `var_low16` | the highest-learnability probe config: alpha 16 over 4096 candidates |

Stage A rules, seeds 1-8, CartPole and Acrobot (`results/<env>/lib/batch5.csv`).

## Batch 5, stage A results (2026-09-29)

8 seeds vs the DR pools (96 CartPole, 48 Acrobot).

| arm | CartPole vd (p) | CartPole r | Acrobot r | verdict |
|---|---|---|---|---|
| `var_low` | +0.073 (0.32) | -0.047 | -0.007 | advance to B |
| `var_low16` | +0.158 (0.04) | +0.061 | -0.050 | advance to B |

`var_low16` is the largest stage A lead so far; earlier leads of this size (+0.14) did not
replicate. Stage B: `results/lib/stageB5.sh`.

## Batch 5, stage B results (2026-09-29)

72 CartPole seeds vs the 96-seed DR pool; 8 seeds on MountainCar and Pendulum.

| arm | CartPole vd (p) | CartPole r | Acrobot r | MountainCar r | Pendulum r | verdict |
|---|---|---|---|---|---|---|
| `var_low` | **+0.082 (0.009)** | -0.014 | -0.007 | +0.097 | +0.048 | **advance to C** |
| `var_low16` | +0.082 (0.004) | -0.001 | -0.050 | +0.097 | +0.028 | abandon: Acrobot guard < -0.03 |

The first arm in 22 to pass stage B. The stage A lead held up at 72 seeds (+0.073 at 8 seeds, +0.082
at 72), unlike every earlier batch. MountainCar is effectively binary: runs freeze at the
"always push right" plateau (0.671) or collapse; DR collapses in 15% of its 48 runs and
neither arm did in 8, so the +0.097 is real but coarse.

## Batch 5, stage C (registered 2026-09-29, before running)

Frozen configuration: `var_low` exactly as in stage B (VarModel window 512, refit 128, tilt 1;
50% uniform / 50% SIR alpha 4 over 64 candidates), commit at launch. DR and `var_low` on
seeds 1001-1100 on CartPole, Acrobot, MountainCar, Pendulum (DR into `runs.csv`, the arm into
`batch5.csv`); `results/lib/stageC5.sh`, ~1.5 h on 6 workers. Declared tests (protocol section 3):
- primary: CartPole hard suite, **test half** (`fin_vt`), gain > 0 at Holm-corrected p < 0.05
  (one arm, so Holm is the raw Welch p);
- guards: random suite on all four envs, one-sided 95% lower bound of the gain > -0.03;
- then held-out Maze and PointNav, the frozen configuration reported as is (registered
  separately once the dev envs pass).

Note added during stage C, before any confirm results were read: `decide --stage C` now implements
the declared tests (Holm over arms' primary p; one-sided 95% Welch lower bound per guard; missing
data counts as failure). On the stage B data the CartPole random-suite guard (-0.014) has a lower
bound of -0.057; at 100 vs 100 seeds the SE drops only ~10%, so the -0.03 non-inferiority
test needs a true guard effect of about +0.01 or better. The protocol is not changed mid-run.

## Batch 5, stage C results (2026-09-30)

Seeds 1001-1100, 100 DR vs 100 `var_low` per env; primary on the CartPole hard suite **test half**.

| test | gain | p / 95% lower bound | pass? |
|---|---|---|---|
| primary: CartPole hard suite (test) | **+0.055** | Holm p = 0.044 | yes |
| guard: CartPole random | -0.044 | -0.085 | no |
| guard: Acrobot random | -0.031 | -0.044 | no |
| guard: MountainCar random | +0.055 | +0.022 | yes |
| guard: Pendulum random | -0.029 | -0.058 | no |

**Verdict: FAIL** (guards). The hard-suite gain is real and replicates on fresh seeds and the
held-out test half: the first confirmed primary gain from any arm. But it is a trade, not a free
gain: leaning training toward low-return tasks costs ~0.03-0.04 on the random suite of three of
the four envs (MountainCar gains, from fewer collapsed runs). The stage B guard point estimates
(-0.014, -0.007, +0.048) were optimistic; stage B's guard test has no power at 8-72 seeds.

Next: keep the tilt's mass shift toward hard tasks while paying back the average-case cost.

## Batch 6 (registered 2026-09-30, before running): tilt, then cool down to DR

Diagnosis from the stage C curves (per check-in gain, 100 vs 100 seeds): `var_low`'s random-suite
cost is not constant. On Pendulum it reaches -0.21 mid-training and is gone by the last check-in
(+0.013); on Acrobot it appears after 50% of the budget. The random suite is DR's own objective,
and typical tasks might be relearned quickly once training returns to them, while the hard-task skill
learned under the tilt may persist. Hypothesis: training that *ends* on plain DR keeps part of the
hard-suite gain and pays back the average-case cost. New `Curriculum(cooldown=(a, b))`: the
non-uniform proposers' share falls linearly to 0 between fractions a and b of the training
budget (environment-agnostic: a fraction of whatever budget the run has).
- `var_low_cool`: `var_low`, cool-down (0.6, 0.8), so the last three check-ins train pure DR.
- `var_low_lin`: `var_low`, SIR share falls linearly from 0.5 to 0 over the whole run.
Stage A as usual (`results/lib/batch6.sh`, seeds 1-8, CartPole + Acrobot, `batch6.csv`). Stage A
cannot resolve a -0.03 guard, so any survivor's stage B also reports the one-sided guard bounds.

### Batch 6, stage A results

| arm | CartPole vd (p) | CartPole r | Acrobot r | verdict |
|---|---|---|---|---|
| `var_low_cool` | -0.022 (0.72) | +0.044 | +0.000 | abandon: primary < 0 |
| `var_low_lin` | -0.019 (0.86) | -0.170 | -0.001 | abandon: primary < 0 |

Both lose the hard-suite gain (`var_low` had +0.073 at the same 8 seeds). Ending on DR does
not bank the hard-task skill: whatever the tilt teaches is forgotten once training returns to
typical tasks. `var_low_cool`'s random-suite point estimate recovered (+0.044), in line with
the hypothesis, but the primary went with it. At 8 seeds this is weak evidence either way; the
protocol abandons it. Conclusion: with this learner, the hard-suite/random-suite trade is set
by the task distribution at the end of training, so a schedule cannot escape it.

### Diagnosis after batch 6: where does `var_low`'s budget go? (mass probe, 2 CartPole seeds)

Share of proposals the greedy agent never solves (hopeless) / always solves (trivial), 4 random
starts each, at check-ins 154k / 307k / 461k / 614k:

| proposer | hopeless | trivial | learn p(1-p) |
|---|---|---|---|
| uniform | 0.05 / 0.04 / 0.27 / 0.01 | 0.41 / 0.42 / 0.40 / 0.64 | 0.11 / 0.12 / 0.07 / 0.07 |
| `var_a4` SIR | 0.08 / 0.05 / 0.33 / 0.06 | 0.41 / 0.24 / 0.26 / 0.27 | 0.11 / 0.15 / 0.08 / 0.14 |
| `var_low` SIR | 0.13 / 0.06 / 0.35 / 0.04 | 0.27 / 0.25 / 0.21 / 0.19 | 0.13 / 0.15 / 0.09 / 0.16 |

(461k: one seed was mid-collapse, hence the hopeless spike in every row.) The tilt wastes little
on hopeless tasks (+0-8 points over uniform); a feasibility gate would not help. What it removes
is trivial tasks, and those are the random suite's typical tasks: with this PPO learner,
competence on them depends on continued practice (batch 6: ending on DR restores it and loses
the hard gain). The stage C trade (+0.055 hard for about -0.035 random) looks inherent to
moving mass at a fixed budget, not a flaw of the sampler.

## Amendment 2 (2026-09-30): relative guards; `var_low` passes post hoc

The user accepts a random-suite cost under 20% of the baseline's own pass rate (protocol,
Amendment 2). Re-judged under it, `var_low`'s stage C on seeds 1001-1100 reads **CONFIRMED**
(every guard's lower bound, worst -0.085 on CartPole, is inside -0.20 x DR's rate). Because the
rule came after those numbers, this does not count as a confirmation.

## `var_low` re-confirmation (registered 2026-09-30, before running)

Stage C exactly as before (test halves, Holm, all four envs) under Amendment 2's guards, on
fresh seeds **1101-1200** for both DR and `var_low` (`results/lib/stageC7.sh`;
`decide --stage C --arms var_low --seeds 1101-1200`). If it confirms, Maze and PointNav
held-out runs are registered next.

## Batch 7 (registered 2026-09-30, before running): efficiency, and a second learner

Batch 6 showed the random-suite loss comes from typical (trivial) tasks losing practice. Two
ways around it that do not simply move less mass:
| arm | hypothesis |
|---|---|
| `var_snip8` | (b) `var_low`, but SIR-proposed episodes stop at 1/8 of the env's horizon and bootstrap from the critic (snippets are not reported to the estimators). Frontier practice costs few steps, so uniform tasks keep most of the step budget (probe, Acrobot 20% budget: 552 episodes vs `var_low`'s 367). |
| `var_snip4` | (b) the same at 1/4 of the horizon |
| `var_low_po` vs `DR_po` | (c) a second learner: task-aware PPO (policy and critic see the task parameters). If forgetting is a capacity/aliasing effect of a task-blind policy, the trade shrinks. Judged against `DR_po` (`--base DR_po`). |

Stage A (seeds 1-8, CartPole + Acrobot, Amendment 2 guards), `results/lib/batch7.sh`. Only PPO
exists in the repo; a value-based learner with replay (DQN) is the next step for (c) if the
task-aware variant is inconclusive.

**Batch 7 rerun (operational).** The first pass lost 32 of 64 jobs to a CSV column clash:
`DR_po` rows (no curriculum) have no `lib_counts` column, and whichever config wrote first
locked the file. `DR_po` now writes to `batch7po.csv` (the 8 CartPole `DR_po` runs already
done moved there unchanged); the failed jobs were rerun with `--resume`. No arm, seed or
budget changed.

### Batch 7 stage A verdicts

| arm | n | primary d | p | cart guard d | acro guard d | verdict |
|---|---|---|---|---|---|---|
| var_snip8 | 8 | -0.022 | 0.81 | +0.043 | +0.006 | ABANDON (primary < 0) |
| var_snip4 | 8 | +0.128 | 0.21 | +0.066 | +0.006 | advance to B |
| var_low_po (vs DR_po) | 8 | +0.115 | 0.16 | -0.015 | -0.008 | advance to B |

`var_snip4` is the first arm whose CartPole random-suite point estimate is clearly
positive at stage A, which is what snippets were meant to do (typical tasks keep the budget).
1/8 of the horizon is too short to reach the failures the hard suite tests.

### Batch 7 stage B (registered before running)

`var_snip4`: CartPole seeds 9-72 (against the DR pool), MountainCar and Pendulum seeds 1-8,
output `batch7.csv`. `var_low_po`: the same seeds, with `DR_po` run alongside as its baseline
(`batch7po.csv`), verdict `decide --stage B --base DR_po`. Rules as Amendment 2 (stage B).

### var_low re-confirmation (seeds 1101-1200): FAIL

Primary +0.028 on the test half (p = 0.34). Guards would pass Amendment 2 (lower bounds:
CartPole -0.042, Acrobot -0.048, MountainCar +0.050, Pendulum -0.077). The primary gain does
not replicate on fresh seeds. For description only (not a registered test): pooled over
seeds 1001-1200 the primary is +0.042 (p = 0.038), CartPole random -0.025. The effect, if
real, is about half the stage B estimate (+0.082) and too small to confirm at 100 seeds.

### Batch 7 stage B: both abandoned

| arm | n | primary d | p | cart | acro | mountaincar | pendulum | verdict |
|---|---|---|---|---|---|---|---|---|
| var_snip4 | 72 | +0.006 | 0.84 | +0.005 | +0.006 | +0.033 | -0.108 | ABANDON |
| var_low_po (vs DR_po) | 72 | +0.037 | 0.17 | -0.004 | -0.008 | +0.036 | -0.214 | ABANDON |

Snippets keep the random suite intact but lose the hard-suite gain (the frontier episodes
are too short to learn the failures). Task-aware PPO does not rescue the method: the primary
gain is smaller than with the blind learner and Pendulum loses 0.21. Option (b) and (c) closed
as run; the stage-A lead of +0.128 is another 8-seed lead that did not replicate.

## Amendment 3 (2026-09-30): training inefficiency acceptable

The user: "modify the goal so that training inefficiencies are acceptable--as long as there
is environment agnostic (or model agnostic) boost". Scouting simulation is no longer charged
(docs/protocol.md, Amendment 3); arms compare at equal training steps and report
`scouted_steps`. This readmits SFL (docs/plr.md: CartPole VerifAI +0.105, Holm p = 0.002;
MountainCar random +0.059; Acrobot random -0.053, which is within Amendment 2's 20% of DR's
0.929 = 0.186). SFL is environment-agnostic and model-agnostic (it only rolls the policy out).
Those results were seen before the amendment, so a claim needs fresh seeds 1101-1200.

### Batch 8 (registered before running)

Arms (plain PPO, no library curriculum; own CSV `batch8.csv`, since their rows have no
`lib_counts`):
- `sfl`: literature SFL (1000 random levels x 8 rollouts every 10 updates, top 100 by
  p(1-p), replay 0.5), scouting uncharged.
- `sfl_p25`: the same at replay 0.25, aimed at Acrobot's cost.
Stage A: CartPole + Acrobot seeds 1-8. Stage B for survivors: CartPole 9-72, MountainCar and
Pendulum 1-8. Stage C: seeds 1101-1200 (`decide --stage C --seeds 1101-1200`). Rules as
Amendments 2 and 3. Probe first: 1 seed, 2 checks on CartPole.

### Batch 8 stage A: both advance

| arm | n | primary d | p | cart | acro | verdict |
|---|---|---|---|---|---|---|
| sfl | 8 | +0.181 | 0.018 | +0.006 | -0.007 | advance to B |
| sfl_p25 | 8 | +0.016 | 0.84 | -0.026 | -0.012 | advance to B |

Scouting uncharged (about 150M steps per CartPole run). Stage B as registered: CartPole
seeds 9-72, MountainCar and Pendulum 1-8, `batch8.csv`, script `results/lib/stageB8.sh`.

### Batch 8 stage B: sfl advances, sfl_p25 abandoned

| arm | n | primary d | p | cart | acro | mountaincar | pendulum | verdict |
|---|---|---|---|---|---|---|---|---|
| sfl | 72 | +0.080 | 0.005 | -0.029 | -0.007 | -0.070 | -0.053 | ADVANCE to C |
| sfl_p25 | 72 | +0.029 | 0.35 | -0.011 | -0.012 | +0.097 | +0.019 | ABANDON |

Tolerances (20% of DR's random rate): CartPole 0.179, Acrobot 0.186, MountainCar 0.115,
Pendulum 0.136. `sfl`'s MountainCar -0.070 (8 seeds) contrasts with the earlier +0.059 at 100
seeds (docs/plr.md). Halving the replay share keeps the guards but loses the hard-suite gain.

**Stage C (registered before running):** `sfl` on seeds 1101-1200, all four envs, against the
DR runs already on those seeds; test halves; `decide --stage C --arms sfl --seeds 1101-1200`
(Holm over the one arm). Script `results/lib/stageC8.sh`.

### Batch 8 stage C: sfl FAILS (fresh seeds 1101-1200)

| arm | n | primary d (test) | Holm p | cart | cart lb | mountaincar | mcar lb | pendulum* | pend lb* | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| sfl | 100 | +0.042 | 0.149 | -0.024 | -0.061 | +0.093 | +0.059 | +0.062 | +0.007 | FAIL |

The primary (CartPole test half) was final once CartPole and MountainCar were done, so the
run was stopped early (abandon rule): *Pendulum had 18 of 100 seeds, Acrobot none; they could
only change the guards, not the verdict. Guards that ran pass with room (MountainCar is even
positive). Scouting cost per run: CartPole about 145M steps, MountainCar 31M, Pendulum 384M.
The pattern repeats: SFL's +0.105 (docs/plr.md, seeds 1001-1100) and +0.080 at stage B shrink
to +0.042 on fresh seeds. The gain is probably real but small (about +0.04), below what 100
seeds confirm at p < 0.05. `results/lib/stageC8.sh` can finish the rest with `--resume`.

## Amendment 4 (2026-09-30)

The user: a continuous environment at each stage. Pendulum (continuous reward) joins stage A
(8 seeds, guard with 0.07 slack); stages B and C already had it. No earlier stage is re-run.

### Batch 9 (registered before running)

Scouting is uncharged (Amendment 3), so search a wider pool. The hard suite's corner holds
~0.3% of uniform mass (massprobe), so SFL's 1000 candidates see ~3 of its tasks per scout;
more candidates make the top-100 frontier reach it more often. Nothing env-specific is added.
- `sfl_n4k`: SFL with 4000 candidates x 4 rollouts (2x SFL's scouting).
- `sfl_n8k`: 8000 x 4 (4x).
CSV `batch9.csv` (plain PPO rows). Stage A (Amendment 4): CartPole, Acrobot, Pendulum seeds
1-8; stage B: CartPole 9-72, MountainCar and Pendulum 1-8 (Pendulum's stage A seeds count);
stage C: seeds 1101-1200. Rules as Amendments 2-4. Probe first: 1 seed, 2 checks, CartPole.

### Batch 10: mix and match (registered before running)

The user asked whether we were just bashing our heads against the problem, and they were
right: batches 4-9 tuned the same two ideas. docs/literature.md maps the field. These arms
combine SFL's scouted frontier (the score that works here) with a part borrowed from each
of two other methods. All are environment- and model-agnostic.
- `sfl_mut` (SFL x ACCEL): 50% of each scout's 1000 candidates are Gaussian edits (sigma 0.05
  x range) of the previous frontier, so the frontier can hill-climb into small regions.
- `sfl_tilt` (SFL x var_low): rank by p(1-p)(1-p), leaning the frontier toward hard tasks.
- `sfl_mut_tilt`: both.
Probe (1 CartPole seed, 10% budget): runs, scouting about 5.2M steps, mutation path exercised.
CSV `batch10.csv`; script `results/lib/batch10.sh` (starts when batch 9 stage A finishes).
Stage A: CartPole, Acrobot, Pendulum seeds 1-8; stage B: CartPole 9-72, MountainCar and
Pendulum 1-8; stage C: 1101-1200. Rules as Amendments 2-4.

**Batch 9 stopped (2026-09-30, the user's call):** stopped during stage A on CartPole (the first env) in favour of batch 10's mix-and-match arms. No verdict was reached; it can be resumed with `results/lib/batch9.sh` (`--resume`).

**Batch 10 stage A** (seeds 1-8; DR pool 96 CartPole, 48 elsewhere):

| arm | hard gain | p | CartPole r | Acrobot r | Pendulum r | verdict |
|---|---|---|---|---|---|---|
| sfl_mut | +0.044 | 0.55 | -0.118 | -0.030 | -0.088 | advance |
| sfl_tilt | +0.021 | 0.80 | -0.144 | -0.033 | +0.052 | advance |
| sfl_mut_tilt | +0.191 | 0.06 | -0.068 | -0.107 | -0.151 | advance |

Each part alone adds little, while the combination leads (the largest stage A lead so far;
earlier leads of this size shrank at stage B). All three run stage B (`results/lib/batch10B.sh`);
`sfl_mut_tilt` goes first.

**Batch 10 stage B** (CartPole 9-72 + stage A seeds; MountainCar and Pendulum 1-8):

| arm | n | hard gain | p | CartPole r | Acrobot r | MountainCar r | Pendulum r | verdict |
|---|---|---|---|---|---|---|---|---|
| sfl_mut_tilt | 72 | +0.090 | 0.008 | -0.099 | -0.107 | +0.097 | -0.151 | ABANDON (Pendulum guard, tol 0.136) |
| sfl_mut | 23 | -0.001 | 0.98 | -0.122 | -0.030 | +0.097 | -0.088 | ABANDON early (stopped at 23 seeds) |
| sfl_tilt | 23 (running) | +0.061 | 0.23 | -0.061 | -0.033 | +0.097 | +0.052 | continues (`batch10B2.sh`) |

- `sfl_mut_tilt` held the largest primary gain of any arm at 72 seeds (+0.090, p 0.008, vs
  `sfl` +0.080). It failed the guard on the continuous-reward env: Pendulum's point estimate
  was -0.151 against a limit of -0.136, from 8 seeds. By the registered rule it is abandoned.
  The pattern matches var_low's: the hard tilt costs random-suite competence.
- MountainCar's identical +0.097 is real: SFL-family runs all reach the env's 0.671 ceiling.
- Lead for a next combination (question 3 in docs/literature.md, grounding): keep the
  mut+tilt frontier but protect the random suite, e.g. a CURROT/DRED-style anchor to the
  target distribution. It must be registered as a new arm and run from stage A, not rescued.

**`sfl_tilt` stage B (72 CartPole seeds): ADVANCE.** Hard gain +0.091 (p 0.005); guards
CartPole -0.029, Acrobot -0.033, MountainCar +0.097, Pendulum +0.052 (all inside the tolerances).
This is the largest stage B gain with passing guards so far (`sfl` +0.080, `var_low` +0.082).

### Batch 10 stage C (registered before running)

`sfl_tilt` (SFL ranking scouted levels by p(1-p)(1-p), frozen as registered) on fresh seeds
1101-1200, all four dev envs, against the DR runs already on those seeds. Test half of the
CartPole hard suite, Holm p < 0.05; every guard's one-sided 95% lower bound >= -tol
(Amendment 2). Order: CartPole, MountainCar, Pendulum, Acrobot. The run stops early if the
primary verdict is final. Script `results/lib/stageC10.sh`; verdict
`decide --stage C --arms sfl_tilt --seeds 1101-1200`.

**Batch 10 stage C result (seeds 1101-1200, 100 vs 100): `sfl_tilt` CONFIRMED.** The first
arm to survive stage C.

| metric | gain vs DR | one-sided 95% bound | 
|---|---|---|
| CartPole hard suite, test half (primary) | +0.102 (Holm p < 0.001) | |
| CartPole random suite | -0.037 | -0.077 |
| Acrobot random suite | -0.020 | -0.034 |
| MountainCar random suite | +0.087 | +0.050 |
| Pendulum random suite | -0.015 | -0.042 |

Every guard bound is inside Amendment 2's tolerance (-20% of DR's own pass rate). The run was
interrupted by a reboot at Pendulum 72/100 and resumed with `--resume` (no seed re-drawn).
Cost (Amendment 3): SFL scouting, about 150M simulated steps per CartPole run, not charged.
Next: the held-out no-harm step (Maze, PointNav), registered below.

### Held-out no-harm step for `sfl_tilt` (registered before running)

The protocol leaves the held-out seed count open ("reported as is"). Registered here:
- `sfl_tilt` frozen as confirmed, DR as baseline, seeds 1101-1148 (48 each), full budgets
  (Maze 5.12M, PointNav as in `BUDGET`), 10 checks. PointNav first, then Maze.
- **No-harm test:** final random-suite success, arm minus DR; pass if the one-sided 95% lower
  bound >= -0.20 x DR's own rate (Amendment 2's rule). Both envs must pass.
- **Reported, not tested:** Maze's 76 named held-out mazes (`heldout/success`), and scouted
  steps per run.
- Probe (1 seed, 5% budget): both envs run; scouting about 32M steps (Maze) and 14M
  (PointNav) at 5%, so about 17 min and 7 min per full `sfl_tilt` run.
Script `results/lib/heldout10.sh`; CSVs `results/{maze,pointnav}/lib/heldout10.csv`.
