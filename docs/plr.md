# Making PLR work: search log

**Question.** The ablation (docs/ablation.md) found the review pile (SIPACL's PLR) never
helps after Holm correction, on any environment or under any picker. It was run at one
setting. Is there *any* PLR setting, from the literature, that works here?

**Method.** Screen in short, cheap runs; follow up only what shows evidence.

- **Harness** (`acl_bench/plr/fast.py`): the same PPO (network, losses, 1024 samples
  per update, 4 x 4 minibatch epochs, truncation as terminal) but the 1024 samples come
  from 8 environments stepped together through the batched exam physics, so one
  forward pass serves eight. About 3.5x faster than the study pipeline; its DR learns
  like the study's N (Acrobot final random success 0.93 in both). Results here are
  compared only with DR run here.
- **Level sampler** (`acl_bench/plr/levels.py`): every knob from PLR (Jiang et al. 2021:
  rank temperature beta, staleness rho, replaced scores, L1 value loss), Robust PLR /
  PLR-perp (Jiang et al. 2021b: admit-by-score buffer, train only on replays, PVL),
  ACCEL (Parker-Holder et al. 2022: MaxMC, high replay rate) and SFL (Rutherford et al.
  2024: pick the top levels of a large random batch by p(1-p), p from k policy rollouts).
- **Screens** (`acl_bench/plr/screen.py`, `summarize.py`): 8 seeds per config, full
  budget, graded on both suites at 10 check-ins; differences from DR with an
  uncorrected Welch p (screening only). Data: `results/<env>/plr/screen.csv`.

Configs: `DR` = no replay. `sipacl` = this repo's setting (beta 1, no staleness, EMA 0.2,
FIFO 5000). `paper` = the papers' defaults (replay 0.5, beta 0.1, staleness 0.1, scores
replaced, buffer 1000 admitted by score). `paper_*` change one thing from `paper`.
`sfl` = replay 0.5 from the top 100 of 1000 random levels by p(1-p) with k = 8,
refreshed every 10 updates.

## Round 1: Acrobot, one change at a time from the paper defaults

Nothing beats DR. `paper` and `sipacl` tie it. Several settings hurt: PLR-perp (curve
success -0.10, p < 0.001; VerifAI -0.02, p = 0.04), negative-return score (curve -0.09),
replay 0.9 (VerifAI final -0.03), buffer 200, MaxMC.

## Round 2: SFL on Acrobot; PLR and SFL on MountainCar

- Acrobot: SFL ties DR; SFL at replay 0.9 hurts the curve (-0.14, p = 0.007).
- MountainCar: SFL at replay 0.9 or keeping the top 20 learns faster on random questions
  (curve +0.15 / +0.13, p = 0.01 / 0.03). Every config's final score is capped at the
  always-push-right policy (0.671); VerifAI success is 0 everywhere.

## Round 3: headroom, and the MountainCar gain at 24 seeds

- **Oracle** (`oracle50`, diagnostic only): half the episodes are on the VerifAI suite's
  own tasks. VerifAI success does *not* rise (Acrobot -0.016, MountainCar 0). So no
  choice of training tasks can lift the VerifAI suite with this agent and budget, which
  rules PLR out there. Both environments hide the task parameters from the policy, and
  the hard corner (Acrobot: low torque, heavy second link; MountainCar: weak engine,
  strong gravity) needs a strategy PPO does not find here even when practising it.
- **MountainCar, 24 seeds:** SFL at replay 0.9 and SFL top-20 reach push-right in 24/24
  seeds, by about 60k steps, and never lose it; DR oscillates and ends below it in 5/24
  (final +0.14, p = 0.009; curve +0.17, p < 0.001). The PVL settings (`paper`, `sipacl`)
  reach it in 8/8. The study pipeline shows the same, more mildly: N 94/100, A 96/100.

So far: replay makes MountainCar converge faster and more reliably to the one policy
PPO finds; nothing moves any suite beyond what DR's best seeds reach.

## Round 4: CartPole

The harness is faithful here too: DR ends at random 0.88 / VerifAI 0.24 (study N: 0.91 /
0.29), with the same large swings between check-ins. Unlike the other two environments,
**CartPole has headroom**: the oracle lifts VerifAI final success by about +0.2. Every
replay config leans the same way at 8 seeds.

## Round 5: CartPole, the round-1 settings plus 24 seeds for the leaders

At ~23 seeds (VerifAI final vs DR): oracle +0.19 (p = 0.001), paper +0.09 (p = 0.08),
SFL top-20 +0.11 (p = 0.056), sipacl -0.03. At 8 seeds: SFL (replay 0.5, top 100)
+0.21 (p = 0.01); paper with MaxMC +0.08, and random final +0.12 (p = 0.047).
PLR-perp and beta 0.3 hurt the VerifAI curve (-0.10, -0.08; p < 0.05).
So the literature's settings (and SFL) move CartPole toward the oracle; SIPACL's
setting, which the ablation tested, does not.

## Round 6: CartPole, around the leaders

At 24 seeds (VerifAI final vs DR): **SFL +0.11 (p = 0.026)**, random final +0.08; SFL
top-50 (8 seeds) +0.15, random +0.13 (p = 0.015); paper +0.09. MaxMC's lead vanished at
24 seeds (-0.01), as did buffer 200's.

## Confirmation (declared before running)

On CartPole, `sfl` and `paper` at their literature defaults (not the best-looking tuned
variant, to avoid the winner's curse) against `DR`, 100 fresh seeds each (101-200),
fast harness. Metrics: final and curve success on the random and VerifAI suites
(`fin_r`, `auc_r`, `fin_v`, `auc_v`); 2 configs x 4 metrics = 8 Welch tests,
Holm-corrected together.

**Result.** SFL lifts CartPole's VerifAI suite: final success 0.313 -> 0.418 (+0.105,
Holm p = 0.002), curve +0.046 (Holm p = 0.008); random suite unchanged (-0.03, n.s.).
The paper defaults do nothing (+0.02, all Holm p = 1): their screening lead was the
winner's curse. So classic PVL-ranked PLR does not work here at any setting tried, and
its successor SFL, which ranks the replayed levels by learnability p(1 - p), does, on
the one environment where choosing training tasks can matter at all (the oracle).

## Cross-environment check (declared before running)

`sfl` against `DR` on Acrobot and MountainCar, 100 fresh seeds each (101-200), the same
four metrics, Holm over 4 per environment: does the CartPole setting help or harm
elsewhere?

**Result.**
- **MountainCar: helps.** Random final +0.059 (Holm p = 0.005), curve +0.114 (Holm
  p < 0.001); 99/100 seeds reach push-right vs 90/100 for DR. VerifAI stays 0.
- **Acrobot: hurts.** Random final -0.053, curve -0.047, VerifAI final -0.012, curve
  -0.011 (all Holm p < 0.001). The 8-seed screen had shown a tie (-0.004); at 100 seeds
  the cost is clear. On Acrobot even the oracle hurt: time spent on selected levels is
  time taken from the broad distribution, and the frontier there teaches nothing the
  suites reward.

Where SFL's levels go on CartPole (one run, medians as a share of each range; uniform
0.50): late in training it favours wide start ranges (init_range 0.75; VerifAI 0.86)
and slightly weaker pushes (force 0.44; VerifAI 0.09), and stays central on pole and
cart. It reaches toward the VerifAI corner along one axis, consistent with getting
about half the oracle's gain.

## Conclusion so far

- Classic PLR (PVL / L1 / MaxMC scores, Robust PLR, SIPACL's setting, the papers'
  defaults) does not beat domain randomization on any environment here.
- SFL, PLR's learnability-ranked successor, at its default settings: **helps CartPole**
  (VerifAI +0.10) and **MountainCar** (random +0.06, more reliable convergence),
  **hurts Acrobot** (random -0.05). Every figure is Holm-corrected over 100 fresh seeds.
- Where choosing training tasks cannot help (the oracle, on Acrobot and on MountainCar's
  VerifAI suite), no replay method helped.

## Why PVL-ranked PLR fails here

One run each with SIPACL's setting, second half of training, every episode's PVL next
to its outcome:

- **Acrobot:** Spearman(PVL, episode return) = **0.99**; failed episodes score ~0. The
  top 10% of levels by PVL are the *easiest* in the box: short, light links (3rd-13th
  percentile of their ranges) and high torque (92nd), the opposite of the VerifAI corner.
- **CartPole:** PVL is higher on passed than failed episodes (3.0 vs 2.3), rank
  correlation with return 0.50; the top levels lean to narrow start ranges (0.38; the
  VerifAI suite sits at 0.86).

The cause: the policy and critic observe only the physical state, never the task
parameters, so the critic can only predict a return averaged over tasks. An easy task
beats that average (large positive advantages, high PVL); a hard one falls short, and
its negative advantages are clipped away. PVL therefore measures "easier than average",
not learning potential, and PLR spends its replays on levels the agent already handles.
In Procgen, where PLR was developed, the level is visible, the critic learns per-level
values, and a positive surprise does signal something left to learn. This also explains
the screens: more replay (0.9, PLR-perp, a small buffer) was worse, and L1 value loss and
MaxMC, which also compare against the same task-blind critic, did no better. SFL escapes
it because it never consults the critic: p(1 - p) from repeated rollouts peaks at levels
passed about half the time.

## Round 7: surprise in either direction (CartPole, 24 seeds)

Which episodes each score ranks highest (one run, top 10%): on CartPole L1 (|advantage|)
picks only failed episodes (pass rate 0.00; PVL's top: 0.23), so it does aim at hard
scenes; on Acrobot L1 and PVL pick the same episodes, all passed, because reaching the
goal early is the big surprise and a 500-step failure is predictable.

VerifAI final vs DR at 24 seeds: L1 +0.067 (p = 0.22); L1 averaged over visits (EMA 0.3)
+0.073 (p = 0.19), random final +0.099 (p = 0.054); negative surprise only +0.031
(p = 0.59); for reference PVL +0.085, SFL +0.110 (p = 0.026). Counting surprise in both
directions does not help. Averaging over visits helps a little, consistent with
single-episode noise (a failure often reflects the start state, which a replay redraws).
PVL's +0.09 at 24 seeds fell to +0.02 at 100, so none of these leans is evidence yet.

## Round 8: fixing PVL-ranked PLR (CartPole)

Fixes for the task-blind critic and single-episode noise, each network change against a DR
baseline with the same network: a privileged critic that sees the task parameters (`_pc`,
the policy stays blind), task parameters visible to both networks (`_po`), exact-start
replay (`_st`: a replay repeats the episode's starting state, as SIPACL's own replays of a
pickled scene do), and a learning-progress score against the level's own last return
(`lp`). Mechanism check (one run each): with a privileged critic, Acrobot's
Spearman(PVL, return) falls from 0.99 to 0.38 and the top levels move from the easy
corner to the middle of the box; CartPole's does not change (0.50 -> 0.61). PVL keeps
favouring levels where the agent is improving, i.e. starting to pass.

At 8 seeds: privileged critic + exact start led (VerifAI +0.14 over its baseline). Task
parameters visible to the policy made learning slower (DR_po VerifAI curve -0.09 vs DR,
p = 0.005). lp: no VerifAI gain.

## Round 9: the leaders at 24 seeds

- Privileged-critic configs: none beats DR_pc on VerifAI (paper_pc_st: +0.14 at 8 seeds,
  -0.04 at 24).
- **`sipacl_st`** (SIPACL's settings + exact-start replay): random final +0.127
  (0.825 -> 0.952, p = 0.015), random curve +0.100 (p = 0.006); continuous (steps
  survived) +0.055 (p = 0.025); VerifAI +0.045 (n.s.). Restoring SIPACL's own replay of
  the exact scene may be what its PVL-ranked PLR needs.

`summarize` now reports a continuous score next to each pass rate (`--continuous`): mean
return where the environment has one, otherwise mean steps scaled to [0, 1], higher
better.

## Confirmation 2 (declared before running)

`sipacl_st` against `DR` on CartPole, 100 fresh seeds (101-200; DR's are the ones from
confirmation 1). Metrics: `fin_r`, `auc_r`, `fin_v`, `auc_v`, and the continuous finals
`fin_rc`, `fin_vc`; Holm over these 6.

## Caveat: SFL's extra simulation

SFL's buffer is filled by scouting rollouts outside training (every 10 updates, 1,000
random levels x 8 rollouts), and they are not charged to the training budget; on
CartPole that is up to ~4M simulated steps per refresh against ~10k training steps
between refreshes. The SFL paper does the same, but it means "SFL beats DR" here is not
an equal-budget comparison: it shows that knowing each level's pass rate is valuable,
not that SFL wins at the same cost. Every other config, including all PVL variants, is
equal-budget. To do: an SFL variant whose scouting counts against the budget.

## Future work

- **LunarLander** (gymnasium Box2D) as a second continuous-reward environment with a
  natural pass/fail (a safe landing) on top of a shaped return, and task parameters
  gymnasium already exposes (gravity, wind power, turbulence). Its physics cannot use
  the batched grader, so its exam would be graded one episode at a time (much slower);
  worth it if Pendulum's results need a second continuous-reward check.

**Result (confirmation 2): not confirmed.** `sipacl_st` vs DR, 100 seeds, Holm over 6:
random final +0.030 (Holm p = 0.64), random curve +0.006 (0.70), VerifAI final +0.061
(raw p = 0.029, Holm 0.18), VerifAI curve +0.021 (0.58), continuous finals +0.014 (0.64)
and +0.049 (0.26). The screen's random-suite gain (+0.127 at 24 seeds) was mostly the
winner's curse. A VerifAI lean of about +0.06 remains, about half of SFL's confirmed
+0.10; it would take several hundred seeds per arm to settle.

Pendulum budget: 2,457,600 steps (2,400 rollouts), from 8 DR runs to 4.9M
(`results/pendulum/plr/calibration_long.csv`): fast learning to ~2M (pass 0.67,
return -415), then a slow creep to 0.85 / -257.

## Pendulum (continuous reward), rounds 1-2, 8 seeds

Final return vs DR (-418): paper + exact start -205 (p = 0.003), paper -148 (p = 0.04),
paper at replay 0.2 -142 (p = 0.03), sipacl -107, sipacl + exact start -126; L1 +18,
learning progress +1, visit-averaged L1 -55, SFL -47 (all n.s.). A privileged critic
helps the baseline (DR_pc -344, +74 vs DR, n.s.) but not PLR on top of it: vs DR_pc,
paper_pc -171 (p = 0.007), paper_pc_st -138 (p = 0.04), sipacl_pc -69 (n.s.).

Mechanism (one run, sipacl, second half): Spearman(PVL, return) = +0.70; PVL's top 10%
pass 77% of the time (average 41%), return -236, on light, low-gravity, strong-torque
levels. L1's top 10% are the opposite: Spearman -0.86, pass 1%, return -1361, on the
heaviest, highest-gravity, weakest-torque levels. PVL replays what the agent already
handles; L1 on a continuous cost chases hopeless levels; neither finds the frontier.
Continuous reward does not rescue PVL, and the loss is larger here because the hard
levels dominate the average return.

## Where PVL-ranked PLR stands

Tried: SIPACL's setting; the PLR / Robust PLR / ACCEL settings; L1, negative-only, MaxMC
and learning-progress scores; visit-averaged scores; a privileged critic; task-aware
networks; exact-scene replay; lower and higher replay rates. On four environments, pass/
fail and continuous: no confirmed gain; the best lead (exact-scene replay, CartPole
VerifAI +0.06) failed confirmation; on Pendulum it hurts. The one method that confirmed,
SFL, drops the value-based score for a measured pass rate (and uses extra simulation).

## Round 10: from the literature, aimed at PVL's easiness bias (8 seeds)

New: PVL gated to the frontier (`pvl_learn`: PVL x 4p(1-p), p = the level's running pass
rate from training episodes only, no extra simulation; `sipacl_learn` with SIPACL's
settings); PVL minus its running linear prediction from the return (`pvl_resid`); the PLR
paper's policy-entropy score (`entropy`); value disagreement across 5 extra critics
(`vds`, Zhang et al. 2020); ACCEL level edits at replay 0.8 with PVL, MaxMC or gated PVL
(`accel*`); staleness 0.3 (`paper_rho03`).

- CartPole (VerifAI final vs DR 0.304): **sipacl_learn +0.16 (p = 0.053)**, the largest
  PVL-form lead so far; pvl_resid +0.09, entropy +0.09, rho 0.3 +0.09, accel +0.05;
  pvl_learn -0.05, vds -0.08. Random final: most +0.07 to +0.11.
- Pendulum (final return vs DR -418): every variant hurts. ACCEL is worst (-537 to -808;
  pass rate 0.68 -> 0.11-0.25): its edits drift into the heavy, weak-torque corner.
  PVL forms -100 to -200; entropy -71 and vds -96 (n.s.) least bad.

Caveat: Pendulum's only exam so far is its random suite, the very distribution DR trains
on, so any shift away from uniform costs there; PLR's claimed benefit is on hard levels,
which CartPole's VerifAI suite measures and Pendulum does not yet. Next: Pendulum's
reference agents (study pipeline) and VerifAI suite; CartPole leaders to 24 seeds.

CartPole leaders at 24 seeds (VerifAI final vs DR 0.304): **sipacl_learn +0.145 (0.449,
p = 0.005)**, random final +0.092 (p = 0.09); pvl_resid +0.100 (p = 0.11); entropy +0.092
(p = 0.12); accel +0.072 (p = 0.22). For reference at 24 seeds: SFL +0.110, paper +0.085.

## Confirmation 3 (declared before running)

`sipacl_learn` (SIPACL's PLR, PVL gated by 4p(1-p) of the level's running pass rate from
training episodes) against `DR` on CartPole, 100 fresh seeds (101-200; DR's from
confirmation 1). Metrics: `fin_r`, `auc_r`, `fin_v`, `auc_v`, `fin_rc`, `fin_vc`; Holm
over these 6.

**Result (confirmation 3): not confirmed.** `sipacl_learn` vs DR, 100 seeds, Holm over 6:
VerifAI final +0.033 (Holm p = 0.72), VerifAI curve +0.024 (0.52), random final +0.023
(0.72), random curve -0.001, continuous finals +0.017 / +0.034 (n.s.). The 24-seed lead
(+0.145) was the winner's curse again.

Across the three PVL-form confirmations on CartPole, VerifAI final lands at +0.020,
+0.061 and +0.033: always positive, never significant. A small real effect (~+0.04, a
third of SFL's confirmed +0.105) is consistent with these; 24-seed screens cannot
separate it from noise, and picking each round's best screen selects the luckiest.

## Round 11: planner regret (8 seeds)

PLR's own justification is minimax regret; PVL, L1 and MaxMC approximate regret with the
task-blind critic. Here the exam's physics planner (beam search, acl_bench.exam.certify)
estimates the best achievable result on each (level, start); score = max(planner, best
seen, this outcome) - this outcome (return on Pendulum, pass indicator otherwise), with
exact-start replay.

- CartPole: VerifAI final +0.119 (paper settings) and +0.100 (SIPACL settings), n.s.: the
  same range as the PVL leads that faded at 100 seeds.
- Pendulum: paper settings -269 (p = 0.001; pass rate 0.68 -> 0.33): its sharp ranking
  fixes on levels the planner solves and the agent cannot yet approach. SIPACL settings
  -46 (n.s.), curve -1: the first PLR variant that does not hurt Pendulum's random suite.
- Acrobot: hurts (curve -0.07 / -0.08), as the oracle predicts.
