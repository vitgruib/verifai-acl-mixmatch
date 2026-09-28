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
