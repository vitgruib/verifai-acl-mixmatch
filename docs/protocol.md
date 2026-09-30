# Training and testing protocol for curriculum wrappers

Declared 2026-09-28, before any arm in `docs/library.md` is run. It replaces the screening
rules of `docs/wrapper_methodology.md` section 6 for all work from here on. Every arm tried
is logged in `docs/library_log.md`, including the abandoned ones, so the number of
comparisons behind any result is visible.

Why a new protocol: rounds 1-12 chose among ~50 configurations using the same hard-suite
questions they later reported, with 24 seeds against a seed-to-seed spread of 0.18-0.19
(CartPole hard suite). Three +0.09 to +0.145 leads fell to +0.02 to +0.06 at 100 seeds.
Picking the best of many noisy screens on the test itself selects luck.

## 1. What is held out

| level | used for design and tuning | held out until confirmation |
|---|---|---|
| **environments** | CartPole (discovery), Acrobot, MountainCar, Pendulum (no-harm) | Maze (random + 76 named held-out mazes), PointNav (no-harm until it has a hard suite) |
| **questions** | the *dev* half of each hard suite (even positions) | the *test* half (odd positions) |
| **seeds** | replicates 1-99 | replicates 1001-1100 |
| **hyperparameters** | chosen on dev environments only | frozen before any held-out run |

- The split of each hard suite is by position in the frozen, fingerprinted section, so it
  is fixed and identical for every run. Every check-in grades both halves (grading costs
  the same); the screening tools read `*_dev` columns only and refuse `*_test` columns
  unless called with `--confirm`.
- The environment-agnostic claim is tested where it has to hold: on environments nobody
  tuned on. A wrapper may read the learner (critic, gradients, parameters) but may use
  nothing environment-specific beyond the task-parameter box and episode outcomes.

## 2. Metrics

- **Primary:** hard-suite final pass rate (mean of each run's last three check-ins) minus
  DR's, on the dev half.
- **Guard:** random-suite final pass rate minus DR's (the "never worse than DR" check).
- **Secondary:** area under the curve (mean over check-ins), and the continuous scores
  (return for Pendulum, scaled steps elsewhere).
- **Cost:** every simulated step a wrapper takes (scouting, evaluation used to choose
  tasks) counts against its training budget. Wall-clock time is reported but not charged.

## 3. Sequential screening with early abandonment

A shared **DR pool** of 48 seeds per dev environment (replicates 1-48) is the baseline for
every screen, so each arm pays only for its own runs. With CartPole's spread (0.19), the
standard error of a gain is about 0.073 at 8 seeds and 0.047 at 24 seeds.

| stage | runs per arm | abandon if | advance if |
|---|---|---|---|
| **A** | 8 seeds on CartPole + 8 on Acrobot | CartPole primary gain < 0; or any guard < -0.10 | otherwise |
| **B** | +16 CartPole seeds (24 in all); 8 seeds on MountainCar and Pendulum | primary gain <= 0 or p >= 0.10; any guard < -0.03 at 24 seeds, or < -0.05 with p < 0.10 on MountainCar / Pendulum | primary gain > 0 at p < 0.10 and all guards pass |
| **C (confirm)** | 100 fresh seeds (1001-1100) on every environment, test halves | fails any declared test | primary gain on the CartPole test half at Holm-corrected p < 0.05; no guard worse than -0.03 (one-sided 95% bound) anywhere; the frozen configuration on Maze reported as is |

- A true +0.10 gain survives stage A's futility rule with probability ~0.91; a true 0
  gain is abandoned there half the time, which is where the speed comes from.
- **Mechanism checks** (cheap, before any training) come first when an arm depends on a
  signal: the offline signal screen (`acl_bench.plr.signals`) on saved snapshots. An arm
  whose signal fails its offline check is not trained.
- At most three arms enter stage C at once, and each is registered in `docs/library_log.md`
  before it runs.

### Amendment 1 (2026-09-29, after batch 1's stage B; applies to every later stage B)

Batch 1's stage B showed stage B was underpowered: every stage-A lead shrank or flipped on
seeds 9-24, and the gains left (+0.05 to +0.07) are about the size this benchmark offers.
At 24 vs 48 seeds (SE 0.047) a true +0.07 passes p < 0.10 only about 45% of the time. Runs
are cheap (a CartPole run takes about 26 s of one core), so:
- the CartPole DR pool grows to **96 seeds** (replicates 1-96);
- stage B runs **72 CartPole seeds** per arm (SE about 0.030; a true +0.07 passes with
  probability about 0.75 and a true +0.10 about 0.95; a true 0 about 0.05), same rules otherwise;
- stage A is unchanged (a cheap futility filter, nothing more).
Stage C still decides every claim on fresh seeds 1001-1100, so screening more generously
costs only compute, not false claims.

### Amendment 2 (2026-09-30, after `var_low`'s stage C; the user relaxed the guard)

The user accepts a small random-suite cost: **anything under 20%** of the baseline's own
random-suite pass rate. Every guard is now relative, per environment, with
tol = 0.20 x (mean random-suite pass rate of the baseline on that environment):
- stage A: abandon if a guard gain < -(tol + 0.07) (the old -0.10 had 0.07 of slack for 8 seeds);
- stage B: abandon if a guard gain < -tol;
- stage C: the one-sided 95% lower bound of every guard gain must be >= -tol.
The primary rules are unchanged. This rule was adopted after seeing `var_low`'s stage C on
seeds 1001-1100 (which passes it post hoc), so a claim needs a **fresh confirmation on seeds
1101-1200** (`decide --stage C --seeds 1101-1200`). A learner variant (e.g. task-aware PPO)
is judged against its own DR baseline (`--base DR_po`).

## 4. Reporting

Per environment: the pass-rate differences with Welch p-values; at confirmation also
interquartile means with stratified-bootstrap confidence intervals (rliable) and Holm
correction across the declared metrics. Every result row records the code commit.
