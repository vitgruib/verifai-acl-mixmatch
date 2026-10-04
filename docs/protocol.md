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

### Amendment 3 (2026-09-30; the user accepts training inefficiency)

The user: "training inefficiencies are acceptable--as long as there is environment agnostic
(or model agnostic) boost". Simulation spent outside the training episodes (scouting rollouts,
evaluations of candidate tasks) is **no longer charged** against the training budget: arms and
the baseline compare at an equal number of *training* steps. Every such arm reports its extra
steps (e.g. `scouted_steps`) as a cost. The method must still be environment-agnostic or
model-agnostic, and all other rules (stages, Amendment 2 guards, fresh seeds 1101-1200 for
claims adopted after seeing 1001-1100) stand. Old results for uncharged-scouting arms (SFL,
docs/plr.md) were seen before this rule, so a claim needs fresh seeds (1101-1200).

### Amendment 4 (2026-09-30; the user: a continuous environment at each stage)

Every action space here is discrete, so "continuous" means continuous reward: Pendulum (the
dev env scored by its return) and PointNav (held out). Stages B and C already include
Pendulum; **stage A now runs 8 Pendulum seeds too** (guard as in Amendment 2, with the 0.07
slack), so no arm advances without a continuous-reward check. `decide --stage A` reads
CartPole, Acrobot and Pendulum. Earlier stage A verdicts stand (no stage is re-run).

### Amendment 5 (proposed 2026-10-01; ADOPTED 2026-10-02 with the user's approval)

The user asked to reconsider the testing suite. Measured on the DR pool (seeds 1-99):

| env | DR hard dev (`fin_vd`) | DR random (`fin_r`) | resolves a gain? |
|---|---|---|---|
| CartPole | 0.333 +- 0.189 (n=96) | 0.893 +- 0.153 | yes |
| Acrobot | 0.083 +- 0.018 (n=48) | 0.929 | barely: near the floor, tiny spread |
| MountainCar | 0.000 +- 0.000 (n=48) | 0.574 +- 0.196 | no: hard suite floored |
| Pendulum | 0.165 +- 0.086 (n=48) | 0.679 +- 0.112 | yes |

MountainCar's random suite has a ceiling: of 784 MountainCar runs, 689 score exactly 0.671
(200 of 298 tasks), none exceeds 0.677, and the rest are collapses (0, 0.224, 0.447). Its
guard measures whether training collapsed, not the curriculum's effect. Its hard suite tops
out at 0.032 over all 784 runs. Seed pairing does
not help: arm and DR results on the same seed are uncorrelated (CartPole r ~ 0 to 0.18).

Consequences: the only primary that can move is CartPole's, so every arm is selected for
CartPole. "CartPole-specific" findings are partly a measurement artifact: the other hard
suites (built from questions 60% of the *reference* agents fail) cannot show a gain.

Proposed (each needs the user's approval because it changes a gate):
1. **Recalibrated hard suites.** Rebuild each dev env's hard suite from questions that
   independent DR agents (seeds outside every pool, e.g. 5001-5010) pass 10-60% of the time,
   from the same VerifAI search pools, winnable-certified as now. DR then sits mid-range with
   real seed-to-seed spread on every env. The old suites stay and are still reported.
2. **Multi-env primary.** The stage A/B primary becomes the mean over dev envs of the gain
   divided by DR's sd, over the suites that resolve (`python -m acl_bench.plr.xenv` computes
   it now, as a report). With today's suites it is dominated by Acrobot's 0.018 sd, so it
   needs item 1 first.
3. **Adversarial evaluation (SFL's CVaR, 2408.15099).** For each final agent, its success on
   the worst 10% of a large fixed level sample, as a robustness metric next to the suites.

**Adopted rules (2026-10-02).** They apply to every stage run from batch 14 on. Earlier
verdicts stand.
- **Calibration agents:** DR on seeds 5001-5010, the final check-in (`results/lib/calib5.sh`,
  snapshots in `results/<env>/snapshots/DR/`). These seeds are in no arm's pool.
- **`calib` suite** (`python -m acl_bench.exam.calib`, seed 20261002): every question in the
  45k-question `SEARCH_{ce,mab,sa}` pool that 1-6 of the 10 calibration agents pass (band
  [0.1, 0.6]). At least one agent passes each question, and grading is deterministic, so
  every question is proven winnable. The suite is capped at 2000 by a fixed-seed sample and
  split even/odd into `calib_dev` and `calib_test`, like the VerifAI suite.
- **`adv` suite (report only):** 1000 uniform levels x 4 starts, kept when a calibration agent
  passes every start, first 500 levels. `adv/cvar` is the mean per-level success over the
  agent's worst 10% of levels (`fin_cvar`, last-3 mean).
- **Primary:** Z = mean over resolving envs of (arm mean - DR mean) / DR sd on `fin_cd`
  (stage C: `fin_ct`). An env resolves when DR's mean is in [0.05, 0.95] and DR's sd is
  >= 0.01. The p value is 2 * norm.sf(|Z| / SE), with SE = sqrt(sum over envs of
  (var_a/n_a + var_DR/n_DR) / sd_DR^2) / (number of envs). Stage gates use Z and its p
  exactly as before: A needs Z >= 0, B needs Z > 0 with p < 0.10, and C needs Z > 0 with
  Holm p < 0.05.
- **Environments:** every stage reads all 4 dev envs. Stage A uses 8 seeds per env. Stage B
  uses 72 CartPole seeds plus 8 per other env, as before, against a new 48-seed DR pool per
  env. Guards (Amendment 2) are unchanged.
- **Storage:** A5 runs go in `results/<env>/a5/*.csv`. The old `lib/` CSVs lack the calib
  columns. `python -m acl_bench.plr.decide --a5 --stage A|B|C ...` applies these rules.
- **Caveat (MountainCar):** its calibration agents are bimodal (pass-count histogram
  [9463, 23283, 0, ..., 0, 10198, 2056]). One agent reached a stronger mode, so the
  MountainCar calib suite is mostly "questions only the strong mode solves". If DR does not
  resolve on it, the env drops out of Z automatically.

Built suites: calib = 2000 questions on every env, with calibration mean pass rates of
0.345 (CartPole), 0.191 (Acrobot) and 0.100 (MountainCar). adv = 500 levels on every env;
the winnable levels among the 1000 drawn were 807, 965, 910 and 738 (CartPole, Acrobot,
MountainCar, Pendulum).

## 4. Reporting

Per environment: the pass-rate differences with Welch p-values; at confirmation also
interquartile means with stratified-bootstrap confidence intervals (rliable) and Holm
correction across the declared metrics. Every result row records the code commit.
