# Handoff (2026-09-30): state of the curriculum-library work

Start here in a new session. Details live in the docs linked below; this page is the map.

## NEW GOAL (user, 2026-10-03) — supersedes the goal below

"Do a documentation of various UED algorithms and show how they fail with VerifAI failure
cases, and why they fail — using the exact environments they tested on — then show how we can
succeed by combining falsifiers and existing PLR techniques." Full statement and success
criterion: `.claude/commands/goal.md` (`/goal`). Start on the grid mazes (`acl_bench/envs/maze.py`,
DCD held-out mazes), then CarRacing F1 tracks, BipedalWalker, JaxNav / XLand-MiniGrid.
Everything below documents the previous goal (batches 1-18); stage C 14 is being finished
for the record, batch 18 cancelled.

**Status 2026-10-03:** `docs/ued_atlas.md` holds the algorithm registry (with paper-exact settings
taken from the DCD configs), how our harness deviates, the 5-stage test pipeline, the compute
table and the failure table (without and with VerifAI). **Waiting on a user decision:** train
with the reference code (JaxUED / SFL / DCD on the professor's cluster; recommended), or extend our
torch harness (LSTM, PAIRED, minimax). Either way, next is a structured maze Scenic spec (wall
bitmap + start/goal + BFS validity) because the current `layout_seed` parameter reduces
falsifiers to random search. Stage C 14: CartPole done (295 runs); Acrobot onward was still
running in `queue16.sh`; the verdict is still to be recorded.

**Batch 19 done 2026-10-04:** maze replay-vs-DR does not reproduce at 10% budget (DR 0.30/0.35 >= RPLR, PLR, ACCEL); PAIRED dropped as too costly locally, deferred to cluster.

**Stage C 14 done 2026-10-04:** `sfl_tilt_carry` (Z +0.058, Holm p 0.51) and `sfl_spread_carry` (Z +0.115, Holm p 0.37) both FAIL on seeds 1101-1200; old library goal closed with no confirmed arm (details in `library_log.md`). The machine is now free for batch 19.

**Status 2026-10-04:** decided (user): train with the papers' own code, scoped budgets. Env x method
matrix and measured reproduction feasibility are in `docs/ued_atlas.md` section 2b (JaxUED maze runs
locally; DCD Bipedal needs py3.11 patches and the cluster for real budgets; DCD CarRacing is
cluster-only since it opens OpenGL windows on macOS, never run it on the Mac). Batch 19 (JaxUED maze,
DR/PLR/RPLR/ACCEL/PAIRED, 3000 updates = 10% budget, seeds 0-1; `results/lib/b19_maze.sh`, code and
outputs in the session scratchpad `jaxued/examples/{checkpoints,results,logs}`) is running. Seed-0 mean
solve on the 8 named mazes: DR 0.30, PLR 0.30, RPLR 0.30, ACCEL 0.05; RPLR seed 1 0.19. Machine is
oversubscribed (10 cores, load ~45) and stage C 14 pauses on low battery.

## The goal (user, 2026-09-28)

A **generalized library that mixes and matches VerifAI, PLR and other techniques** into a
wrapper that boosts any RL agent's performance for ~free.
- May be model-specific (read the critic, gradients, parameters); must be
  **environment-agnostic** (nothing names an environment; no per-env buckets or fix-ups).
- Working style asked for: a very clean training/testing procedure first; test many
  algorithms; **abandon early** when preliminary tests look hopeless; be creative, including
  mathematical modifications of existing techniques.
- Inspiration: github.com/Kv139/ACL-experiments, branch `joelle` (MetaDrive PVL buffer +
  Scenic mutation/crossover). Its logs are not comparable across arms (each logs its own
  training distribution, one seed).
- Earlier questions answered in-session: why PVL-scored PLR works in the literature but not
  here (`docs/plr.md`, "Why PVL-ranked PLR fails here": a task-blind critic makes PVL track
  easiness; it works with a level-visible critic and sparse terminal rewards); the
  standardized testing suite (`docs/wrapper_methodology.md`, `docs/protocol.md`).

## Where everything is

| what | where |
|---|---|
| testing procedure (held-out envs, dev/test halves, seed ranges, staged screening, Amendment 1) | `docs/protocol.md` |
| library design and maths (SIR density, tempered importance correction, sqrt(p(1-p)), posterior replay) | `docs/library.md` |
| every arm tried, registered before running, with results and verdicts | `docs/library_log.md` |
| signal screen (pass model `model_1ep` is the best offline signal), PVL analysis, oracle headroom | `docs/plr.md` |
| earlier wrapper work, calibrations (Maze 5.12M, PointNav 1.97M) | `docs/wrapper_log.md`, `docs/wrapper_methodology.md` |
| library code | `acl_bench/curriculum/` (`core.py`, `estimators.py`, `proposers.py`, `__init__.py` ARMS + `build`) |
| harness integration | `acl_bench/plr/fast.py` (`LevelConfig.lib`), `screen.py` (library arms registered, dev/test split, commit column) |
| stage verdicts | `python -m acl_bench.plr.decide --stage A|B --arms ...` (`--confirm` for stage C) |
| tests | `tests/test_curriculum.py` |
| run scripts and logs | `results/lib/*.sh`, `results/lib/*.log`; results `results/<env>/lib/*.csv` (DR pool in `runs.csv`) |

## Protocol in one paragraph

Dev envs CartPole (discovery; primary = hard VerifAI suite, dev half, `fin_vd`), Acrobot,
MountainCar, Pendulum (no-harm guards on the random suite, `fin_r`); Maze and PointNav are
held out. Seeds 1-99 screen, 1001-1100 confirm. Shared DR pool: 96 CartPole seeds, 48
elsewhere. Stage A: 8 seeds CartPole + Acrobot, abandon if primary < 0 or a guard < -0.10.
Stage B (Amendment 1): 72 CartPole seeds + 8 on MountainCar/Pendulum, advance only at
primary > 0 with p < 0.10 and guards passing. Stage C: 100 fresh seeds, test halves, Holm.
A CartPole run takes ~26 s of one core; 6 workers.

## Results so far

21 arms tried. **`var_low` (batch 5) passed stage B and failed stage C on guards**; batch 6 (cool-down variants) died at stage A.
- Batch 1 (`sir_a1`, `sir_a4`, `sir_is`, `sir_is05`, `replay_post`, `mutate_post`,
  `verifai_surr`, reference `fmodel`): five died at stage A; `fmodel`, `sir_a4`, `sir_is`
  died at stage B; re-screen at 72 seeds: `sir_a4` -0.002, `sir_is` +0.039 (p = 0.22).
- Batch 2 (`grad_is`, `align`: per-episode gradient size / alignment): abandoned before
  training; the signal is not predictable from the task (cross-validated R^2 below noise).
- Batch 3 (`sir_is_a1`, `sir_is_u50`, `rw_a4`, `ens_a4`, `prog_a4`): two died at stage A
  (`rw_a4` hurt Acrobot -0.154); the rest at stage B, all within +/-0.02 at 72 seeds.
- Batch 4 (`VarModel`: predicted std of the standardized return, `var_a2/a4/a8/is`): better
  offline signal, but `var_a4` +0.008 and `var_is` -0.000 at 72 seeds. Closed.
- Mass probe (`acl_bench/curriculum/massprobe.py`): the hard suite sits in a corner holding
  0.3% of uniform mass. VarModel SIR doubles the true learnability of proposals but does not
  move mass there; tilting toward low predicted return (`std*exp(-mean_z)`) moves ~5x more
  mass there and raises learnability further, still only ~1-2% of proposals.
- Batch 5 (`VarModel(tilt=1)`: score = std * exp(-predicted standardized return)):
  `var_low` +0.082 (p = 0.009) at 72 seeds, guards pass -> stage C (`results/lib/stageC5.sh`,
  seeds 1001-1100 on all four dev envs, DR and the arm, ~1.5 h; verdict
  `python -m acl_bench.plr.decide --stage C --arms var_low`). `var_low16` failed the Acrobot
  guard. **Stage C (100 vs 100, seeds 1001-1100): FAIL.** Primary confirmed (+0.055 on the
  test half, p = 0.044), but random-suite guards: CartPole -0.044 (bound -0.085), Acrobot
  -0.031 (-0.044), Pendulum -0.029 (-0.058); MountainCar +0.055 passes. A trade, not free.
- Batch 6 (`var_low_cool`: tilt, then SIR share -> 0 between 60% and 80% of the budget;
  `var_low_lin`: SIR share 0.5 -> 0 linearly): does ending on DR pay back the guard cost?
  Stage A: both lose the primary (-0.022, -0.019); the gain is forgotten once training ends on DR.
- Lessons: 8-seed stage A leads (up to +0.14) never replicated; CartPole has headroom
  (oracle +0.19, scouted SFL +0.09); a general frontier sampler finds *a* frontier, not the
  hard suite's corner. Diagnose mass placement with `massprobe` before training.
- Ops lesson: `pgrep -f name.sh` also matches any waiting shell whose command contains the
  name (caused a deadlock); use `pgrep -f "name[.]sh"`.

## Status (2026-09-30, end of batch 7)

- **User decision (2026-09-30):** a random-suite cost below 20% of DR's own pass rate is
  acceptable (Amendment 2 in `docs/protocol.md`, implemented in `decide.py`).
- `var_low` re-confirmation on fresh seeds 1101-1200: **FAIL**, primary +0.028 (p = 0.34);
  guards would pass. Pooled 1001-1200 (descriptive only): +0.042, p = 0.038. The real effect
  is probably ~+0.04, too small to confirm at 100 seeds.
- Batch 7 stage B: `var_snip4` (snippets) +0.006 and `var_low_po` (task-aware PPO) +0.037
  (Pendulum -0.21): both abandoned. Options (b) and (c) closed as run.
- `DR_po` rows have no `lib_counts` column: keep them in separate CSVs (`batch7po.csv`).
- Standing pattern: 8-seed stage-A leads (+0.11 to +0.14) shrink to +0.00-0.04 at 72+ seeds.

## Status (2026-09-30, Amendment 3 / batch 8)

- The user relaxed the goal again: training inefficiency (extra simulation, compute) is fine as
  long as the boost is environment-agnostic or model-agnostic. Protocol Amendment 3: scouting
  is not charged (equal training steps) but `scouted_steps` is reported.
- This readmits SFL (about 150M scouting steps per CartPole run, about 4 min wall-clock).
  Batch 8 (`sfl`, `sfl_p25`, CSV `batch8.csv`): `sfl_p25` abandoned at stage B; `sfl`
  passed stage B (+0.080, p 0.005) but FAILED stage C on seeds 1101-1200 (CartPole test
  +0.042, Holm p 0.149; guards fine). Stage C was stopped once the primary was final
  (Pendulum 18/100 seeds, Acrobot 0). No arm has yet survived stage C under Amendments 2-3.
- `massprobe.py` gained a Metropolis sampler diagnostic (MCMC); a 1-seed smoke test showed
  little extra mass near the hard suite, which is inconclusive and was not pursued.

## Status (2026-10-01, Amendment 4 / batches 9-10)

- Amendment 4: Pendulum (continuous reward) joins stage A; `decide --stage A` reads it.
- Batch 9 (`sfl_n4k`, `sfl_n8k`: wider scouting) was stopped in favour of batch 10.
- The user pushed back on incremental knob tweaks. `docs/literature.md` maps the field
  (proposer / score / grounding) and records what has been tried. Batch 10 combines SFL with
  ACCEL edits (`sfl_mut`), var_low's hard tilt (`sfl_tilt`) and both (`sfl_mut_tilt`).
- Batch 10 stage B: `sfl_mut_tilt` +0.090 (p 0.008) but fails the Pendulum guard (-0.151);
  `sfl_mut` abandoned early (about 0); **`sfl_tilt` advances: +0.091 (p 0.005), all guards pass.**
- **`sfl_tilt` stage C on fresh seeds 1101-1200: CONFIRMED** (first arm ever): CartPole hard
  test half +0.102 (Holm p < 0.001); guard bounds CartPole -0.077, Acrobot -0.034,
  MountainCar +0.050, Pendulum -0.042, all inside -20% of DR.
- **Held-out no-harm: PASS** (`python -m acl_bench.plr.heldout --arm sfl_tilt`, 48 vs 48
  seeds 1101-1148): PointNav random +0.007 (bound +0.004, tol -0.196); Maze random -0.022
  (bound -0.033, tol -0.132), named held-out mazes +0.036 (p < 0.001). **`sfl_tilt` meets
  the success criterion.** Next: port it into `acl_bench/curriculum/` as a proposer.
- **Thorough testing of `sfl_tilt` (2026-10-01):**
  - Unit tests: `tests/test_sfl.py`, 7 tests.
  - Reproducibility: stored rows reproduce bit-exactly at HEAD.
  - Robust statistics: the IQM and Mann-Whitney results agree.
  - Control: the tilt beats plain `sfl` by +0.061 (p 0.03).
  - Mechanism: the buffer stays on the frontier and drifts toward the hard corner.
  - Robustness (batch 11): task-aware PPO +0.167 (p < 0.0001), lr 1e-3 +0.049 (n.s.),
    half budget +0.029 (n.s.); guards pass in all.
  - Across dev envs the hard-suite boost is CartPole-specific: Acrobot is slightly worse,
    Pendulum is flat, and MountainCar gains only on its random suite.
  - So: a real, mechanism-backed but modest effect, not a universal boost.
- **Batch 12 (2026-10-01):** `sfl_tilt_soft` (NCC score-proportional replay) and
  `sfl_tilt_carry` (persistent frontier). Both passed stage A (CartPole hard +0.183 p 0.021
  and +0.105). **Stage B running:** `results/lib/stageB12.sh`, log `results/lib/stageB12.log`;
  then `decide --stage B --arms sfl_tilt_soft sfl_tilt_carry`. The user dropped the "must
  beat `sfl_tilt`" rule: record every arm that passes.
- **Batch 13 (stage A running: `results/lib/batch13.sh`, log `results/lib/batch13.log`; verdict with `decide --stage A --arms sfl_tilt_verify sfl_tilt_amort`; from the "innovation moves" table in `docs/literature.md`):**
  `sfl_tilt_verify` (re-roll a 200-level shortlist 24x, against the winner's curse) and
  `sfl_tilt_amort` (k-NN on the last 3 scouts pre-screens 10k draws). Code and tests are in;
  see `docs/library_log.md` for status.
- **Testing suite:** hard suites floor outside CartPole (MountainCar DR 0.000, Acrobot
  0.083 +- 0.018), so only CartPole's primary can move. Proposed Amendment 5 in
  `docs/protocol.md` (recalibrated suites, multi-env primary, CVaR evaluation) **awaits the
  user's approval**. `python -m acl_bench.plr.xenv` prints the cross-env standardized report.

## Status (2026-10-02, Amendment 5 ADOPTED / batch 14)

- **Amendment 5 adopted** with the user's approval (`docs/protocol.md`). The `calib` suites
  were rebuilt from DR seeds 5001-5010 (`acl_bench/exam/calib.py`, `results/lib/calib5.sh`)
  and the `adv` CVaR suites are report-only. The primary is a multi-env Z on `fin_cd`,
  every stage uses all 4 dev envs, and results go in `results/<env>/a5/`.
- Batch 12 stage B under the old rules: soft +0.142 and carry +0.126 (both p < 0.001),
  ADVANCE. Batch 13 stage A: verify +0.092 and amort +0.219, both advance. All four are
  re-screened under A5 in batch 14 rather than going on to old-rule stage C.
- **Batch 14 (running, pid 45034, `results/lib/a5_pool.sh`, log `results/lib/a5_pool.log`):**
  - First the A5 DR pool (seeds 1-48 x 4 envs, `a5/dr.csv`).
  - Then stage A (seeds 1-8 x 4 envs, `a5/b14.csv`) of the five new "bold" arms
    (`sfl_halving`: successive-halving scout; `sfl_ghost`: learning-progress bonus against
    the previous policy; `sfl_spread`: farthest-point diverse buffer; `sfl_bisect`: bisect
    failed/solved pairs onto the frontier; `sfl_tilt_sc`: soft + carry), plus `sfl_tilt`,
    soft, carry, verify and amort.
  - Code: `acl_bench/plr/fast.py` (`_halving`, `_bisect`, `_farthest`). Tests:
    `tests/test_sfl.py`; the full suite passes (166).
  - Check: `pgrep -f "a5_pool[.]sh"`. Verdict: `python -m acl_bench.plr.decide --a5 --stage A
    --arms sfl_halving sfl_ghost sfl_spread sfl_bisect sfl_tilt_sc sfl_tilt sfl_tilt_soft
    sfl_tilt_carry sfl_tilt_verify sfl_tilt_amort`.
- 1-seed CartPole scratch probe (not counted; DR seed 1 calib_dev 0.029): sc 0.844, spread
  0.550, ghost 0.540, halving 0.391, bisect 0.239. Bisect also collapsed random (0.546) and
  CVaR (0.060), so it is likely to die at stage A.
- **Batch 14 stage A verdicts** (`docs/library_log.md`): advance halving, spread, soft,
  carry, verify; abandon ghost, bisect, sc, `sfl_tilt` (Z -0.137) and amort. Every arm gains on
  CartPole, but hard-focused selection costs Acrobot (z -0.6 to -1.3). Spread, carry and verify
  spare it.
- **Stage B 14 (done):** only `sfl_tilt_carry` advances (Z +0.333, p 0.040, positive on all 3
  resolving envs). Halving (+0.273, p 0.20), verify, spread and tilt_soft are abandoned.
- **Batch 15 (done):** all 5 pass stage A. `sfl_spread_carry` is best (Z +0.450, p 0.048).
  `sfl` +0.274 and `sfl_spread_verify` +0.177 advance. `sfl_auto` and `sfl_spread0` are dropped (Z ~0.1 or less).
  The tilt causes the Acrobot cost; untilted `sfl` loses Pendulum instead.
- **Running: `results/lib/queue16.sh` (log `queue16.log`).** It runs in order:
  1. Stage B 15 (done): `sfl_spread_carry` ADVANCES (Z +0.330, p 0.073); `sfl` (+0.104) and
     `sfl_spread_verify` (+0.100) abandoned.
  2. Batch 16 stage A: `sfl_carry_mem` (new `sfl_memory`: carried levels pool discounted earlier
     rollouts), `sfl_carry0`, `sfl_spread_carry0`, into `a5/b16.csv`.
  3. Stage C 14: `results/lib/stageC14.sh`, DR + `sfl_tilt_carry` + `sfl_spread_carry` (added and
     registered), seeds 1101-1200, `a5/c14.csv`. Verdict:
     `decide --a5 --stage C --seeds 1101-1200 --arms sfl_tilt_carry sfl_spread_carry`.
- **Batch 16 stage A (done):** `sfl_carry0` abandoned (-0.32); `sfl_carry_mem` (+0.119, cart +1.66) and
  `sfl_spread_carry0` (+0.097) dropped as hopeless. Carry needs the tilt; memory helps CartPole only.
- **Batch 17 cancelled** (knob-turning; the user asked for more self-evaluation and new mechanisms).
- **Queued: `results/lib/queue18.sh` (log `queue18.log`)** waits for stage C 14, then runs batch 18:
  start-state SFL (`sfl_states`: half the scout candidates are (level, visited state) pairs from a
  training archive, replayed from that state), seeds 1-16, plus `sfl_tilt` seeds 9-16, into `a5/b18.csv`.
  Stage A gate is seeds 1-8. Seeds 1-16 give an exploratory same-seed `sfl_states` vs `sfl_tilt` comparison.
  The self-evaluation and falsifiers are in `library_log.md`.
- **Uncommitted:** a git commit/push of this work was refused by the auto-mode permission
  check; everything since `ab16711` is in the working tree only. Commit it next session.

## Next ideas

After batch 6 the mass probe shows `var_low` removes *trivial* tasks (41-64% of uniform's
proposals, 19-27% of the tilted SIR's), not hopeless ones, and PPO's random-suite competence
needs that practice. Moving mass at a fixed budget seems to trade hard for random. Options:
(a) accept a declared trade (a different success criterion, which is the user's call);
(b) raise the budget's efficiency instead of moving mass (e.g. shorter tilted episodes);
(c) a second learner, to see whether the forgetting is PPO-specific.

If batch 5 fails: a proposer that can concentrate mass in small regions (cross-entropy /
GMM on high-score tasks, or local MCMC around them) under the tilted score; budgeted
scouting (~5%) of a shortlist; retarget to no-harm variance reduction of DR's objective.
Also on the shelf: bandit mixer over proposers, a second RL algorithm, the standard-tier
benchmarks.

## Standing rules

Runs launch with `--allow-battery`. Keep shell commands short; do not retry-loop. Commit and
push `wip` after each tested change, then fast-forward `main` (`git push origin wip:main`);
`main` was created on the remote on 2026-09-29. Register each batch in `docs/library_log.md`
before running it.
