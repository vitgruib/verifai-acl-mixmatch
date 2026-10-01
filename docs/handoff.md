# Handoff (2026-09-30): state of the curriculum-library work

Start here in a new session. Details live in the docs linked below; this page is the map.

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
- **Running: held-out no-harm step** (`results/lib/heldout10.sh`, log `heldout10.log`,
  CSVs `results/{pointnav,maze}/lib/heldout10.csv`; 48 seeds DR vs `sfl_tilt`, PointNav then
  Maze, likely 8-12 h; launched with `nohup`; `--resume` if interrupted; verdict `python -m acl_bench.plr.heldout --arm sfl_tilt`). Rule registered in `docs/library_log.md`. `decide.py`
  does not read these envs; compute the final random-suite gain and its lower bound directly.
- Lead: `sfl_mut_tilt` plus a grounding anchor (CURROT/DRED style) to fix its Pendulum guard.
- Next in line if batch 10 fails: a bandit mixer over proposers, a CURROT-style
  success floor, and PACE scoring on the frontier (`docs/literature.md`, end).

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
