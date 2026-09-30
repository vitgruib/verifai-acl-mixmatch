# Handoff (2026-09-29): state of the curriculum-library work

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

17 arms tried in batches 1-4; **none survives stage B**. Batch 5 is in stage A.
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
- Batch 5 (`var_low`, `var_low16`: `VarModel(tilt=1)`), stage A running
  (`results/lib/batch5.sh`, log `batch5.log`).
- Lessons: 8-seed stage A leads (up to +0.14) never replicated; CartPole has headroom
  (oracle +0.19, scouted SFL +0.09); a general frontier sampler finds *a* frontier, not the
  hard suite's corner. Diagnose mass placement with `massprobe` before training.
- Ops lesson: `pgrep -f name.sh` also matches any waiting shell whose command contains the
  name (caused a deadlock); use `pgrep -f "name[.]sh"`.

## Next ideas

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
