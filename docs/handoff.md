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

13 arms tried; **none survives stage B**.
- Batch 1 (`sir_a1`, `sir_a4`, `sir_is`, `sir_is05`, `replay_post`, `mutate_post`,
  `verifai_surr`, reference `fmodel`): five died at stage A; `fmodel`, `sir_a4`, `sir_is`
  died at stage B; re-screen at 72 seeds: `sir_a4` -0.002, `sir_is` +0.039 (p = 0.22).
- Batch 2 (`grad_is`, `align`: per-episode gradient size / alignment): abandoned before
  training; the signal is not predictable from the task (cross-validated R^2 below noise).
- Batch 3 (`sir_is_a1`, `sir_is_u50`, `rw_a4`, `ens_a4`, `prog_a4`): two died at stage A
  (`rw_a4` hurt Acrobot -0.154); the rest at stage B, all within +/-0.02 at 72 seeds.
- Lessons: 8-seed stage A leads (up to +0.14) never replicated; CartPole has headroom
  (oracle +0.19, scouted SFL +0.09) but sampling by the pass model's p(1-p), in every form
  tried, does not find the hard suite's region. That family is closed.
- Ops lesson: `pgrep -f name.sh` also matches any waiting shell whose command contains the
  name (caused a deadlock); use `pgrep -f "name[.]sh"`.

## Open decision (asked of the user, unanswered)

1. **Diagnose** (recommended, cheap, no training): does the pass model rate the hard suite's
   tasks as frontier? Compare its p(1-p) on hard-suite tasks vs uniform tasks over training.
   If it calls them easy or impossible, the fix is the signal, not the sampler.
2. **Budgeted scouting**: spend a small fixed budget (~5%, vs SFL's ~220x) scouting
   candidates the pass model shortlists; trades "free" for part of SFL's +0.09.
3. **Retarget**: aim at no-harm variance reduction of DR's own objective (`sir_is` was the
   only arm still positive at 72 seeds).

Unregistered ideas still on the shelf: recency-weighted pass model, bandit mixer over
proposers, maze oracle diagnostic (~12 min/run), a PointNav hard suite, a second RL
algorithm, the standard-tier benchmarks.

## Standing rules

Runs launch with `--allow-battery`. Keep shell commands short; do not retry-loop. Commit and
push `wip` after each tested change, then fast-forward `main` (`git push origin wip:main`);
`main` was created on the remote on 2026-09-29. Register each batch in `docs/library_log.md`
before running it.
