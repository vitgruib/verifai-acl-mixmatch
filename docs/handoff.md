# Handoff (2026-10-11)

Start here in a new session.

## Goal

Find where published UED algorithms fail, using VerifAI falsification on their papers'
environments and the papers' own code; explain why; then fix with falsifier + PLR.
Full statement and success criterion: `.claude/commands/goal.md`.

## Status

- Local work is done. The cluster suite (`cluster/README.md`) is built and tested headless at
  tiny budgets: 3 parts (1 Maze, 2 Kinetix, 3 JaxNav), DR / PLR / Robust PLR / ACCEL / SFL,
  10 seeds, then a `ce` falsifier with certified-solvable failures.
- **Waiting on:** the professor running Maze in full on the cluster (`submit.sh maze-smoke`, then
  `submit.sh 1`: 6 algorithms x 10 seeds, 60 GPU jobs). It tests H2, pre-registered as batch 20 in
  `docs/library_log.md`. Kinetix and JaxNav (`submit.sh smoke`, `2`, `3`) come after. Only the
  cluster can verify the CUDA venv builds, GPU training speed and falsification rates on fully
  trained agents.
- **After the tarball returns:** diagnose each failure (`docs/failure_records.md` checklist),
  write it into `docs/ued_atlas.md`, then design the falsifier + PLR fix.

## Scope decisions

- 2026-10-04: train with the papers' own code (not our torch harness). PAIRED dropped (cost).
- 2026-10-06: CarRacing and BipedalWalker (DCD) dropped for Kinetix (~2/3 of GPU-h).
  `atlas/carracing/` and `atlas/bipedal/` remain as working but unused falsifiers.
- XLand-MiniGrid not pursued.

## Evidence so far

- Batch 19 (JaxUED maze, 10% budget, 2 seeds): replay methods do not beat DR
  (DR 0.30/0.35 on the 8 named mazes >= PLR, Robust PLR, ACCEL).
- Maze falsifier prelim (`docs/ued_atlas.md` 4b): `ce` is the best sampler. Failing levels force
  detours. PLR / Robust PLR failures are out of distribution; "seen but scored low" is not supported
  at 10% budget.

## Where things are

| what | where |
|---|---|
| algorithm registry, paper settings, results | `docs/ued_atlas.md` |
| failure-record schema, diagnosis checklist | `docs/failure_records.md` |
| cluster run (compute, steps, recovery, validation) | `cluster/README.md` |
| falsifiers per env | `atlas/{maze,kinetix,jaxnav}/` |
| previous goal (curriculum library, batches 1-18, closed with no confirmed arm) | `docs/library_log.md`, `docs/protocol.md` |

## Standing rules

Runs launch with `--allow-battery`. Keep shell commands short; do not retry-loop. Work on `main`
(the GitHub default since 2026-10-11; `wip` is retired): commit and push after each tested change.
Register each batch in `docs/library_log.md` before running it.
