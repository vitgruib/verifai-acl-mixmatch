---
description: Restate the ACL27 curriculum-library goal and keep iterating toward it
argument-hint: "[optional focus or extra constraint]"
---

# Goal: an environment-agnostic curriculum library

Build a **generalized curriculum library** that mixes and matches VerifAI, PLR and other
techniques into a wrapper that **boosts any RL agent's performance**.
Training inefficiency is acceptable (2026-09-30): extra simulation or compute is fine as long
as the boost is agnostic.
Keep iterating until a PLR-style method is found that, while staying reasonably simple,
gives broad, environment-agnostic gains.

## Hard constraints

- **Agnostic boost.** The method must be environment-agnostic *or* model-agnostic (at least
  one). Environment-agnostic: nothing names an environment (no per-env buckets, thresholds,
  fix-ups or tuned constants); it may then read the critic, gradients or parameters.
  Model-agnostic: it treats the learner as a black box (only episode outcomes).
- **Inefficiency is acceptable (Amendment 3, 2026-09-30).** Extra simulation beyond training
  episodes (e.g. scouting rollouts, evaluations) and extra wall-clock need not be small; it is
  not charged against the training budget (arms compare at equal training steps) but must be
  reported as a cost.
- **Simple.** Prefer a method that can be stated in a paragraph and plugged into
  `acl_bench/curriculum/` as a proposer/estimator.

## Success criterion

An arm counts as a success only if it survives the full protocol in `docs/protocol.md`:
- Stage A: 8 seeds on CartPole + Acrobot + Pendulum (Amendment 4: a continuous-reward env at every stage). Abandon if primary (`fin_vd`) < 0 or a guard fails.
- Stage B: 72 CartPole seeds against the DR pool, plus 8 on MountainCar/Pendulum. Advance only
  at primary > 0 with p < 0.10 and guards passing.
- Stage C: 100 fresh seeds (1001+), test halves, Holm correction.
- **Guards are relative (Amendment 2, 2026-09-30):** a random-suite cost is acceptable if it is
  under 20% of the baseline's own random-suite pass rate on that env (stage C: one-sided 95%
  lower bound >= -0.20 x DR's rate; stage B: point estimate; stage A: with 0.07 extra slack).
  Claims made under this rule need fresh seeds (1101-1200), since it was adopted after
  `var_low`'s stage C on 1001-1100.
- It must then not hurt the held-out envs (Maze, PointNav).

## Working style

- Clean protocol first. **Register every batch in `docs/library_log.md` before running it.**
- Test many arms; **abandon early** when preliminary results look hopeless.
- **Prod before long runs.** Before launching any long job (a stage, a batch, a diagnostic
  over many seeds), run a small, fast probe first: 1-2 seeds, a short budget or few checks,
  a unit test, or an offline check of the signal. Use it to catch crashes, wrong configs and
  obviously dead ideas in minutes, and only commit the full compute once the probe looks sane.
- Be creative, including mathematical modifications of existing techniques (e.g. VarModel
  generalizing p(1-p) to return variance). Diagnose signals offline before paying for training.
- Record every result and verdict in `docs/library_log.md`; keep `docs/handoff.md` current.

## Operations

- Start by reading `docs/handoff.md`, then the tail of `docs/library_log.md`, and check for
  running jobs with `pgrep -f "name[.]sh"` (plain `name.sh` also matches waiting shells).
- Launch runs with `--allow-battery`; use `--resume` to continue interrupted runs.
- Verdicts: `python -m acl_bench.plr.decide --stage A|B --arms ...` (`--confirm` for stage C).
- Activate the venv first: `source .venv/bin/activate`.
- Commit and push `wip` after each tested change, then `git push origin wip:main`.

## Current focus

$ARGUMENTS
