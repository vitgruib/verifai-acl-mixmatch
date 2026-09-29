# Curriculum library: log of every arm tried

Protocol: `docs/protocol.md`. Design: `docs/library.md`. Each batch is registered here
before it runs; results and verdicts are added after. Nothing is deleted.

## Batch 1 (registered 2026-09-28): the library's first arms, stage A

DR pool: 48 seeds per dev environment (`results/<env>/lib/runs.csv`). Arms: 8 seeds on
CartPole and on Acrobot (`results/<env>/lib/batch1.csv`).

| arm | hypothesis |
|---|---|
| `fmodel` | reference: the pre-library frontier picker (best of 64 uniform tasks by model p(1-p), 50% uniform) |
| `sir_a1` | sampling in proportion to model learnability (soft SFL signal, no scouting) helps |
| `sir_a4` | the same, sharper (alpha 4) |
| `sir_is` | variance-optimal unbiased sampling: q ~ sqrt(p(1-p)) with full importance correction speeds up learning DR's own objective without changing it (no harm by construction) |
| `sir_is05` | half-corrected frontier bias: keeps part of the frontier gain, limits harm |
| `replay_post` | a PLR buffer ranked by posterior learnability (model prior + the task's own visits) |
| `mutate_post` | ACCEL-style edits of the posterior buffer's picks |
| `verifai_surr` | VerifAI's cross-entropy sampler, searching the pass model instead of the simulator |

## Batch 2 (registered 2026-09-28): model-specific gradient signals -- abandoned at the mechanism check

| arm | hypothesis |
|---|---|
| `grad_is` | variance-optimal unbiased sampling with each episode's measured policy-gradient size (actor last layer, closed form), q ~ E‖g(task)‖, full importance correction |
| `align` | sample tasks whose gradient aligns with the uniform tasks' gradient in the same rollout |

**Mechanism check (before any screen):** is the per-episode signal predictable from the
task? Cross-validated R² of the task regressor at the end of short runs (2 seeds each):
CartPole grad size -0.25 / -0.22, alignment -0.62 / -0.71; Acrobot grad size -0.77 / -0.84,
alignment -0.72 / -1.23. On the same check a pure-noise target scores -0.29 and a real
signal +0.18 (the regressor overfits, so the check is conservative). Neither signal can be
predicted from the task: single-episode gradients are dominated by noise and episode
length. **Both abandoned untrained.** Note: batch 1's Acrobot half runs on code that
initializes the pass model without touching torch's global random stream (CartPole's half
did touch it): a different random stream, not a different algorithm.

## Batch 1, stage A results (2026-09-29)

8 seeds per arm against the 48-seed DR pool. `vd`: CartPole hard suite, dev half (primary);
`r`: random suite (guard). Welch p in brackets.

| arm | CartPole vd | CartPole r | Acrobot r | verdict |
|---|---|---|---|---|
| `fmodel` | +0.103 (0.16) | +0.029 (0.63) | -0.011 (0.64) | advance to B |
| `sir_a1` | -0.004 (0.95) | +0.005 (0.94) | -0.013 (0.48) | abandon: primary < 0 |
| `sir_a4` | +0.018 (0.82) | +0.040 (0.39) | -0.021 (0.35) | advance to B |
| `sir_is` | +0.007 (0.94) | **+0.091 (0.02)** | -0.028 (0.28) | advance to B |
| `sir_is05` | -0.042 (0.51) | +0.003 (0.96) | -0.069 (0.08) | abandon: primary < 0 |
| `replay_post` | -0.155 (0.003) | +0.066 (0.17) | -0.024 (0.23) | abandon: primary < 0 |
| `mutate_post` | -0.015 (0.80) | +0.023 (0.69) | -0.114 (0.001) | abandon: primary < 0, Acrobot guard |
| `verifai_surr` | -0.008 (0.91) | +0.020 (0.74) | +0.002 (0.89) | abandon: primary < 0 |

Reading: soft sampling (alpha 1) and the replay buffers do nothing or harm; only the sharp
picker (`fmodel`, and weakly `sir_a4`) moves the hard suite. `sir_is` is the one result
matching its own theory: it is built to learn DR's objective faster, and DR's objective is
the random suite, where it gains +0.091 -- but this was not its declared primary, and it is
one of 16 guard comparisons (uncorrected).

**Registered before stage B's data (secondary hypothesis H-is):** on CartPole seeds 9-24
only (fresh), `sir_is` improves the random suite over DR (one-sided Welch, p < 0.05). If it
holds, `sir_is` goes to stage C with the random suite as its primary, on all envs.

Stage B (`results/lib/stageB1.sh`): the three survivors, CartPole seeds 9-24, MountainCar
and Pendulum seeds 1-8.
