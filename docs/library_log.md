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
