# Literature map: curriculum / UED techniques, tried and untried here

Written 2026-09-30 after batch 9, when the user asked whether we were "just bashing our head
into the problem" instead of reading the literature and mixing techniques. They were right:
batches 4-9 were knob changes to two ideas, VarModel's score and SFL's search width. This
page maps the field onto three questions a curriculum has to answer. Each technique is
marked tried or untried here. The page ends with combinations to test.

## The three questions a curriculum answers

1. **Where do candidate tasks come from?** (the proposer)
2. **How is a task scored?** (the signal)
3. **What keeps training anchored to the target distribution?** (the guard)

Most of our arms changed only (2). The mass probe says (1) is the bottleneck. The hard
suite's corner holds about 0.3% of uniform mass, and no score can rank a task the proposer
never draws.

## Techniques

| technique | Q | idea | here |
|---|---|---|---|
| PLR / robust PLR (Jiang et al. 2021) | 2 | replay levels with high PVL / regret proxy | tried; PVL tracks easiness with a task-blind critic (docs/plr.md) |
| ACCEL (Parker-Holder et al. 2022) | 1 | mutate replayed high-regret levels | tried with PLR scores (`accel`); library `Mutate` on posterior; never on SFL's frontier |
| SFL / No Regrets (Rutherford et al. 2024, [arXiv 2408.15099](https://arxiv.org/abs/2408.15099)) | 1+2 | scout random levels with rollouts, keep top p(1-p) | tried: about +0.04 real, fails stage C |
| VDS (Zhang et al. 2020) | 2 | value-ensemble disagreement | tried, no gain |
| ALP-GMM (Portelas et al. 2019) | 1+2 | GMM over absolute learning progress | tried as `ProgressModel`, about 0 |
| VerifAI-style samplers (cross-entropy, bandit) | 1 | steer sampling toward falsifying regions | tried (`picker ce/mab`, `verifai_surr`), no gain |
| var_low (ours) | 2 | return-std x exp(-predicted return): learnable *and* hard | about +0.04, fails stage C |
| Self-paced RL / SPDL (Klink et al. 2020) | 1+3 | context distribution moves from easy to target under a KL bound | untried |
| CURROT / GRADIENT (Klink et al. 2022, [PMLR](https://proceedings.mlr.press/v162/klink22a/klink22a.pdf); Huang et al. 2023, [arXiv 2309.14091](https://arxiv.org/abs/2309.14091)) | 1+3 | optimal-transport interpolation from easy tasks to the target, constrained to tasks above a success threshold | untried |
| DRED (Garcin et al. 2024, [arXiv 2402.03479](https://arxiv.org/abs/2402.03479)) | 3 | generate levels from a model fitted to the target distribution (grounding) | untried; SFL's 50% DR draws already ground partially |
| TSCL / bandit teachers (Matiisen et al. 2017; DUMP 2025, [arXiv 2504.09710](https://arxiv.org/abs/2504.09710)) | 1 | a non-stationary bandit over task sources, rewarded by learning progress | untried (on the shelf since batch 3) |
| Actor-Curator (2026, [arXiv 2602.20532](https://arxiv.org/abs/2602.20532)) | 2 | neural curator that predicts policy improvement, trained as a bandit with mirror descent | untried (LLM setting; idea = a learned improvement predictor) |
| PACE (2026, [arXiv 2605.01358](https://arxiv.org/abs/2605.01358)) | 2 | score a level by the squared norm of the parameter update it induces | close to batch 2's `grad_is`, which failed because the signal is not predictable from task parameters; PACE scores *visited* levels, so it would fit a replay buffer |
| UED as min-max optimisation (2025, [arXiv 2505.20659](https://arxiv.org/abs/2505.20659)) | 2 | nonconvex-strongly-concave objective with convergence guarantees | read the abstract only |
| Diffusion curricula (NeurIPS 2024) | 1 | a generative model proposes tasks | out of scope (too heavy for "simple") |

## What our results say about mixing

- Every working component gives about +0.04, and the gains come from different places. SFL
  *finds* learnable tasks by scouting. var_low *leans* toward hard ones. Neither moves real
  mass into the hard corner (massprobe).
- ACCEL's whole point is reach: edits compound, so a frontier can walk into small regions
  that uniform sampling almost never hits. We only ran ACCEL with PLR's broken PVL score,
  never with SFL's scouted p(1-p), which is the score that works here.
- The guards fail when the mix leaves DR too far behind (var_low's stage C). The pieces from
  SPDL, CURROT and DRED are about exactly that. SFL already keeps 50% DR draws, and its guards pass.

## Combinations registered (batch 10, docs/library_log.md)

All three are SFL with one or two borrowed parts. Every part is environment-agnostic: it uses
only the task box and episode outcomes, so it is model-agnostic too.

- `sfl_mut` = **SFL x ACCEL**: half of each scout's candidates are Gaussian edits
  (sigma 0.05 x range, clipped) of the last frontier; the other half stay uniform.
- `sfl_tilt` = **SFL x var_low**: rank scouted levels by p(1-p)(1-p), so the frontier leans
  hard. This is var_low's tilt applied to measured pass rates, not a model's predictions.
- `sfl_mut_tilt`: both. The tilt points the hill-climb toward the hard side.

## Next if these fail

1. A **bandit mixer** over proposers (uniform, SFL frontier, mutated frontier, replay),
   rewarded by the per-source change in scouted pass rate (TSCL/DUMP).
2. **CURROT-style constraint**: only propose tasks with scouted p above a floor. The
   frontier then moves from solved tasks toward the target, rather than jumping to it.
3. **PACE on the replay buffer**: rank SFL's frontier by realized update norm instead of p(1-p).

## Lineage: who builds on whom, and where SFL sits (2026-10-01)

- **PLR** (Jiang et al. 2021, [2010.03934](https://arxiv.org/abs/2010.03934)) replays levels
  ranked by a regret proxy (positive value loss). **Robust PLR** ([2110.02439](https://arxiv.org/abs/2110.02439))
  trains only on replayed levels, which gives a minimax-regret guarantee.
- **ACCEL** ([2203.01302](https://arxiv.org/abs/2203.01302)) adds small edits to
  high-regret levels, so the frontier can walk to places uniform sampling misses.
- **SFL** (Rutherford et al. 2024, [2408.15099](https://arxiv.org/abs/2408.15099)) shows
  that the regret proxies mostly track success rate, not regret. It replaces them with
  scouted learnability p(1-p) and beats PLR and ACCEL on JaxNav and XLand-Minigrid.
- **Users of SFL:**
  - LILO ([2502.12272](https://arxiv.org/abs/2502.12272)) carries p(1-p) over to LLM
    reasoning RL.
  - Kinetix ([2410.23208](https://arxiv.org/abs/2410.23208)) finds SFL the best UED method
    on its physics tasks.
- **Extensions of SFL:**
  - **NCC** ([2505.20659](https://arxiv.org/abs/2505.20659)) turns the hard top-k into an
    entropy-regularized soft adversary: replay probability is proportional to (a tempered)
    score. It also generalizes learnability to sigma x N(mu) and beats SFL on XLand-Minigrid.
  - **TRACED** ([2506.19997](https://arxiv.org/abs/2506.19997)) adds transition-prediction
    error and "co-learnability" (does training on A help B) to the score, and uses ACCEL edits.
- **Related scores:**
  - ProCuRL / ProCuRL-Target ([2304.12877](https://arxiv.org/abs/2304.12877),
    [2405.02481](https://arxiv.org/abs/2405.02481)) use the ZPD score p(1-p), which is SFL's
    score, and the Target variant multiplies it by correlation with target tasks. Here the
    targets are the exam, so the Target variant would be leakage: excluded.
  - MAGELLAN ([2502.07709](https://arxiv.org/abs/2502.07709)) predicts learning progress
    with a learned model, rather than scouting.
  - Dreaming in Code / DiCode ([2602.08194](https://arxiv.org/abs/2602.08194)) has an LLM
    write new environments.
- **Ours:** `sfl_tilt` (p(1-p)(1-p), leaning hard) is my own combination of SFL's score with
  `var_low`'s exp(-mean) tilt. I have not seen it in a paper. Easy-to-hard LLM work
  ([2506.06632](https://arxiv.org/abs/2506.06632)) weights harder tasks in a related way.

### Batch 12: build on SFL from its follow-ups

- `sfl_tilt_soft` (from NCC): keep every scouted level with score > 0 and replay it with
  probability proportional to its score, instead of uniformly from the top 100.
- `sfl_tilt_carry` (persistent frontier, as in PLR's buffer and TRACED): the last buffer is
  re-scouted among the 1000 candidates, so frontier levels survive until learned.
- Not ported: TRACED's transition-error score (it needs a world model), co-learnability
  (it needs pairwise transfer runs), and NCC's sigma x N(mu) (here it reduces to p(1-p) for
  binary outcomes).
