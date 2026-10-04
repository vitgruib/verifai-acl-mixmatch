# Failure records: schema and how to diagnose from them

Every falsification run writes one directory, self-contained enough to replay and explain each
counterexample on a laptop without the cluster that found it. Writer: `atlas/record.py`
(`Recorder`); reader: `atlas.record.load(dir) -> (header, rows, summary)`.

Directory: `<out>/<algo>/s<train_seed>/<space>_<sampler>_r<falsifier_seed>/`

| file | contents |
|---|---|
| `run.json` | header: what was falsified and how (below) |
| `samples.jsonl` | one line per proposed point, **valid or not**, in proposal order |
| `traces.npz` | agent trajectories for counterexamples only: `i<index>_pos` (step x attempt x 2 positions, int8), `i<index>_len` |
| `summary.json` | aggregates, written when the run closes (absent = run crashed or still going) |

## run.json

| key | meaning |
|---|---|
| `schema_version` | bump when fields change; readers should check it |
| `created`, `host` | when and where |
| `env`, `algo`, `train_seed` | which agent (e.g. `maze`, `rplr`, `3`) |
| `checkpoint`, `checkpoint_step` | path and update number of the weights falsified |
| `train_config` | the training run's own `config.json`, copied in full |
| `upstream` | repo + commit of the paper code that trained it (jaxued / dcd / sfl) |
| `atlas_commit` | commit of this repo that ran the falsifier |
| `space` | name and bounds of the falsification box (`dr` or `seg` for maze; see `atlas/maze/space.py`) |
| `falsifier` | `{name, params, via}`: sampler (random, halton, ce, mab, sa), its parameters, and whether it ran through VerifAI's Scenic external-parameter API |
| `spec` | the property: `solve_rate >= tau` over `attempts` stochastic rollouts; `rho = solve_rate - tau` |
| `budget`, `seed` | number of proposals, falsifier seed |
| `train_context` | what the `train` fields below were measured against (DR reference size, buffer present or not) |

## samples.jsonl (one row per proposal)

| field | meaning | diagnostic use |
|---|---|---|
| `i`, `t` | proposal index, seconds since start | search speed; `first_cex_i` |
| `params` | the falsifier's raw point | re-run the exact proposal; see what the sampler converged to |
| `valid`, `invalid_reason` | hard constraints (goal reachable, agent != goal, ...) | a sampler wasting budget on invalid points |
| `level` | replayable level encoding (maze: wall string + agent/goal/dir) | reload in the env to watch the failure |
| `desc` | interpretable features: `n_walls, path_len, manhattan, detour, dead_ends, corridor_cells` | **what kind** of level fails |
| `returns`, `lengths` | per attempt | timeout vs wandering vs near-miss |
| `solve_rate`, `mean_return` | over the attempts | |
| `rho`, `cex` | robustness (< 0 = violated), counterexample flag | |
| `hard` | `solve_rate == 0` | solid failures vs coin-flips near the threshold |
| `train` | where the level sits relative to training (below) | **why** it fails |

`train` fields (`atlas/maze/context.py`), all in feature space scaled by the DR spread:

- `dr_knn`: mean distance to the 10 nearest levels from the DR generator. Large means the
  agent rarely or never saw anything like it (out-of-distribution failure).
- `buf_knn`: same, against the PLR/ACCEL level buffer saved in the checkpoint. PLR-family only.
- `buf_score_pct`: percentile, within the buffer, of those neighbours' replay scores. Low means
  the curriculum saw this region and ranked it unimportant (hypothesis H2: the score is blind to it).
- `buf_age`: staleness of those neighbours. High means the curriculum learned it once and stopped
  replaying it (forgetting).

## summary.json

`n, n_valid, invalid_rate, n_cex, cex_rate, n_hard, first_cex_i, first_cex_t, min_rho, mean_rho,
distinct_cex_cells` (counterexamples binned coarsely on `desc`: distinct failure modes),
`desc_cex` / `desc_ok` (mean features of failing vs passing levels), `train_cex` / `train_ok`
(the same for `train` fields), `wall_s`.

## Diagnosing a failure: a checklist

1. **Is it real?** `hard` counterexamples (0/10 solved) are failures; `solve_rate` of 0.4 is
   a coin-flip near tau. Re-evaluate the `level` with more attempts before writing it up.
2. **What kind?** Compare `desc_cex` against `desc_ok`. In the maze prelim, failing levels have
   about twice the walls and path length of passing ones. Group counterexamples by coarse
   `desc` cells (`atlas/prelim.py: mode_of`) and look at one level per cell.
3. **Why: never seen, or seen and ignored?**
   - High `dr_knn` and high `buf_knn`: out of distribution. The curriculum never generated it,
     so the fix is generation (a falsifier as the level generator).
   - Low `buf_knn` but low `buf_score_pct`: the curriculum had it and scored it low (H2). The fix
     is the score.
   - Low `buf_knn` and high `buf_age`: it was learned and then forgotten. The fix is replay
     staleness.
4. **How does it fail?** Load `traces.npz["i<index>_pos"]`. If `lengths` sit at the step cap, the
   agent wanders or loops; short episodes with no return mean a wrong turn into a dead end.
5. **Across algorithms:** run `python -m atlas.prelim <root>`, which pools runs. It prints the
   counterexample rate and number of distinct failure modes per (algo, space, sampler), both
   relative to random search, plus the H2 table.
