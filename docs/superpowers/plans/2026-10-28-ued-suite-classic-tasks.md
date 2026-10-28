# UED Generalization Suite, Part 1 (Classic Tasks): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train DR, PLR, Robust PLR, ACCEL and SFL on CartPole, LunarLander and FrozenLake, grade every agent on a frozen three-section exam (random, VerifAI from DR failures, custom), and run the pre-registered screen: does any method beat DR on the VerifAI section?

**Architecture:** Everything lives in the existing PyTorch package `acl_bench`. Two new environments follow the existing env contract (`acl_bench/envs/__init__.py`). A new subpackage `acl_bench/suite/` holds this study's arms, exam construction, runner and analysis. It reuses `acl_bench/plr/fast.py` (training), `acl_bench/exam/` (sets, grader, search) and `acl_bench/study/safety.py` (parallel runs with safety pauses). The new exams go under `frozen_sets/suite/<env>/`, so the earlier CartPole exam in `frozen_sets/cartpole/` stays untouched.

**Tech Stack:** Python 3.11 (`.venv`), PyTorch, NumPy, Gymnasium 1.3.0 (+ Box2D for LunarLander), Scenic 3.1.1 / VerifAI 2.2.0, SciPy 1.17.1, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-ued-generalization-suite-design.md`. Maze is out of scope here; it gets its own plan after this part's screen result.

## Global Constraints

- No JAX. No VerifAI in training; VerifAI only builds the exam.
- Methods: DR, PLR, Robust PLR, ACCEL, SFL. Equal total environment steps per method, including SFL's scouting (`charge_scouting=True`).
- Seeds: 20 scored per method per task; 10 reference DR agents per task, separate seeds, never scored.
- Compute: the classic tasks get about 12 hours of the Mac in total, within a 2-day total. Runs use the safety guard and are launched with `--allow-battery`. If a measured cost breaks the 12-hour share, stop and report to the user before cutting seeds or budget.
- Exam, all tasks: a question = one parameter setting played from 10 start states (10 plays). An agent fails a question when it passes fewer than 5 of the 10 plays.
- VerifAI section: cross-entropy sampler against each of the 10 reference DR agents, same budget per agent; keep a question if at least 3 of the 10 reference agents fail it. FrozenLake also drops questions that fail the path check or whose optimal success probability is below 0.5.
- Random section: uniform over the training box, no solvability filter on continuous tasks (FrozenLake keeps the path check and the optimal-success filter).
- Custom section, continuous settings: default case, then per setting (others at default): min edge, max edge, slightly beyond each edge (10% of the range width), way beyond each edge (50% of the range width). Skip invalid values. No corners.
- Pre-registration (Task 8) is committed before any scored run (Task 10).

## Review Focus

1. **Re-grading an agent twice must give identical scores**, including on slippery FrozenLake and windy LunarLander: the randomness comes from the start state, not from global RNG. Test in Tasks 2 and 3.
2. **Beyond-range values the library rejects** (LunarLander gravity outside (-12, 0); negative wind; FrozenLake size below 2) are skipped, not crashes. Test in Task 4.
3. **Degenerate FrozenLake maps** (start on a hole, start = goal, a map with no path) never reach an exam. Test in Task 3.
4. **A resumed run after a safety pause or crash** neither duplicates nor drops CSV rows. Test in Task 6.
5. **Agents saved by the runner load back and grade the same**, including the `make_agent` layout from `fast.py`. Test in Task 6.

---

### Task 1: "return" goal and per-env reward in training and grading

LunarLander passes on episode return >= 200, which none of the existing goals ("survive", "reach", "balance") express. SFL scouting also looks rewards up in `REWARD` by module name (`fast.py:337`), which fails for envs that define `reward`.

**Files:**
- Modify: `acl_bench/exam/grader.py:21-65` (`rollout`, `passed`)
- Modify: `acl_bench/plr/fast.py:195-205` (episode success), `fast.py:337` (SFL scouting reward)
- Test: `tests/test_grader.py`, `tests/test_sfl.py`

**Interfaces:**
- Produces: env contract extension. `GOAL = "return"` plus `RETURN_THRESHOLD: float` and a batched `reward(state, action, params)`. `rollout(..., extras)` fills `extras["return"]` (N,) when the env has `reward`. `passed(env, steps, ended, max_steps=None, streak=None, ret=None) -> np.ndarray` returns `ret >= env.RETURN_THRESHOLD` for GOAL "return".

- [ ] **Step 1: Write the failing tests**

```python
def test_return_goal_passes_on_threshold():
    env = _toy_return_env(threshold=2.0)          # tiny fake module in the test: reward 1 per step
    ok = passed(env, steps=np.array([1, 3]), ended=np.array([True, True]), ret=np.array([1.0, 3.0]))
    assert ok.tolist() == [False, True]

def test_sfl_scouting_uses_env_reward():
    # an env with reward() and no REWARD entry: sfl_select runs and returns a finite score
    top, sim_steps, w = sfl_select(_toy_return_env(2.0), agent, levels, lc, rng)
    assert sim_steps > 0 and np.isfinite(w).all()
```

- [ ] **Step 2: Run them; expect failure** (`passed()` has no `ret`; `sfl_select` returns zero rewards for the toy env)

Run: `source .venv/bin/activate && python -m pytest tests/test_grader.py tests/test_sfl.py -k "return_goal or env_reward" -v`

- [ ] **Step 3: Implement.** In `rollout`, accumulate `env.reward(...)` into `extras["return"]` when present. In `passed`, add the "return" branch. In `fast.py` episode success, use `ret >= env.RETURN_THRESHOLD` for "return". In `sfl_select`, use `env.reward` when the env has one, else `REWARD`.

- [ ] **Step 4: Run the full suite; expect PASS** (existing envs unchanged)

Run: `python -m pytest tests -q`

- [ ] **Step 5: Commit** `git commit -m "Grader and trainer: GOAL 'return' with RETURN_THRESHOLD; SFL scouting uses env.reward"`

### Task 2: LunarLander environment

**Files:**
- Create: `acl_bench/envs/lunarlander.py`, `acl_bench/envs/lunarlander.scenic`
- Modify: `acl_bench/envs/__init__.py:21` (add `"lunarlander"` to `NAMES`), `requirements.txt` (Box2D)
- Test: `tests/test_lunarlander.py`; the env-parametrized tests in `tests/test_envs.py` and `tests/test_grader.py` now include it

**Interfaces:**
- Consumes: Task 1 (`GOAL = "return"`, `reward`).
- Produces: `PARAM_BOUNDS = {"gravity": (-12.0, -2.0), "wind_power": (0.0, 20.0), "turbulence_power": (0.0, 2.0)}`, `DEFAULTS = {"gravity": -10.0, "wind_power": 0.0, "turbulence_power": 0.0}`, `GOAL = "return"`, `RETURN_THRESHOLD = 200.0`, `MAX_EPISODE_STEPS = 1000`, `OBS_DIM = 8`, `ACTION_DIM = 4` (discrete), plus `valid(params: dict) -> bool` (used by Task 4: gravity in (-12, 0), wind >= 0, turbulence >= 0).

Box2D physics can't be batched in NumPy, so the module keeps a pool of Gymnasium `LunarLander` instances. A state row is `[pool_id, reset_seed, t, ret]`. `sample_starts(task_row, n, rng)` draws `reset_seed`s. `step(state, action, params)` steps each row's pooled env, creating it from `(params, reset_seed)` on first use. `enable_wind = wind_power > 0 or turbulence_power > 0`. Wind and turbulence draw from the env's own seeded `np_random`, so a play is reproducible from `(params, reset_seed)`. Pool entries are freed when their row terminates. `planning_cost` is not provided (no certification on continuous tasks).

- [ ] **Step 1: Install Box2D and verify.** `pip install "gymnasium[box2d]==1.3.0"` (needs `swig`; `brew install swig` if the build fails), then `python -c "import gymnasium as g; g.make('LunarLander-v3', gravity=-5, enable_wind=True, wind_power=10).reset(seed=0)"`. If the install can't be made to work, stop and report to the user.

- [ ] **Step 2: Write the failing tests**

```python
def test_params_reach_the_simulator():
    env = lunarlander.make_env({"gravity": -3.0, "wind_power": 12.0, "turbulence_power": 1.0})
    u = env.unwrapped
    assert u.gravity == -3.0 and u.enable_wind and u.wind_power == 12.0

def test_play_is_reproducible_from_its_start():
    p = np.array([[-10.0, 15.0, 1.5]]); s0 = lunarlander.sample_starts(p[0], 1, np.random.default_rng(3))
    a = grade(lunarlander, agent, p, s0, extras={}); b = grade(lunarlander, agent, p, s0, extras={})
    assert a[0].tolist() == b[0].tolist() and a[1].tolist() == b[1].tolist()

def test_scenic_box_matches_bounds(): ...   # covered by test_envs.py once registered
```

- [ ] **Step 3: Run; expect FAIL** (`ModuleNotFoundError: acl_bench.envs.lunarlander`)

Run: `python -m pytest tests/test_lunarlander.py -v`

- [ ] **Step 4: Implement** the module and the `.scenic` box (one `VerifaiRange` per bound), and register it.

- [ ] **Step 5: Run; expect PASS:** `python -m pytest tests -q`

- [ ] **Step 6: Commit** `git commit -m "LunarLander env: Gymnasium Box2D pool behind the batched env contract; gravity/wind/turbulence box"`

### Task 3: FrozenLake environment (random maps, slip, egocentric view)

**Files:**
- Create: `acl_bench/envs/frozenlake.py`, `acl_bench/envs/frozenlake.scenic`
- Modify: `acl_bench/envs/__init__.py:21` (add `"frozenlake"`)
- Test: `tests/test_frozenlake.py`

**Interfaces:**
- Produces: `PARAM_BOUNDS = {"size": (4, 10), "hole_density": (0.1, 0.3), "slip": (0.0, 0.67), "layout_seed": (0, 1_000_000)}`, `DEFAULTS = {"size": 8, "hole_density": 0.2, "slip": 0.67}` (`layout_seed` has no default; the custom section uses fixed seeds), `GOAL = "reach"`, `MAX_EPISODE_STEPS = 100`, `VIEW = 5`, `MAX_SIZE = 15`. Also `layout(size, hole_density, layout_seed) -> (holes: bool[size, size], start, goal)`, `certify_exact(params, s0) -> bool` (BFS: a hole-free path exists), `optimal_success(params) -> float` (value iteration with the slip model, horizon `MAX_EPISODE_STEPS`), `held_out_questions()` (Gymnasium's classic 4x4 and 8x8 maps and a single-safe-path 8x8 map, as `(params, s0)` rows), `valid(params) -> bool`.

Batched NumPy, in `maze.py`'s style. The state row is `[x, y, t, size, play_seed, goal_x, goal_y, 225 hole cells]` (`MAX_SIZE^2`, padded). `size` is rounded to an int. The start is the top-left cell and the goal the bottom-right; the generator clears both. Slip: with probability `slip` the move goes to one of the two perpendicular directions (equal odds), as in Gymnasium (where 0.67 is its slippery ice). Slip randomness is a deterministic hash of `(play_seed, t)`, so plays reproduce from `s0` and `step` needs no RNG argument. The observation is the 5x5 window centred on the agent, with hole, goal and off-map channels (`OBS_DIM = 3 * 25`). Falling in a hole terminates with no reward. Reaching the goal terminates with reward `1 - 0.9 t / MAX_EPISODE_STEPS`.

- [ ] **Step 1: Write the failing tests**

```python
def test_start_and_goal_are_never_holes():
    for seed in range(200):
        holes, start, goal = frozenlake.layout(10, 0.3, seed)
        assert not holes[start] and not holes[goal] and start != goal

def test_no_slip_matches_deterministic_moves():
    # slip 0: action "right" from (0,0) on an empty 4x4 map lands on (1,0)

def test_optimal_success_on_gymnasium_4x4_slippery():
    # Gymnasium's classic 4x4 map at slip 2/3 has a known optimal success of about 0.74 (horizon 100)
    p, _ = frozenlake.held_out_questions()["4x4"]
    assert abs(frozenlake.optimal_success(p) - 0.74) < 0.02

def test_unsolvable_map_is_not_certified():
    # a hand-built map with the goal walled off by holes: certify_exact is False, optimal_success is 0

def test_play_is_reproducible_from_its_start(): ...   # as in Task 2
```

Before relying on the 0.74 value, confirm it by running value iteration on Gymnasium's own `FrozenLake-v1` transition table (`env.unwrapped.P`) and use that number.

- [ ] **Step 2: Run; expect FAIL** (module missing): `python -m pytest tests/test_frozenlake.py -v`
- [ ] **Step 3: Implement** the module and the `.scenic` box, and register it.
- [ ] **Step 4: Run; expect PASS:** `python -m pytest tests -q`
- [ ] **Step 5: Commit** `git commit -m "FrozenLake env: random maps, slip as in Gymnasium, egocentric 5x5 view, BFS + value-iteration checks"`

### Task 4: Custom exam section

**Files:**
- Create: `acl_bench/suite/__init__.py`, `acl_bench/suite/custom.py`
- Test: `tests/test_suite_custom.py`

**Interfaces:**
- Consumes: each env's `PARAM_BOUNDS`, `DEFAULTS`, `valid`, `held_out_questions` (FrozenLake).
- Produces: `custom_cases(env) -> list[tuple[str, dict]]`, labels like `"default"` or `"force:max+50%"`; `custom_suite(env, starts=10, seed=...) -> PairSet` named `"custom"`, with case labels saved in the manifest `extra`.

Offsets are `0, +10%, +50%` of `hi - lo` past each edge. CartPole `start spread` and FrozenLake `layout_seed` are not swept: FrozenLake custom cases use 3 fixed layout seeds per case, and only cases whose maps pass `certify_exact` and `optimal_success >= 0.5` are kept. FrozenLake also gets its `held_out_questions` maps and the beyond-range values from the spec (size 11 and 15; hole density 0.32, 0.4, 0.08, 0.0; slip 0.73, 0.9).

- [ ] **Step 1: Write the failing tests**

```python
def test_cartpole_has_default_plus_six_per_swept_setting():
    labels = [l for l, _ in custom_cases(cartpole)]
    assert labels[0] == "default" and len(labels) <= 1 + 4 * 6     # start spread not swept
    assert "force_mag:max+50%" in labels

def test_beyond_values_are_offsets_of_the_range():
    p = dict(custom_cases(cartpole))["force_mag:max+10%"]
    assert p["force_mag"] == pytest.approx(16 + 0.1 * 12)

def test_invalid_beyond_values_are_skipped():
    labels = [l for l, _ in custom_cases(lunarlander)]
    assert "gravity:min+10%" not in labels        # -13 is outside Gymnasium's (-12, 0)
    assert "wind_power:min+10%" not in labels     # negative wind

def test_frozenlake_custom_maps_are_all_solvable():
    s = custom_suite(frozenlake)
    assert all(frozenlake.optimal_success(p) >= 0.5 for p in np.unique(s.params, axis=0))
```

- [ ] **Step 2: Run; expect FAIL** (module missing): `python -m pytest tests/test_suite_custom.py -v`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run; expect PASS**
- [ ] **Step 5: Commit** `git commit -m "Suite: custom exam section (default, edges, +10% and +50% beyond, invalid values skipped)"`

### Task 5: Random and VerifAI sections with per-question pass rates

**Files:**
- Create: `acl_bench/suite/exam.py`
- Test: `tests/test_suite_exam.py`

**Interfaces:**
- Consumes: `acl_bench/exam/search.py` `search(env, sampler_name, agents, n_iters, starts_per_task, rng)`, `acl_bench/exam/sets.py` (`PairSet`, `save_sets`, `load_sets`), Task 3 (`certify_exact`, `optimal_success`), Task 4.
- Produces: `exam_dir(env_name) -> Path` (`frozen_sets/suite/<env>`); `random_section(env, n_questions=300, starts=10, seed) -> PairSet`; `verifai_section(env, agents, n_iters, starts=10, min_fail_agents=3, seed) -> PairSet`; `build_exam(env_name, ref_agents, n_iters, seed)`, which writes all three sections with `save_sets`; `question_scores(env, agent, pairset) -> np.ndarray` (per-question pass rate over its 10 plays); CLI `python -m acl_bench.suite.exam --env X --iters N --seed S`.

The VerifAI section runs `search(..., "ce", ...)` separately against each of the 10 reference agents with the same `n_iters` (equal budget per agent), pools the visited tasks, and computes each task's pass rate per reference agent over its 10 starts. A task is kept when at least `min_fail_agents` agents pass fewer than 5 of 10. FrozenLake then applies the path check and the `optimal_success >= 0.5` filter. The random section uses no solvability filter on CartPole and LunarLander.

- [ ] **Step 1: Write the failing tests**

```python
def test_question_kept_only_when_three_of_ten_agents_fail():
    # 10 fake agents (fixed pass/fail per task); task A failed by 3 -> kept, task B by 2 -> dropped

def test_agent_fails_a_question_below_five_of_ten_plays():
    # fake agent passing exactly 4 of 10 starts counts as failing; 5 of 10 counts as passing

def test_random_section_is_unfiltered_on_continuous_tasks():
    s = random_section(cartpole, n_questions=50, starts=10, seed=0)
    assert len(s.params) == 50 * 10

def test_exam_round_trips_with_checksums(tmp_path): ...   # save_sets -> load_sets, digests equal
```

- [ ] **Step 2: Run; expect FAIL:** `python -m pytest tests/test_suite_exam.py -v`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run; expect PASS**
- [ ] **Step 5: Commit** `git commit -m "Suite: random and DR-falsified VerifAI sections with per-question pass rates (3 of 10 agents, below 5 of 10 plays)"`

### Task 6: Arms and runner

**Files:**
- Create: `acl_bench/suite/arms.py`, `acl_bench/suite/run.py`
- Test: `tests/test_suite_run.py`

**Interfaces:**
- Consumes: `acl_bench/plr/fast.py` (`FastConfig`, `train`, `evaluate`), `acl_bench/plr/levels.py` (`LevelConfig`), `acl_bench/study/safety.py` (`run_jobs`, `Watchdog`, `Limits`), `acl_bench/snapshots.py` (`save_run`, `load_run`), Task 5 (`load_sets(exam_dir(env))`).
- Produces: `ARMS: dict[str, LevelConfig]` with keys `DR`, `PLR`, `RPLR`, `ACCEL`, `SFL`, `REF` (REF = the DR config on separate seeds); `BUDGET: dict[str, int]` (steps per run per env, filled in Task 7); CLI `python -m acl_bench.suite.run --env X --arms DR PLR ... --seeds 1-20 --workers N --allow-battery --resume [--steps S]`. It writes `results/suite/<env>/scores.csv`, one row per (arm, seed, section) with columns `arm, seed, section, n_questions, pass_rate, steps, scouted_steps, wall_time_sec`, and snapshots to `results/suite/<env>/snapshots/<arm>/<seed>.npz`.

Arm settings (record each in the pre-registration):
- DR: `replay_prob=0`.
- PLR: `replay_prob=0.8, score="maxmc", beta=0.3, staleness=0.3, buffer=1000` (JaxUED's Maze defaults; buffer at the existing classic-task size).
- RPLR: PLR + `robust=True`.
- ACCEL: RPLR + `accel=True` (`mut_sigma` as in the existing `accel` arm).
- SFL: the existing `SFL` config with `charge_scouting=True`.

If `maxmc` misbehaves on a task's reward scale (for example LunarLander's negative returns), fall back to `pvl` for that task only and record it.

- [ ] **Step 1: Write the failing tests**

```python
def test_smoke_run_writes_one_row_per_section(tmp_path):
    # run.py on cartpole, arms DR SFL, seeds 1-2, steps 4096, exam built from a tiny fixture
    rows = read_csv(tmp_path / "scores.csv")
    assert {(r["arm"], r["seed"], r["section"]) for r in rows} == {
        (a, s, sec) for a in ("DR", "SFL") for s in ("1", "2") for sec in ("random", "verifai", "custom")}

def test_sfl_scouting_counts_against_the_budget(tmp_path):
    # SFL's steps + scouted_steps <= the step budget DR gets

def test_resume_neither_duplicates_nor_drops_rows(tmp_path):
    # run seeds 1-2, then rerun seeds 1-3 with --resume: exactly 3 seeds x 2 arms x 3 sections rows

def test_saved_agent_grades_the_same_after_reload(tmp_path):
    # question_scores before saving == question_scores after load_run
```

- [ ] **Step 2: Run; expect FAIL:** `python -m pytest tests/test_suite_run.py -v`
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run; expect PASS**
- [ ] **Step 5: Commit** `git commit -m "Suite: arms (DR, PLR, RPLR, ACCEL, SFL charged, REF) and safety-guarded runner writing per-section scores"`

### Task 7: Calibrate training lengths and time the whole part

Operational: no new code beyond filling `BUDGET`.

- [ ] **Step 1: CartPole budget.** Take the training length the earlier study calibrated (`results/cartpole/calibration/`). Put it in `BUDGET["cartpole"]`.
- [ ] **Step 2: LunarLander and FrozenLake budgets.** Run 3 DR seeds on each at a generous length with check-ins (`--steps` large, 10 check-ins), using a temporary random section only. The budget is the first check-in after which the random-section score rises less than 0.02 over the next two. Put the values in `BUDGET`.
- [ ] **Step 3: Time it.** From the Step 2 wall times, estimate total classic-task time: 3 tasks × (10 REF + 5 arms × 20 seeds) runs, plus exam search, at the worker count the Mac sustains. **If the estimate is above 12 hours, stop and report to the user** with the numbers and options (fewer seeds, fewer reference agents).
- [ ] **Step 4: Commit** `git commit -m "Suite: calibrated budgets (cartpole, lunarlander, frozenlake) and timing estimate"`, with the timing estimate in the commit message.

### Task 8: Pre-registration

- [ ] **Step 1: Append batch 21 to `docs/library_log.md`**, "registered 2026-10-28 (or the actual date), before running". It contains:
  - The question: does any UED method beat DR on the DR-falsified VerifAI section?
  - Arms and settings from Task 6, budgets from Task 7, seeds (20 scored, 10 reference per task).
  - The exam rules from Global Constraints.
  - **Primary test:** for each method × task (4 × 3 = 12), a one-sided Mann-Whitney U test across seeds of the method's VerifAI-section pass rate > DR's, Holm-corrected across the 12 at α = 0.05.
  - **Decision:** any Holm-significant improvement → follow-up testing on the random and custom sections (designed then). None → this line is dropped early.
  - Also reported, not tested: random and custom section scores per method; each custom case against the best method; custom cases no agent passes, flagged as possibly impossible.
- [ ] **Step 2: Commit** `git commit -m "Pre-register batch 21: DR-falsified VerifAI screen on CartPole, LunarLander, FrozenLake"`

### Task 9: Build the exams

Operational.

- [ ] **Step 1: Train 10 REF agents per task:** `python -m acl_bench.suite.run --env <task> --arms REF --seeds 1-10 --allow-battery --resume` (REF seeds are hashed apart from scored seeds by arm name, as in `run_seed`).
- [ ] **Step 2: Build each exam:** `python -m acl_bench.suite.exam --env <task> --iters <n> --seed 20261028`, with `<n>` chosen in Task 7 to fit the time share.
- [ ] **Step 3: Verify.** `load_sets(exam_dir(task))` loads all three sections with checksums. Report each section's question count, and the VerifAI section's REF fail rates as a sanity check (every kept question is failed by at least 3 reference agents).
- [ ] **Step 4: Commit the frozen exams** (`frozen_sets/suite/`) `git commit -m "Suite: frozen exams for cartpole, lunarlander, frozenlake"`

### Task 10: Scored runs

Operational. Starts only after Task 8 is committed.

- [ ] **Step 1: Run** `python -m acl_bench.suite.run --env <task> --arms DR PLR RPLR ACCEL SFL --seeds 1-20 --workers <from Task 7> --allow-battery --resume` for each task, in the background, checking in about every 30 minutes.
- [ ] **Step 2: Verify completeness:** each `scores.csv` has 5 arms × 20 seeds × 3 sections rows and no failed jobs in the log.
- [ ] **Step 3: Commit the results** (`results/suite/*/scores.csv`; snapshots stay uncommitted, as before).

### Task 11: Analysis and report

**Files:**
- Create: `acl_bench/suite/analyze.py`
- Test: `tests/test_suite_analyze.py`

**Interfaces:**
- Consumes: `results/suite/<env>/scores.csv`, per-case custom scores (from `question_scores` over the saved agents).
- Produces: `screen(scores) -> DataFrame` (method, task, U, p, Holm p, improved: bool); `table(scores) -> DataFrame` (method × section, mean and spread over seeds); `custom_breakdown(env)` (per case: each method's mean, the best method, a flag when no agent passes); CLI `python -m acl_bench.suite.analyze` writing `results/suite/analysis/`.

- [ ] **Step 1: Write the failing tests**

```python
def test_screen_flags_a_clear_improvement_and_not_a_tie():
    # synthetic: method A = DR + 0.2 on verifai for 20 seeds -> improved; method B identical to DR -> not

def test_holm_is_applied_across_all_twelve_comparisons():
    # p-values [0.01, 0.04, ...] -> Holm-adjusted values computed by hand

def test_custom_case_no_agent_passes_is_flagged(): ...
```

- [ ] **Step 2: Run; expect FAIL**
- [ ] **Step 3: Implement** (`scipy.stats.mannwhitneyu(alternative="greater")`; Holm by hand or with `statsmodels` if already installed)
- [ ] **Step 4: Run; expect PASS**, then run the CLI on the real results
- [ ] **Step 5: Report to the user:** the screen decision (improved or dropped), the methods × sections tables per task, and the flagged custom cases. Append the result to batch 21 in `docs/library_log.md`.
- [ ] **Step 6: Commit** `git commit -m "Suite: analysis, batch 21 screen result"`
