# ACL x VerifAI on CartPole

Does **automatic curriculum learning** help an RL agent generalize across task variants,
and does it help more when paired with a **VerifAI sampler** that picks new tasks by
where the agent is still learning? An ablation on CartPole with five physical task
parameters (pole length and mass, cart mass, push force, start range), sampled through
Scenic and VerifAI.

**Result:** the **cross-entropy picker improves final performance** on the general
section: +0.055 success over plain training (0.961 vs 0.906, p = 0.0003; Holm-adjusted
p = 0.013 across all 39 primary tests), and +0.036 when combined with replay (p = 0.019). Replay
alone, the bandit picker and the annealing picker had no detectable effect, and no
method improved hard-question success. Feedback for both components is
learning-potential (PVL). Details in [docs/ablation.md](docs/ablation.md);
interactive report (private until shared): https://claude.ai/artifact/TCcHBgEY9kiA8oDq4kdsNm

## What is compared

| method | picker (new tasks) | review pile (replay) |
|---|---|---|
| N | random | off |
| A | random | on |
| S_ce / S_mab / S_sa | cross-entropy / bandit / simulated annealing | off |
| B_ce / B_mab / B_sa | same three | on |

- **Review pile (ACL):** SIPACL's prioritized replay, `P(i) ~ 1/rank` by a
  learning-potential score (positive value loss). Differences from SIPACL, including
  truncation and sampler feedback: [docs/sipacl.md](docs/sipacl.md).
- **Picker:** a VerifAI sampler behind Scenic's external-parameter API. **Feedback is the
  same positive value loss** that ranks replay: after each new task, `rho = -z(PVL)/3`,
  so the picker seeks tasks with high learning potential, *not* tasks the agent fails.
  Failure-seeking feedback is not tested here.
- **Exam:** every run is graded on a locked set of questions: a general section, a hard
  section, an easy section and four edge sections. The hard ones were *found* by
  searching for failures with the samplers and *proven* winnable with a planner and
  interval reachability: [docs/exam.md](docs/exam.md).

## Layout

```
acl_bench/                 training
  cartpole.py, .scenic     the task-parameterized environment and its sampled task box
  sampling.py              Scenic/VerifAI samplers (random, ce, mab, sa)
  scoring.py               task scores: pvl_gae (SIPACL), neg_return
  acl.py                   the review pile (prioritized replay) and sampler feedback
  ppo.py                   CleanRL-style PPO driven by the review pile
  snapshots.py             save/load an agent at every check-in
acl_bench/exam/            the exam
  grader.py                batched physics + deterministic rollouts
  sets.py                  locked sections: save, load (checksum-verified), grade
  search.py                samplers hunting for questions the reference agents fail
  feasibility.py           proofs: winnable (replayed plan) / impossible (interval reachability)
  build.py                 search pools -> E6, E7, edge sections
acl_bench/study/           the ablation
  arms.py                  the 8 methods + reference agents
  run.py                   train arms in parallel (independent seeds, safeguards)
  regrade.py               grade saved snapshots on the exam
  compare.py               group comparison, Holm / Benjamini-Hochberg, checkpoint grid
  analyze.py               the analysis -> results/analysis/
  safety.py                pause on battery, heat, low memory or disk
frozen_sets/cartpole/      the locked exam (E0, E1v-E4v, E6, E7)
frozen_sets/search/        raw search pools the exam was built from
results/training.csv       training log: E0 at every check-in, episodes, wall time
results/grades.csv         every run graded on the full exam at 10 check-ins
results/analysis/          per-run metrics, per-method summary, test tables, report data
results/calibration/       evidence for the training length and learning rate
results/snapshots/         saved agents (not committed; ~1 GB)
docs/                      ablation.md (study + results), exam.md, sipacl.md
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m acl_bench.study.run --arms N A B_ce --seeds 1-100 \
    --out results/training.csv --snapshots results/snapshots   # train (resumable: --resume)
python -m acl_bench.exam.search --snapshots results/snapshots   # search pools (needs REF runs)
python -m acl_bench.exam.build                                  # exam sections from the pools
python -m acl_bench.study.regrade --snapshots results/snapshots \
    --n-checkpoints 10 --workers 8 --out results/grades.csv     # grade every saved run
python -m acl_bench.study.analyze                               # tests + tables
```

Every command is `python -m <module>` because the code is one package whose modules
import each other as `acl_bench.<...>`. `-m` runs a module by its import name from the
repository root, so those imports resolve; `python acl_bench/study/run.py` would put
`acl_bench/study/` on the path instead and fail with `No module named 'acl_bench'`.

Long runs pause themselves on battery, heat, low memory or low disk; `touch results/STOP`
stops training cleanly (`results/STOP_REGRADE` for grading).

## Tests

```bash
python -m pytest tests                     # everything (~1 min)
python -m pytest tests -v                  # one line per test, pass/fail
python -m pytest tests/test_grader.py      # one file
python -m pytest tests -k feasibility      # tests whose name matches
python -m pytest tests -x --pdb            # stop at the first failure, open a debugger there
python -m pytest tests --collect-only -q   # list what would run, without running it
```

pytest finds every `tests/test_*.py`, runs each `test_*` function in it, and builds
anything a test names as an argument from the `@pytest.fixture` of that name. `-s`
shows `print` output; `--durations=5` shows the slowest tests.

| file | checks | code under test |
|---|---|---|
| `test_grader.py` | batched physics equals gymnasium's CartPole step by step; batched rollouts equal a one-at-a-time loop | `exam/grader.py` |
| `test_sets.py` | the locked exam: checksums, sizes, rules per section, no overlap, samples proven winnable | `frozen_sets/cartpole`, `exam/sets.py` |
| `test_feasibility.py` | proofs are sound: interval step contains the true next state; certificates replay; impossible starts are caught | `exam/feasibility.py` |
| `test_search_build.py` | the search records every question with the right fail share; cluster naming | `exam/search.py`, `exam/build.py` |
| `test_sipacl_fidelity.py` | replay ranking, smoothing, placeholder and PVL match SIPACL's transcribed code | `acl.py`, `scoring.py` |
| `test_seeding.py` | a seed fully determines a run; arms get independent seeds | `ppo.py`, `study/run.py` |
| `test_snapshots.py` | saved agents regrade to exactly the live grades; checkpoint grid | `snapshots.py`, `study/regrade.py` |
| `test_compare.py` | group comparison, Holm and Benjamini-Hochberg against textbook values and scipy | `study/compare.py` |
| `test_analyze.py` | the 13 comparisons; edge-section discovery; the checkpoint grid refuses gaps | `study/analyze.py` |
| `test_safety.py` | the watchdog pauses on each hazard; failed jobs are skipped, not fatal | `study/safety.py` |
