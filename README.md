# verifai-acl-mixmatch

Mix-and-match testing of three things that can each be switched on, off, or
swapped, on fast RL environments whose task parameters are sampled through
Scenic and VerifAI:

1. **A VerifAI sampler** that proposes new tasks (random, Halton, cross-entropy,
   multi-armed bandit, simulated annealing).
2. **ACL**: prioritized replay of already-seen tasks, ported from
   [vitgruib/SIPACL](https://github.com/vitgruib/SIPACL) (Prioritized Level Replay).
3. **A scoring function** that says how worth revisiting a task is (SIPACL's
   GAE-based Positive Value Loss, five alternatives from the literature, and a
   raw-return baseline).

**Status: no results yet.** An earlier round of runs was deleted because the
code that produced it had three bugs (see "What was wrong before" below). The
framework is ready to run; nothing in this repo yet supports a claim about
which combination works best.

## The factors, and the grid

| factor | values |
|---|---|
| environment | `cartpole`, `acrobot` |
| sampler | `random`, `halton` (non-adaptive); `ce`, `mab`, `sa` (adaptive) |
| ACL | `off` (never replay), `on` (replay with probability 0.5) |
| scoring function | `pvl_gae`, `l1_value_loss`, `max_mc`, `td_error_l2`, `alp`, `intermediate_difficulty`, `neg_return` |

The scoring function has up to two consumers: it **ranks tasks for replay**
(only if ACL is on) and, z-scored and negated into `rho`, it is **the feedback
an adaptive sampler steers by** (only if the sampler is adaptive). A
non-adaptive sampler with ACL off has neither consumer, so the function cannot
matter there and that cell runs once, labeled `unused`, instead of seven
identical times. That gives 2 + 14 + 21 + 21 = **58 cells per environment**.

"Not using one or both" is therefore built into the grid rather than bolted on:

| | ACL off | ACL on |
|---|---|---|
| **non-adaptive sampler** (`random`/`halton`) | neither component | ACL only |
| **adaptive sampler** (`ce`/`mab`/`sa`) | adaptive sampler only | both |

Each run trains a standard CleanRL-style PPO agent (two 64-unit tanh layers for actor and critic;
lr 3e-4, gamma 0.99, GAE lambda 0.95, clip 0.2, entropy 0.01; 1024-step
rollouts, 4 minibatches, 4 epochs), then scores the final policy on a fixed
held-out set of 15 tasks x 3 episodes drawn uniformly from the parameter box,
independent of whichever sampler trained it.

## How this relates to SIPACL

This is a re-implementation, not a fork, and **the only thing carried over from
SIPACL is the ACL-carrying gym code**: `Work/custom/custom_gym.py` (`MetaDriveEnv`)
and the Scenic gym base class it extends (`Scenic/src/scenic/gym/envs/scenic_gym.py`,
`ScenicGymEnv`). Nothing comes from `Work/policy/ppo.py`, the scenarios or the
controllers; PPO here is a standard CleanRL-style implementation, and the
Scenic/VerifAI sampling uses Scenic's own documented API. (`ppo.py` was read only
to see how `MetaDriveEnv` is driven, which is how the PVL off-by-one below was
found.) `tests/test_sipacl_fidelity.py` transcribes SIPACL's
`_replay_probs_from_lp`, `_compute_learning_progress` and `_lp_delta` as the
reference and checks this repo's against them (perturbing a constant makes the
tests fail, so they are not vacuous).

| behavior | SIPACL | here | status |
|---|---|---|---|
| replay ranking | `P(i) ~ 1/rank^alpha`, alpha = 1.0, stable sort so ties go to the lower index | same | matches (tested) |
| score smoothing | EMA beta = 0.2; first visit uses the raw score | same | matches (tested) |
| new-slot placeholder | `1e10`, so unscored slots rank first | same | matches (tested) |
| buffer | 5000, FIFO; scenes pickled to disk | 5000, FIFO; task parameters in memory | storage differs |
| replay probability | 0.5; `-1` disables replay | 0.5; ACL off | matches (tested) |
| replay episodes | do not call the sampler | same | matches (tested) |
| PVL formula | mean(max(GAE, 0)); last step terminal | same when the next value is 0 | matches (tested) |
| **PVL episode span** | **T-1 steps.** `logScores()` fires inside `step()` before PPO's `log_step_data` for the final step, so the last step is left out and step T-2 is treated as terminal. In MetaDrive the final step carries the crash penalty, the biggest surprise | all T steps | **deliberate difference; looks like an off-by-one in SIPACL** |
| truncation | treated as terminal | bootstraps from the critic | differs |
| sampler feedback | `feedback_fn(simulation.result)` (identity; `ppo.py` never overrides it), set only on true termination and delivered at the next `generate()` even if the last episode was a replay | one scoring function; delivered only for tasks the sampler itself proposed | design choice |
| evaluation | no replay | fixed held-out task set | differs by design |
| aborted episodes | a slot whose episode is aborted by `ResetException` keeps LP `1e10` and is replayed first | not applicable: episodes always run to completion | n/a |
| `DOUBLE` mode | constant declared, never used | not implemented | n/a |
| PPO | not carried over | independent CleanRL-style PPO (1024-step rollouts, 4 minibatches, 4 epochs) | n/a |

The off-by-one is worth checking in SIPACL itself: it changes what PVL measures
whenever the last step is the informative one.

## Samplers

[VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) ships eight sampler
types. All of them go through **Scenic** (`param x = VerifaiRange(...)` in a
`.scenic` file, sampled by `scenario.generate(feedback=rho)`), which is Scenic's own
external-sampler API (`scenic.scenarioFromFile(..., params={"verifaiSamplerType":
...})`) with no simulator model attached. Each was tested directly:

| sampler | in grid? | why |
|---|---|---|
| `random` | yes | i.i.d. uniform; ignores feedback |
| `halton` | yes | quasi-random low-discrepancy sequence; ignores feedback |
| `ce` (cross-entropy) | yes | refits a distribution toward low-`rho` regions |
| `mab` (multi-armed bandit) | yes | UCB1 over discretized buckets, toward low `rho` |
| `sa` (simulated annealing) | yes | single-chain local search that cools over time. `verifai.server.choose_sampler` has no `'sa'` branch, so it is injected through Scenic's documented `externalSampler` parameter (`SimulatedAnnealingSampler` in `scenic_sampling.py`) and still uses the same `generate(feedback=...)` loop |
| `bo` (Bayesian optimization) | no | works, but its GP refit grew with task history and consumed 83% of an earlier grid's compute (mean 227s per CartPole run vs ~5s for the others), and needs `GPyOpt`+`GPy` plus a `setuptools<81` pin |
| `eg` (epsilon-greedy) | no | raises `NotImplementedError: tried to use abstract BoxSampler` in the installed VerifAI release |
| `grid` | no | exhaustive by design (terminates once covered; 58 ms/sample even before that); wrong tool for an open-ended training loop |

VerifAI's convention is that **low `rho` means a counterexample**, i.e. a
region worth sampling more. Every scoring function here is oriented so that a
high score means "worth revisiting", so the sampler receives `rho = -z(score)`
with `z` a running z-score (Welford). This also removes any per-environment
reward-scale constant.

## Scoring functions

| function | idea | source |
|---|---|---|
| `pvl_gae` (SIPACL's current) | mean(max(GAE advantage, 0)): the critic is still surprised upward | Schulman et al. 2016; SIPACL `_lp_delta` |
| `l1_value_loss` | mean(\|GAE advantage\|): surprise in either direction | Jiang et al., Prioritized Level Replay, ICML 2021 |
| `max_mc` | mean(max(Monte-Carlo return - V(s), 0)): actual return, not a bootstrapped target | Jiang et al., Robust PLR, NeurIPS 2021 |
| `td_error_l2` | mean squared one-step TD error | Schaul et al., Prioritized Experience Replay, ICLR 2016 |
| `alp` | \|return now - EMA of returns on this task\|: change, not level | Portelas, Colas et al., ALP-GMM, CoRL 2020 |
| `intermediate_difficulty` | `1 - 2\|success_rate - 0.5\|`, peaking at 50% success; "solved" is per-environment (`EnvSpec.success_return`) | Florensa et al. 2017; Wang et al., POET, 2019; Du et al., VACL, 2022 |
| `neg_return` | **baseline, not learning progress**: minus the episode return, so worse performance scores higher. Classic falsification / hard-example mining | -- |

## Environments

Both are Gymnasium `classic_control` (no Box2D, no MuJoCo dependency), with
physical parameters exposed as VerifAI-sampled task variables. Episodes truncate
at 500 steps (the raw classes have no cap outside their registered `-v1`
wrapper, so the training loop applies it).

**Do the task parameters matter?** Measured with `acl_bench/param_sensitivity.py`:
one policy per seed trained under plain domain randomization (random sampler, ACL
off; 400k steps CartPole, 250k Acrobot), evaluated on 300 Halton-sampled tasks x 6
episodes, then a random forest on the parameters (5-fold out-of-sample) with
permutation importance. Two seeds per environment.

*CartPole* -- parameters explain out-of-sample R^2 = 0.61 / 0.74 (seed 1 / seed 2);
the policy fails (mean return < 195) on 15% / 2% of tasks:

| parameter | bounds | importance (seed 1 / 2) | effect on return | verdict |
|---|---|---|---|---|
| `init_range` | 0.05-0.3 | 1.24 / 0.13 | worse when larger | matters for raw return, but see below |
| `force_mag` | 4-16 | 0.04 / 1.09 | better when larger | matters |
| `masscart` | 0.5-2.0 | 0.03 / 0.67 | worse when larger | matters |
| `length` | 0.25-1.5 | -0.00 / 0.09 | inconsistent | weak |
| `masspole` | 0.05-0.5 | 0.01 / 0.00 | none | inert |

A later oracle probe (see [docs/cartpole_suite.md](docs/cartpole_suite.md)) found
that ~5% of uniformly sampled starts are guaranteed one-step failures, concentrated
at large `init_range`, and that on oracle-solvable pairs `init_range` barely
changes the policy's failure rate (10.0% to 12.9% across quartiles) while
`force_mag` (37.1% to 0.9%) and `masscart` (1.7% to 22.3%) dominate. So
`init_range`'s importance above mostly reflects unrecoverable starts, not policy
weakness. This is one policy and one seed.

*Acrobot* -- R^2 = 0.44 / 0.51 (noise ceilings 0.74 / 0.64); the policy solves
every sampled task at 250k steps, but return still ranges from about -150 to -60:

| parameter | bounds | importance (seed 1 / 2) | effect on return | verdict |
|---|---|---|---|---|
| `link_length_1` | 0.5-1.5 | 0.59 / 0.59 | worse when longer | dominant |
| `link_moi` | 0.5-1.5 | 0.17 / 0.20 | worse when larger | matters |
| `link_mass_2` | 0.5-1.5 | 0.13 / 0.30 | worse when heavier | matters |
| `link_length_2` | 0.5-1.5 | 0.06 / 0.04 | none clear | weak |
| `link_mass_1` | 0.5-1.5 | 0.02 / 0.03 | slightly worse | weak |
| `torque_noise_max` | 0-0.3 | -0.00 / -0.01 | none | inert |

Caveats: importance is the drop in held-out R^2 when a parameter is shuffled, so
values can exceed 1 (a shuffled model can be worse than predicting the mean) and
should be read as a ranking, not a share of variance. Where replicate
evaluations disagree (CartPole seed 1, r = 0.31) the noise-ceiling estimate is
itself unreliable -- there R^2 (0.61) exceeds the "ceiling" (0.48). The two
CartPole policies differ a lot in quality (mean return 251 vs 465) and in which
parameters bind (seed 1: `init_range`; seed 2: `force_mag` and `masscart`), so
the difficulty landscape depends on what the policy has learned. Acrobot's
ranking is stable across both. Two policies and one budget per environment is a
thin basis; `masspole` and `torque_noise_max` look inert but are still sampled.

**Considered and not included.** Speed is the top constraint, so each candidate
was benchmarked with a random policy on this machine (Apple silicon, Python 3.14).
Our training loop is capped near 11k steps/s by the per-step policy forward
pass, so anything faster than that is not the bottleneck:

| candidate | raw steps/s | outcome |
|---|---|---|
| MiniGrid (DoorKey, LavaGap, MultiRoom, KeyCorridor, Dynamic-Obstacles) | 15,000-23,000 | fast and installs; layouts are naturally placeable by Scenic. Not yet built or tested; untested risk is that sparse reward and partial observability make an MLP policy slow to train |
| MuJoCo (Hopper, Walker2d, HalfCheetah) | 17,000-39,000 | fast, but its diversity would be smooth physics parameters like CartPole's, and it needs millions of steps |
| highway-env | 13-210 | too slow |
| Pendulum | (fast) | dropped: did not learn in 60k steps and was still improving at 3M under the shared hyperparameters |
| LunarLander (Box2D) | not measured | will not build without `swig` |
| Procgen | not measured | no distribution for arm64 + Python 3.14; its levels are opaque integer seeds, so VerifAI samplers would have nothing to search |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# one cell: environment, sampler, scoring function, ACL on/off, seed
python -c "
from acl_bench.experiment import run_one, make_fixed_eval_set
from acl_bench.ppo import PPOConfig
print(run_one('acrobot', 'mab', 'max_mc', True, 1,
              PPOConfig(total_timesteps=100_000), make_fixed_eval_set('acrobot')))
"

# the full grid (58 cells per environment x seeds)
python -m acl_bench.experiment --seeds 1 2 3 --total-timesteps 100000

# how many steps does an environment need? (long runs with periodic evaluation)
python -m acl_bench.convergence --env cartpole --seed 1 --total-timesteps 1000000 \
    --out results/convergence/cartpole_s1.csv

# do an environment's task parameters matter?
python -m acl_bench.param_sensitivity --env cartpole --total-timesteps 400000

python -m pytest tests                # ACL fidelity vs. SIPACL + oracle checks
python -m acl_bench.timing_report     # compute breakdown + signal-vs-noise tables
python -m acl_bench.plot_results      # heatmap + 2x2 ablation charts
python -m acl_bench.plot_convergence  # learning curves
```

## Test suite

A design for a single-environment suite (CartPole) lives in
[docs/cartpole_suite.md](docs/cartpole_suite.md): an LQR oracle
(`acl_bench/oracle.py`) that separates fixable failures from infeasible starts,
frozen evaluation sets for specific edge cases (weak actuation, heavy cart, large
recoverable disturbances, corners) and for measured hard tasks, falsification-based
evaluation with the VerifAI samplers, convergence-speed metrics, and a staged
protocol (calibrate, screen, confirm on disjoint seeds). **Only the oracle is
built**; the evaluation sets, metrics and runners are not.

## TODO / future work

- **Build the CartPole suite** above, starting with a *batched* evaluator (stepping
  many task pairs in lockstep with one batched policy forward pass); naive
  evaluation would cost as much as training.
- **MiniGrid** (Farama). Measured 15,000-23,000 raw steps/s on DoorKey, LavaGap,
  MultiRoom, KeyCorridor and Dynamic-Obstacles, and it installs. Plan: subclass an
  environment whose `_gen_grid` places the key, door, goal, lava and obstacles at
  Scenic-sampled `VerifaiRange` positions (object placement is what Scenic is for),
  plus grid size and obstacle count. Edge cases to falsify: a key behind its own
  locked door or an unreachable goal, a goal next to lava, narrow gaps. A BFS solver
  would give exact solvability, the analogue of the CartPole LQR oracle. Not yet
  done: parameter sensitivity, a convergence check (sparse reward and partial
  observability may make an MLP policy slow to train; `FullyObsWrapper` may be
  needed). Unsolvable layouts will look like permanent counterexamples to
  falsification samplers, which is where `intermediate_difficulty` may matter.
- **MuJoCo** (Hopper, Walker2d, HalfCheetah). Measured 17,000-39,000 raw steps/s;
  `pip install mujoco` works. Plan: randomize physics (friction, mass, gravity,
  damping) through the model attributes, with falls as the falsifiable failure.
  Needs continuous actions, so the retained but currently unexercised continuous
  PPO head must be tested first. Caveats: the parameter space is smooth like
  CartPole's, and these typically need millions of steps, so run a convergence check
  before committing a budget.
- **SIPACL parity switch**: an option to reproduce SIPACL's exact PVL span (T-1
  steps, last step terminal) for bit-for-bit comparison, if that is wanted.

## What was wrong before

The deleted round of results should not be trusted, for three concrete reasons
found on review:

1. **Inverted sampler feedback in the "sampler only" cells.** With the potential
   function off, the sampler was fed `-z(raw return)`, so a *high* return produced a
   *low* `rho`: adaptive samplers were steered toward easy tasks, the opposite of
   falsification. Fixed by making the raw-return baseline `neg_return` (worse
   performance scores higher, like every other function).
2. **`intermediate_difficulty` was constant on Acrobot.** Its "solved" threshold was
   hard-coded to CartPole's 195, but Acrobot's returns are all negative, so it never
   registered a success. Now per-environment.
3. **The ACL constants and claims did not match SIPACL.** The code used rank alpha
   0.9, EMA beta 0.5 and a 200-task buffer while the README said "exactly as SIPACL"
   (which uses 1.0, 0.2 and 5000). ACL was also folded into the scoring-function axis
   as a `none` value, so it was not a separate factor and could not be crossed with
   the others. Both fixed above.

Separately, PPO steps-to-plateau estimates from an earlier convergence run (CartPole
~400k steps, Acrobot ~200k, Pendulum not converged at 3M) came from the old code;
the plateaus should be re-checked before fixing a budget.

## Layout

```
acl_bench/
  envs/param_cartpole.py, param_acrobot.py   task-parameterized gym envs
  oracle.py                                  CartPole solvability oracle (LQR) for the test suite
  envs/registry.py                           env specs (bounds, step cap, success return, .scenic file)
  scenic_scenarios/*.scenic                  VerifaiRange-declared task parameters, one per env
  scenic_sampling.py                         Scenic-mediated samplers + the generate()/feedback loop
  potential/functions.py                     the 7 scoring functions
  curriculum/plr.py                          ACL: prioritized replay buffer + sampler feedback coupling
  ppo.py                                     PPO training loop
  experiment.py                              env x sampler x ACL x function x seeds grid
  convergence.py                             long runs with periodic held-out evaluation
  param_sensitivity.py                       do the task parameters matter?
  timing_report.py, plot_results.py, plot_convergence.py
tests/                                       SIPACL-fidelity and oracle tests
docs/cartpole_suite.md                       test-suite design
results/                                     (empty; outputs of the scripts above)
```
