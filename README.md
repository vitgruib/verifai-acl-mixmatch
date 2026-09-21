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

Each run trains a PPO agent (two 64-unit tanh layers for actor and critic;
lr 3e-4, gamma 0.99, GAE lambda 0.95, clip 0.2, entropy 0.01; 1024-step
rollouts, 4 minibatches, 4 epochs), then scores the final policy on a fixed
held-out set of 15 tasks x 3 episodes drawn uniformly from the parameter box,
independent of whichever sampler trained it.

## How this relates to SIPACL (what matches and what doesn't)

SIPACL trains a PPO driving policy in Scenic + MetaDrive with a Prioritized
Level Replay buffer of scenes. This repo reuses its structure on cheaper
environments; it is a re-implementation, not a fork. Checked against SIPACL's
`Work/custom/custom_gym.py`:

**Matches SIPACL:** rank-based replay `P(i) ~ 1/rank^alpha` with SIPACL's
`alpha=1.0`, stable tie-breaking by buffer index, EMA smoothing of the per-visit
score with SIPACL's `beta=0.2` (first visit skips the EMA), a `1e10` placeholder
so new slots rank first, buffer size 5000, replay probability 0.5, and PVL =
mean(max(GAE advantage, 0)) over the episode. ACL off corresponds to SIPACL's
`replay_resample_prob = -1`.

**Deliberately differs:**
- The buffer holds task parameters in memory; SIPACL pickles Scenic scenes to disk.
- SIPACL treats the last step of every episode as terminal (next value 0). Here a
  time-limit *truncation* bootstraps from the critic, which matters once policies
  survive to the step cap. All scoring functions share this convention.
- **Sampler feedback.** SIPACL does route feedback into Scenic
  (`scenario.generate(feedback=...)`), but the value is `feedback_fn(simulation.result)`,
  an identity function `ppo.py` never overrides, and the default sampler ignores it.
  PVL only drives replay. Here one scoring function feeds both replay ranking and
  the sampler (as `rho = -z(score)`), and feedback is delivered only for tasks the
  sampler itself proposed, not replays. That coupling is this repo's design choice.
- PPO uses the same network and core hyperparameters but smaller rollout/update
  settings than SIPACL's (4096 steps, 32 minibatches, 10 epochs).

## Samplers

[VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) ships eight sampler
types. All of them go through **Scenic** (`param x = VerifaiRange(...)` in a
`.scenic` file, sampled by `scenario.generate(feedback=rho)`), the same mechanism
as SIPACL's `scenic.scenarioFromFile(..., params={"verifaiSamplerType": ...})`,
just with no MetaDrive/CARLA model attached. Each was tested directly:

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
| `init_range` | 0.05-0.3 | 1.24 / 0.13 | worse when larger | matters |
| `force_mag` | 4-16 | 0.04 / 1.09 | better when larger | matters |
| `masscart` | 0.5-2.0 | 0.03 / 0.67 | worse when larger | matters |
| `length` | 0.25-1.5 | -0.00 / 0.09 | inconsistent | weak |
| `masspole` | 0.05-0.5 | 0.01 / 0.00 | none | inert |

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

python -m acl_bench.timing_report     # compute breakdown + signal-vs-noise tables
python -m acl_bench.plot_results      # heatmap + 2x2 ablation charts
python -m acl_bench.plot_convergence  # learning curves
```

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
results/                                     (empty; outputs of the scripts above)
```
