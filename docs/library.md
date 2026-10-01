# The curriculum library (`acl_bench/curriculum`)

A wrapper that decides which task each training episode gets, built from parts that can
be mixed and matched. Protocol: `docs/protocol.md`; every arm tried: `docs/library_log.md`.

## Contract

- **Environment-agnostic.** A component sees the box of task parameters, each episode's
  task and outcome (pass/fail, return, length), and nothing that names an environment.
  VerifAI's samplers are reached through the environment's Scenic file, the one place
  where a task space is declared.
- **May be model-specific.** A component may read the learner: per-step values of its
  critic now; gradients and parameters in later batches.
- **Budget-fair.** Any simulation a component runs to choose tasks is added to
  `Curriculum.sim_steps` and charged to the training budget. All batch-1 parts use none.

## Parts

| part | kind | what it does |
|---|---|---|
| `TaskSpace` | core | the box; uniform draws; normalization |
| `Episode` | core | what every part learns from |
| `Curriculum` | core | picks a proposer per task (fixed mixture), reports every episode to all parts, returns an importance weight per episode |
| `PassModel` | estimator | MLP pass-rate model p(task) over the latest 4,096 training episodes, refit every 256 (the offline screen's best signal) |
| `ProgressModel` | estimator | learning progress \|p_now - p_2-refits-ago\| from the pass model's own history (model-based ALP-GMM), 1,024-episode window |
| `EnsembleModel` | estimator | 4 pass models on Poisson bootstraps; score = mean p(1-p) + std of p (exploration where models disagree) |
| `GradSignal` | estimator (model-specific) | per-episode policy-gradient size or alignment through the actor's last layer, regressed over tasks; failed its mechanism check (not predictable from the task, `docs/library_log.md` batch 2) |
| `Uniform` | proposer | domain randomization |
| `SIR` | proposer | sampling-importance-resampling: draw 64 uniform tasks, pick one with probability proportional to (p(1-p) + 0.01)^alpha |
| `Replay` | proposer | PLR-style buffer of visited tasks, replayed in proportion to posterior learnability^alpha |
| `Mutate` | proposer | ACCEL-style: Gaussian edit (5% of each range) of a task `Replay` would pick |
| `VerifAISurrogate` | proposer | a VerifAI sampler (ce / mab / sa through Scenic) whose feedback is the pass model's learnability, so it takes several search steps per task at no simulation cost |

Arms are specs in `acl_bench.curriculum.ARMS`; each is also a config of the fast harness
(`python -m acl_bench.plr.screen --configs <arm>`).

## The maths behind the new parts

**SIR density.** With M uniform candidates and weights s^alpha, the chosen task's density
tends to q(x) = s(x)^alpha / Z, Z = E_uniform[s^alpha] (relative to uniform), as M grows.
`SIR.density` estimates Z from 4,096 uniform draws at each model refit
(`tests/test_curriculum.py` checks the density against the samples).

**Tempered importance correction.** A curriculum q changes the objective PPO optimizes,
from DR's E_uniform[J(x)] to E_q[J(x)]. Weighting each episode's policy loss by
w = (1 / q_mix(x))^lambda, q_mix = sum_i m_i q_i, gives:
- lambda = 1: an unbiased estimate of DR's gradient. The curriculum can then change only
  the *variance* of the gradient, not what is learned: harm is ruled out to first order.
- lambda = 0: the plain curriculum (SFL-style bias toward the frontier).
- in between: a controlled amount of bias.
Mixing in uniform with share m_u bounds every weight by 1 / m_u.

**Why sqrt(p(1-p)).** For a fixed estimator budget, the proposal minimizing the variance
of an importance-sampled mean of g(x) is q* ~ uniform(x) ||g(x)|| (the classic optimal
importance-sampling result). With a pass/fail outcome and a baseline that knows the task,
the size of a task's policy-gradient contribution scales with the outcome's standard
deviation, sqrt(p(1-p)). So `sir_is` (alpha = 0.5, lambda = 1) is the variance-optimal way
to learn DR's objective, if that scaling holds. It does not hold exactly with a critic
that cannot see the task (its advantages on easy tasks are not zero; `docs/plr.md`), which
is what the model-specific gradient-norm proposer of a later batch measures directly.

**Posterior learnability.** A buffer task with w wins in n visits and model prediction
p0 gets Beta(a, b), a = m p0 + w, b = m (1 - p0) + n - w (m = 2 pseudo-visits), and score
E[p(1-p)] = ab / ((a+b)(a+b+1)). The model settles tasks with few visits; the task's own
visits take over as they accumulate (hierarchical shrinkage). Visits are discounted by
0.8 each time, to follow the changing policy.

**Reweighting instead of resampling.** `reweight_alpha` keeps the proposed tasks and
weights each episode's policy loss by s(x)^alpha / Z. In expectation this is the gradient
SIR would give, but the agent still sees every region of the box as often as under DR.

## The confirmed method: `sfl_tilt` (batch 10, stage C 2026-10-01)

**In one paragraph.** Every 10 PPO updates, draw 1,000 levels from the environment's own
level distribution and roll each out 8 times with the current stochastic policy (batched,
outside the training budget). Estimate each level's success rate p from the environment's
own success test, score it p(1-p)(1-p) (SFL's learnability, tilted toward harder levels by an
extra factor (1-p), i.e. a Bernoulli variance weighted by the failure rate), and keep the top
100. Each training episode then plays one of these 100 levels with probability 0.5 and a fresh
random level otherwise. Nothing names an environment: the only interface is "sample a level"
and "did the episode succeed", so it is environment-agnostic and model-agnostic (episode
outcomes only; no critic, gradients or parameters).

- Code: `acl_bench/plr/fast.py` (`_scout`, score line `p * (1 - p) * (1 - p) ** lc.sfl_tilt`),
  config `LevelConfig(sfl=True, sfl_tilt=1.0, replay_prob=0.5)` in `acl_bench/plr/levels.py`;
  arm `sfl_tilt` in `acl_bench/plr/screen.py`. It is not yet ported into `acl_bench/curriculum/`.
- Evidence: stage C on seeds 1101-1200, CartPole hard suite test half +0.102 (Holm p < 0.001),
  every random-suite guard inside -20% of DR (`docs/library_log.md`, batch 10).
- Why the tilt helps: plain SFL's p(1-p) is symmetric, so it practises easy-but-unreliable
  levels as much as hard-but-learnable ones; the extra (1-p) moves mass toward the failure
  side where the hard suite lives, while the 50% random share keeps random-suite competence
  (var_low's tilt without that anchor cost guards).
- Cost: about 115-150M scouted steps per run (CartPole, PointNav), reported, not charged.
