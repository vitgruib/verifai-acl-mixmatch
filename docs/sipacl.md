# How this repo's ACL relates to SIPACL

This is a re-implementation, not a fork. The only thing carried over from
[SIPACL](https://github.com/vitgruib/SIPACL) is the ACL logic in
`Work/custom/custom_gym.py` (`MetaDriveEnv`). PPO is a standard CleanRL-style
implementation (SIPACL's `Work/policy/ppo.py` was read only to see how `MetaDriveEnv` is
driven). Samplers go through Scenic's own documented external-sampler API.
`tests/test_sipacl_fidelity.py` transcribes SIPACL's `_replay_probs_from_lp`,
`_compute_learning_progress` and `_lp_delta` and checks this repo's code against them.

## Side by side

| behavior | SIPACL | here | status |
|---|---|---|---|
| replay ranking | `P(i) ~ 1/rank^alpha`, alpha 1.0, ties to the lower index | same | matches (tested) |
| score smoothing | EMA, beta 0.2; first visit uses the raw score | same | matches (tested) |
| new-slot placeholder | `1e10`, so unscored slots rank first | same | matches (tested) |
| buffer | 5,000, FIFO eviction | same | matches (tested) |
| replay probability | 0.5; `-1` disables replay | 0.5; ACL off disables it | matches (tested) |
| replay episodes | do not call the sampler | same | matches (tested) |
| score formula (PVL) | mean(max(GAE advantage, 0)); recursion carries the unclipped advantage | same | matches (tested) |
| **what a replay repeats** | the exact Scenic scene, pickled to disk (setup and starting state) | the task **parameters** only; the episode starts from a fresh random state within `init_range` | **differs**: a replay here is "same physics, new start" |
| **score span** | T-1 steps: `logScores()` runs inside `step()` before PPO logs the final step, so the last step is left out and step T-2 is treated as terminal (in MetaDrive the last step carries the crash penalty) | all T steps | **differs; looks like an off-by-one in SIPACL** |
| **truncation** (hitting the step cap) | treated as terminal everywhere: the score and PPO both use next value 0 | the **score** bootstraps from the critic, `V(s_T)`; **PPO's update** still treats it as terminal, like SIPACL and CleanRL | **differs**, and inconsistent inside this repo (below) |
| **sampler feedback** | `feedback_fn(simulation.result)`, identity by default (a Scenic result object, not a number), set only on termination; the scenarios use `verifaiSamplerType = 'halton'`, which ignores feedback | after every NEW episode, the task score becomes `rho` (below); ce, mab and sa steer by it | **differs**: SIPACL never steered a sampler |
| structure | ACL lives inside a `ScenicGymEnv` subclass | a standalone `PLRCurriculum` (`acl_bench/acl.py`) driven by the training loop | deliberate |
| aborted episodes | an episode aborted by `ResetException` leaves its slot at `1e10`, replayed first | not applicable: episodes always finish | n/a |

## Truncation, in detail

A CartPole episode ends either by *termination* (the pole falls or the cart leaves the
track) or by *truncation* (it survives to the 500-step cap). After a truncation the pole
is still up, so reward would keep coming.

- **SIPACL** treats both as the end of the world. For an episode that survives to the
  cap, the last step's target is `r` instead of `r + gamma * V(s_T)`. With a critic that
  expects about 1/(1-gamma) of future reward, that is a large negative surprise that the
  GAE recursion carries back through roughly the last 1/(1 - gamma * lambda), about 20,
  steps. Those negative advantages are clipped to zero, so a fully successful episode
  scores a little lower than it otherwise would.
- **Here**, the replay score bootstraps: `next_value = V(s_T)` on truncation, 0 on
  termination (`acl_bench/ppo.py`), so a survived episode shows no artificial surprise
  at the cap.
- **But the PPO update here does not bootstrap** (`b_dones = terminated or truncated`,
  as in CleanRL). So the curriculum's score and the learner disagree about truncation.
  The effect on CartPole is small, and changing PPO now would make new runs
  incomparable with the 800 already trained, so it is documented rather than fixed.

## Sampler feedback, in detail

`acl_bench/acl.py`, `PLRCurriculum.report_episode`:

1. Only after an episode on a **newly drawn** task (never a replay), the task's score
   (`pvl_gae` for the A, S and B arms) is taken.
2. It is z-scored against every feedback value so far in the run (Welford running mean
   and SD), clipped to +-3, divided by 3 and negated: `rho = -clip(z, -3, 3) / 3`, in [-1, 1].
3. It is delivered on the sampler's next draw (`scenario.generate(feedback=rho)`), which
   VerifAI pairs with the task it last proposed.

VerifAI's convention is that low `rho` is a counterexample, a region worth sampling more
of. So:

- **ce** and **mab** (threshold 0) treat a task whose score is above the run's average
  so far as a counterexample and shift probability toward its bucket.
- **sa** accepts a proposed task if its `rho` is lower than the current one's, or
  otherwise with probability `exp((rho_old - rho_new) / T)`.

The mapping (z-score, clip, sign) is this repo's own; SIPACL has nothing to copy here.
