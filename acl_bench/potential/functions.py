"""Learning-potential / feedback functions for prioritized replay.

`pvl_gae` is a direct port of SIPACL's `_lp_delta` (SIPACL/Work/custom/custom_gym.py):
mean(max(GAE_delta, 0)) over an episode, i.e. Positive Value Loss (PVL) computed
from Generalized Advantage Estimation (Schulman et al., 2016). The rest are
alternative "how worth replaying is this task" signals pulled from the
curriculum-learning / Unsupervised Environment Design literature:

  - l1_value_loss, max_mc         : Jiang, Grefenstette & Rocktaschel,
                                     "Prioritized Level Replay" (ICML 2021) and
                                     Jiang et al., "Replay-Guided Adversarial
                                     Environment Design" (Robust PLR, NeurIPS 2021)
                                     introduce GAE/PVL, L1 value loss, and MaxMC
                                     as interchangeable score functions for the
                                     same replay buffer -- exactly the slot PVL
                                     fills in SIPACL's MetaDriveEnv.
  - td_error_l2                   : plain squared one-step TD error, the
                                     classical "surprise" signal behind
                                     prioritized experience replay
                                     (Schaul et al., 2016).
  - absolute_learning_progress    : ALP-GMM (Portelas, Colas et al., 2020),
                                     "Teacher algorithms for curriculum
                                     learning of Deep RL in continuously
                                     parameterized environments" -- score a
                                     task by how much the return on it has
                                     recently *changed*, not by its absolute
                                     value.
  - intermediate_difficulty       : Zone-of-proximal-development /
                                     regret-style curricula (Florensa et al.,
                                     "Reverse Curriculum Generation", 2017;
                                     Wang et al., POET, 2019; Du et al., VACL,
                                     2022) reward tasks whose success rate is
                                     near 50%: neither trivially solved nor
                                     hopeless.

Every function has the same contract: given one episode's rollout and a
mutable per-task `slot_state` dict (scratch memory the buffer keeps next to
each task), return `(raw_score, slot_state)`. The buffer then EMA-smooths
`raw_score` across visits exactly as SIPACL's `_compute_learning_progress`
does, regardless of which function produced it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

EMA_ALPHA = 0.5  # smoothing for functions that keep their own running stats


def _gae(rewards: np.ndarray, values: np.ndarray, next_value: float,
         gamma: float, gae_lambda: float) -> np.ndarray:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < T else next_value
        delta = rewards[t] + gamma * v_next - values[t]
        lastgaelam = delta + gamma * gae_lambda * lastgaelam
        advantages[t] = lastgaelam
    return advantages


def pvl_gae(rewards, values, next_value, gamma, gae_lambda, slot_state):
    """SIPACL's current default: mean(max(GAE_delta, 0))."""
    adv = _gae(np.asarray(rewards), np.asarray(values), next_value, gamma, gae_lambda)
    return float(np.mean(np.clip(adv, 0.0, None))), slot_state


def l1_value_loss(rewards, values, next_value, gamma, gae_lambda, slot_state):
    """PLR's L1 value loss: mean(|GAE_delta|) -- penalizes over- and
    under-estimation alike, instead of only "still improving" surprises."""
    adv = _gae(np.asarray(rewards), np.asarray(values), next_value, gamma, gae_lambda)
    return float(np.mean(np.abs(adv))), slot_state


def max_mc(rewards, values, next_value, gamma, gae_lambda, slot_state):
    """Robust PLR's MaxMC: mean(max(MonteCarlo_return - V(s), 0)), using the
    actual discounted return-to-go instead of the bootstrapped GAE target.
    Less sensitive to a miscalibrated critic than PVL."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    T = len(rewards)
    mc_returns = np.zeros(T, dtype=np.float64)
    running = next_value
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        mc_returns[t] = running
    return float(np.mean(np.clip(mc_returns - values, 0.0, None))), slot_state


def td_error_l2(rewards, values, next_value, gamma, gae_lambda, slot_state):
    """Mean squared one-step TD error -- the classical "surprise" signal from
    prioritized experience replay (Schaul et al., 2016), applied per-episode."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    T = len(rewards)
    next_values = np.empty(T, dtype=np.float64)
    next_values[:-1] = values[1:]
    next_values[-1] = next_value
    delta = rewards + gamma * next_values - values
    return float(np.mean(delta ** 2)), slot_state


def absolute_learning_progress(rewards, values, next_value, gamma, gae_lambda, slot_state):
    """ALP-GMM (Portelas et al., 2020): |return_now - EMA(return_on_this_task)|."""
    episode_return = float(np.sum(rewards))
    prev_ema = slot_state.get("ema_return")
    if prev_ema is None:
        score = 0.0  # no history yet -- first visit carries no "progress" signal
    else:
        score = abs(episode_return - prev_ema)
    new_ema = episode_return if prev_ema is None else (
        EMA_ALPHA * episode_return + (1 - EMA_ALPHA) * prev_ema
    )
    slot_state = {**slot_state, "ema_return": new_ema}
    return score, slot_state


def make_intermediate_difficulty(success_return_threshold: float) -> Callable:
    """Factory: closes over the env-specific "solved" return threshold."""

    def intermediate_difficulty(rewards, values, next_value, gamma, gae_lambda, slot_state):
        episode_return = float(np.sum(rewards))
        success = 1.0 if episode_return >= success_return_threshold else 0.0
        prev_rate = slot_state.get("success_rate")
        new_rate = success if prev_rate is None else (
            EMA_ALPHA * success + (1 - EMA_ALPHA) * prev_rate
        )
        score = 1.0 - 2.0 * abs(new_rate - 0.5)  # peaks at rate == 0.5
        slot_state = {**slot_state, "success_rate": new_rate}
        return float(score), slot_state

    return intermediate_difficulty


# CartPole-v1-style episodes cap at 500 steps/reward=1 per step; treat "solved"
# as comfortably above the classic v0 bar (195) but below the hard cap.
POTENTIAL_FUNCTIONS: dict[str, Callable] = {
    "pvl_gae": pvl_gae,
    "l1_value_loss": l1_value_loss,
    "max_mc": max_mc,
    "td_error_l2": td_error_l2,
    "alp": absolute_learning_progress,
    "intermediate_difficulty": make_intermediate_difficulty(success_return_threshold=195.0),
}

# "none" is the ACL ablation: no potential function, no replay buffer at all
# (see acl_bench.curriculum.plr.PLRCurriculum) -- isolates "sampler alone"
# from "sampler + ACL" in the mix-and-match grid.
POTENTIAL_FUNCTION_NAMES: tuple[str, ...] = ("none",) + tuple(POTENTIAL_FUNCTIONS.keys())


def resolve_potential_fn(name: str) -> Callable | None:
    return None if name == "none" else POTENTIAL_FUNCTIONS[name]
