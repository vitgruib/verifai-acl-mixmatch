"""Task scores: how worth revisiting a task is, computed from one finished episode.

The score has up to two consumers (acl_bench.acl): it ranks tasks for replay when ACL
is on, and it becomes the feedback an adaptive sampler steers by. Higher always means
"more worth revisiting".

  - `pvl_gae` ports SIPACL's `_lp_delta` (SIPACL/Work/custom/custom_gym.py): Positive
    Value Loss, mean(max(GAE advantage, 0)) over the episode (Schulman et al., 2016;
    Jiang et al., Prioritized Level Replay, 2021). The GAE recursion carries the raw
    advantage; only the averaged output is clipped. One deliberate difference from
    SIPACL: a time-limit truncation bootstraps from the critic (`next_value`) instead
    of treating the last step as terminal (docs/sipacl.md).
  - `neg_return` is not a learning-progress signal: minus the episode return, so worse
    performance scores higher (hard-example mining). It is the reference arm's score,
    which no component consumes (random sampler, ACL off).

Contract: `fn(rewards, values, next_value, gamma, gae_lambda) -> float`.
"""
from __future__ import annotations

import numpy as np


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


def pvl_gae(rewards, values, next_value, gamma, gae_lambda) -> float:
    adv = _gae(np.asarray(rewards), np.asarray(values), next_value, gamma, gae_lambda)
    return float(np.mean(np.clip(adv, 0.0, None)))


def neg_return(rewards, values, next_value, gamma, gae_lambda) -> float:
    return -float(np.sum(rewards))


SCORES = {"pvl_gae": pvl_gae, "neg_return": neg_return}
