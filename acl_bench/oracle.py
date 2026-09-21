"""Ground-truth solvability for the parameterized CartPole.

The learned policy failing on a task can mean two very different things: the
policy is weak there (fixable -- what a curriculum should target), or the task
is unsolvable (an infeasible start, or too little actuator authority -- not
fixable, and a falsification sampler will chase it forever). This module labels
which is which with a model-based controller:

    LQR on the linearization of CartPole's own dynamics, executed bang-bang
    (CartPole's action is a fixed-magnitude push left/right), run in the real
    environment from the exact initial state.

`survival_steps == max_steps` is a *sufficient* condition for solvable: if this
controller balances the pole, a policy exists. Failure of this controller is
only evidence of unsolvability (a smarter controller might still succeed), so
callers should treat "oracle fails" as "not known solvable", not "impossible".
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from acl_bench.envs.param_cartpole import make_env

G = 9.8
TAU = 0.02
THETA_THRESHOLD = 12 * 2 * np.pi / 360   # CartPoleEnv's failure angle (0.2095 rad)
X_THRESHOLD = 2.4
_Q = np.diag([1.0, 1.0, 10.0, 1.0])
_R = np.array([[0.1]])


def lqr_gain(params: dict) -> np.ndarray:
    """Discrete LQR gain for CartPole's Euler-integrated dynamics, linearized
    at the upright equilibrium (state = [x, x_dot, theta, theta_dot])."""
    m_p, m_c, length = params["masspole"], params["masscart"], params["length"]
    total = m_p + m_c
    pml = m_p * length
    l_eff = length * (4.0 / 3.0 - m_p / total)
    a_theta = G / l_eff
    b_theta = -1.0 / (l_eff * total)
    a_x = -pml * G / (total * l_eff)
    b_x = 1.0 / total + pml / (total ** 2 * l_eff)

    a = np.eye(4) + TAU * np.array([[0, 1, 0, 0], [0, 0, a_x, 0], [0, 0, 0, 1], [0, 0, a_theta, 0]])
    b = TAU * np.array([[0.0], [b_x], [0.0], [b_theta]])
    p = solve_discrete_are(a, b, _Q, _R)
    return np.linalg.solve(_R + b.T @ p @ b, b.T @ p @ a)


def survival_steps(params: dict, s0: np.ndarray, max_steps: int = 500) -> int:
    """Steps the LQR bang-bang controller keeps the pole up, starting from the
    exact state `s0`, in the real (nonlinear) parameterized environment."""
    env = make_env(params)
    env.reset(seed=0)
    env.state = np.array(s0, dtype=np.float64)
    gain = lqr_gain(params)
    for step in range(1, max_steps + 1):
        action = 1 if float((-gain @ env.state)[0]) > 0 else 0
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            return step
    return max_steps


def infeasible_start(s0: np.ndarray) -> bool:
    """True if the episode is guaranteed to end on the very first step whatever
    the policy does. CartPole's Euler step updates position from the *old*
    velocity, so the state whose bounds are checked after step 1 is
    (x0 + tau*x_dot0, theta0 + tau*theta_dot0), independent of the action. A
    pole that starts past the failure angle is therefore not automatically lost:
    velocity pointing back inside can rescue it."""
    x1 = s0[0] + TAU * s0[1]
    theta1 = s0[2] + TAU * s0[3]
    return bool(abs(theta1) > THETA_THRESHOLD or abs(x1) > X_THRESHOLD)
