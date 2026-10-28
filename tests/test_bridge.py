"""CartPole bridge (atlas/bridge): JaxUED's PLR code trains on our CartPole, our grader grades.

The JAX tests run in .venv-jax (no torch), the torch tests in .venv (no JAX); each skips in
the other environment.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

from acl_bench.envs import cartpole as ours
from atlas.bridge import compare, export

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDS = np.array([ours.PARAM_BOUNDS[k] for k in ours.PARAM_ORDER])


def _jax_cartpole():
    jax = pytest.importorskip("jax")
    from jaxued.environments.gymnax import cartpole
    return jax, cartpole


def _box_levels(jax, cartpole, n):
    levels = jax.vmap(cartpole.make_level_generator_box())(jax.random.split(jax.random.PRNGKey(0), n))
    return levels, np.stack([np.asarray(getattr(levels, k), np.float64) for k in ours.PARAM_ORDER], axis=1)


def test_physics_matches_ours():
    jax, cartpole = _jax_cartpole()
    levels, params = _box_levels(jax, cartpole, 50)
    assert ((params >= BOUNDS[:, 0]) & (params <= BOUNDS[:, 1])).all()
    rng = np.random.default_rng(0)
    level = jax.tree_util.tree_map(lambda x: np.repeat(np.asarray(x), 200), levels)
    p = np.repeat(params, 200, axis=0)
    state = rng.uniform([-2.6, -2, -0.25, -2], [2.6, 2, 0.25, 2], size=(len(p), 4)).astype(np.float32)
    action = rng.integers(0, 2, size=len(p))
    env = cartpole.CartPole()
    js = cartpole.EnvState(*state.T, time=np.zeros(len(p), np.int32), level_params=level)
    obs, _, _, done, _ = jax.vmap(env.step_env, in_axes=(None, 0, 0, None))(
        jax.random.PRNGKey(0), js, action, env.default_params)
    expected = ours.step(state.astype(np.float64), action, p)
    np.testing.assert_allclose(np.asarray(obs), expected, atol=1e-5)
    assert np.array_equal(np.asarray(done), ours.terminated(expected))


def test_start_spread_is_the_levels_init_range():
    jax, cartpole = _jax_cartpole()
    levels, params = _box_levels(jax, cartpole, 50)
    level = jax.tree_util.tree_map(lambda x: np.repeat(np.asarray(x), 400), levels)
    env = cartpole.CartPole()
    _, st = jax.vmap(env.reset_env_to_level, in_axes=(0, 0, None))(
        jax.random.split(jax.random.PRNGKey(1), 50 * 400), level, env.default_params)
    s = np.stack([np.asarray(st.x), np.asarray(st.x_dot), np.asarray(st.theta), np.asarray(st.theta_dot)], 1)
    spread = np.abs(s).reshape(50, 400 * 4).max(axis=1)
    init_range = params[:, ours.PARAM_ORDER.index("init_range")]
    assert (spread <= init_range).all() and (spread > 0.95 * init_range).all()


def test_bridge_network_has_the_agents_layout():
    jax = pytest.importorskip("jax")
    sys.path.insert(0, os.path.join(ROOT, "third_party", "jaxued", "examples", "gymnax"))
    from gymnax_plr import ActorCriticMLP
    obs = np.random.default_rng(2).normal(size=(100, 4)).astype(np.float32)
    net = ActorCriticMLP(2)
    params = net.init(jax.random.PRNGKey(0), (obs, None), None)
    _, pi, value = net.apply(params, (obs, None), None)
    sd = export.to_state_dict(params)

    def torch_layout_forward(branch):
        h = np.tanh(obs @ sd[f"{branch}.0.weight"].T + sd[f"{branch}.0.bias"])
        h = np.tanh(h @ sd[f"{branch}.2.weight"].T + sd[f"{branch}.2.bias"])
        return h @ sd[f"{branch}.4.weight"].T + sd[f"{branch}.4.bias"]

    logits = torch_layout_forward("actor")
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))   # distrax normalizes its logits
    np.testing.assert_allclose(np.asarray(pi.logits), log_probs, atol=1e-5)
    np.testing.assert_allclose(np.asarray(value), torch_layout_forward("critic")[:, 0], atol=1e-5)


def test_export_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from atlas.bridge.grade_cartpole import load_agent
    rng = np.random.default_rng(1)
    shapes = {"actor_0": (4, 64), "actor_2": (64, 64), "actor_4": (64, 2),
              "critic_0": (4, 64), "critic_2": (64, 64), "critic_4": (64, 1)}
    p = {name: {"kernel": rng.normal(0, 0.3, size=s).astype(np.float32),
                "bias": rng.normal(0, 0.3, size=s[1]).astype(np.float32)} for name, s in shapes.items()}
    export.save(tmp_path / "params.npz", {"params": p})
    agent = load_agent(tmp_path / "params.npz")
    obs = rng.normal(size=(100, 4)).astype(np.float32)
    h = np.tanh(obs @ p["actor_0"]["kernel"] + p["actor_0"]["bias"])
    h = np.tanh(h @ p["actor_2"]["kernel"] + p["actor_2"]["bias"])
    expected = h @ p["actor_4"]["kernel"] + p["actor_4"]["bias"]
    with torch.no_grad():
        logits = agent.actor(torch.as_tensor(obs)).numpy()
    np.testing.assert_allclose(logits, expected, atol=1e-5)


def test_settings_match_our_training():
    pytest.importorskip("torch")
    from acl_bench.plr.fast import FastConfig
    from acl_bench.suite.arms import ARMS, BUDGET
    from atlas.bridge.cartpole_jaxued import ARM_FLAGS, PLR_FLAGS, PPO_FLAGS
    cfg, plr = FastConfig(), ARMS["PLR"]
    assert PPO_FLAGS == {
        "num_train_envs": cfg.n_envs, "num_steps": 1024 // cfg.n_envs, "num_minibatches": cfg.minibatches,
        "epoch_ppo": cfg.epochs, "lr": cfg.lr, "gamma": cfg.gamma, "gae_lambda": cfg.lam,
        "clip_eps": cfg.clip, "max_grad_norm": cfg.max_grad_norm, "entropy_coeff": cfg.ent,
        "critic_coeff": cfg.vf}
    assert PLR_FLAGS == {
        "score_function": "MaxMC", "level_buffer_capacity": plr.buffer, "temperature": plr.beta,
        "staleness_coeff": plr.staleness, "minimum_fill_ratio": (plr.buffer // 10) / plr.buffer,
        "prioritization": "rank"}
    assert (plr.score, plr.min_fill, plr.score_ema, plr.admit) == ("maxmc", 0, 1.0, "min")
    assert ARM_FLAGS == {
        "DR": {"replay_prob": ARMS["DR"].replay_prob, "exploratory_grad_updates": True},
        "PLR": {"replay_prob": plr.replay_prob, "exploratory_grad_updates": not plr.robust},
        "RPLR": {"replay_prob": ARMS["RPLR"].replay_prob, "exploratory_grad_updates": not ARMS["RPLR"].robust}}
    assert BUDGET["cartpole"] == 614_400


def _scores(means: dict) -> pd.DataFrame:
    """Two seeds per (arm, section) at `mean` +- 0.02, at the full budget."""
    return pd.DataFrame([{"arm": arm, "seed": seed, "section": section, "check": 614_400, "budget": 614_400,
                          "pass_rate": m + d} for (arm, section), m in means.items()
                         for seed, d in ((1, -0.02), (2, 0.02))])


OURS = {("DR", "random"): 0.94, ("PLR", "random"): 0.88, ("RPLR", "random"): 0.83,
        ("DR", "verifai"): 0.57, ("PLR", "verifai"): 0.47, ("RPLR", "verifai"): 0.47,
        ("DR", "custom"): 0.91, ("PLR", "custom"): 0.87, ("RPLR", "custom"): 0.85}


def test_compare_reading_passes_when_close_and_ordered():
    bridge = {**OURS, ("DR", "random"): 0.89, ("PLR", "verifai"): 0.52, ("DR", "verifai"): 0.62,
              ("RPLR", "custom"): 0.40}
    result = compare.reading(_scores(OURS), _scores(bridge), budget=614_400)
    assert result["verdict"] == "PASS"
    assert all(rule["pass"] for rule in result["rules"])


def test_compare_reading_fails_when_one_arm_is_off():
    bridge = {**OURS, ("RPLR", "verifai"): 0.32}
    result = compare.reading(_scores(OURS), _scores(bridge), budget=614_400)
    assert result["verdict"] == "FAIL"
    failed = [rule["name"] for rule in result["rules"] if not rule["pass"]]
    assert failed == ["RPLR verifai within 0.10"]


def test_compare_reading_fails_when_dr_is_not_best_on_verifai():
    bridge = {**OURS, ("DR", "verifai"): 0.52, ("PLR", "verifai"): 0.56}
    result = compare.reading(_scores(OURS), _scores(bridge), budget=614_400)
    assert result["verdict"] == "FAIL"
    assert [rule["name"] for rule in result["rules"] if not rule["pass"]] == ["DR > PLR on verifai"]
