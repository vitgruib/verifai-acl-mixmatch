import numpy as np
import pandas as pd
import torch

from acl_bench.exam.sets import PairSet, save_sets
from acl_bench.ppo import Agent
from acl_bench.snapshots import save_run
import pytest

from acl_bench.suite.analyze import (
    custom_breakdown,
    load_scores,
    require_complete,
    screen,
    table,
)

SEEDS = range(1, 21)


def scores_frame(verifai_by_arm, task="cartpole"):
    rows = []
    for arm, rates in verifai_by_arm.items():
        for seed, rate in zip(SEEDS, rates):
            rows.append(dict(task=task, arm=arm, seed=seed, section="verifai", pass_rate=rate))
            rows.append(dict(task=task, arm=arm, seed=seed, section="random", pass_rate=0.9))
    return pd.DataFrame(rows)


def test_screen_flags_a_clear_improvement_and_not_a_tie():
    base = np.linspace(0.30, 0.40, 20)
    df = scores_frame({"DR": base, "PLR": base + 0.2, "SFL": base})
    out = screen(df).set_index("method")
    assert out.loc["PLR", "improved"] and not out.loc["SFL", "improved"]
    assert out.loc["PLR", "p"] < 1e-6 and out.loc["SFL", "p"] > 0.4
    assert "DR" not in out.index
    assert (out["n_dr"] == 20).all() and (out["n_method"] == 20).all()


def test_holm_is_applied_across_every_test_run():
    # Two clear improvements (identical data, so identical p) and two ties (p near 0.5): four tests.
    # Step-down: smallest p x 4, next x 3 but kept monotone (so also 4p), ties x2 and x1 are >= 1 capped at 1.
    base = np.linspace(0.30, 0.40, 20)
    df = pd.concat([scores_frame({"DR": base, "PLR": base + 0.2, "SFL": base}, task=t)
                    for t in ("cartpole", "frozenlake")])
    out = screen(df)
    assert len(out) == 4
    clear = out[out["method"] == "PLR"]
    assert np.allclose(clear["holm_p"], 4 * clear["p"])
    assert (out[out["method"] == "SFL"]["holm_p"] == 1.0).all()


def test_table_gives_mean_std_and_seed_count_per_task_method_section():
    df = scores_frame({"DR": [0.2] * 10 + [0.4] * 10})
    out = table(df).set_index(["task", "arm", "section"])
    row = out.loc[("cartpole", "DR", "verifai")]
    assert np.isclose(row["mean"], 0.3) and np.isclose(row["std"], np.std([0.2] * 10 + [0.4] * 10, ddof=1))
    assert row["n_seeds"] == 20


def test_custom_case_no_agent_passes_is_flagged(tmp_path):
    exam = tmp_path / "exam"
    ps = PairSet("custom", np.zeros((20, 1)), np.zeros((20, 1)))
    save_sets({"custom": ps}, str(exam), extra={"custom_cases": ["easy", "impossible"]})
    snaps = tmp_path / "snapshots"
    for arm, marker in (("DR", 0.2), ("PLR", 0.9)):
        for seed in (1, 2):
            agent = Agent(2, 2)
            with torch.no_grad():
                agent.actor[4].bias[0] = marker
            save_run(str(snaps / arm / f"{seed}.npz"),
                     {10: {k: v.detach().numpy() for k, v in agent.state_dict().items()}})

    def score(env, agent, pairset):
        marker = float(agent.actor[4].bias[0].detach())
        return np.array([marker, 0.1])        # "easy" scored by marker, "impossible" never passed

    out = custom_breakdown("cartpole", snapshots=snaps, exam=exam, score=score).set_index("case")
    assert out.loc["easy", "best"] == "PLR" and not out.loc["easy", "no_agent_passes"]
    assert np.isclose(out.loc["easy", "DR"], 0.2, atol=1e-6)
    assert out.loc["impossible", "no_agent_passes"]


def test_tied_best_methods_are_all_reported(tmp_path):
    exam = tmp_path / "exam"
    save_sets({"custom": PairSet("custom", np.zeros((10, 1)), np.zeros((10, 1)))}, str(exam),
              extra={"custom_cases": ["tie"]})
    for arm in ("DR", "ACCEL", "PLR"):
        agent = Agent(2, 2)
        with torch.no_grad():
            agent.actor[4].bias[0] = 0.3 if arm == "PLR" else 0.8
        save_run(str(tmp_path / "snap" / arm / "1.npz"),
                 {10: {k: v.detach().numpy() for k, v in agent.state_dict().items()}})

    out = custom_breakdown("cartpole", snapshots=tmp_path / "snap", exam=exam,
                           score=lambda env, agent, ps: np.array([float(agent.actor[4].bias[0].detach())]))
    assert out.loc[0, "best"] == "ACCEL=DR"


def write_scores(path, drop=None, extra_rows=()):
    rows = []
    for arm in ("DR", "PLR", "RPLR", "ACCEL", "SFL"):
        for seed in SEEDS:
            for section in ("random", "verifai", "custom"):
                rows.append(dict(arm=arm, seed=seed, section=section, check=100, budget=100, pass_rate=0.5))
    df = pd.DataFrame(rows + list(extra_rows))
    if drop:
        df = df[~((df["arm"] == drop[0]) & (df["seed"] == drop[1]))]
    path.parent.mkdir(parents=True)
    df.to_csv(path, index=False)


def test_load_scores_drops_ref_and_keeps_the_final_check(tmp_path):
    extra = [dict(arm="REF", seed=1, section="verifai", check=100, budget=100, pass_rate=1.0),
             dict(arm="DR", seed=1, section="verifai", check=50, budget=100, pass_rate=0.0)]
    write_scores(tmp_path / "cartpole" / "scores.csv", extra_rows=extra)
    df = load_scores(tmp_path)
    assert "REF" not in set(df["arm"])
    assert len(df) == 5 * 20 * 3 and (df["check"] == 100).all()


def test_incomplete_scores_stop_the_analysis_naming_the_gap(tmp_path):
    write_scores(tmp_path / "cartpole" / "scores.csv", drop=("SFL", 7))
    with pytest.raises(SystemExit, match=r"cartpole.*SFL.*random.*19"):
        require_complete(load_scores(tmp_path))
