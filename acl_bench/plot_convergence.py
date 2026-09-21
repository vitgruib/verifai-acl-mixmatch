"""Learning curves from results/convergence/*.csv: held-out eval return vs.
environment steps, mean +/- SD across seeds, one panel per environment (own
axes -- the three envs have different reward scales and step budgets), with
the grid's original 60k-step budget marked.

    python -m acl_bench.plot_convergence
"""
from __future__ import annotations

import glob

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from acl_bench.plot_results import ENV_ORDER, THEMES, style

GRID_BUDGET = 60_000
CONFIG_COLORS = {"random:none": "#2a78d6", "random:pvl_gae": "#eb6834", "sa:pvl_gae": "#1baf7a"}
CONFIG_LABELS = {"random:none": "neither (random, no ACL)", "random:pvl_gae": "ACL only (random + pvl_gae)",
                 "sa:pvl_gae": "both (sa + pvl_gae)"}


def plot(df: pd.DataFrame, theme_name: str, out_path: str):
    theme = THEMES[theme_name]
    envs = [e for e in ENV_ORDER if e in df["env"].unique()]
    fig, axes = plt.subplots(1, len(envs), figsize=(4.6 * len(envs), 4.2), dpi=160)
    axes = np.atleast_1d(axes)

    for ax, env in zip(axes, envs):
        sub = df[df["env"] == env]
        for config, color in CONFIG_COLORS.items():
            g = sub[sub["config"] == config].groupby("step")["eval_return"]
            mean, sd = g.mean(), g.std()
            ax.plot(mean.index / 1e3, mean.values, color=color, linewidth=2, label=CONFIG_LABELS[config])
            ax.fill_between(mean.index / 1e3, (mean - sd).values, (mean + sd).values,
                            color=color, alpha=0.15, linewidth=0)
        ax.axvline(GRID_BUDGET / 1e3, color=theme["muted"], linestyle="--", linewidth=1)
        ax.set_title(env, fontsize=11)
        ax.set_xlabel("environment steps (thousands)")
        ax.grid(True, color=theme["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax, theme)
        if ax is axes[0]:
            ax.set_ylabel("held-out eval return (mean +/- SD, 3 seeds)")
            ax.legend(frameon=False, labelcolor=theme["secondary"], fontsize=8, loc="lower right")
            ax.annotate("grid budget (60k)", (GRID_BUDGET / 1e3, ax.get_ylim()[0]),
                        xytext=(6, 8), textcoords="offset points", fontsize=8, color=theme["muted"])

    fig.suptitle("Convergence: how many steps does each environment need?", fontsize=12,
                 color=theme["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, facecolor=theme["surface"])
    plt.close(fig)


def main():
    df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("results/convergence/*.csv")))
    df["config"] = df["sampler"] + ":" + df["potential_fn"]
    for theme_name in ("light", "dark"):
        plot(df, theme_name, f"results/convergence_{theme_name}.png")
    print("wrote results/convergence_{light,dark}.png")


if __name__ == "__main__":
    main()
