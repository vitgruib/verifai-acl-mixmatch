"""Render results/grid_results.csv into the two charts used in the README:

  1. A sequential-blue heatmap per environment (own color scale -- reward
     scales aren't comparable across environments), sampler x
     potential-function-condition (the 6 named functions plus "none", the
     ACL-off ablation), averaged over seeds.
  2. A 2x2 ablation summary per environment: {non-adaptive sampler, adaptive
     sampler} x {ACL off ("none"), ACL on (avg of the 6 potential functions)}
     -- directly isolating "not using one or both" of the two components.

Produces light- and dark-mode PNGs so the README can pick the right one via
a <picture> tag, per the palette in the dataviz skill (references/palette.md).
"""
from __future__ import annotations

import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- palette (see dataviz skill references/palette.md) ---
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ABLATION_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a"),
}

POTENTIAL_ORDER = ["none", "pvl_gae", "l1_value_loss", "max_mc", "td_error_l2", "alp",
                    "intermediate_difficulty"]
# "bo" only appears in the first grid (replaced by "sa"); kept so old results still render.
SAMPLER_ORDER = ["random", "halton", "ce", "mab", "sa", "bo"]
NON_ADAPTIVE = {"random", "halton"}
ADAPTIVE = {"ce", "mab", "sa", "bo"}
ENV_ORDER = ["cartpole", "acrobot", "pendulum"]


def style(ax, theme):
    ax.set_facecolor(theme["surface"])
    ax.figure.set_facecolor(theme["surface"])
    ax.tick_params(colors=theme["secondary"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(theme["grid"])
    ax.title.set_color(theme["primary"])
    ax.xaxis.label.set_color(theme["secondary"])
    ax.yaxis.label.set_color(theme["secondary"])


def plot_heatmap(df: pd.DataFrame, theme_name: str, out_path: str):
    theme = THEMES[theme_name]
    envs = [e for e in ENV_ORDER if e in df["env"].unique()]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    fig, axes = plt.subplots(len(envs), 1, figsize=(10, 3.4 * len(envs)), dpi=160)
    axes = np.atleast_1d(axes)

    for ax, env in zip(axes, envs):
        sub = df[df["env"] == env]
        samplers = [x for x in SAMPLER_ORDER if x in sub["sampler"].unique()]
        pivot = (
            sub.groupby(["sampler", "potential_fn"])["eval_return_mean"]
            .mean()
            .reindex(pd.MultiIndex.from_product([samplers, POTENTIAL_ORDER]))
            .unstack()
            .reindex(index=samplers, columns=POTENTIAL_ORDER)
        )
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(POTENTIAL_ORDER)))
        ax.set_xticklabels(POTENTIAL_ORDER, rotation=25, ha="right")
        ax.set_yticks(range(len(samplers)))
        ax.set_yticklabels(samplers)
        ax.set_title(f"{env} -- mean held-out eval return (own color scale)",
                     fontsize=11, pad=10)

        vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isnan(v):
                    continue
                frac = (v - vmin) / (vmax - vmin + 1e-9)
                label_color = theme["primary"] if frac < 0.6 else theme["surface"]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=label_color, fontsize=9)

        style(ax, theme)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color=theme["secondary"], labelcolor=theme["secondary"])
        cbar.outline.set_edgecolor(theme["grid"])

    fig.tight_layout()
    fig.savefig(out_path, facecolor=theme["surface"])
    plt.close(fig)


def plot_ablation_bars(df: pd.DataFrame, theme_name: str, out_path: str):
    """2x2 per env: {non-adaptive, adaptive} sampler x {ACL off, ACL on}."""
    theme = THEMES[theme_name]
    envs = [e for e in ENV_ORDER if e in df["env"].unique()]
    conditions = ["neither (baseline)", "adaptive sampler only", "ACL only",
                  "both (sampler + ACL)"]

    def cell(sub, adaptive: bool, acl: bool):
        samplers = ADAPTIVE if adaptive else NON_ADAPTIVE
        mask = sub["sampler"].isin(samplers)
        mask &= (sub["potential_fn"] != "none") if acl else (sub["potential_fn"] == "none")
        vals = sub.loc[mask, "eval_return_mean"]
        return vals.mean(), vals.std()

    fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 4.6), dpi=160)
    axes = np.atleast_1d(axes)

    for ax, env in zip(axes, envs):
        sub = df[df["env"] == env]
        means, stds = zip(
            cell(sub, False, False), cell(sub, True, False),
            cell(sub, False, True), cell(sub, True, True),
        )
        x = np.arange(4)
        # Dots + error bars, not bars: Acrobot's returns are negative, and a bar
        # hanging from 0 on a truncated axis implies a baseline that isn't meaningful.
        for xi, m, s, color in zip(x, means, stds, ABLATION_COLORS):
            ax.errorbar(xi, m, yerr=s, fmt="o", markersize=9, capsize=5, linewidth=1.6,
                        color=color, ecolor=theme["secondary"], markeredgecolor=theme["surface"],
                        markeredgewidth=1.5)
        ax.margins(x=0.15)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8.5)
        ax.set_title(env, fontsize=11)
        ax.grid(True, axis="y", color=theme["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax, theme)
        if ax is axes[0]:
            ax.set_ylabel("held-out eval return (mean +/- std)")

    fig.suptitle("Ablation: not using one or both components", fontsize=12, color=theme["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, facecolor=theme["surface"])
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/grid_results.csv"
    df = pd.read_csv(path)
    for theme_name in ("light", "dark"):
        plot_heatmap(df, theme_name, f"results/heatmap_{theme_name}.png")
        plot_ablation_bars(df, theme_name, f"results/ablation_{theme_name}.png")
    print("wrote results/heatmap_{light,dark}.png and results/ablation_{light,dark}.png")


if __name__ == "__main__":
    main()
