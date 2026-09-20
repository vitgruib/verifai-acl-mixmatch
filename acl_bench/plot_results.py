"""Render results/grid_results.csv into the two charts used in the README:

  1. A sequential-blue heatmap of mean held-out eval return, sampler x
     learning-potential function (averaged over seeds).
  2. Training-return curves for the three samplers under the default
     potential function (pvl_gae), one categorical hue per sampler.

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
CATEGORICAL = {"random": "#2a78d6", "halton": "#eb6834", "mab": "#1baf7a"}

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a"),
}

POTENTIAL_ORDER = ["pvl_gae", "l1_value_loss", "max_mc", "td_error_l2", "alp",
                    "intermediate_difficulty"]
SAMPLER_ORDER = ["random", "halton", "mab"]


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
    pivot = (
        df.groupby(["sampler", "potential_fn"])["eval_return_mean"]
        .mean()
        .reindex(pd.MultiIndex.from_product([SAMPLER_ORDER, POTENTIAL_ORDER]))
        .unstack()
        .reindex(index=SAMPLER_ORDER, columns=POTENTIAL_ORDER)
    )

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=160)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(POTENTIAL_ORDER)))
    ax.set_xticklabels(POTENTIAL_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(len(SAMPLER_ORDER)))
    ax.set_yticklabels(SAMPLER_ORDER)
    ax.set_title("Mean held-out eval return: sampler x learning-potential function",
                 fontsize=11, pad=12)

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
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=theme["secondary"], labelcolor=theme["secondary"])
    cbar.outline.set_edgecolor(theme["grid"])
    fig.tight_layout()
    fig.savefig(out_path, facecolor=theme["surface"])
    plt.close(fig)


def plot_sampler_bars(df: pd.DataFrame, theme_name: str, out_path: str):
    """Grouped bars: mean eval return +/- std across seeds, one group per
    potential function, one bar per sampler -- the same data as the heatmap,
    with the seed-to-seed spread made explicit instead of averaged away."""
    theme = THEMES[theme_name]
    stats = (
        df.groupby(["potential_fn", "sampler"])["eval_return_mean"]
        .agg(["mean", "std"])
        .reindex(pd.MultiIndex.from_product([POTENTIAL_ORDER, SAMPLER_ORDER]))
    )

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    n_groups, n_bars = len(POTENTIAL_ORDER), len(SAMPLER_ORDER)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, sampler in enumerate(SAMPLER_ORDER):
        means = [stats.loc[(pfn, sampler), "mean"] for pfn in POTENTIAL_ORDER]
        stds = [stats.loc[(pfn, sampler), "std"] for pfn in POTENTIAL_ORDER]
        offset = (i - (n_bars - 1) / 2) * width
        ax.bar(x + offset, means, width * 0.9, yerr=stds, capsize=3,
               color=CATEGORICAL[sampler], label=sampler,
               error_kw=dict(ecolor=theme["secondary"], linewidth=1))

    ax.set_xticks(x)
    ax.set_xticklabels(POTENTIAL_ORDER, rotation=20, ha="right")
    ax.set_ylabel("held-out eval return (mean +/- std over 5 seeds)")
    ax.set_title("Eval return by potential function, grouped by sampler", fontsize=11)
    ax.grid(True, axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=theme["secondary"], ncol=3, loc="upper left")
    style(ax, theme)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=theme["surface"])
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/grid_results.csv"
    df = pd.read_csv(path)
    for theme_name in ("light", "dark"):
        plot_heatmap(df, theme_name, f"results/heatmap_{theme_name}.png")
        plot_sampler_bars(df, theme_name, f"results/sampler_bars_{theme_name}.png")
    print("wrote results/heatmap_{light,dark}.png and sampler_bars_{light,dark}.png")


if __name__ == "__main__":
    main()
