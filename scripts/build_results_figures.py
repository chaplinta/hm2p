#!/usr/bin/env python3
"""Generate figures (PNGs) for the results deck from the committed analysis CSVs
and the behaviour JSON. Output: docs/manuscripts/figures/*.png.

Figures are paired within-session slopegraphs (one line per session, light->dark),
which is the honest visual for the paired non-parametric tests used throughout.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DARK_HYP = ROOT / "results" / "dark_hypotheses"
MAPENG = ROOT / "results" / "map_engagement" / "map_engagement.csv"
BEHAV = ROOT / "docs" / "manuscripts" / "behaviour-results.json"
FIGDIR = ROOT / "docs" / "manuscripts" / "figures"

BLUE = "#2E75B6"
GREY = "#888888"
GREEN = "#2D8B57"
RED = "#C0392B"

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})


def _slope(ax, light, dark, color=GREY, point_color=BLUE):
    light, dark = np.asarray(light, float), np.asarray(dark, float)
    ok = np.isfinite(light) & np.isfinite(dark)
    light, dark = light[ok], dark[ok]
    for l, d in zip(light, dark):
        ax.plot([0, 1], [l, d], color=color, alpha=0.35, lw=1, zorder=1)
    ax.scatter(np.zeros_like(light), light, color=point_color, s=22, zorder=2)
    ax.scatter(np.ones_like(dark), dark, color=point_color, s=22, zorder=2)
    # condition medians
    ax.plot([-0.12, 0.12], [np.median(light)] * 2, color=RED, lw=3, zorder=3)
    ax.plot([0.88, 1.12], [np.median(dark)] * 2, color=RED, lw=3, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Light", "Dark"])
    ax.set_xlim(-0.35, 1.35)


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def fig_gauntlet():
    """Raw vs occupancy-matched MVL — the effect and its disappearance."""
    df = pd.read_csv(DARK_HYP / "A1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    _slope(axes[0], df["mvl_light_raw"], df["mvl_dark_raw"])
    axes[0].set_title("Raw MVL\n(dark > light, p = 0.001)", fontsize=12)
    axes[0].set_ylabel("HD mean vector length")
    _slope(axes[1], df["mvl_light_matched"], df["mvl_dark_matched"])
    axes[1].set_title("Occupancy-matched MVL\n(n.s., p = 0.16)", fontsize=12)
    fig.suptitle("Matching head-direction sampling removes the dark>light effect",
                 fontsize=13, y=1.02)
    _save(fig, "fig_gauntlet.png")


def fig_mvl_vs_mi():
    """Matched MVL and matched Skaggs MI — both null."""
    mvl = pd.read_csv(DARK_HYP / "A1.csv")
    mi = pd.read_csv(DARK_HYP / "A1_mi.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    _slope(axes[0], mvl["mvl_light_matched"], mvl["mvl_dark_matched"])
    axes[0].set_title("Matched MVL\n(p = 0.16)", fontsize=12)
    axes[0].set_ylabel("debiased, matched value")
    _slope(axes[1], mi["mvl_light_matched"], mi["mvl_dark_matched"])
    axes[1].set_title("Matched Skaggs information\n(p = 0.64)", fontsize=12)
    fig.suptitle("Two independent metrics agree: no dark enhancement after matching",
                 fontsize=13, y=1.02)
    _save(fig, "fig_mvl_vs_mi.png")


def _behav_pairs(key_l, key_d):
    sess = [s for s in json.load(open(BEHAV))["per_session"] if not s.get("exclude")]
    l = [s.get(key_l) for s in sess]
    d = [s.get(key_d) for s in sess]
    return l, d


def fig_behaviour():
    """Coverage-vs-random-walk-null and occupancy entropy, light vs dark."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    zl, zd = _behav_pairs("mean_epoch_zcov_light", "mean_epoch_zcov_dark")
    _slope(axes[0], zl, zd, point_color=GREEN)
    axes[0].axhline(0, color="k", lw=0.8, ls=":")
    axes[0].set_title("Coverage vs random-walk null\n(z; p_adj = 0.018)", fontsize=12)
    axes[0].set_ylabel("z vs random walk")
    el, ed = _behav_pairs("mean_epoch_entropy_light", "mean_epoch_entropy_dark")
    _slope(axes[1], el, ed, point_color=GREEN)
    axes[1].set_title("Occupancy entropy\n(bits; p_adj = 0.009)", fontsize=12)
    fig.suptitle("Exploration: directed in light, near-random in dark", fontsize=13, y=1.02)
    _save(fig, "fig_behaviour.png")


def fig_map_engagement():
    """Map engagement debiased consistency, light vs dark (null)."""
    df = pd.read_csv(MAPENG)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    _slope(ax, df["debiased_light"], df["debiased_dark"])
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title("Map engagement\n(within-cell consistency; p = 0.49, n.s.)", fontsize=12)
    ax.set_ylabel("debiased population-vector consistency")
    fig.suptitle("Spatial map equally engaged with and without vision",
                 fontsize=12, y=1.0)
    _save(fig, "fig_map_engagement.png")


def main():
    fig_gauntlet()
    fig_mvl_vs_mi()
    fig_behaviour()
    fig_map_engagement()


if __name__ == "__main__":
    main()
