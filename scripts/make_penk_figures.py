#!/usr/bin/env python3
"""Figures for the Penk+ vs Penk⁻CamKII+ results summary.

Reads the local result tables written by the cell-type programme runners
(``run_celltype_programme.py``, ``run_celltype_composites.py``,
``run_patching_adaptation.py``) and writes PNG figures to
``docs/figures/penk/`` plus, for each figure, the plotted data as a small JSON
file under ``docs/figures/penk/data/`` (animal-level points, time courses and
annotation statistics; no per-frame data).

Statistics shown on the figures are recomputed here from animal medians:
two-sided exact Mann-Whitney U on the animal medians and the common-language
effect size CLES = P(Penk+ animal > Penk⁻CamKII+ animal), ties counted 0.5.
They can differ slightly from the programme reports, which summarise each
animal by the mean over its cells. The composite figure uses the exact
permutation p values from ``results/celltype_programme_composite/summary.json``.

No S3 access; local CSV/JSON files only.

Usage
-----
    python scripts/make_penk_figures.py
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

from hm2p.constants import CELLTYPE_LABEL, HEX_NONPENK, HEX_PENK  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = REPO / "docs" / "figures" / "penk"
DATA_OUT = OUT / "data"

GROUPS = ("penk", "nonpenk")
COLOURS = {"penk": HEX_PENK, "nonpenk": HEX_NONPENK}
PATCH_MAP = {"penkpos": "penk", "penkneg": "nonpenk"}
DPI = 300


# ── Pure helpers ───────────────────────────────────────────────────────────


def animal_medians(
    df: pd.DataFrame,
    col: str,
    group_col: str = "celltype",
    animal_col: str = "animal_id",
    agg: str = "median",
) -> pd.DataFrame:
    """Summarise ``col`` per animal.

    Returns a DataFrame with columns ``[group_col, animal_col, "value",
    "n"]`` (``n`` = number of non-NaN rows contributing), one row per animal,
    sorted by group then animal. Animals with no finite value are dropped.
    """
    sub = df[[group_col, animal_col, col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub[np.isfinite(sub[col])]
    out = (
        sub.groupby([group_col, animal_col])[col]
        .agg([agg, "count"])
        .reset_index()
        .rename(columns={agg: "value", "count": "n"})
    )
    return out.sort_values([group_col, animal_col]).reset_index(drop=True)


def fraction_significant(
    df: pd.DataFrame,
    p_col: str,
    alpha: float = 0.05,
    group_col: str = "celltype",
    animal_col: str = "animal_id",
) -> pd.DataFrame:
    """Fraction of rows with ``p_col < alpha`` per animal (NaN p dropped)."""
    sub = df[[group_col, animal_col, p_col]].dropna(subset=[p_col]).copy()
    sub["_sig"] = (sub[p_col] < alpha).astype(float)
    out = (
        sub.groupby([group_col, animal_col])["_sig"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "value", "count": "n"})
    )
    return out.sort_values([group_col, animal_col]).reset_index(drop=True)


def cles(a: Sequence[float], b: Sequence[float]) -> float:
    """Common-language effect size P(a > b) + 0.5 P(a == b); NaN if empty."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def group_test(animals: pd.DataFrame, group_col: str = "celltype") -> dict[str, Any]:
    """Animal-level Penk+ vs Penk⁻CamKII+ comparison of ``animals['value']``.

    Two-sided Mann-Whitney U (exact when there are no ties) and CLES for
    Penk+ > Penk⁻CamKII+. Returns NaN statistics if either group is empty.
    """
    a = animals.loc[animals[group_col] == "penk", "value"].to_numpy(float)
    b = animals.loc[animals[group_col] == "nonpenk", "value"].to_numpy(float)
    res: dict[str, Any] = {
        "n_penk": int(a.size),
        "n_nonpenk": int(b.size),
        "median_penk": float(np.median(a)) if a.size else float("nan"),
        "median_nonpenk": float(np.median(b)) if b.size else float("nan"),
        "cles_penk_gt_nonpenk": cles(a, b),
        "mwu_p_two_sided": float("nan"),
    }
    if a.size and b.size:
        res["mwu_p_two_sided"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return res


def format_stats(res: dict[str, Any]) -> str:
    """Short annotation string, e.g. ``'CLES 0.82, p = 0.040 (11 vs 4)'``."""
    return (
        f"CLES {res['cles_penk_gt_nonpenk']:.2f}, p = {res['mwu_p_two_sided']:.3f} "
        f"({res['n_penk']} vs {res['n_nonpenk']} animals)"
    )


def jitter(n: int, width: float = 0.25, seed: int = 0) -> np.ndarray:
    """Deterministic uniform jitter in [-width, width] of length ``n``."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def cell_quartiles(
    df: pd.DataFrame, col: str, group_col: str = "celltype"
) -> dict[str, dict[str, float]]:
    """Per-group cell-level n, 25th, 50th and 75th percentiles of ``col``."""
    out: dict[str, dict[str, float]] = {}
    for g, sub in df.groupby(group_col):
        v = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        out[str(g)] = {
            "n_cells": int(v.size),
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
        }
    return out


def timecourse_by_group(
    tc: dict[str, Any], channel: str
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Extract (time, {group: mean trace}, {group: n_sessions}) for a channel.

    ``tc`` has the layout of ``event_triggered_timecourses.json``:
    ``{"time_s": [...], channel: {group: {"mean": [...], "n_sessions": n}}}``.
    """
    t = np.asarray(tc["time_s"], dtype=float)
    means: dict[str, np.ndarray] = {}
    ns: dict[str, int] = {}
    for g, d in tc[channel].items():
        m = np.asarray(d["mean"], dtype=float)
        if m.shape != t.shape:
            raise ValueError(f"{channel}/{g}: trace length {m.size} != time {t.size}")
        means[g] = m
        ns[g] = int(d.get("n_sessions", 0))
    return t, means, ns


def has_group_traces(tc: dict[str, Any], key: str) -> bool:
    """True if ``tc[key]`` is a per-cell-type mapping rather than one trace."""
    v = tc.get(key)
    return isinstance(v, dict) and any(g in v for g in GROUPS)


def animals_to_records(animals: pd.DataFrame, group_col: str = "celltype") -> list[dict]:
    """Animal table → list of JSON-ready dicts (animal_id, celltype, value, n)."""
    recs = []
    for _, r in animals.iterrows():
        recs.append(
            {
                "animal_id": str(r["animal_id"]),
                "celltype": str(r[group_col]),
                "value": None if not np.isfinite(r["value"]) else float(r["value"]),
                "n": int(r["n"]) if "n" in r and pd.notna(r["n"]) else None,
            }
        )
    return recs


def _clean(obj: Any) -> Any:
    """Recursively replace NaN/inf with None and numpy scalars with Python ones."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ── Plotting (I/O) ─────────────────────────────────────────────────────────


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def _strip(
    ax: plt.Axes,
    animals: pd.DataFrame,
    cells: pd.DataFrame | None = None,
    cell_col: str | None = None,
    log: bool = False,
) -> None:
    """Cell-level dots (faint) under animal-level dots (outlined) with medians."""
    for i, g in enumerate(GROUPS):
        if cells is not None and cell_col is not None:
            v = pd.to_numeric(cells.loc[cells["celltype"] == g, cell_col], errors="coerce")
            v = v[np.isfinite(v)].to_numpy(float)
            if log:
                v = v[v > 0]
            ax.scatter(
                i + jitter(v.size, 0.28, seed=i),
                v,
                s=3,
                color=COLOURS[g],
                alpha=0.18,
                linewidths=0,
                zorder=1,
            )
        a = animals.loc[animals["celltype"] == g, "value"].to_numpy(float)
        ax.scatter(
            i + jitter(a.size, 0.12, seed=10 + i),
            a,
            s=28,
            facecolor=COLOURS[g],
            edgecolor="black",
            linewidths=0.6,
            zorder=3,
        )
        if a.size:
            ax.hlines(np.median(a), i - 0.3, i + 0.3, color="black", lw=1.2, zorder=4)
    ax.set_xticks(range(len(GROUPS)), [CELLTYPE_LABEL[g] for g in GROUPS])
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    if log:
        _log_axis(ax)
    ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)


def _log_axis(ax: plt.Axes) -> None:
    """Log y axis with plain-number tick labels."""
    ax.set_yscale("log")
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _save(fig: plt.Figure, name: str, data: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    with open(DATA_OUT / f"{name}.json", "w") as fh:
        json.dump(_clean(data), fh, indent=1)
    print(f"wrote {OUT / name}.png and data/{name}.json")


def fig_composites() -> None:
    d = RES / "celltype_programme_composite"
    summary = json.loads((d / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.3))
    data: dict[str, Any] = {"panels": {}}
    for ax, key, title in zip(
        axes, ("running", "light"), ("Running coupling", "Light coupling"), strict=True
    ):
        tab = pd.read_csv(d / f"{key}_animals.csv")
        measure_cols = [c for c in tab.columns if re.match(r"^[RL]\d_", c)]
        long = tab.melt(
            id_vars=["animal_id", "celltype"], value_vars=measure_cols, value_name="rank"
        ).dropna(subset=["rank"])
        animals = tab[["celltype", "animal_id", "composite", "n_measures"]].rename(
            columns={"composite": "value", "n_measures": "n"}
        )
        _strip(ax, animals, long, "rank")
        s = summary[key]
        higher = CELLTYPE_LABEL[s["higher_group"]]
        ax.set_title(
            f"{title}: {higher} higher\n"
            f"exact perm p = {s['perm_p_one_sided']:.3f}, CLES {s['cles']:.2f}\n"
            f"{s['measures_agreeing']}/{len(s['measures_used'])} measures agree",
        )
        ax.set_ylabel("Rank (higher = more coupled)")
        ax.set_ylim(0, 16)
        data["panels"][key] = {
            "animals": animals_to_records(animals),
            "per_measure_ranks": [
                {
                    "animal_id": str(r.animal_id),
                    "celltype": r.celltype,
                    "measure": r.variable,
                    "rank": float(r["rank"]),
                }
                for _, r in long.iterrows()
            ],
            "stats": s,
        }
    fig.text(
        0.01,
        -0.04,
        "Large dots: composite (mean rank over measures) per animal; faint dots: "
        "per-measure ranks; bar: group median. One-sided tests; exploratory "
        "(measures chosen after seeing individual results).",
        fontsize=6,
    )
    fig.tight_layout()
    _save(fig, "composites", data)


def fig_event_kinetics() -> None:
    cells = pd.read_csv(RES / "celltype_programme_ctl" / "ctl" / "cells.csv")
    panels = [
        ("ev_mean_duration_s", "Event duration (s)", False),
        ("ev_mean_iei_s", "Inter-event interval (s)", False),
        ("ev_mean_amplitude", "Event amplitude (dF/F units)", True),
        ("iso_decay_s", "Isolated small-event decay (s)", True),
        ("f_baseline", "Baseline fluorescence (raw F)", True),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(10.0, 2.9))
    data: dict[str, Any] = {"panels": {}}
    for ax, (col, label, log) in zip(axes, panels, strict=True):
        animals = animal_medians(cells, col)
        res = group_test(animals)
        _strip(ax, animals, cells, col, log=log)
        ax.set_ylabel(label)
        ax.set_title(f"CLES {res['cles_penk_gt_nonpenk']:.2f}\np = {res['mwu_p_two_sided']:.3f}")
        data["panels"][col] = {
            "label": label,
            "animals": animals_to_records(animals),
            "cell_quartiles": cell_quartiles(cells, col),
            "stats": res,
        }
    fig.text(
        0.01,
        -0.05,
        "Faint dots: cells; outlined dots: animal medians; bar: median of animals. "
        "CLES = P(Penk+ animal > Penk⁻CamKII+ animal); p: two-sided Mann-Whitney U on "
        "animal medians (11 vs 4). Log y axes.",
        fontsize=6,
    )
    fig.tight_layout()
    _save(fig, "event_kinetics", data)


def fig_running_around_events() -> None:
    etb = RES / "celltype_programme_etb" / "etb"
    tc = json.loads((etb / "event_triggered_timecourses.json").read_text())
    cells = pd.read_csv(etb / "cells.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    data: dict[str, Any] = {"timecourses": {}}
    for ax, ch, label in zip(
        axes[:2], ("speed_cm_s", "abs_ahv"), ("Speed (cm/s)", "|AHV| (deg/s)"), strict=True
    ):
        t, means, ns = timecourse_by_group(tc, ch)
        for g in GROUPS:
            if g in means:
                ax.plot(
                    t,
                    means[g],
                    color=COLOURS[g],
                    lw=1.5,
                    label=f"{CELLTYPE_LABEL[g]} ({ns[g]} sessions)",
                )
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ax.set_xlabel("Time from event onset (s)")
        ax.set_ylabel(label)
        ax.grid(color="0.92", lw=0.5)
        data["timecourses"][ch] = {
            "time_s": t,
            "mean_by_celltype": means,
            "n_sessions": ns,
        }
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.36, 1.08),
    )
    axes[0].set_title("Event-triggered speed (session means)")
    axes[1].set_title("Event-triggered |AHV| (session means)")

    sp = cells[cells["channel"] == "speed_cm_s"]
    animals = animal_medians(sp, "during_z")
    res = group_test(animals)
    _strip(axes[2], animals, sp, "during_z")
    axes[2].axhline(0, color="0.5", lw=0.6)
    axes[2].set_ylabel("Speed during events (z vs circular shift)")
    axes[2].set_title(f"During-event speed\n{format_stats(res)}")
    data["during_speed_z"] = {
        "animals": animals_to_records(animals),
        "cell_quartiles": cell_quartiles(sp, "during_z"),
        "stats": res,
    }
    fig.text(
        0.01,
        -0.05,
        "Left, middle: mean over sessions of the event-triggered average (dF/F events). "
        "Right: faint dots cells, outlined dots animal medians; z relative to each cell's "
        "circular-shift null (0 = chance).",
        fontsize=6,
    )
    fig.tight_layout()
    _save(fig, "running_around_events", data)


def fig_running_state() -> None:
    cells = pd.read_csv(RES / "celltype_programme_runshape" / "runshape" / "cells.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
    data: dict[str, Any] = {"panels": {}}
    specs = [
        ("step_index", "Run-vs-still step index", None),
        ("graded_rho", "Graded speed rho (running bins)", None),
        ("step_index_p", "Fraction of cells with significant step index", "frac"),
    ]
    for ax, (col, label, kind) in zip(axes, specs, strict=True):
        if kind == "frac":
            animals = fraction_significant(cells, col)
            _strip(ax, animals)
            ax.axhline(0.05, color="0.5", lw=0.6, ls="--")
            ax.set_ylim(0, 1)
        else:
            animals = animal_medians(cells, col)
            _strip(ax, animals, cells, col)
            ax.axhline(0, color="0.5", lw=0.6)
        res = group_test(animals)
        ax.set_ylabel(label)
        ax.set_title(
            f"CLES {res['cles_penk_gt_nonpenk']:.2f}, p = {res['mwu_p_two_sided']:.3f}\n"
            f"({res['n_penk']} vs {res['n_nonpenk']} animals)"
        )
        entry: dict[str, Any] = {
            "label": label,
            "animals": animals_to_records(animals),
            "stats": res,
        }
        if kind is None:
            entry["cell_quartiles"] = cell_quartiles(cells, col)
        data["panels"][col if kind is None else "frac_sig_step_index"] = entry
    fig.text(
        0.01,
        -0.05,
        "CASCADE spikes, all non-artefact frames. Faint dots: cells; outlined dots: animals "
        "(medians, or fraction with p < 0.05 against 300 circular shifts; dashed line = 0.05 "
        "chance). Penk+ animals with a defined index: 9 of 11.",
        fontsize=6,
    )
    fig.tight_layout()
    _save(fig, "running_state", data)


def fig_light_transitions() -> None:
    h5 = RES / "celltype_programme" / "h5"
    tc = json.loads((h5 / "population_timecourses.json").read_text())
    data: dict[str, Any] = {}
    if has_group_traces(tc, "ltd"):  # pragma: no cover - layout not produced yet
        raise NotImplementedError("per-cell-type population traces not handled")
    # Fallback: session-level amplitudes; pooled trace shown for context only.
    sessions = pd.read_csv(h5 / "sessions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    t = np.asarray(tc["time_s"], dtype=float)
    axes[0].plot(t, tc["ltd"], color="0.2", lw=1.2, label="light → dark")
    axes[0].plot(t, tc["dtl"], color="0.6", lw=1.2, label="dark → light")
    axes[0].axvline(0, color="0.5", lw=0.6, ls="--")
    axes[0].set_xlabel("Time from transition (s)")
    axes[0].set_ylabel("Population dF/F (baseline-subtracted)")
    axes[0].set_title("Both groups pooled (context only)")
    axes[0].legend(frameon=False)
    axes[0].grid(color="0.92", lw=0.5)
    data["pooled_timecourse"] = {
        "note": "pooled over all sessions; not split by cell type in the source file",
        "time_s": t,
        "ltd": tc["ltd"],
        "dtl": tc["dtl"],
    }
    data["panels"] = {}
    for ax, col, label in zip(
        axes[1:],
        ("ltd_early_amplitude", "dtl_late_amplitude"),
        ("Light-off early response (dF/F)", "Light-on late response (dF/F)"),
        strict=True,
    ):
        animals = animal_medians(sessions, col)
        res = group_test(animals)
        _strip(ax, animals, sessions, col)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylabel(label)
        ax.set_title(
            f"CLES {res['cles_penk_gt_nonpenk']:.2f}, p = {res['mwu_p_two_sided']:.3f}\n"
            f"({res['n_penk']} vs {res['n_nonpenk']} animals)"
        )
        data["panels"][col] = {
            "label": label,
            "animals": animals_to_records(animals),
            "sessions": [
                {
                    "animal_id": str(r.animal_id),
                    "celltype": r.celltype,
                    "exp_id": r.exp_id,
                    "value": float(r[col]),
                }
                for _, r in sessions.iterrows()
                if np.isfinite(r[col])
            ],
            "stats": res,
        }
    fig.text(
        0.01,
        -0.05,
        "Middle, right: faint dots sessions, outlined dots animal medians. The source file "
        "has no per-cell-type population traces, so the time course is pooled.",
        fontsize=6,
    )
    fig.tight_layout()
    _save(fig, "light_transitions", data)


def fig_patching_adaptation() -> None:
    cells = pd.read_csv(RES / "patching" / "celltype_programme" / "adaptation_cells.csv")
    cells = cells.assign(celltype=cells["cell_type"].map(PATCH_MAP))
    animal_ids = sorted(cells["animal_id"].unique())
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    style = {a: (cmap(i % 10), markers[i % len(markers)]) for i, a in enumerate(animal_ids)}
    specs = [
        ("matched_sfa_ratio", "Last / first ISI (≥ 6-spike sweep)", True),
        ("matched_first_isi_ms", "First ISI (ms, ≥ 6-spike sweep)", True),
        ("max_n_spikes", "Maximum spikes per sweep", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.0))
    data: dict[str, Any] = {"panels": {}, "cells": []}
    for ax, (col, label, log) in zip(axes, specs, strict=True):
        for i, g in enumerate(GROUPS):
            sub = cells[cells["celltype"] == g]
            xs = i + jitter(len(sub), 0.2, seed=i)
            for x, (_, r) in zip(xs, sub.iterrows(), strict=True):
                if not np.isfinite(r[col]):
                    continue
                c, m = style[r["animal_id"]]
                ax.scatter(
                    x, r[col], s=18, color=c, marker=m, edgecolor="black", linewidths=0.3, zorder=3
                )
            v = sub[col].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                ax.hlines(np.median(v), i - 0.3, i + 0.3, color=COLOURS[g], lw=1.5, zorder=4)
        a = cells.loc[cells["celltype"] == "penk", col].to_numpy(float)
        b = cells.loc[cells["celltype"] == "nonpenk", col].to_numpy(float)
        delta = 2 * cles(a, b) - 1
        ax.set_xticks(range(2), ["Penk+", "Penk\u207b"])
        ax.set_xlim(-0.6, 1.6)
        if log:
            _log_axis(ax)
        ax.set_ylabel(label)
        ax.set_title(f"Cliff's δ {delta:+.2f} (cells; mouse-confounded)")
        ax.grid(axis="y", color="0.9", lw=0.5, zorder=0)
        animals = animal_medians(cells, col)
        data["panels"][col] = {
            "label": label,
            "cliffs_delta_cells": delta,
            "animal_medians": animals_to_records(animals),
            "n_cells": {
                g: int(np.isfinite(cells.loc[cells.celltype == g, col]).sum()) for g in GROUPS
            },
        }
    handles = [
        plt.Line2D(
            [],
            [],
            color=style[a][0],
            marker=style[a][1],
            ls="",
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=a,
        )
        for a in animal_ids
    ]
    fig.legend(
        handles=handles, title="Mouse", frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5)
    )
    fig.text(
        0.01,
        -0.05,
        "Ex vivo, 23 Penk+ vs 14 Penk⁻ cells from 6 mice; one mouse contributed both types. "
        "Dots: cells, coloured and shaped by mouse; bar: cell median. Cell types segregate by "
        "mouse, so cell-level effect sizes are descriptive.",
        fontsize=6,
    )
    fig.tight_layout()
    data["cells"] = [
        {
            "cell_index": int(r.cell_index),
            "animal_id": r.animal_id,
            "celltype": r.celltype,
            **{c: (float(r[c]) if np.isfinite(r[c]) else None) for c, _, _ in specs},
        }
        for _, r in cells.iterrows()
    ]
    _save(fig, "patching_adaptation", data)


def main() -> None:
    _style()
    for fn in (
        fig_composites,
        fig_event_kinetics,
        fig_running_around_events,
        fig_running_state,
        fig_light_transitions,
        fig_patching_adaptation,
    ):
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: missing {exc.filename}")


if __name__ == "__main__":
    main()
