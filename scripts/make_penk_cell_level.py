#!/usr/bin/env python3
"""Cell-level summary of the Penk+ vs Penk-CamKII+ programme for the results page.

Pools every soma ROI across sessions and animals and compares the two cell
types cell by cell, ignoring which animal a cell came from. These tests treat
cells as independent, which they are not (cells share an animal, a session and
an expression level), so p values here describe the pooled cell distributions
and overstate certainty about the populations. The animal-level tests are in
``scripts/make_penk_figures.py`` and part 2 of the page.

Statistics (all non-parametric):

- between cell types: two-sided Mann-Whitney U on cells, Cliff's delta,
  Benjamini-Hochberg FDR within each section;
- within a cell type, for signed measures centred on zero: two-sided Wilcoxon
  signed-rank of the cell values against 0;
- fraction of cells individually significant in each cell type (the runner's
  per-cell shuffle p < 0.05, with the stated sign), compared with a two-sided
  Fisher exact test.

Writes ``docs/figures/penk/data/cell_level.json``.

References
----------
Benjamini Y, Hochberg Y. 1995. "Controlling the false discovery rate: a
practical and powerful approach to multiple testing." J R Stat Soc B
57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

Cliff N. 1993. "Dominance statistics: ordinal analyses to answer ordinal
questions." Psychological Bulletin 114:494-509. doi:10.1037/0033-2909.114.3.494
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = REPO / "docs" / "figures" / "penk" / "data" / "cell_level.json"
GROUPS = ("penk", "nonpenk")
ALPHA = 0.05


@dataclass(frozen=True)
class Measure:
    """One per-cell measure read from a programme result table."""

    key: str
    section: str
    label: str
    path: str
    column: str
    signal: str
    centred: bool = False
    row_filter: tuple[str, str] | None = None
    sig_column: str | None = None
    sig_sign: int = 0  # +1: count only positive values, -1: only negative, 0: either
    sig_value_column: str | None = None
    log: bool = False


SECTIONS = {
    "signature": "Activity signature",
    "running": "Running",
    "light": "Light",
    "other": "Head direction, space and population",
}

MEASURES: tuple[Measure, ...] = (
    # activity signature
    Measure(
        "ev_mean_duration_s",
        "signature",
        "Event duration (s)",
        "celltype_programme_ctl/ctl/cells.csv",
        "ev_mean_duration_s",
        "dF/F events",
    ),
    Measure(
        "ev_mean_iei_s",
        "signature",
        "Inter-event interval (s)",
        "celltype_programme_ctl/ctl/cells.csv",
        "ev_mean_iei_s",
        "dF/F events",
        log=True,
    ),
    Measure(
        "iso_decay_s",
        "signature",
        "Isolated small-event decay (s)",
        "celltype_programme_ctl/ctl/cells.csv",
        "iso_decay_s",
        "dF/F events",
    ),
    Measure(
        "ev_mean_amplitude",
        "signature",
        "Event amplitude (dF/F)",
        "celltype_programme_ctl/ctl/cells.csv",
        "ev_mean_amplitude",
        "dF/F events",
        log=True,
    ),
    Measure(
        "f_baseline",
        "signature",
        "Baseline fluorescence (raw F)",
        "celltype_programme_ctl/ctl/cells.csv",
        "f_baseline",
        "raw F",
        log=True,
    ),
    Measure(
        "spike_rate_hz",
        "signature",
        "Inferred spike rate (spikes/s)",
        "celltype_programme_ctl/ctl/cells.csv",
        "spike_rate_hz",
        "CASCADE",
        log=True,
    ),
    Measure(
        "tr_skewness",
        "signature",
        "dF/F skewness",
        "celltype_programme/h2/cells.csv",
        "tr_skewness",
        "dF/F",
    ),
    Measure(
        "ev_fraction_active",
        "signature",
        "Fraction of time in events",
        "celltype_programme/h2/cells.csv",
        "ev_fraction_active",
        "dF/F events",
    ),
    # running
    Measure(
        "speed_during_z",
        "running",
        "Speed during events (z vs shift null)",
        "celltype_programme_etb/etb/cells.csv",
        "during_z",
        "dF/F events",
        centred=True,
        row_filter=("channel", "speed_cm_s"),
        sig_column="during_p",
        sig_sign=1,
        sig_value_column="during_z",
    ),
    Measure(
        "active_during_z",
        "running",
        "Moving during events (z vs shift null)",
        "celltype_programme_etb/etb/cells.csv",
        "during_z",
        "dF/F events",
        centred=True,
        row_filter=("channel", "active"),
        sig_column="during_p",
        sig_sign=1,
        sig_value_column="during_z",
    ),
    Measure(
        "step_index",
        "running",
        "Run vs still step index",
        "celltype_programme_runshape/runshape/cells.csv",
        "step_index",
        "CASCADE",
        centred=True,
        sig_column="step_index_p",
        sig_sign=1,
    ),
    Measure(
        "graded_rho",
        "running",
        "Graded speed correlation while running",
        "celltype_programme_runshape/runshape/cells.csv",
        "graded_rho",
        "CASCADE",
        centred=True,
        sig_column="graded_rho_p",
        sig_sign=0,
    ),
    Measure(
        "bout_adaptation_index",
        "running",
        "Early vs late activity within runs",
        "celltype_programme_runshape/runshape/cells.csv",
        "bout_adaptation_index",
        "CASCADE",
        centred=True,
    ),
    Measure(
        "bout_dur_rho_sum",
        "running",
        "Summed activity vs run length (rho)",
        "celltype_programme_runshape/runshape/cells.csv",
        "bout_dur_rho_sum",
        "CASCADE",
        centred=True,
    ),
    Measure(
        "act_movement_modulation",
        "running",
        "Movement modulation index",
        "celltype_programme/h2/cells.csv",
        "act_movement_modulation",
        "dF/F",
        centred=True,
    ),
    Measure(
        "loc_index",
        "running",
        "Corridor vs junction at matched speed",
        "celltype_programme_locrun/locrun/cells.csv",
        "loc_index",
        "CASCADE",
        centred=True,
        sig_column="loc_index_p",
        sig_sign=0,
    ),
    Measure(
        "run_index_corr",
        "running",
        "Fast vs slow movement in corridors",
        "celltype_programme_locrun/locrun/cells.csv",
        "run_index_corr",
        "CASCADE",
        centred=True,
        sig_column="run_index_corr_p",
        sig_sign=1,
    ),
    # light
    Measure(
        "act_light_modulation",
        "light",
        "Light modulation index",
        "celltype_programme/h2/cells.csv",
        "act_light_modulation",
        "dF/F",
        centred=True,
    ),
    Measure(
        "light_during_z",
        "light",
        "Lights on during events (z vs shift null)",
        "celltype_programme_etb/etb/cells.csv",
        "during_z",
        "dF/F events",
        centred=True,
        row_filter=("channel", "light_on"),
        sig_column="during_p",
        sig_sign=1,
        sig_value_column="during_z",
    ),
    Measure(
        "dtl_late_amplitude",
        "light",
        "Lights-on sustained response (dF/F)",
        "celltype_programme/h5/cells.csv",
        "dtl_late_amplitude",
        "dF/F",
        centred=True,
        sig_column="dtl_p_value",
        sig_sign=1,
    ),
    Measure(
        "ltd_early_amplitude",
        "light",
        "Lights-off early response (dF/F)",
        "celltype_programme/h5/cells.csv",
        "ltd_early_amplitude",
        "dF/F",
        centred=True,
        sig_column="ltd_p_value",
        sig_sign=-1,
    ),
    Measure(
        "ahv_dark_minus_light_matched",
        "light",
        "AHV depth, dark minus light (matched)",
        "celltype_programme/h3/cells.csv",
        "ahv_dark_minus_light_matched",
        "dF/F",
        centred=True,
    ),
    Measure(
        "part_light",
        "light",
        "GLM light share of deviance",
        "celltype_programme/h8/cells.csv",
        "part_light",
        "dF/F",
    ),
    # head direction, space, population
    Measure(
        "tun_mvl",
        "other",
        "HD mean vector length",
        "celltype_programme/h2/cells.csv",
        "tun_mvl",
        "dF/F",
    ),
    Measure(
        "ahv_modulation_depth",
        "other",
        "AHV modulation depth",
        "celltype_programme/h3/cells.csv",
        "ahv_modulation_depth",
        "dF/F",
    ),
    Measure(
        "ebc_mrl",
        "other",
        "Egocentric boundary tuning (MRL)",
        "celltype_programme_ego/ego/cells.csv",
        "ebc_mrl",
        "CASCADE",
        sig_column="ebc_p",
        sig_sign=0,
    ),
    Measure(
        "place_skaggs_info",
        "other",
        "Place information (bits/event)",
        "celltype_programme/h10/cells.csv",
        "place_skaggs_info",
        "dF/F events",
    ),
    Measure(
        "drift_abs_rho",
        "other",
        "Slow drift over session (|rho|)",
        "celltype_programme_tctx/tctx/cells.csv",
        "drift_abs_rho",
        "CASCADE",
        sig_column="drift_p",
        sig_sign=0,
    ),
    Measure(
        "pop_coupling_all",
        "other",
        "Population coupling",
        "celltype_programme/h9/cells.csv",
        "pop_coupling_all",
        "dF/F",
        centred=True,
    ),
    Measure(
        "mean_noise_corr",
        "other",
        "Mean noise correlation",
        "celltype_programme/h9/cells.csv",
        "mean_noise_corr",
        "dF/F",
        centred=True,
    ),
)


def load_measure(m: Measure, root: Path = RES) -> pd.DataFrame | None:
    """Per-cell values of one measure: columns animal_id, exp_id, roi_idx, celltype, value, sig.

    Returns None when the table or column is missing. ``sig`` is True when the
    cell is individually significant with the required sign, False when it is
    not, and NaN when the measure has no per-cell test.
    """
    path = root / m.path
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if m.column not in df.columns:
        return None
    if m.row_filter is not None:
        col, val = m.row_filter
        df = df[df[col] == val]
    out = df[["animal_id", "exp_id", "roi_idx", "celltype"]].copy()
    out["animal_id"] = out["animal_id"].astype(str)
    out["value"] = pd.to_numeric(df[m.column], errors="coerce")
    out["sig"] = np.nan
    if m.sig_column is not None and m.sig_column in df.columns:
        p = pd.to_numeric(df[m.sig_column], errors="coerce")
        sv = pd.to_numeric(df[m.sig_value_column or m.column], errors="coerce")
        sig = p < ALPHA
        if m.sig_sign > 0:
            sig &= sv > 0
        elif m.sig_sign < 0:
            sig &= sv < 0
        out["sig"] = np.where(p.notna() & sv.notna(), sig.astype(float), np.nan)
    out = out[np.isfinite(out["value"])]
    return out[out["celltype"].isin(GROUPS)].reset_index(drop=True)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta = P(a > b) - P(a < b) over all pairs; NaN if either is empty."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def quartiles(x: np.ndarray) -> dict[str, float]:
    """25th, 50th and 75th percentiles."""
    q25, q50, q75 = np.percentile(x, [25, 50, 75]) if len(x) else (np.nan,) * 3
    return {"q25": float(q25), "median": float(q50), "q75": float(q75)}


def within_test(x: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank of x against 0; NaN with fewer than 5 non-zero values."""
    x = np.asarray(x, float)
    x = x[x != 0]
    if x.size < 5:
        return float("nan")
    return float(stats.wilcoxon(x).pvalue)


def fraction_test(sig_a: np.ndarray, sig_b: np.ndarray) -> dict[str, Any]:
    """Fraction of significant cells per group and a two-sided Fisher exact p."""
    sig_a = np.asarray(sig_a, float)
    sig_b = np.asarray(sig_b, float)
    sig_a, sig_b = sig_a[np.isfinite(sig_a)], sig_b[np.isfinite(sig_b)]
    if sig_a.size == 0 or sig_b.size == 0:
        return {}
    ka, kb = int(sig_a.sum()), int(sig_b.sum())
    p = stats.fisher_exact([[ka, sig_a.size - ka], [kb, sig_b.size - kb]]).pvalue
    return {
        "k_penk": ka,
        "n_penk": int(sig_a.size),
        "frac_penk": ka / sig_a.size,
        "k_nonpenk": kb,
        "n_nonpenk": int(sig_b.size),
        "frac_nonpenk": kb / sig_b.size,
        "fisher_p": float(p),
    }


def bh_fdr(p: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p values; NaN entries stay NaN."""
    arr = np.asarray(p, float)
    out = np.full(arr.shape, np.nan)
    ok = np.isfinite(arr)
    if not ok.any():
        return out.tolist()
    v = arr[ok]
    order = np.argsort(v)
    n = v.size
    adj = v[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out.tolist()


def summarise(m: Measure, cells: pd.DataFrame) -> dict[str, Any]:
    """Cell-level statistics and per-cell values for one measure."""
    a = cells.loc[cells.celltype == "penk", "value"].to_numpy(float)
    b = cells.loc[cells.celltype == "nonpenk", "value"].to_numpy(float)
    mwu = (
        float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        if a.size and b.size
        else float("nan")
    )
    res: dict[str, Any] = {
        "key": m.key,
        "section": m.section,
        "label": m.label,
        "signal": m.signal,
        "centred": m.centred,
        "log": m.log,
        "n_penk": int(a.size),
        "n_nonpenk": int(b.size),
        "n_animals_penk": int(cells.loc[cells.celltype == "penk", "animal_id"].nunique()),
        "n_animals_nonpenk": int(cells.loc[cells.celltype == "nonpenk", "animal_id"].nunique()),
        "quartiles": {"penk": quartiles(a), "nonpenk": quartiles(b)},
        "mwu_p": mwu,
        "cliffs_delta": cliffs_delta(a, b),
        "within_p": {"penk": within_test(a), "nonpenk": within_test(b)} if m.centred else None,
        "fraction_sig": None,
        "values": {
            g: [round(float(v), 5) for v in cells.loc[cells.celltype == g, "value"]]
            for g in GROUPS
        },
        "animals": {g: cells.loc[cells.celltype == g, "animal_id"].tolist() for g in GROUPS},
    }
    if m.sig_column is not None:
        res["fraction_sig"] = (
            fraction_test(
                cells.loc[cells.celltype == "penk", "sig"].to_numpy(),
                cells.loc[cells.celltype == "nonpenk", "sig"].to_numpy(),
            )
            or None
        )
    return res


def build(root: Path = RES, measures: tuple[Measure, ...] = MEASURES) -> dict[str, Any]:
    """Summaries for every available measure, with FDR within each section."""
    rows = []
    for m in measures:
        cells = load_measure(m, root)
        if cells is None or cells.empty:
            continue
        rows.append(summarise(m, cells))
    for sec in SECTIONS:
        idx = [i for i, r in enumerate(rows) if r["section"] == sec]
        adj = bh_fdr([rows[i]["mwu_p"] for i in idx])
        for i, q in zip(idx, adj, strict=True):
            rows[i]["mwu_fdr"] = q
    run = _running_scatter(root)
    return {"sections": SECTIONS, "measures": rows, "running_scatter": run}


def _running_scatter(root: Path) -> dict[str, Any] | None:
    """Per-cell step index vs graded speed rho (CASCADE) for the run-state scatter."""
    path = root / "celltype_programme_runshape/runshape/cells.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    need = {"step_index", "graded_rho", "step_index_p", "celltype"}
    if not need <= set(df.columns):
        return None
    df = df[df.celltype.isin(GROUPS)].dropna(subset=["step_index", "graded_rho"])
    return {
        g: {
            "step_index": df.loc[df.celltype == g, "step_index"].round(5).tolist(),
            "graded_rho": df.loc[df.celltype == g, "graded_rho"].round(5).tolist(),
            "step_sig": (
                (df.loc[df.celltype == g, "step_index_p"] < ALPHA)
                & (df.loc[df.celltype == g, "step_index"] > 0)
            ).tolist(),
            "animal_id": df.loc[df.celltype == g, "animal_id"].astype(str).tolist(),
        }
        for g in GROUPS
    }


def _clean(obj: Any) -> Any:
    """Replace NaN/inf with None recursively so the output is strict JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


def main() -> None:  # pragma: no cover - file I/O entry point
    data = _clean(build())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({len(data['measures'])} measures)")


if __name__ == "__main__":  # pragma: no cover
    main()
