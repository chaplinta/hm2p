"""Assemble per-cell metrics into a flat dict and build a population DataFrame.

Reimplements the metrics assembly logic from ``procPC.m`` lines 100-385 and
the summary table from ``sumTableNum.m``.

MATLAB sources (read-only reference):
    - procPC.m — per-cell metric struct assembly
    - sumTableNum.m — population summary stats and Mann-Whitney U
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# Metric column definitions (matching MATLAB naming)
# ============================================================================

#: Metadata columns
_META_COLS = [
    "cell_index",
    "animal_id",
    "slice_id",
    "cell_slice_id",
    "hemisphere",
    "cell_type",
    "depth_slice",
    "depth_pial",
    "area",
    "layer",
]

#: Passive electrophysiology columns
_PASSIVE_EPHYS_COLS = [
    "ephys_passive_RMP",
    "ephys_passive_rin",
    "ephys_passive_tau",
    "ephys_passive_incap",
    "ephys_passive_sag",
    "ephys_passive_rhreo",
    "ephys_passive_maxsp",
]

#: Active electrophysiology columns
_ACTIVE_EPHYS_COLS = [
    "ephys_active_minVm",
    "ephys_active_peakVm",
    "ephys_active_maxVmSlope",
    "ephys_active_halfVm",
    "ephys_active_amplitude",
    "ephys_active_maxAHP",
    "ephys_active_halfWidth",
]

#: Morphology column suffixes (applied for both apical and basal)
_MORPH_STAT_SUFFIXES = [
    "len",
    "max_plen",
    "bpoints",
    "mpeucl",
    "maxbo",
    "mblen",
    "mplen",
    "mbo",
    "width",
    "height",
    "depth",
    "wh",
    "wd",
    "shlpeakcr",
    "shlpeakcrdist",
    "ext_super",
    "ext_deep",
]


def _morph_cols(prefix: str) -> list[str]:
    """Generate morph column names for a given prefix ('morph_api' or 'morph_bas')."""
    cols = [f"{prefix}_{s}" for s in _MORPH_STAT_SUFFIXES]
    if prefix == "morph_bas":
        cols.append("morph_bas_ntrees")
    return cols


#: All apical morphology columns
_APICAL_MORPH_COLS = _morph_cols("morph_api")

#: All basal morphology columns
_BASAL_MORPH_COLS = _morph_cols("morph_bas")

#: Complete ordered list of all metric columns
ALL_METRIC_COLS = (
    _META_COLS
    + _PASSIVE_EPHYS_COLS
    + _ACTIVE_EPHYS_COLS
    + _APICAL_MORPH_COLS
    + _BASAL_MORPH_COLS
)

# ============================================================================
# Stat-name mapping from compute_tree_stats to MATLAB column suffixes
# ============================================================================

_TREE_STAT_TO_SUFFIX: dict[str, str] = {
    "total_length": "len",
    "max_path_length": "max_plen",
    "n_branch_points": "bpoints",
    "mean_path_eucl_ratio": "mpeucl",
    "max_branch_order": "maxbo",
    "mean_branch_length": "mblen",
    "mean_path_length": "mplen",
    "mean_branch_order": "mbo",
    "width": "width",
    "height": "height",
    "depth": "depth",
    "width_height_ratio": "wh",
    "width_depth_ratio": "wd",
}


# ============================================================================
# Public API
# ============================================================================


def build_cell_metrics(
    ephys_data: dict[str, Any] | None,
    morph_data: dict[str, Any] | None,
    cell_info: dict[str, Any],
) -> dict[str, Any]:
    """Assemble all metrics for one cell into a flat dict.

    Parameters
    ----------
    ephys_data : dict or None
        Electrophysiology results.  Expected keys:

        - ``passive`` — dict with ``RMP``, ``rin``, ``tau``, ``sag``
        - ``active`` — dict with ``minVm``, ``peakVm``, ``maxVmSlope``,
          ``halfVm``, ``amplitude``, ``maxAHP``, ``halfWidth``
        - ``rheobase`` — float
        - ``max_spike_rate`` — float

        If *None*, all ephys columns are set to NaN.

    morph_data : dict or None
        Morphology results.  Expected keys:

        - ``apical_stats`` — dict from :func:`~hm2p.patching.morphology.compute_tree_stats`
        - ``basal_stats`` — dict from :func:`~hm2p.patching.morphology.compute_tree_stats`
        - ``apical_sholl`` — dict with ``peak_crossings``, ``peak_distance``
        - ``basal_sholl`` — dict with ``peak_crossings``, ``peak_distance``
        - ``apical_surface_dist`` — dict with ``dist_superficial``, ``dist_deep``
        - ``basal_surface_dist`` — dict with ``dist_superficial``, ``dist_deep``
        - ``n_basal_trees`` — int

        If *None*, all morph columns are set to NaN.

    cell_info : dict
        Metadata fields: ``cell_index``, ``animal_id``, ``slice_id``,
        ``cell_slice_id``, ``hemisphere``, ``cell_type``, ``depth_slice``,
        ``depth_pial``, ``area``, ``layer``.

    Returns
    -------
    dict
        Flat dict with one key per column in :data:`ALL_METRIC_COLS`.
    """
    m: dict[str, Any] = {}

    # --- Metadata ---
    for key in _META_COLS:
        m[key] = cell_info.get(key, np.nan)

    # --- Passive ephys ---
    if ephys_data is not None:
        passive = ephys_data.get("passive", {})
        m["ephys_passive_RMP"] = passive.get("RMP", np.nan)
        m["ephys_passive_rin"] = passive.get("rin", np.nan)
        m["ephys_passive_tau"] = passive.get("tau", np.nan)

        rin = m["ephys_passive_rin"]
        tau = m["ephys_passive_tau"]
        if _is_valid(rin) and _is_valid(tau) and rin != 0:
            m["ephys_passive_incap"] = tau / rin
        else:
            m["ephys_passive_incap"] = np.nan

        m["ephys_passive_sag"] = passive.get("sag", np.nan)
        m["ephys_passive_rhreo"] = ephys_data.get("rheobase", np.nan)
        m["ephys_passive_maxsp"] = ephys_data.get("max_spike_rate", np.nan)

        # --- Active ephys ---
        active = ephys_data.get("active", {})
        m["ephys_active_minVm"] = active.get("minVm", np.nan)
        m["ephys_active_peakVm"] = active.get("peakVm", np.nan)
        m["ephys_active_maxVmSlope"] = active.get("maxVmSlope", np.nan)
        m["ephys_active_halfVm"] = active.get("halfVm", np.nan)
        m["ephys_active_amplitude"] = active.get("amplitude", np.nan)
        m["ephys_active_maxAHP"] = active.get("maxAHP", np.nan)
        m["ephys_active_halfWidth"] = active.get("halfWidth", np.nan)
    else:
        for col in _PASSIVE_EPHYS_COLS + _ACTIVE_EPHYS_COLS:
            m[col] = np.nan

    # --- Morphology ---
    if morph_data is not None:
        # Apical
        _fill_morph_stats(m, "morph_api", morph_data.get("apical_stats", {}))
        api_sholl = morph_data.get("apical_sholl", {})
        m["morph_api_shlpeakcr"] = api_sholl.get("peak_crossings", np.nan)
        m["morph_api_shlpeakcrdist"] = api_sholl.get("peak_distance", np.nan)
        api_surf = morph_data.get("apical_surface_dist", {})
        m["morph_api_ext_super"] = api_surf.get("dist_superficial", np.nan)
        m["morph_api_ext_deep"] = api_surf.get("dist_deep", np.nan)

        # Basal
        _fill_morph_stats(m, "morph_bas", morph_data.get("basal_stats", {}))
        bas_sholl = morph_data.get("basal_sholl", {})
        m["morph_bas_shlpeakcr"] = bas_sholl.get("peak_crossings", np.nan)
        m["morph_bas_shlpeakcrdist"] = bas_sholl.get("peak_distance", np.nan)
        bas_surf = morph_data.get("basal_surface_dist", {})
        m["morph_bas_ext_super"] = bas_surf.get("dist_superficial", np.nan)
        m["morph_bas_ext_deep"] = bas_surf.get("dist_deep", np.nan)
        m["morph_bas_ntrees"] = morph_data.get("n_basal_trees", np.nan)
    else:
        for col in _APICAL_MORPH_COLS + _BASAL_MORPH_COLS:
            m[col] = np.nan

    return m


def _fill_morph_stats(m: dict, prefix: str, stats: dict[str, Any]) -> None:
    """Map compute_tree_stats keys into the flat metric dict."""
    for stat_key, suffix in _TREE_STAT_TO_SUFFIX.items():
        col = f"{prefix}_{suffix}"
        m[col] = stats.get(stat_key, np.nan)


def _is_valid(val: Any) -> bool:
    """Check if a value is a valid (non-NaN, non-None) number."""
    if val is None:
        return False
    try:
        return not math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def build_metrics_table(cells: list[dict[str, Any]]) -> pd.DataFrame:
    """Combine all cell metric dicts into a DataFrame (one row per cell).

    Parameters
    ----------
    cells : list of dict
        Each dict as returned by :func:`build_cell_metrics`.

    Returns
    -------
    DataFrame
        One row per cell, columns matching :data:`ALL_METRIC_COLS` (plus any
        extra keys present in the dicts).
    """
    if not cells:
        return pd.DataFrame(columns=ALL_METRIC_COLS)
    df = pd.DataFrame(cells)
    # Reorder so standard columns come first, keeping any extras at the end
    ordered = [c for c in ALL_METRIC_COLS if c in df.columns]
    extras = [c for c in df.columns if c not in ALL_METRIC_COLS]
    return df[ordered + extras].reset_index(drop=True)


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to a metrics DataFrame.

    Currently computes:

    - ``ephys_passive_incap`` — input capacitance = tau / rin (recomputed to
      ensure consistency even if already present).
    - Width/height and width/depth ratios for apical and basal (recomputed).

    Parameters
    ----------
    df : DataFrame
        As returned by :func:`build_metrics_table`.

    Returns
    -------
    DataFrame
        Copy with derived columns added or updated.
    """
    out = df.copy()

    # Input capacitance
    if "ephys_passive_tau" in out.columns and "ephys_passive_rin" in out.columns:
        tau = out["ephys_passive_tau"].values.astype(float)
        rin = out["ephys_passive_rin"].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            incap = np.where((rin != 0) & np.isfinite(rin) & np.isfinite(tau), tau / rin, np.nan)
        out["ephys_passive_incap"] = incap

    # Morph width/height and width/depth ratios
    for prefix in ("morph_api", "morph_bas"):
        w_col = f"{prefix}_width"
        h_col = f"{prefix}_height"
        d_col = f"{prefix}_depth"
        wh_col = f"{prefix}_wh"
        wd_col = f"{prefix}_wd"

        if w_col in out.columns and h_col in out.columns:
            w = out[w_col].values.astype(float)
            h = out[h_col].values.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[wh_col] = np.where((h != 0) & np.isfinite(h) & np.isfinite(w), w / h, 0.0)

        if w_col in out.columns and d_col in out.columns:
            w = out[w_col].values.astype(float)
            d = out[d_col].values.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[wd_col] = np.where((d != 0) & np.isfinite(d) & np.isfinite(w), w / d, 0.0)

    return out
