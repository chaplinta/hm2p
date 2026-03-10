"""Orchestrator for the patching electrophysiology and morphology pipeline.

Loads metadata, loops over cells, processes each one (ephys + morphology),
assembles per-cell metrics, and saves the output table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hm2p.patching.config import PatchConfig, SAMPLE_RATE
from hm2p.patching.metrics import (
    build_cell_metrics,
    build_metrics_table,
    compute_derived_metrics,
)
from hm2p.patching.morphology import (
    compute_sholl,
    compute_surface_distance,
    compute_tree_stats,
    load_morphology,
    rotate_to_surface,
    soma_subtract,
)
from hm2p.patching.protocols import process_all_protocols
from hm2p.patching.spike_features import extract_spike_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def load_metadata(config: PatchConfig) -> pd.DataFrame:
    """Load and merge animals.csv and cells.csv from *config.metadata_dir*.

    Parameters
    ----------
    config : PatchConfig
        Pipeline configuration with ``metadata_dir`` pointing to the
        directory containing ``animals.csv`` and ``cells.csv``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame on ``animal_id``.

    Raises
    ------
    FileNotFoundError
        If either CSV file is missing.
    """
    animals_path = config.metadata_dir / "animals.csv"
    cells_path = config.metadata_dir / "cells.csv"

    if not animals_path.exists():
        raise FileNotFoundError(f"animals.csv not found: {animals_path}")
    if not cells_path.exists():
        raise FileNotFoundError(f"cells.csv not found: {cells_path}")

    animals = pd.read_csv(animals_path)
    cells = pd.read_csv(cells_path)

    merged = cells.merge(animals, on="animal_id", how="left")
    logger.info(
        "Loaded metadata: %d cells, %d animals -> %d rows",
        len(cells),
        len(animals),
        len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Per-cell processing
# ---------------------------------------------------------------------------


def _build_ephys_data(protocol_results: dict) -> dict[str, Any]:
    """Convert protocol result dataclasses into the dict expected by build_cell_metrics."""
    ephys: dict[str, Any] = {}

    # Passive properties
    passive: dict[str, Any] = {}
    if "passive" in protocol_results:
        pr = protocol_results["passive"]
        passive["RMP"] = pr.rmp
        passive["rin"] = pr.rin
        passive["tau"] = float(np.nanmean(pr.tau))
    elif "iv" in protocol_results:
        passive["RMP"] = protocol_results["iv"].rmp
    if "sag" in protocol_results:
        sr = protocol_results["sag"]
        passive["sag"] = float(np.nanmean(sr.sag_ratio))
    ephys["passive"] = passive

    # Rheobase
    if "rheobase" in protocol_results:
        ephys["rheobase"] = protocol_results["rheobase"].rheo_current
    else:
        ephys["rheobase"] = np.nan

    # Max spike rate from IV
    if "iv" in protocol_results:
        iv = protocol_results["iv"]
        ephys["max_spike_rate"] = float(np.max(iv.spike_counts))
    else:
        ephys["max_spike_rate"] = np.nan

    return ephys


def _extract_active_features(
    protocol_results: dict,
) -> dict[str, float] | None:
    """Extract spike waveform features from the rheobase sweep.

    Returns the active features dict or None if no spikes found.
    """
    if "rheobase" not in protocol_results:
        return None

    rheo = protocol_results["rheobase"]
    # Find the first sweep with a spike
    spike_idx = np.where(rheo.spike_counts > 0)[0]
    if len(spike_idx) == 0:
        return None

    sweep_idx = spike_idx[0]
    trace = rheo.traces[:, sweep_idx]
    sr = SAMPLE_RATE
    time_ms = np.arange(len(trace)) / (sr / 1000.0)

    # Stimulus starts after the delay (approximate: use half the trace)
    stim_start = time_ms[len(time_ms) // 10]
    stim_end = time_ms[-1]

    features = extract_spike_features(
        trace, time_ms, stim_start, stim_end, spike_index=1
    )
    if features is None:
        return None

    return {
        "minVm": features["min_vm"],
        "peakVm": features["peak_vm"],
        "maxVmSlope": features["max_vm_slope"],
        "halfVm": features["half_vm"],
        "amplitude": features["amplitude"],
        "maxAHP": features["max_ahp"],
        "halfWidth": features["half_width"],
    }


def _build_morph_data(
    tracing_path: Path,
) -> dict[str, Any] | None:
    """Load morphology from *tracing_path* and compute all metrics.

    Returns
    -------
    dict or None
        Dict with keys expected by ``build_cell_metrics`` for morphology,
        or *None* if the directory doesn't exist or contains no valid data.
    """
    if not tracing_path.is_dir():
        logger.warning("Tracing directory not found: %s", tracing_path)
        return None

    neurons = load_morphology(tracing_path)
    if "soma" not in neurons:
        logger.warning("No soma SWC in %s — skipping morphology", tracing_path)
        return None

    # Soma subtraction
    neurons = soma_subtract(neurons)
    soma_center = np.array([0.0, 0.0, 0.0])  # after subtraction

    # Rotate to surface if surface data available
    if "surface" in neurons:
        surface_xy = neurons["surface"]["nodes"][["x", "y"]].values
        if len(surface_xy) > 2:
            neurons, _ = rotate_to_surface(neurons, surface_xy)

    result: dict[str, Any] = {}

    # Sholl radii
    radii = np.arange(10, 510, 10, dtype=float)

    # Apical
    if "apical" in neurons:
        api = neurons["apical"]
        result["apical_stats"] = compute_tree_stats(api["nodes"], api["edges"])
        sholl_counts = compute_sholl(api["nodes"], soma_center, radii, api["edges"])
        peak_idx = int(np.argmax(sholl_counts))
        result["apical_sholl"] = {
            "peak_crossings": int(sholl_counts[peak_idx]),
            "peak_distance": float(radii[peak_idx]),
        }
        if "surface" in neurons:
            surface_xy = neurons["surface"]["nodes"][["x", "y"]].values
            dendrite_xy = api["nodes"][["x", "y"]].values
            result["apical_surface_dist"] = compute_surface_distance(
                surface_xy, dendrite_xy
            )
        else:
            result["apical_surface_dist"] = {}
    else:
        result["apical_stats"] = {}
        result["apical_sholl"] = {}
        result["apical_surface_dist"] = {}

    # Basal
    if "basal" in neurons:
        bas = neurons["basal"]
        result["basal_stats"] = compute_tree_stats(bas["nodes"], bas["edges"])
        sholl_counts = compute_sholl(bas["nodes"], soma_center, radii, bas["edges"])
        peak_idx = int(np.argmax(sholl_counts))
        result["basal_sholl"] = {
            "peak_crossings": int(sholl_counts[peak_idx]),
            "peak_distance": float(radii[peak_idx]),
        }
        if "surface" in neurons:
            surface_xy = neurons["surface"]["nodes"][["x", "y"]].values
            dendrite_xy = bas["nodes"][["x", "y"]].values
            result["basal_surface_dist"] = compute_surface_distance(
                surface_xy, dendrite_xy
            )
        else:
            result["basal_surface_dist"] = {}
    else:
        result["basal_stats"] = {}
        result["basal_sholl"] = {}
        result["basal_surface_dist"] = {}

    # Count basal trees from original files
    basal_swc_files = list(tracing_path.glob("Basal*.swc"))
    result["n_basal_trees"] = len(basal_swc_files)

    return result


def _build_cell_info(cell_row: pd.Series) -> dict[str, Any]:
    """Extract metadata fields from a cell row for build_cell_metrics."""
    info: dict[str, Any] = {}
    meta_keys = [
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
    for key in meta_keys:
        val = cell_row.get(key, np.nan)
        # Convert pandas NA / NaN-like to np.nan
        if pd.isna(val):
            val = np.nan
        info[key] = val
    return info


def process_cell(cell_row: pd.Series, config: PatchConfig) -> dict | None:
    """Process a single cell: extract ephys and/or morphology, build metrics.

    Parameters
    ----------
    cell_row : pd.Series
        A row from the merged metadata DataFrame.
    config : PatchConfig
        Pipeline configuration.

    Returns
    -------
    dict or None
        Flat metrics dict for this cell (as returned by
        :func:`~hm2p.patching.metrics.build_cell_metrics`), or *None*
        if the cell has no processable data.
    """
    cell_id = cell_row.get("cell_index", "unknown")
    logger.info("Processing cell %s", cell_id)

    has_ephys = False
    has_morph = False

    ephys_id = cell_row.get("ephys_id", "")
    if isinstance(ephys_id, str) and ephys_id.strip():
        has_ephys = True
    elif not pd.isna(ephys_id) and str(ephys_id).strip():
        has_ephys = True

    good_morph = cell_row.get("good_morph", False)
    if good_morph is True or good_morph == 1:
        has_morph = True

    if not has_ephys and not has_morph:
        logger.info("Cell %s has no ephys or morphology data — skipping", cell_id)
        return None

    ephys_data: dict[str, Any] | None = None
    morph_data: dict[str, Any] | None = None

    # --- Electrophysiology ---
    if has_ephys:
        try:
            ephys_dir = config.ephys_dir / str(ephys_id).strip()
            protocol_results = process_all_protocols(ephys_dir)

            if protocol_results:
                ephys_data = _build_ephys_data(protocol_results)

                # Extract active (spike) features
                try:
                    active_features = _extract_active_features(protocol_results)
                    if active_features is not None:
                        ephys_data["active"] = active_features
                except Exception:
                    logger.exception(
                        "Failed to extract spike features for cell %s", cell_id
                    )
            else:
                logger.warning("No protocols extracted for cell %s", cell_id)
        except Exception:
            logger.exception("Failed to process ephys for cell %s", cell_id)

    # --- Morphology ---
    if has_morph:
        try:
            morph_id = cell_row.get("morph_id", cell_row.get("cell_slice_id", ""))
            if pd.isna(morph_id) or not str(morph_id).strip():
                morph_id = str(cell_id)
            tracing_path = config.morph_dir / str(morph_id).strip()
            morph_data = _build_morph_data(tracing_path)
        except Exception:
            logger.exception("Failed to process morphology for cell %s", cell_id)

    if ephys_data is None and morph_data is None:
        logger.info("Cell %s: no data could be extracted", cell_id)
        return None

    cell_info = _build_cell_info(cell_row)
    return build_cell_metrics(ephys_data, morph_data, cell_info)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: PatchConfig | None = None) -> pd.DataFrame:
    """Run the full patching pipeline: metadata -> process cells -> metrics table.

    Parameters
    ----------
    config : PatchConfig or None
        Pipeline configuration. If *None*, loads from the default
        ``config/patching.yaml`` location.

    Returns
    -------
    pd.DataFrame
        Metrics table with one row per cell.
    """
    if config is None:
        from hm2p.patching.config import load_config

        config = load_config(Path("config/patching.yaml"))

    metadata = load_metadata(config)
    logger.info("Processing %d cells...", len(metadata))

    cell_metrics: list[dict[str, Any]] = []
    for idx, row in metadata.iterrows():
        try:
            result = process_cell(row, config)
            if result is not None:
                cell_metrics.append(result)
        except Exception:
            logger.exception(
                "Unexpected error processing cell at index %s", idx
            )

    logger.info(
        "Processed %d / %d cells successfully", len(cell_metrics), len(metadata)
    )

    df = build_metrics_table(cell_metrics)
    df = compute_derived_metrics(df)

    # Save output
    config.analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.analysis_dir / "metrics.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved metrics table to %s (%d rows)", output_path, len(df))

    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def run_statistics(metrics_df: pd.DataFrame, config: PatchConfig) -> None:
    """Compute and save summary statistics and group comparisons.

    Imports a ``statistics`` module (expected at
    ``hm2p.patching.statistics``) and calls:
    - ``compute_summary_stats(metrics_df)`` -> saved as ``summary_stats.csv``
    - ``compute_mannwhitney(metrics_df)`` -> saved as ``mannwhitney.csv``

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics table as returned by :func:`run_pipeline`.
    config : PatchConfig
        Pipeline configuration.
    """
    from hm2p.patching import statistics as stats_mod

    config.analysis_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect numeric metric columns (exclude metadata columns)
    meta_cols = {
        "cell_index", "animal_id", "slice_id", "cell_slice_id",
        "hemisphere", "cell_type", "area", "layer", "ephys_id",
        "has_morph", "good_morph",
    }
    metric_cols = [
        c for c in metrics_df.columns
        if c not in meta_cols and metrics_df[c].dtype.kind in ("f", "i")
    ]

    summary = stats_mod.compute_summary_stats(metrics_df, metric_cols)
    summary_path = config.analysis_dir / "summary_stats.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved summary stats to %s", summary_path)

    mw = stats_mod.mann_whitney_comparison(metrics_df, metric_cols)
    mw_path = config.analysis_dir / "mannwhitney.csv"
    mw.to_csv(mw_path, index=False)
    logger.info("Saved Mann-Whitney results to %s", mw_path)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def run_pca_analysis(metrics_df: pd.DataFrame, config: PatchConfig) -> None:
    """Run PCA on metric subsets and save results.

    Imports a ``pca`` module (expected at ``hm2p.patching.pca``) and calls
    ``run_pca(metrics_df, subset)`` for each of ``"ephys"``, ``"morph"``,
    and ``"all"`` subsets.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics table as returned by :func:`run_pipeline`.
    config : PatchConfig
        Pipeline configuration.
    """
    from hm2p.patching import pca as pca_mod

    pca_dir = config.analysis_dir / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    # Define metric column subsets
    ephys_cols = [c for c in metrics_df.columns if c.startswith("ephys_")]
    morph_cols = [c for c in metrics_df.columns if c.startswith("morph_")]
    all_cols = ephys_cols + morph_cols
    subsets = {"ephys": ephys_cols, "morph": morph_cols, "all": all_cols}

    for subset_name, cols in subsets.items():
        if not cols:
            logger.info("No %s columns for PCA — skipping", subset_name)
            continue
        try:
            result = pca_mod.run_pca(metrics_df, metric_cols=cols)
            output_path = pca_dir / f"pca_{subset_name}.csv"
            result.scores.to_csv(output_path, index=False)
            logger.info("Saved PCA (%s) to %s", subset_name, output_path)
        except Exception:
            logger.exception("PCA failed for subset '%s'", subset_name)
