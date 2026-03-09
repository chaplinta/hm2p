"""Save and load analysis results to/from HDF5.

Persists per-cell analysis results for every signal type (dff, deconv, events)
so the frontend can compare conclusions across calcium measures without
re-running the analysis.

Output format: analysis.h5
    /{signal_type}/activity/{metric}           (n_rois,)
    /{signal_type}/hd/{condition}/tuning_curve  (n_rois, n_bins)
    /{signal_type}/hd/{condition}/mvl           (n_rois,)
    /{signal_type}/hd/{condition}/pd            (n_rois,)
    /{signal_type}/hd/{condition}/width         (n_rois,)
    /{signal_type}/hd/{condition}/p_value       (n_rois,)
    /{signal_type}/hd/{condition}/significant   (n_rois,)
    /{signal_type}/hd/comparison/{metric}       (n_rois,)
    /{signal_type}/place/{condition}/rate_map    list of 2D arrays
    /{signal_type}/place/{condition}/si          (n_rois,)
    /{signal_type}/place/{condition}/coherence   (n_rois,)
    /{signal_type}/place/{condition}/sparsity    (n_rois,)
    /{signal_type}/place/{condition}/p_value     (n_rois,)
    /{signal_type}/place/comparison/correlation  (n_rois,)
    /bin_centers_hd                              (n_bins,)
    /params/*                                    analysis parameters as attrs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hm2p.analysis.run import AnalysisParams, CellResult

log = logging.getLogger(__name__)


def save_analysis_results(
    output_path: Path,
    results_by_signal: dict[str, list[CellResult]],
    params: AnalysisParams,
    session_id: str,
    n_rois: int,
    n_frames: int,
    fps: float,
    signal_types_available: list[str],
) -> None:
    """Save analysis results for all signal types to a single HDF5 file.

    Args:
        output_path: Destination analysis.h5 file path.
        results_by_signal: Dict mapping signal_type -> list of CellResult.
        params: Analysis parameters used.
        session_id: Session identifier.
        n_rois: Number of ROIs.
        n_frames: Number of frames.
        fps: Imaging frame rate.
        signal_types_available: List of signal types that were available.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Store metadata as root attributes
        f.attrs["session_id"] = session_id
        f.attrs["n_rois"] = n_rois
        f.attrs["n_frames"] = n_frames
        f.attrs["fps"] = fps
        f.attrs["signal_types_available"] = signal_types_available

        # Store analysis parameters
        pg = f.create_group("params")
        pg.attrs["signal_types_run"] = list(results_by_signal.keys())
        pg.attrs["speed_threshold"] = params.speed_threshold
        pg.attrs["hd_n_bins"] = params.hd_n_bins
        pg.attrs["hd_smoothing_sigma_deg"] = params.hd_smoothing_sigma_deg
        pg.attrs["place_bin_size"] = params.place_bin_size
        pg.attrs["place_smoothing_sigma"] = params.place_smoothing_sigma
        pg.attrs["place_min_occupancy_s"] = params.place_min_occupancy_s
        pg.attrs["n_shuffles"] = params.n_shuffles
        pg.attrs["alpha"] = params.alpha

        for signal_type, results in results_by_signal.items():
            sg = f.create_group(signal_type)
            _save_signal_results(sg, results, params)

    log.info(
        "Saved analysis results to %s (%d signal types, %d ROIs)",
        output_path, len(results_by_signal), n_rois,
    )


def _save_signal_results(
    group: h5py.Group,
    results: list[CellResult],
    params: AnalysisParams,
) -> None:
    """Save results for one signal type into an HDF5 group."""
    n_rois = len(results)

    # --- Activity metrics ---
    ag = group.create_group("activity")
    if results and results[0].activity:
        metric_keys = list(results[0].activity.keys())
        for key in metric_keys:
            vals = np.array([r.activity.get(key, np.nan) for r in results], dtype=np.float32)
            ag.create_dataset(key, data=vals, compression="gzip")

    # --- HD tuning ---
    hg = group.create_group("hd")
    for condition in ("all", "light", "dark"):
        attr_name = f"hd_{condition}"
        cg = hg.create_group(condition)

        # Collect tuning curves and scalar metrics
        tuning_curves = []
        mvls = np.full(n_rois, np.nan, dtype=np.float32)
        pds = np.full(n_rois, np.nan, dtype=np.float32)
        widths = np.full(n_rois, np.nan, dtype=np.float32)
        p_values = np.full(n_rois, np.nan, dtype=np.float32)
        significant = np.zeros(n_rois, dtype=bool)

        for i, r in enumerate(results):
            hd_data = getattr(r, attr_name)
            if hd_data:
                tc = hd_data.get("tuning_curve")
                if tc is not None:
                    tuning_curves.append(tc)
                else:
                    tuning_curves.append(np.full(params.hd_n_bins, np.nan))
                mvls[i] = hd_data.get("mvl", np.nan)
                pds[i] = hd_data.get("preferred_direction", np.nan)
                widths[i] = hd_data.get("tuning_width", np.nan)
                p_values[i] = hd_data.get("p_value", np.nan)
                significant[i] = hd_data.get("significant", False)
            else:
                tuning_curves.append(np.full(params.hd_n_bins, np.nan))

        if tuning_curves:
            cg.create_dataset(
                "tuning_curves", data=np.array(tuning_curves, dtype=np.float32),
                compression="gzip",
            )
            # Store bin centers (same for all)
            if results and results[0].hd_all and "bin_centers" in results[0].hd_all:
                centers = results[0].hd_all["bin_centers"]
                if "bin_centers" not in hg:
                    hg.create_dataset("bin_centers", data=centers.astype(np.float32))

        cg.create_dataset("mvl", data=mvls, compression="gzip")
        cg.create_dataset("preferred_direction", data=pds, compression="gzip")
        cg.create_dataset("tuning_width", data=widths, compression="gzip")
        cg.create_dataset("p_value", data=p_values, compression="gzip")
        cg.create_dataset("significant", data=significant, compression="gzip")

    # HD comparison (light vs dark)
    hcomp = hg.create_group("comparison")
    corrs = np.full(n_rois, np.nan, dtype=np.float32)
    pd_shifts = np.full(n_rois, np.nan, dtype=np.float32)
    mvl_ratios = np.full(n_rois, np.nan, dtype=np.float32)
    for i, r in enumerate(results):
        if r.hd_comparison:
            corrs[i] = r.hd_comparison.get("correlation", np.nan)
            pd_shifts[i] = r.hd_comparison.get("pd_shift", np.nan)
            mvl_ratios[i] = r.hd_comparison.get("mvl_ratio_dark_over_light", np.nan)
    hcomp.create_dataset("correlation", data=corrs, compression="gzip")
    hcomp.create_dataset("pd_shift", data=pd_shifts, compression="gzip")
    hcomp.create_dataset("mvl_ratio", data=mvl_ratios, compression="gzip")

    # --- Place tuning ---
    pg = group.create_group("place")
    for condition in ("all", "light", "dark"):
        attr_name = f"place_{condition}"
        cg = pg.create_group(condition)

        si_vals = np.full(n_rois, np.nan, dtype=np.float32)
        coherence_vals = np.full(n_rois, np.nan, dtype=np.float32)
        sparsity_vals = np.full(n_rois, np.nan, dtype=np.float32)
        p_values = np.full(n_rois, np.nan, dtype=np.float32)
        significant = np.zeros(n_rois, dtype=bool)

        for i, r in enumerate(results):
            place_data = getattr(r, attr_name)
            if place_data:
                si_vals[i] = place_data.get("spatial_info", np.nan)
                coherence_vals[i] = place_data.get("spatial_coherence", np.nan)
                sparsity_vals[i] = place_data.get("sparsity", np.nan)
                p_values[i] = place_data.get("p_value", np.nan)
                significant[i] = place_data.get("significant", False)

        cg.create_dataset("spatial_info", data=si_vals, compression="gzip")
        cg.create_dataset("spatial_coherence", data=coherence_vals, compression="gzip")
        cg.create_dataset("sparsity", data=sparsity_vals, compression="gzip")
        cg.create_dataset("p_value", data=p_values, compression="gzip")
        cg.create_dataset("significant", data=significant, compression="gzip")

    # Place comparison
    pcomp = pg.create_group("comparison")
    place_corrs = np.full(n_rois, np.nan, dtype=np.float32)
    for i, r in enumerate(results):
        if r.place_comparison:
            place_corrs[i] = r.place_comparison.get("correlation", np.nan)
    pcomp.create_dataset("correlation", data=place_corrs, compression="gzip")


def load_analysis_results(path: Path) -> dict[str, Any]:
    """Load analysis results from analysis.h5 into a nested dict.

    Returns:
        Dict with structure:
            'meta': {session_id, n_rois, n_frames, fps, signal_types_available}
            'params': {speed_threshold, hd_n_bins, ...}
            '{signal_type}': {
                'activity': {metric_name: (n_rois,) array},
                'hd': {
                    'bin_centers': (n_bins,),
                    'all'/'light'/'dark': {
                        'tuning_curves': (n_rois, n_bins),
                        'mvl': (n_rois,), 'pd': (n_rois,), ...
                    },
                    'comparison': {correlation, pd_shift, mvl_ratio}
                },
                'place': { similar structure }
            }
    """
    if not path.exists():
        raise FileNotFoundError(f"Analysis file not found: {path}")

    data: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        # Root metadata
        data["meta"] = dict(f.attrs)

        # Parameters
        if "params" in f:
            data["params"] = dict(f["params"].attrs)

        # Signal type results
        signal_types = data.get("meta", {}).get("signal_types_available", [])
        for st in f.keys():
            if st == "params":
                continue
            sg = f[st]
            if not isinstance(sg, h5py.Group):
                continue
            data[st] = _load_group_recursive(sg)

    return data


def _load_group_recursive(group: h5py.Group) -> dict[str, Any]:
    """Recursively load an HDF5 group into a nested dict."""
    result: dict[str, Any] = {}
    # Load attributes
    for k, v in group.attrs.items():
        result[f"_attr_{k}"] = v
    # Load datasets and subgroups
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = item[:]
        elif isinstance(item, h5py.Group):
            result[key] = _load_group_recursive(item)
    return result
