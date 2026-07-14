"""Analysis orchestration — load synced data, run activity + tuning analyses.

Loads ca.h5 + kinematics.h5 (resampled to imaging rate), splits by condition,
computes HD and place tuning curves with significance, and compares light vs dark.
Supports multiple signal types and parameter grids for robustness checking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AnalysisParams:
    """Configurable analysis parameters for robustness checking."""

    signal_type: str = "dff"  # "dff", "deconv", "events"
    speed_threshold: float = 2.5  # cm/s
    # HD tuning
    hd_n_bins: int = 36
    hd_smoothing_sigma_deg: float = 6.0
    # Place tuning
    place_bin_size: float = 2.5  # cm
    place_smoothing_sigma: float = 3.0  # cm
    place_min_occupancy_s: float = 0.5
    # Significance
    n_shuffles: int = 1000
    alpha: float = 0.05


@dataclass
class CellResult:
    """Analysis results for a single cell."""

    roi_idx: int
    # Activity by condition
    activity: dict = field(default_factory=dict)
    # HD tuning (all frames, light, dark)
    hd_all: dict = field(default_factory=dict)
    hd_light: dict = field(default_factory=dict)
    hd_dark: dict = field(default_factory=dict)
    # Place tuning (all frames, light, dark)
    place_all: dict = field(default_factory=dict)
    place_light: dict = field(default_factory=dict)
    place_dark: dict = field(default_factory=dict)
    # Comparison metrics
    hd_comparison: dict = field(default_factory=dict)
    place_comparison: dict = field(default_factory=dict)
    # Gain modulation (light vs dark peak amplitude) — H-N12
    gain: dict = field(default_factory=dict)
    # Junction / decision-point coding — H-N13
    junction: dict = field(default_factory=dict)


def _get_signal(
    dff: np.ndarray,
    deconv: np.ndarray | None,
    event_masks: np.ndarray | None,
    roi_idx: int,
    signal_type: str,
    extra_signals: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Extract the signal array for one ROI based on signal_type.

    Supported signal types:
        dff          — raw dF/F0
        deconv       — Suite2p deconvolved spikes (raw)
        deconv_norm  — Suite2p deconv normalized to [0, 1] per ROI
        events       — V&H binary event mask
        events_sd    — SD-threshold binary event mask
        spikes       — CASCADE calibrated spike rates (spikes/s)
    """
    if extra_signals is None:
        extra_signals = {}

    if signal_type == "dff":
        return dff[roi_idx]
    elif signal_type == "deconv":
        if deconv is None:
            raise ValueError("deconv not available")
        return deconv[roi_idx]
    elif signal_type == "deconv_norm":
        arr = extra_signals.get("deconv_norm")
        if arr is None:
            raise ValueError("deconv_norm not available")
        return arr[roi_idx]
    elif signal_type == "events":
        if event_masks is None:
            raise ValueError("event_masks not available")
        return event_masks[roi_idx].astype(np.float32)
    elif signal_type == "events_sd":
        arr = extra_signals.get("event_masks_sd")
        if arr is None:
            raise ValueError("event_masks_sd not available")
        return arr[roi_idx].astype(np.float32)
    elif signal_type == "spikes":
        arr = extra_signals.get("spikes")
        if arr is None:
            raise ValueError("CASCADE spikes not available")
        return arr[roi_idx]
    else:
        raise ValueError(f"Unknown signal_type: {signal_type!r}")


def _compute_hd_for_condition(
    signal: np.ndarray,
    hd_deg: np.ndarray,
    condition_mask: np.ndarray,
    params: AnalysisParams,
    rng: np.random.Generator,
) -> dict:
    """Compute HD tuning + significance for one condition mask."""
    from hm2p.analysis.significance import hd_tuning_significance
    from hm2p.analysis.tuning import (
        compute_hd_tuning_curve,
        mean_vector_length,
        preferred_direction,
        tuning_width_fwhm,
    )

    tc, centers = compute_hd_tuning_curve(
        signal,
        hd_deg,
        condition_mask,
        n_bins=params.hd_n_bins,
        smoothing_sigma_deg=params.hd_smoothing_sigma_deg,
    )
    mvl = mean_vector_length(tc, centers)
    pd = preferred_direction(tc, centers)
    width = tuning_width_fwhm(tc, centers)

    sig = hd_tuning_significance(
        signal,
        hd_deg,
        condition_mask,
        n_shuffles=params.n_shuffles,
        n_bins=params.hd_n_bins,
        smoothing_sigma_deg=params.hd_smoothing_sigma_deg,
        rng=rng,
    )

    return {
        "tuning_curve": tc,
        "bin_centers": centers,
        "mvl": mvl,
        "preferred_direction": pd,
        "tuning_width": width,
        "p_value": sig["p_value"],
        "significant": sig["p_value"] < params.alpha,
    }


def _compute_place_for_condition(
    signal: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    condition_mask: np.ndarray,
    fps: float,
    params: AnalysisParams,
    rng: np.random.Generator,
) -> dict:
    """Compute place tuning + significance for one condition mask."""
    from hm2p.analysis.significance import place_tuning_significance
    from hm2p.analysis.tuning import (
        compute_place_rate_map,
        spatial_coherence,
        spatial_information,
        spatial_sparsity,
    )

    rate_map, occ_map, bx, by = compute_place_rate_map(
        signal,
        x,
        y,
        condition_mask,
        bin_size=params.place_bin_size,
        smoothing_sigma=params.place_smoothing_sigma,
        min_occupancy_s=params.place_min_occupancy_s,
        fps=fps,
    )
    si = spatial_information(rate_map, occ_map)
    coherence = spatial_coherence(rate_map)
    sparsity = spatial_sparsity(rate_map, occ_map)

    sig = place_tuning_significance(
        signal,
        x,
        y,
        condition_mask,
        n_shuffles=params.n_shuffles,
        bin_size=params.place_bin_size,
        smoothing_sigma=params.place_smoothing_sigma,
        min_occupancy_s=params.place_min_occupancy_s,
        fps=fps,
        rng=rng,
    )

    return {
        "rate_map": rate_map,
        "occupancy_map": occ_map,
        "spatial_info": si,
        "spatial_coherence": coherence,
        "sparsity": sparsity,
        "p_value": sig["p_value"],
        "significant": sig["p_value"] < params.alpha,
    }


def _compute_junction_for_cell(
    signal: np.ndarray,
    hd_deg: np.ndarray,
    location_masks: dict[str, np.ndarray],
    moving: np.ndarray,
    light_on: np.ndarray,
    params: AnalysisParams,
    min_frames: int = 50,
) -> dict:
    """Per-cell activity and HD tuning at junctions vs corridors (H-N13).

    For each of light and dark, computes mean signal and HD tuning MVL
    restricted to (a) junction frames and (b) corridor frames, using the same
    movement gate as the other tuning analyses. Conditions with fewer than
    ``min_frames`` samples yield NaN so downstream tests drop them.

    References
    ----------
    Koren Iton A et al. 2025. "NaviGraph: A graph-based framework for
    multimodal analysis of spatial decision-making." bioRxiv.
    doi:10.1101/2025.05.18.654725
    """
    from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

    junction = np.asarray(location_masks.get("junction"), dtype=bool)
    corridor = np.asarray(location_masks.get("corridor"), dtype=bool)
    out: dict = {}
    for cond_name, cond in (("light", light_on), ("dark", ~light_on)):
        for loc_name, loc in (("junction", junction), ("corridor", corridor)):
            m = moving & cond & loc
            if int(m.sum()) >= min_frames:
                act = float(np.mean(signal[m]))
                tc, centers = compute_hd_tuning_curve(
                    signal,
                    hd_deg,
                    m,
                    n_bins=params.hd_n_bins,
                    smoothing_sigma_deg=params.hd_smoothing_sigma_deg,
                )
                mvl = float(np.clip(mean_vector_length(tc, centers), 0.0, 1.0))
            else:
                act = np.nan
                mvl = np.nan
            out[f"act_{loc_name}_{cond_name}"] = act
            out[f"mvl_{loc_name}_{cond_name}"] = mvl
    return out


def analyze_cell(
    roi_idx: int,
    dff: np.ndarray,
    deconv: np.ndarray | None,
    event_masks: np.ndarray | None,
    hd_deg: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    speed: np.ndarray,
    light_on: np.ndarray,
    active_mask: np.ndarray,
    fps: float,
    params: AnalysisParams | None = None,
    seed: int = 42,
    extra_signals: dict[str, np.ndarray] | None = None,
    location_masks: dict[str, np.ndarray] | None = None,
) -> CellResult:
    """Run full analysis for one cell.

    Args:
        roi_idx: ROI index into dff/deconv/event_masks arrays.
        dff: (n_rois, n_frames) dF/F array.
        deconv: (n_rois, n_frames) deconvolved spikes, or None.
        event_masks: (n_rois, n_frames) bool event masks, or None.
        hd_deg: (n_frames,) head direction in degrees.
        x_cm: (n_frames,) x position in cm.
        y_cm: (n_frames,) y position in cm.
        speed: (n_frames,) speed in cm/s.
        light_on: (n_frames,) bool.
        active_mask: (n_frames,) bool — True for valid (not bad_behav) frames.
        fps: Imaging frame rate.
        params: Analysis parameters.
        seed: Random seed for reproducibility.
        extra_signals: Optional extra signal arrays keyed by signal_type.
        location_masks: Optional per-frame maze node-type masks
            ("junction", "corridor", ...) from
            ``hm2p.maze.neural.classify_frames_by_node_type``. When provided,
            junction/decision-point metrics (H-N13) are computed.

    Returns:
        CellResult with all metrics.
    """
    from hm2p.analysis.activity import compute_cell_activity
    from hm2p.analysis.comparison import (
        mvl_ratio,
        preferred_direction_shift,
        rate_map_correlation,
        tuning_curve_correlation,
    )

    if params is None:
        params = AnalysisParams()
    rng = np.random.default_rng(seed + roi_idx)

    signal = _get_signal(
        dff, deconv, event_masks, roi_idx, params.signal_type, extra_signals=extra_signals
    )
    evt = event_masks[roi_idx] if event_masks is not None else np.zeros_like(signal, dtype=bool)

    result = CellResult(roi_idx=roi_idx)

    # --- Activity by condition ---
    result.activity = compute_cell_activity(
        signal,
        evt,
        speed,
        light_on,
        active_mask,
        fps,
        speed_threshold=params.speed_threshold,
    )

    # --- Masks for HD/place ---
    moving = (speed >= params.speed_threshold) & active_mask
    moving_light = moving & light_on
    moving_dark = moving & ~light_on

    # --- HD tuning ---
    if moving.sum() > 100:
        result.hd_all = _compute_hd_for_condition(signal, hd_deg, moving, params, rng)
    if moving_light.sum() > 100:
        result.hd_light = _compute_hd_for_condition(signal, hd_deg, moving_light, params, rng)
    if moving_dark.sum() > 100:
        result.hd_dark = _compute_hd_for_condition(signal, hd_deg, moving_dark, params, rng)

    # --- HD comparison (light vs dark) ---
    if result.hd_light and result.hd_dark:
        tc_l = result.hd_light["tuning_curve"]
        tc_d = result.hd_dark["tuning_curve"]
        centers = result.hd_light["bin_centers"]
        result.hd_comparison = {
            "correlation": tuning_curve_correlation(tc_l, tc_d),
            "pd_shift": preferred_direction_shift(tc_l, tc_d, centers),
            "mvl_ratio_dark_over_light": mvl_ratio(tc_l, tc_d, centers),
        }

    # --- Gain modulation (light vs dark peak amplitude) — H-N12 ---
    if moving.sum() > 100:
        from hm2p.analysis.gain import gain_modulation_index

        result.gain = gain_modulation_index(
            signal,
            hd_deg,
            moving,
            light_on,
            n_bins=params.hd_n_bins,
            smoothing_sigma_deg=params.hd_smoothing_sigma_deg,
        )

    # --- Place tuning ---
    if moving.sum() > 100:
        result.place_all = _compute_place_for_condition(
            signal,
            x_cm,
            y_cm,
            moving,
            fps,
            params,
            rng,
        )
    if moving_light.sum() > 100:
        result.place_light = _compute_place_for_condition(
            signal,
            x_cm,
            y_cm,
            moving_light,
            fps,
            params,
            rng,
        )
    if moving_dark.sum() > 100:
        result.place_dark = _compute_place_for_condition(
            signal,
            x_cm,
            y_cm,
            moving_dark,
            fps,
            params,
            rng,
        )

    # --- Place comparison (light vs dark) ---
    if result.place_light and result.place_dark:
        result.place_comparison = {
            "correlation": rate_map_correlation(
                result.place_light["rate_map"],
                result.place_dark["rate_map"],
            ),
        }

    # --- Junction / decision-point coding (H-N13) ---
    if location_masks is not None and moving.sum() > 100:
        result.junction = _compute_junction_for_cell(
            signal,
            hd_deg,
            location_masks,
            moving,
            light_on,
            params,
        )

    return result


def check_sync_status(
    sync_h5_path: Path,
    *,
    include_failed_sync: bool = False,
) -> tuple[bool, str, str]:
    """Inspect ``sync.h5`` and decide whether Stage 6 should consume it.

    Per ``docs/sync-pipeline-design.md`` §5, sessions whose ``sync_status``
    starts with ``FAILED_`` are skipped by default. The
    ``include_failed_sync`` override forces analysis to proceed regardless.

    Parameters
    ----------
    sync_h5_path:
        Path to the sync.h5 file produced by Stage 5.
    include_failed_sync:
        When True, return ``(True, status, reason)`` even for FAILED_*.

    Returns
    -------
    proceed, status, reason:
        ``proceed`` is True when analysis should run. ``status`` is the
        ``sync_status`` attr (or ``"NO_SYNC_FILE"``). ``reason`` is a
        human-readable explanation, used by the caller to populate
        ``skipped_reason`` in the sentinel ``analysis.h5``.
    """
    from hm2p.io.hdf5 import read_attrs
    from hm2p.sync.diagnostics import decode_codes_json

    if not Path(sync_h5_path).exists():
        return False, "NO_SYNC_FILE", "sync.h5 not found — Stage 5 has not been run"
    try:
        attrs = read_attrs(sync_h5_path)
    except Exception as exc:  # pragma: no cover — defensive
        return False, "READ_ERROR", f"failed to read sync.h5 attrs: {exc}"
    raw_status = attrs.get("sync_status")
    if raw_status is None:
        return (
            False,
            "NO_STATUS",
            (
                "sync.h5 lacks sync_status attr — file predates the diagnostics "
                "rollout, re-run Stage 5"
            ),
        )
    status = raw_status.decode("utf-8") if isinstance(raw_status, bytes) else str(raw_status)
    if not status.startswith("FAILED_"):
        return True, status, ""
    if include_failed_sync:
        return True, status, "override active — include_failed_sync=True"
    failures_raw = attrs.get("sync_failures", "[]")
    if isinstance(failures_raw, bytes):
        failures_raw = failures_raw.decode("utf-8")
    try:
        failures = decode_codes_json(failures_raw)
        first = failures[0] if failures else status
    except Exception:
        first = status
    return False, status, f"{status}: {first}"


def write_skipped_analysis_h5(
    output_path: Path,
    session_id: str,
    skipped_reason: str,
    sync_status: str,
) -> None:
    """Write a sentinel analysis.h5 for a skipped session.

    The sentinel contains only the metadata needed for downstream
    aggregations to recognise the skip and avoid re-running. Pages that
    consume analysis.h5 must check the ``skipped_reason`` attr and
    suppress per-cell rendering when present.
    """
    from hm2p.io.hdf5 import write_h5

    write_h5(
        output_path,
        arrays={},
        attrs={
            "session_id": session_id,
            "skipped_reason": skipped_reason,
            "sync_status": sync_status,
        },
    )


def analyze_session(
    ca_h5_path: Path,
    kinematics_h5_path: Path,
    timestamps_h5_path: Path,
    params: AnalysisParams | None = None,
) -> list[CellResult]:
    """Run analysis for all cells in one session.

    Loads data, resamples kinematics to imaging rate, and calls analyze_cell
    for each ROI.

    Args:
        ca_h5_path: Path to ca.h5 (Stage 4 output).
        kinematics_h5_path: Path to kinematics.h5 (Stage 3 output).
        timestamps_h5_path: Path to timestamps.h5 (Stage 0 output).
        params: Analysis parameters.

    Returns:
        List of CellResult, one per ROI.
    """
    from hm2p.io.hdf5 import read_h5
    from hm2p.sync.align import resample_to_imaging_rate

    if params is None:
        params = AnalysisParams()

    # Load calcium data
    ca = read_h5(ca_h5_path)
    dff = ca["dff"]  # (n_rois, n_frames)
    deconv = ca.get("spks")
    event_masks = ca.get("event_masks")
    fps = float(ca.get("fps_imaging", 9.8))

    # Load kinematics
    kin = read_h5(kinematics_h5_path)

    # Load timestamps for resampling
    ts = read_h5(timestamps_h5_path)
    cam_times = ts["frame_times_camera"]
    img_times = ts["frame_times_imaging"]

    # Resample kinematics to imaging frame times
    hd_deg = resample_to_imaging_rate(kin["hd_deg"], cam_times, img_times)
    x_mm = resample_to_imaging_rate(kin["x_mm"], cam_times, img_times)
    y_mm = resample_to_imaging_rate(kin["y_mm"], cam_times, img_times)
    speed = resample_to_imaging_rate(kin["speed_cm_s"], cam_times, img_times)
    light_on = (
        resample_to_imaging_rate(
            kin["light_on"].astype(np.float64),
            cam_times,
            img_times,
        )
        > 0.5
    )
    bad_behav = (
        resample_to_imaging_rate(
            kin["bad_behav"].astype(np.float64),
            cam_times,
            img_times,
        )
        > 0.5
    )
    active_mask = ~bad_behav

    # Convert mm to cm for place analysis
    x_cm = x_mm / 10.0
    y_cm = y_mm / 10.0

    # Truncate to common length
    n = min(dff.shape[1], len(hd_deg))
    dff = dff[:, :n]
    if deconv is not None:
        deconv = deconv[:, :n]
    if event_masks is not None:
        event_masks = event_masks[:, :n]
    hd_deg = hd_deg[:n]
    x_cm = x_cm[:n]
    y_cm = y_cm[:n]
    speed = speed[:n]
    light_on = light_on[:n]
    active_mask = active_mask[:n]

    # --- Maze node-type masks for junction analysis (H-N13) ---
    # Requires the maze-registered coordinates written by Stage 3. When absent
    # (older kinematics.h5), junction metrics are skipped and location_masks
    # stays None.
    location_masks = None
    if "x_maze" in kin and "y_maze" in kin:
        from hm2p.maze.discretize import discretize_position_fast
        from hm2p.maze.neural import classify_frames_by_node_type
        from hm2p.maze.topology import build_rose_maze

        maze = build_rose_maze()
        x_maze = resample_to_imaging_rate(kin["x_maze"], cam_times, img_times)[:n]
        y_maze = resample_to_imaging_rate(kin["y_maze"], cam_times, img_times)[:n]
        cell_indices = discretize_position_fast(x_maze, y_maze, maze)
        cell_indices[~active_mask] = -1
        location_masks = classify_frames_by_node_type(cell_indices, maze)

    n_rois = dff.shape[0]
    log.info("Analyzing %d ROIs, %d frames at %.1f Hz", n_rois, n, fps)

    results = []
    for i in range(n_rois):
        log.info("  ROI %d/%d", i + 1, n_rois)
        r = analyze_cell(
            roi_idx=i,
            dff=dff,
            deconv=deconv,
            event_masks=event_masks,
            hd_deg=hd_deg,
            x_cm=x_cm,
            y_cm=y_cm,
            speed=speed,
            light_on=light_on,
            active_mask=active_mask,
            fps=fps,
            params=params,
            location_masks=location_masks,
        )
        results.append(r)

    return results
