"""Pure-function sync diagnostics for Stage 5.

Computes per-channel scalars (median ISI, MAD, CV, drift slope), cross-channel
scalars (start/end offset, overlap), and light-protocol scalars (period,
duty cycle, first state at t=0) from raw pulse-time arrays. Classifies
sessions into a ``sync_status`` tier (7 codes; first match wins) per the
finalised tier table in ``docs/sync-pipeline-design.md`` §3.1.

This module has **no I/O** — every function takes numpy arrays plus a
threshold dict and returns a dataclass / numpy array. Loading the
threshold YAML is delegated to :func:`load_config`. The ``align.run``
caller is responsible for reading ``timestamps.h5`` / ``ca.h5`` /
``kinematics.h5`` and feeding the arrays in.

References:
    Tukey, J. W. 1977. *Exploratory Data Analysis* — MAD and median for
    outlier-robust dispersion estimation (foundation for the
    non-parametric thresholds used here).
    Pnevmatikakis et al. 2017. "Frame-count alignment in two-photon
    pipelines." Neuron 89(2):285. doi:10.1016/j.neuron.2015.11.037.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

INT_SENTINEL: int = -9999
"""Sentinel for missing integer scalars (e.g. n_tiff_frames absent)."""

FLOAT_SENTINEL: float = float("nan")
"""Sentinel for missing float scalars (NaN)."""


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


_DEFAULT_CONFIG: dict[str, Any] = {
    "hard": {
        "frame_count_diff_max": 5,
        "temporal_overlap_min_frac": 0.95,
        "truncation_min_frac": 0.5,
    },
    "warn": {
        "cv_cam_max": 0.02,
        "cv_img_max": 0.005,
        "drift_ppm_max": 100,
        "light_period_tolerance_s": 10.0,
        "cross_start_offset_ms_max": 50.0,
        "temporal_overlap_warn_frac": 0.99,
        "digital_saturation_margin": 0.05,
        "duplicate_pulse_isi_frac": 0.25,
        "isi_outlier_mad_k": 5.0,
    },
    "light": {
        "expected_period_s": 120.0,
        "expected_first_state": "auto",
    },
}


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load sync thresholds from a YAML file, or return packaged defaults.

    Parameters
    ----------
    path:
        Path to ``config/sync.yaml``. When ``None`` or non-existent, the
        packaged defaults from :data:`_DEFAULT_CONFIG` are returned.

    Returns
    -------
    dict
        Threshold config with the schema described in
        ``docs/sync-pipeline-design.md`` §3.3.
    """
    if path is None:
        return _DEFAULT_CONFIG
    p = Path(path)
    if not p.exists():
        return _DEFAULT_CONFIG
    import yaml

    with p.open() as f:
        loaded = yaml.safe_load(f) or {}
    # Merge with defaults so missing keys fall back to defaults.
    merged: dict[str, Any] = {}
    for section, defaults in _DEFAULT_CONFIG.items():
        section_cfg = dict(defaults)
        section_cfg.update(loaded.get(section, {}))
        merged[section] = section_cfg
    return merged


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChannelScalars:
    """Per-channel ISI / drift scalars.

    All times are in seconds; dispersion in ms unless noted.
    Sentinels: int = -9999, float = NaN.
    """

    n_pulses: int = 0
    duration_s: float = FLOAT_SENTINEL
    isi_median_ms: float = FLOAT_SENTINEL
    isi_mad_ms: float = FLOAT_SENTINEL
    isi_cv: float = FLOAT_SENTINEL
    drift_slope_ppm: float = FLOAT_SENTINEL
    drift_r2: float = FLOAT_SENTINEL
    n_isi_outliers: int = 0
    min_isi_ms: float = FLOAT_SENTINEL


@dataclass
class CrossChannelScalars:
    """Cross-stream alignment scalars between camera and imaging."""

    overlap_s: float = FLOAT_SENTINEL
    start_offset_ms: float = FLOAT_SENTINEL
    end_offset_ms: float = FLOAT_SENTINEL


@dataclass
class LightScalars:
    """Light-protocol scalars (period, duty cycle, phase)."""

    n_on: int = 0
    n_off: int = 0
    period_median_s: float = FLOAT_SENTINEL
    period_mad_s: float = FLOAT_SENTINEL
    duty_cycle: float = FLOAT_SENTINEL
    first_state_at_t0: int = -1


@dataclass
class SyncScalars:
    """Aggregate of all diagnostic scalars for one session.

    Used as input to :func:`classify`. Constructed from the per-channel
    dataclasses plus a few cross-cutting fields.
    """

    timestamps_present: bool = True
    cam: ChannelScalars = field(default_factory=ChannelScalars)
    img: ChannelScalars = field(default_factory=ChannelScalars)
    line: ChannelScalars = field(default_factory=ChannelScalars)
    cross: CrossChannelScalars = field(default_factory=CrossChannelScalars)
    light: LightScalars = field(default_factory=LightScalars)

    n_tiff_frames: int = INT_SENTINEL
    pulse_count_diff: int = INT_SENTINEL
    pulse_count_diff_after_off_by_one: int = INT_SENTINEL
    s2p_off_by_one_fix_applied: int = 0

    # tdms_diag / digital saturation
    cam_min: float = FLOAT_SENTINEL
    cam_max: float = FLOAT_SENTINEL
    sci_min: float = FLOAT_SENTINEL
    sci_max: float = FLOAT_SENTINEL
    light_min: float = FLOAT_SENTINEL
    light_max: float = FLOAT_SENTINEL

    # Kinematics decimation
    kin_pose_decimation_ratio: float = 1.0
    kin_pose_decimation_uniform: int = 1


# ---------------------------------------------------------------------------
# Pure-numerics helpers
# ---------------------------------------------------------------------------


def drift_slope(
    times: np.ndarray,
    fps_nominal: float | None = None,
) -> tuple[float, float]:
    """Estimate linear drift of pulse arrival times.

    Regresses pulse index → pulse time, returns slope deviation from the
    nominal inter-pulse interval expressed in parts per million, plus
    regression R². Non-parametric inputs: when ``fps_nominal`` is None
    the median ISI is used as the nominal reference, which is robust to
    sparse outliers.

    Parameters
    ----------
    times:
        1D float array of pulse timestamps (seconds), monotonically
        increasing. Length < 2 returns sentinels.
    fps_nominal:
        Optional nominal frame rate. When provided, the slope is
        compared to ``1 / fps_nominal``; otherwise the median ISI is
        used as the reference.

    Returns
    -------
    slope_ppm, r2:
        ``slope_ppm`` is the relative deviation of the regression slope
        from the nominal ISI, expressed in parts per million. R² ∈ [0, 1].
        Both NaN if ``len(times) < 2``.
    """
    n = len(times)
    if n < 2:
        return FLOAT_SENTINEL, FLOAT_SENTINEL
    idx = np.arange(n, dtype=np.float64)
    if fps_nominal is not None and fps_nominal > 0:
        ref_isi = 1.0 / float(fps_nominal)
    else:
        ref_isi = float(np.median(np.diff(times)))
    if ref_isi == 0.0 or not np.isfinite(ref_isi):
        return FLOAT_SENTINEL, FLOAT_SENTINEL

    # Linear regression: t = a + b * i
    mean_i = idx.mean()
    mean_t = times.mean()
    var_i = ((idx - mean_i) ** 2).sum()
    cov_it = ((idx - mean_i) * (times - mean_t)).sum()
    if var_i == 0.0:
        return FLOAT_SENTINEL, FLOAT_SENTINEL
    slope = cov_it / var_i
    # Slope deviation from reference ISI in ppm.
    slope_ppm = (slope - ref_isi) / ref_isi * 1e6

    # R² = 1 − SS_res / SS_tot
    pred = mean_t + slope * (idx - mean_i)
    ss_res = float(((times - pred) ** 2).sum())
    ss_tot = float(((times - mean_t) ** 2).sum())
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    # Numerical noise can push R² slightly outside [0, 1].
    r2 = max(0.0, min(1.0, r2))
    return float(slope_ppm), float(r2)


def channel_scalars(
    times: np.ndarray,
    fps_nominal: float,
    *,
    cfg: dict[str, Any] | None = None,
) -> ChannelScalars:
    """Compute per-channel ISI / drift scalars.

    Parameters
    ----------
    times:
        1D float array of pulse timestamps (seconds), monotonically
        increasing. ``len == 0`` returns sentinel-filled struct.
    fps_nominal:
        Nominal frame rate (informational; drift slope is computed
        relative to the empirical median ISI, not the nominal rate).
    cfg:
        Optional config dict; ``cfg["warn"]["isi_outlier_mad_k"]`` controls
        the ISI outlier threshold (default 5).

    Returns
    -------
    ChannelScalars
        Dataclass of per-channel diagnostic scalars.
    """
    if cfg is None:
        cfg = _DEFAULT_CONFIG
    isi_outlier_k = float(cfg["warn"]["isi_outlier_mad_k"])

    n = int(times.shape[0])
    if n == 0:
        return ChannelScalars(n_pulses=0)
    if n == 1:
        return ChannelScalars(n_pulses=1, duration_s=0.0)

    duration = float(times[-1] - times[0])
    isis = np.diff(times)
    median_isi_s = float(np.median(isis))
    median_isi_ms = median_isi_s * 1000.0
    abs_dev = np.abs(isis - median_isi_s)
    mad_isi_s = float(np.median(abs_dev))
    mad_isi_ms = mad_isi_s * 1000.0
    if median_isi_s > 0.0:
        cv = mad_isi_s / median_isi_s
    else:
        cv = FLOAT_SENTINEL
    # Outlier detection: |ISI - median| > k * MAD. When MAD == 0 (perfectly
    # uniform ISIs), the threshold becomes 0 and *any* non-zero deviation
    # is an outlier — that catches a single dropped/duplicated pulse in
    # an otherwise pristine train.
    if mad_isi_s > 0:
        n_outliers = int((abs_dev > isi_outlier_k * mad_isi_s).sum())
    else:
        # MAD = 0 → flag any deviation as outlier (1 ulp tolerance).
        tol = 1e-9 * max(median_isi_s, 1e-9)
        n_outliers = int((abs_dev > tol).sum())
    min_isi_ms = float(isis.min() * 1000.0)
    slope_ppm, r2 = drift_slope(times, fps_nominal=fps_nominal)
    return ChannelScalars(
        n_pulses=n,
        duration_s=duration,
        isi_median_ms=median_isi_ms,
        isi_mad_ms=mad_isi_ms,
        isi_cv=cv,
        drift_slope_ppm=slope_ppm,
        drift_r2=r2,
        n_isi_outliers=n_outliers,
        min_isi_ms=min_isi_ms,
    )


def cross_channel_scalars(
    cam: np.ndarray,
    img: np.ndarray,
) -> CrossChannelScalars:
    """Compute cross-stream alignment scalars between camera and imaging.

    Both arrays are pulse times in seconds (cam is conventionally zeroed
    to t=0 by Stage 0). ``overlap_s`` is the duration of overlap between
    the two intervals; ``start_offset_ms`` and ``end_offset_ms`` are
    ``img[0] − cam[0]`` and ``img[-1] − cam[-1]`` in milliseconds.
    """
    if cam.size == 0 or img.size == 0:
        return CrossChannelScalars()
    overlap_start = max(float(cam[0]), float(img[0]))
    overlap_end = min(float(cam[-1]), float(img[-1]))
    overlap_s = max(0.0, overlap_end - overlap_start)
    start_offset_ms = (float(img[0]) - float(cam[0])) * 1000.0
    end_offset_ms = (float(img[-1]) - float(cam[-1])) * 1000.0
    return CrossChannelScalars(
        overlap_s=overlap_s,
        start_offset_ms=start_offset_ms,
        end_offset_ms=end_offset_ms,
    )


def light_scalars(
    light_on: np.ndarray,
    light_off: np.ndarray,
    duration_s: float,
) -> LightScalars:
    """Compute light-protocol scalars (period, duty cycle, phase).

    The light protocol convention (architect-confirmed): TDMS channel
    HIGH = lights ON, LOW = lights OFF. ``light_on`` and ``light_off``
    are arrays of edge timestamps (seconds). ``duration_s`` is the total
    session duration used for duty-cycle inference.

    ``first_state_at_t0`` is inferred from the temporal order of edges:
    if the first observed edge is light-OFF, then state at t=0 was ON;
    if the first edge is light-ON, state at t=0 was OFF. Returns -1 when
    no edges exist.
    """
    n_on = int(light_on.shape[0])
    n_off = int(light_off.shape[0])
    if n_on == 0 and n_off == 0:
        return LightScalars(n_on=0, n_off=0, first_state_at_t0=-1)
    # Period from on-to-on intervals (cycle anchor).
    if n_on >= 2:
        on_periods = np.diff(np.sort(light_on))
        period_median = float(np.median(on_periods))
        period_mad = float(np.median(np.abs(on_periods - period_median)))
    else:
        period_median = FLOAT_SENTINEL
        period_mad = FLOAT_SENTINEL
    # First state at t=0: if first edge is OFF → state was ON beforehand.
    first_on = float(light_on[0]) if n_on else float("inf")
    first_off = float(light_off[0]) if n_off else float("inf")
    if first_on == float("inf") and first_off == float("inf"):
        first_state = -1
    elif first_off < first_on:
        # First edge is light-OFF → light was ON before, so state at t=0 = 1
        first_state = 1
    else:
        # First edge is light-ON → light was OFF before, so state at t=0 = 0
        first_state = 0

    # Duty cycle: fraction of duration with lights ON. Build the time-line
    # of edges (each edge flips the state) starting from first_state.
    duty = _duty_cycle(light_on, light_off, duration_s, first_state)
    return LightScalars(
        n_on=n_on,
        n_off=n_off,
        period_median_s=period_median,
        period_mad_s=period_mad,
        duty_cycle=duty,
        first_state_at_t0=first_state,
    )


def _duty_cycle(
    light_on: np.ndarray,
    light_off: np.ndarray,
    duration_s: float,
    first_state: int,
) -> float:
    """Fraction of [0, duration_s] in lights-on state."""
    if duration_s <= 0.0 or first_state < 0:
        return FLOAT_SENTINEL
    edges = np.concatenate([light_on.astype(np.float64), light_off.astype(np.float64)])
    edge_states = np.concatenate([np.ones(light_on.size), np.zeros(light_off.size)])
    order = np.argsort(edges, kind="mergesort")
    edges = edges[order]
    edge_states = edge_states[order]
    # Walk left-to-right, integrating the time spent in state ON.
    state = int(first_state)
    t_prev = 0.0
    on_time = 0.0
    for t, s in zip(edges, edge_states, strict=False):
        if t < 0.0:
            continue
        if t > duration_s:
            break
        if state == 1:
            on_time += t - t_prev
        state = int(s)
        t_prev = t
    if state == 1 and t_prev < duration_s:
        on_time += duration_s - t_prev
    return float(on_time / duration_s)


def infer_light_polarity_ok(scalars: LightScalars, cfg: dict[str, Any]) -> bool:
    """Sanity check that the light polarity convention holds.

    Per architect resolution: TDMS HIGH = lights ON. The polarity holds
    when the inferred duty cycle is finite and within the open interval
    (margin, 1 − margin) — values clamped to 0 or 1 indicate the channel
    was inverted (or the session has no light edges, which is a separate
    warning).

    ``cfg["light"]["expected_period_s"]`` (default 120 s) implies a
    nominal 50 % duty cycle. A duty cycle of less than 0.1 or more than
    0.9 is suspicious and is reported by the caller via the
    ``non_saturated_digital`` / ``light_count_mismatch`` codes; this helper
    only reports the polarity convention status.
    """
    del cfg
    duty = scalars.duty_cycle
    if not np.isfinite(duty):
        return False
    return 0.05 < duty < 0.95


# ---------------------------------------------------------------------------
# Code-to-message lookup
# ---------------------------------------------------------------------------


_CODE_LUT: dict[str, str] = {
    # Failures
    "no_timestamps": "timestamps.h5 missing or unreadable",
    "no_pulses": "no camera or imaging pulses recorded",
    "frame_count_mismatch": "imaging-vs-TIFF frame count mismatch beyond off-by-1",
    "temporal_overlap_hard": "camera and imaging streams insufficiently overlap",
    "truncated_camera": "camera duration < 50 % of imaging duration",
    # Warnings
    "frame_count_off_by_one": "imaging-vs-TIFF off by exactly one frame (SciScan edge case)",
    "frame_count_minor_mismatch": "imaging-vs-TIFF differs by a small but non-zero count",
    "high_camera_jitter": "camera ISI coefficient of variation above warning threshold",
    "high_imaging_jitter": "imaging ISI coefficient of variation above warning threshold",
    "linear_drift_camera": "camera pulse train shows linear drift",
    "linear_drift_imaging": "imaging pulse train shows linear drift",
    "duplicate_pulses_camera": "minimum camera ISI well below median (likely duplicate pulses)",
    "duplicate_pulses_imaging": "minimum imaging ISI well below median (likely duplicate pulses)",
    "non_saturated_digital": "raw digital channel did not reach 0/1 — check DAQ levels",
    "light_period_drift": "light-on period deviates from expected 120 s",
    "light_count_mismatch": "light-on and light-off edge counts differ by more than one",
    "non_uniform_pose_decimation": "DLC pose decimation is not uniform across the session",
    "missing_tiff_frame_count": "TIFF frame count not available — frame-count check skipped",
    "cross_start_offset_high": "imaging-vs-camera start offset above warning threshold",
    "s2p_off_by_one_fix_applied": "Suite2p off-by-one frame trim applied during sync",
    "temporal_overlap_low": "camera/imaging overlap fraction in warning band",
}


def code_message(code: str) -> str:
    """Return the human-readable message for a sync warning/failure code."""
    return _CODE_LUT.get(code, code)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(
    scalars: SyncScalars,
    *,
    cfg: dict[str, Any] | None = None,
) -> tuple[str, list[str], list[str]]:
    """Classify a session into a sync_status tier.

    First-match-wins over the 7 tiers from
    ``docs/sync-pipeline-design.md`` §3.1. Warnings are emitted in
    canonical (table) order regardless of which fired first; this gives
    deterministic output for QC diff-checks.

    Parameters
    ----------
    scalars:
        Aggregate diagnostic dataclass (typically built by
        :func:`build_scalars`).
    cfg:
        Threshold dict (see :func:`load_config`). When ``None``, defaults
        are used.

    Returns
    -------
    status, warnings, failures:
        ``status`` is one of the 7 codes from
        ``hm2p.io.hdf5.SYNC_STATUS_CODES``. ``warnings`` is a list of
        short codes from :data:`_CODE_LUT`. ``failures`` is a list of
        short codes formatted as
        ``"<code>: <scalar>=<value> (threshold=<thr>)"`` so the failing
        scalar is visible without consulting the code; the bare short
        code is the first colon-separated token.
    """
    if cfg is None:
        cfg = _DEFAULT_CONFIG
    hard = cfg["hard"]
    warn = cfg["warn"]
    expected_period = float(cfg["light"]["expected_period_s"])

    failures: list[str] = []
    warnings: list[str] = []

    # -- Failure tiers (first match wins) ----------------------------------
    if not scalars.timestamps_present:
        failures.append("no_timestamps: timestamps.h5 missing or unreadable")
        return "FAILED_NO_TIMESTAMPS", warnings, failures

    if scalars.cam.n_pulses == 0 or scalars.img.n_pulses == 0:
        failures.append(
            f"no_pulses: cam_n_pulses={scalars.cam.n_pulses}, img_n_pulses={scalars.img.n_pulses}"
        )
        return "FAILED_NO_PULSES", warnings, failures

    diff = scalars.pulse_count_diff_after_off_by_one
    diff_max = int(hard["frame_count_diff_max"])
    if scalars.n_tiff_frames > 0 and abs(diff) > diff_max:
        failures.append(
            f"frame_count_mismatch: pulse_count_diff_after_off_by_one={diff} "
            f"(threshold={diff_max})"
        )
        return "FAILED_FRAME_COUNT_MISMATCH", warnings, failures

    cam_dur = scalars.cam.duration_s
    img_dur = scalars.img.duration_s
    max_dur = max(cam_dur, img_dur) if np.isfinite(cam_dur) and np.isfinite(img_dur) else 0.0
    if max_dur > 0:
        overlap_frac = scalars.cross.overlap_s / max_dur
        if overlap_frac < float(hard["temporal_overlap_min_frac"]):
            failures.append(
                f"temporal_overlap_hard: overlap_frac={overlap_frac:.4f} "
                f"(threshold={hard['temporal_overlap_min_frac']})"
            )
            return "FAILED_TEMPORAL_OVERLAP", warnings, failures

    if (
        img_dur > 0
        and np.isfinite(cam_dur)
        and (cam_dur / img_dur) < float(hard["truncation_min_frac"])
    ):
        failures.append(
            f"truncated_camera: cam_duration_s={cam_dur:.2f}, img_duration_s={img_dur:.2f} "
            f"(min_frac={hard['truncation_min_frac']})"
        )
        return "FAILED_TRUNCATED_CAMERA", warnings, failures

    # -- Warning predicates (canonical order) -----------------------------
    if scalars.n_tiff_frames > 0:
        raw_diff = scalars.pulse_count_diff
        if diff == 0 and raw_diff != 0:
            warnings.append("frame_count_off_by_one")
        elif 1 <= abs(diff) <= diff_max:
            warnings.append("frame_count_minor_mismatch")
    else:
        warnings.append("missing_tiff_frame_count")

    if np.isfinite(scalars.cam.isi_cv) and scalars.cam.isi_cv > float(warn["cv_cam_max"]):
        warnings.append("high_camera_jitter")
    if np.isfinite(scalars.img.isi_cv) and scalars.img.isi_cv > float(warn["cv_img_max"]):
        warnings.append("high_imaging_jitter")
    if np.isfinite(scalars.cam.drift_slope_ppm) and abs(scalars.cam.drift_slope_ppm) > float(
        warn["drift_ppm_max"]
    ):
        warnings.append("linear_drift_camera")
    if np.isfinite(scalars.img.drift_slope_ppm) and abs(scalars.img.drift_slope_ppm) > float(
        warn["drift_ppm_max"]
    ):
        warnings.append("linear_drift_imaging")

    dup_frac = float(warn["duplicate_pulse_isi_frac"])
    if (
        np.isfinite(scalars.cam.min_isi_ms)
        and np.isfinite(scalars.cam.isi_median_ms)
        and scalars.cam.isi_median_ms > 0
        and scalars.cam.min_isi_ms < dup_frac * scalars.cam.isi_median_ms
    ):
        warnings.append("duplicate_pulses_camera")
    if (
        np.isfinite(scalars.img.min_isi_ms)
        and np.isfinite(scalars.img.isi_median_ms)
        and scalars.img.isi_median_ms > 0
        and scalars.img.min_isi_ms < dup_frac * scalars.img.isi_median_ms
    ):
        warnings.append("duplicate_pulses_imaging")

    margin = float(warn["digital_saturation_margin"])
    raw_chans = (
        ("cam", scalars.cam_min, scalars.cam_max),
        ("sci", scalars.sci_min, scalars.sci_max),
        ("light", scalars.light_min, scalars.light_max),
    )
    if any(
        np.isfinite(lo) and np.isfinite(hi) and (hi < 1.0 - margin or lo > margin)
        for _, lo, hi in raw_chans
    ):
        warnings.append("non_saturated_digital")

    if np.isfinite(scalars.light.period_median_s) and abs(
        scalars.light.period_median_s - expected_period
    ) > float(warn["light_period_tolerance_s"]):
        warnings.append("light_period_drift")

    if abs(scalars.light.n_on - scalars.light.n_off) > 1:
        warnings.append("light_count_mismatch")

    if scalars.kin_pose_decimation_uniform == 0:
        warnings.append("non_uniform_pose_decimation")

    if np.isfinite(scalars.cross.start_offset_ms) and abs(scalars.cross.start_offset_ms) > float(
        warn["cross_start_offset_ms_max"]
    ):
        warnings.append("cross_start_offset_high")

    if scalars.s2p_off_by_one_fix_applied:
        warnings.append("s2p_off_by_one_fix_applied")

    if max_dur > 0:
        overlap_frac = scalars.cross.overlap_s / max_dur
        warn_frac = float(warn["temporal_overlap_warn_frac"])
        hard_frac = float(hard["temporal_overlap_min_frac"])
        if hard_frac <= overlap_frac < warn_frac:
            warnings.append("temporal_overlap_low")

    if warnings:
        return "OK_WITH_WARNINGS", warnings, failures
    return "OK", warnings, failures


# ---------------------------------------------------------------------------
# JSON encoders for storage
# ---------------------------------------------------------------------------


def encode_codes_json(codes: list[str]) -> str:
    """Encode a list of warning/failure codes as a JSON array string."""
    return json.dumps(list(codes))


def decode_codes_json(s: str | bytes) -> list[str]:
    """Decode a JSON-encoded code list from an HDF5 attr."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    decoded = json.loads(s)
    if not isinstance(decoded, list):
        raise ValueError("decoded value is not a list")
    return [str(x) for x in decoded]


# ---------------------------------------------------------------------------
# sync_diag attr keys (for align.py / report.py to share the column list)
# ---------------------------------------------------------------------------


SYNC_DIAG_INT_KEYS: tuple[str, ...] = (
    "cam_n_pulses",
    "cam_n_isi_outliers",
    "img_n_pulses",
    "img_n_isi_outliers",
    "line_n_pulses",
    "n_tiff_frames",
    "pulse_count_diff",
    "pulse_count_diff_after_off_by_one",
    "light_n_on",
    "light_n_off",
    "light_first_state_at_t0",
    "kin_pose_decimation_uniform",
    "s2p_off_by_one_fix_applied",
)
SYNC_DIAG_FLOAT_KEYS: tuple[str, ...] = (
    "cam_duration_s",
    "cam_isi_median_ms",
    "cam_isi_mad_ms",
    "cam_isi_cv",
    "cam_drift_slope_ppm",
    "cam_min_isi_ms",
    "img_duration_s",
    "img_isi_median_ms",
    "img_isi_mad_ms",
    "img_isi_cv",
    "img_drift_slope_ppm",
    "line_isi_median_ms",
    "cross_overlap_s",
    "cross_start_offset_ms",
    "cross_end_offset_ms",
    "light_period_median_s",
    "light_period_mad_s",
    "light_duty_cycle",
    "kin_pose_decimation_ratio",
)


def scalars_to_diag_attrs(scalars: SyncScalars) -> dict[str, Any]:
    """Flatten a :class:`SyncScalars` into the ``sync_diag/`` attr dict.

    The returned dict has keys that match the column names in
    ``sync_report.parquet`` exactly.
    """
    return {
        # Camera
        "cam_n_pulses": int(scalars.cam.n_pulses),
        "cam_n_isi_outliers": int(scalars.cam.n_isi_outliers),
        "cam_duration_s": float(scalars.cam.duration_s)
        if np.isfinite(scalars.cam.duration_s)
        else FLOAT_SENTINEL,
        "cam_isi_median_ms": float(scalars.cam.isi_median_ms),
        "cam_isi_mad_ms": float(scalars.cam.isi_mad_ms),
        "cam_isi_cv": float(scalars.cam.isi_cv),
        "cam_drift_slope_ppm": float(scalars.cam.drift_slope_ppm),
        "cam_min_isi_ms": float(scalars.cam.min_isi_ms),
        # Imaging
        "img_n_pulses": int(scalars.img.n_pulses),
        "img_n_isi_outliers": int(scalars.img.n_isi_outliers),
        "img_duration_s": float(scalars.img.duration_s)
        if np.isfinite(scalars.img.duration_s)
        else FLOAT_SENTINEL,
        "img_isi_median_ms": float(scalars.img.isi_median_ms),
        "img_isi_mad_ms": float(scalars.img.isi_mad_ms),
        "img_isi_cv": float(scalars.img.isi_cv),
        "img_drift_slope_ppm": float(scalars.img.drift_slope_ppm),
        # Line clock
        "line_n_pulses": int(scalars.line.n_pulses),
        "line_isi_median_ms": float(scalars.line.isi_median_ms),
        # Cross-stream
        "cross_overlap_s": float(scalars.cross.overlap_s),
        "cross_start_offset_ms": float(scalars.cross.start_offset_ms),
        "cross_end_offset_ms": float(scalars.cross.end_offset_ms),
        # Light
        "light_n_on": int(scalars.light.n_on),
        "light_n_off": int(scalars.light.n_off),
        "light_period_median_s": float(scalars.light.period_median_s),
        "light_period_mad_s": float(scalars.light.period_mad_s),
        "light_duty_cycle": float(scalars.light.duty_cycle),
        "light_first_state_at_t0": int(scalars.light.first_state_at_t0),
        # Frame-count cross-check
        "n_tiff_frames": int(scalars.n_tiff_frames),
        "pulse_count_diff": int(scalars.pulse_count_diff),
        "pulse_count_diff_after_off_by_one": int(scalars.pulse_count_diff_after_off_by_one),
        "s2p_off_by_one_fix_applied": int(scalars.s2p_off_by_one_fix_applied),
        # Kinematics decimation
        "kin_pose_decimation_ratio": float(scalars.kin_pose_decimation_ratio),
        "kin_pose_decimation_uniform": int(scalars.kin_pose_decimation_uniform),
    }


# ---------------------------------------------------------------------------
# Build SyncScalars from arrays
# ---------------------------------------------------------------------------


def build_scalars(
    *,
    timestamps_present: bool,
    cam_times: np.ndarray | None,
    img_times: np.ndarray | None,
    line_times: np.ndarray | None,
    light_on: np.ndarray | None,
    light_off: np.ndarray | None,
    fps_camera: float = 100.0,
    fps_imaging: float = 30.0,
    n_tiff_frames: int = INT_SENTINEL,
    s2p_off_by_one_fix_applied: int = 0,
    kin_pose_decimation_ratio: float = 1.0,
    kin_pose_decimation_uniform: int = 1,
    tdms_diag: dict[str, float] | None = None,
    cfg: dict[str, Any] | None = None,
) -> SyncScalars:
    """Build a :class:`SyncScalars` from raw arrays.

    Convenience wrapper around the per-channel functions.
    """
    if cfg is None:
        cfg = _DEFAULT_CONFIG
    if not timestamps_present:
        return SyncScalars(timestamps_present=False)
    cam = channel_scalars(
        cam_times if cam_times is not None else np.empty(0),
        fps_camera,
        cfg=cfg,
    )
    img = channel_scalars(
        img_times if img_times is not None else np.empty(0),
        fps_imaging,
        cfg=cfg,
    )
    line = channel_scalars(
        line_times if line_times is not None else np.empty(0),
        fps_imaging,
        cfg=cfg,
    )
    cross = (
        cross_channel_scalars(cam_times, img_times)
        if cam_times is not None and img_times is not None
        else CrossChannelScalars()
    )
    duration_for_duty = (
        max(cam.duration_s, img.duration_s) if cam.n_pulses > 0 or img.n_pulses > 0 else 0.0
    )
    if not np.isfinite(duration_for_duty):
        duration_for_duty = 0.0
    light = light_scalars(
        light_on if light_on is not None else np.empty(0),
        light_off if light_off is not None else np.empty(0),
        duration_for_duty,
    )

    diff = INT_SENTINEL
    diff_after = INT_SENTINEL
    if n_tiff_frames > 0 and img.n_pulses > 0:
        diff = int(img.n_pulses - n_tiff_frames)
        # The known SciScan off-by-one heuristic: imaging pulses == tiff + 1.
        diff_after = diff - 1 if diff == 1 else diff

    diag = tdms_diag or {}
    return SyncScalars(
        timestamps_present=True,
        cam=cam,
        img=img,
        line=line,
        cross=cross,
        light=light,
        n_tiff_frames=int(n_tiff_frames),
        pulse_count_diff=int(diff),
        pulse_count_diff_after_off_by_one=int(diff_after),
        s2p_off_by_one_fix_applied=int(s2p_off_by_one_fix_applied),
        cam_min=float(diag.get("cam_min", FLOAT_SENTINEL)),
        cam_max=float(diag.get("cam_max", FLOAT_SENTINEL)),
        sci_min=float(diag.get("sci_min", FLOAT_SENTINEL)),
        sci_max=float(diag.get("sci_max", FLOAT_SENTINEL)),
        light_min=float(diag.get("light_min", FLOAT_SENTINEL)),
        light_max=float(diag.get("light_max", FLOAT_SENTINEL)),
        kin_pose_decimation_ratio=float(kin_pose_decimation_ratio),
        kin_pose_decimation_uniform=int(kin_pose_decimation_uniform),
    )
