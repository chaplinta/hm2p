"""Tracking quality diagnostics — detect poor pose estimation.

Pure numpy functions for evaluating DLC / pose tracker output quality.
Identifies frames with jumps, low confidence, anatomical violations,
and temporal inconsistencies. All functions operate on raw tracker
output arrays (x, y, likelihood per keypoint).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Per-keypoint quality metrics
# ---------------------------------------------------------------------------


def likelihood_summary(
    likelihood: npt.NDArray[np.floating],
) -> dict:
    """Summary statistics for a keypoint's likelihood values.

    Parameters
    ----------
    likelihood : (n_frames,) float
        Per-frame confidence from tracker (0–1).

    Returns
    -------
    dict
        ``"mean"``, ``"median"``, ``"std"`` — basic stats.
        ``"pct_above_90"`` — fraction of frames with likelihood >= 0.9.
        ``"pct_above_50"`` — fraction of frames with likelihood >= 0.5.
        ``"n_frames"`` — total frames.
    """
    n = len(likelihood)
    if n == 0:
        return {
            "mean": float("nan"), "median": float("nan"), "std": float("nan"),
            "pct_above_90": 0.0, "pct_above_50": 0.0, "n_frames": 0,
        }
    return {
        "mean": float(np.nanmean(likelihood)),
        "median": float(np.nanmedian(likelihood)),
        "std": float(np.nanstd(likelihood)),
        "pct_above_90": float(np.mean(likelihood >= 0.9)),
        "pct_above_50": float(np.mean(likelihood >= 0.5)),
        "n_frames": n,
    }


def detect_jumps(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    threshold_px: float = 50.0,
) -> npt.NDArray[np.bool_]:
    """Detect frames where a keypoint jumps unrealistically far.

    A "jump" is a frame-to-frame displacement exceeding threshold_px.

    Parameters
    ----------
    x, y : (n_frames,) float
        Keypoint pixel coordinates.
    threshold_px : float
        Maximum plausible displacement per frame (pixels).

    Returns
    -------
    is_jump : (n_frames,) bool
        True for frames that follow an implausible jump.
        First frame is always False.
    """
    n = len(x)
    is_jump = np.zeros(n, dtype=bool)
    if n < 2:
        return is_jump

    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    is_jump[1:] = dist > threshold_px
    return is_jump


def detect_frozen_keypoint(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    window: int = 30,
    max_displacement_px: float = 0.5,
) -> npt.NDArray[np.bool_]:
    """Detect frames where a keypoint is frozen (stuck in place).

    A keypoint is "frozen" if it moves less than max_displacement_px
    over a sliding window. This can indicate the detector is returning
    a fixed position (e.g. corner of frame) rather than tracking.

    Parameters
    ----------
    x, y : (n_frames,) float
        Keypoint pixel coordinates.
    window : int
        Number of consecutive frames to check.
    max_displacement_px : float
        Maximum total displacement within window to be considered frozen.

    Returns
    -------
    is_frozen : (n_frames,) bool
        True for frames within a frozen stretch.
    """
    n = len(x)
    is_frozen = np.zeros(n, dtype=bool)
    if n < window:
        return is_frozen

    for start in range(n - window + 1):
        end = start + window
        x_range = np.nanmax(x[start:end]) - np.nanmin(x[start:end])
        y_range = np.nanmax(y[start:end]) - np.nanmin(y[start:end])
        total_range = np.sqrt(x_range**2 + y_range**2)
        if total_range < max_displacement_px:
            is_frozen[start:end] = True

    return is_frozen


# ---------------------------------------------------------------------------
# Anatomical constraint validation
# ---------------------------------------------------------------------------


def ear_distance(
    left_ear_x: npt.NDArray[np.floating],
    left_ear_y: npt.NDArray[np.floating],
    right_ear_x: npt.NDArray[np.floating],
    right_ear_y: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute inter-ear distance per frame.

    Parameters
    ----------
    left_ear_x, left_ear_y, right_ear_x, right_ear_y : (n_frames,) float

    Returns
    -------
    distance : (n_frames,) float
        Euclidean distance between ears in pixels.
    """
    dx = left_ear_x - right_ear_x
    dy = left_ear_y - right_ear_y
    return np.sqrt(dx**2 + dy**2)


def detect_ear_distance_outliers(
    left_ear_x: npt.NDArray[np.floating],
    left_ear_y: npt.NDArray[np.floating],
    right_ear_x: npt.NDArray[np.floating],
    right_ear_y: npt.NDArray[np.floating],
    z_threshold: float = 3.0,
) -> dict:
    """Detect frames where inter-ear distance is anomalous.

    Ear distance should be roughly constant for a rigid mouse head.
    Large deviations suggest one or both ears are mis-tracked.

    Parameters
    ----------
    left_ear_x, left_ear_y, right_ear_x, right_ear_y : (n_frames,) float
    z_threshold : float
        Number of MAD (median absolute deviation) units for outlier detection.

    Returns
    -------
    dict
        ``"distance"`` — (n_frames,) ear distance array.
        ``"median"`` — median ear distance.
        ``"mad"`` — median absolute deviation.
        ``"is_outlier"`` — (n_frames,) bool, True if anomalous.
        ``"n_outliers"`` — count of outlier frames.
    """
    dist = ear_distance(left_ear_x, left_ear_y, right_ear_x, right_ear_y)
    valid = np.isfinite(dist)

    if valid.sum() < 10:
        return {
            "distance": dist,
            "median": float("nan"),
            "mad": float("nan"),
            "is_outlier": np.zeros(len(dist), dtype=bool),
            "n_outliers": 0,
        }

    med = float(np.median(dist[valid]))
    mad = float(np.median(np.abs(dist[valid] - med)))
    if mad < 1e-10:
        mad = 1.0  # Avoid division by zero

    z_scores = np.abs(dist - med) / mad
    is_outlier = z_scores > z_threshold
    is_outlier[~valid] = True  # NaN frames are also flagged

    return {
        "distance": dist,
        "median": med,
        "mad": mad,
        "is_outlier": is_outlier,
        "n_outliers": int(is_outlier.sum()),
    }


def body_length_consistency(
    head_x: npt.NDArray[np.floating],
    head_y: npt.NDArray[np.floating],
    tail_x: npt.NDArray[np.floating],
    tail_y: npt.NDArray[np.floating],
    z_threshold: float = 3.0,
) -> dict:
    """Check head-to-tail distance consistency.

    Parameters
    ----------
    head_x, head_y : (n_frames,) float
        Head keypoint (e.g. mouse_center or mid_back).
    tail_x, tail_y : (n_frames,) float
        Tail base keypoint.
    z_threshold : float
        MAD units for outlier detection.

    Returns
    -------
    dict
        ``"length"`` — (n_frames,) body length.
        ``"median"`` — median body length.
        ``"mad"`` — MAD of body length.
        ``"is_outlier"`` — (n_frames,) bool.
        ``"n_outliers"`` — outlier count.
    """
    dx = head_x - tail_x
    dy = head_y - tail_y
    length = np.sqrt(dx**2 + dy**2)
    valid = np.isfinite(length)

    if valid.sum() < 10:
        return {
            "length": length,
            "median": float("nan"),
            "mad": float("nan"),
            "is_outlier": np.zeros(len(length), dtype=bool),
            "n_outliers": 0,
        }

    med = float(np.median(length[valid]))
    mad = float(np.median(np.abs(length[valid] - med)))
    if mad < 1e-10:
        mad = 1.0

    z_scores = np.abs(length - med) / mad
    is_outlier = z_scores > z_threshold
    is_outlier[~valid] = True

    return {
        "length": length,
        "median": med,
        "mad": mad,
        "is_outlier": is_outlier,
        "n_outliers": int(is_outlier.sum()),
    }


# ---------------------------------------------------------------------------
# Session-level quality report
# ---------------------------------------------------------------------------


def session_quality_report(
    keypoint_data: dict[str, dict[str, npt.NDArray[np.floating]]],
    fps: float = 30.0,
    jump_threshold_px: float = 50.0,
) -> dict:
    """Generate a comprehensive quality report for one session.

    Parameters
    ----------
    keypoint_data : dict
        Mapping from bodypart name to dict with keys ``"x"``, ``"y"``,
        ``"likelihood"``.
    fps : float
        Tracking frame rate (for interpreting jump thresholds).
    jump_threshold_px : float
        Per-frame pixel displacement threshold for jump detection.

    Returns
    -------
    dict
        ``"per_keypoint"`` — dict of per-keypoint metrics.
        ``"overall_score"`` — 0–100 session quality score.
        ``"n_frames"`` — total frames.
        ``"problem_frames"`` — (n_frames,) bool, union of all problems.
        ``"pct_good"`` — fraction of clean frames.
        ``"issues"`` — list of human-readable issue descriptions.
    """
    n_frames = 0
    problem_frames = None
    per_kp = {}
    issues = []

    for bp_name, bp_data in keypoint_data.items():
        x = np.asarray(bp_data["x"], dtype=np.float64)
        y = np.asarray(bp_data["y"], dtype=np.float64)
        lik = np.asarray(bp_data["likelihood"], dtype=np.float64)
        n_frames = len(x)

        if problem_frames is None:
            problem_frames = np.zeros(n_frames, dtype=bool)

        # Likelihood
        lik_stats = likelihood_summary(lik)
        low_lik = lik < 0.9
        problem_frames |= low_lik

        # Jumps
        jumps = detect_jumps(x, y, threshold_px=jump_threshold_px)
        problem_frames |= jumps

        per_kp[bp_name] = {
            "likelihood": lik_stats,
            "n_jumps": int(jumps.sum()),
            "pct_low_confidence": float(np.mean(low_lik)),
        }

        # Warnings
        if lik_stats["pct_above_90"] < 0.8:
            issues.append(
                f"{bp_name}: only {lik_stats['pct_above_90']*100:.0f}% frames "
                f"above 0.9 confidence"
            )
        if jumps.sum() > n_frames * 0.01:
            issues.append(
                f"{bp_name}: {int(jumps.sum())} jump frames "
                f"({jumps.sum()/n_frames*100:.1f}%)"
            )

    if problem_frames is None:
        problem_frames = np.zeros(0, dtype=bool)

    n = len(problem_frames)
    pct_good = float(1.0 - np.mean(problem_frames)) if n > 0 else 0.0

    # Score: 100 = perfect, 0 = terrible
    # Weighted combination of per-keypoint quality
    if per_kp:
        mean_lik = np.mean([v["likelihood"]["mean"] for v in per_kp.values()])
        mean_pct_good = np.mean(
            [1.0 - v["pct_low_confidence"] for v in per_kp.values()]
        )
        jump_penalty = min(
            1.0,
            sum(v["n_jumps"] for v in per_kp.values()) / max(n, 1) * 10,
        )
        score = (0.5 * mean_lik + 0.3 * mean_pct_good + 0.2 * (1.0 - jump_penalty)) * 100
    else:
        score = 0.0

    return {
        "per_keypoint": per_kp,
        "overall_score": float(np.clip(score, 0, 100)),
        "n_frames": n_frames,
        "problem_frames": problem_frames,
        "pct_good": pct_good,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Frame selection for retraining
# ---------------------------------------------------------------------------


def worst_frames(
    likelihood: npt.NDArray[np.floating],
    n_frames: int = 20,
    min_spacing: int = 30,
) -> npt.NDArray[np.intp]:
    """Select the worst-tracked frames for manual review / retraining.

    Picks frames with lowest mean likelihood, enforcing minimum spacing
    so frames aren't all from the same bad stretch.

    Parameters
    ----------
    likelihood : (n_frames, n_keypoints) or (n_frames,) float
        Per-frame confidence. If 2D, uses the mean across keypoints.
    n_frames : int
        Number of frames to select.
    min_spacing : int
        Minimum frame gap between selected frames.

    Returns
    -------
    indices : (n_selected,) int
        Frame indices sorted by ascending quality.
    """
    if likelihood.ndim == 2:
        mean_lik = np.nanmean(likelihood, axis=1)
    else:
        mean_lik = likelihood.copy()

    # Sort by ascending likelihood (worst first)
    order = np.argsort(mean_lik)

    selected = []
    for idx in order:
        if len(selected) >= n_frames:
            break
        # Check spacing
        if all(abs(int(idx) - int(s)) >= min_spacing for s in selected):
            selected.append(idx)

    return np.array(sorted(selected), dtype=np.intp)


def stratified_frame_selection(
    likelihood: npt.NDArray[np.floating],
    n_per_bin: int = 5,
    n_bins: int = 4,
    min_spacing: int = 30,
) -> dict:
    """Select frames stratified across quality bins for retraining.

    Selects frames from different quality levels: worst, poor, moderate,
    and good — to ensure retraining data covers the full range.

    Parameters
    ----------
    likelihood : (n_frames, n_keypoints) or (n_frames,) float
    n_per_bin : int
        Frames to select per quality bin.
    n_bins : int
        Number of quality bins (e.g. 4 = worst/poor/moderate/good).
    min_spacing : int
        Minimum frames between selected frames.

    Returns
    -------
    dict
        ``"indices"`` — (n_selected,) selected frame indices.
        ``"bins"`` — list of (label, indices) per quality bin.
        ``"total_selected"`` — total frames selected.
    """
    if likelihood.ndim == 2:
        mean_lik = np.nanmean(likelihood, axis=1)
    else:
        mean_lik = likelihood.copy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = ["worst", "poor", "moderate", "good"][:n_bins]
    bins_result = []
    all_selected = set()

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (mean_lik >= lo) & (mean_lik <= hi)
        else:
            in_bin = (mean_lik >= lo) & (mean_lik < hi)

        bin_indices = np.where(in_bin)[0]
        if len(bin_indices) == 0:
            bins_result.append((bin_labels[i], np.array([], dtype=np.intp)))
            continue

        # Pick n_per_bin with spacing
        order = np.argsort(mean_lik[bin_indices])
        candidates = bin_indices[order]

        selected = []
        for idx in candidates:
            if len(selected) >= n_per_bin:
                break
            if (all(abs(int(idx) - int(s)) >= min_spacing for s in selected)
                    and idx not in all_selected):
                selected.append(idx)
                all_selected.add(idx)

        bins_result.append((bin_labels[i], np.array(sorted(selected), dtype=np.intp)))

    all_indices = np.array(sorted(all_selected), dtype=np.intp)
    return {
        "indices": all_indices,
        "bins": bins_result,
        "total_selected": len(all_indices),
    }
