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
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "pct_above_90": 0.0,
            "pct_above_50": 0.0,
            "n_frames": 0,
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


def detect_ear_swaps(
    left_ear_x: npt.NDArray[np.floating],
    left_ear_y: npt.NDArray[np.floating],
    right_ear_x: npt.NDArray[np.floating],
    right_ear_y: npt.NDArray[np.floating],
    axis_x1: npt.NDArray[np.floating],
    axis_y1: npt.NDArray[np.floating],
    axis_x2: npt.NDArray[np.floating],
    axis_y2: npt.NDArray[np.floating],
) -> dict:
    """Detect frames where left/right ears are swapped relative to the body axis.

    The body axis is defined by two midline keypoints (e.g. nose→head_midpoint,
    nose→neck, or head_midpoint→tail_base).  For each frame, the function
    computes the signed cross-product of (axis direction) × (ear - axis
    origin) to determine which side of the axis each ear falls on.
    If the left ear is on the right side (or vice versa), the ears are
    flagged as swapped.

    Parameters
    ----------
    left_ear_x, left_ear_y : (n_frames,) float
        Left ear positions.
    right_ear_x, right_ear_y : (n_frames,) float
        Right ear positions.
    axis_x1, axis_y1 : (n_frames,) float
        Anterior midline keypoint (e.g. nose_tip or head_midpoint).
    axis_x2, axis_y2 : (n_frames,) float
        Posterior midline keypoint (e.g. head_midpoint or tail_base).

    Returns
    -------
    dict
        ``"is_swapped"`` — (n_frames,) bool, True where ears are on the
        wrong side of the body axis.
        ``"n_swapped"`` — count of swapped frames.
        ``"pct_swapped"`` — fraction of valid frames that are swapped.
        ``"left_sign"`` — (n_frames,) float, signed side of left ear
        (positive = left of axis when facing from axis_x1 to axis_x2).
    """
    n = len(left_ear_x)

    # Body axis vector: anterior → posterior
    ax = axis_x2 - axis_x1
    ay = axis_y2 - axis_y1

    # Vector from axis origin to each ear
    le_dx = left_ear_x - axis_x1
    le_dy = left_ear_y - axis_y1
    re_dx = right_ear_x - axis_x1
    re_dy = right_ear_y - axis_y1

    # Signed cross product: positive = left of axis, negative = right
    # (when looking from anterior to posterior)
    left_sign = ax * le_dy - ay * le_dx
    right_sign = ax * re_dy - ay * re_dx

    # Ears are swapped when they are on the same side, or when left ear
    # is on the right side (negative sign) and right ear is on the left
    # (positive sign).  The canonical arrangement has left_sign > 0 for
    # the majority of frames.  Determine the expected sign from the
    # majority vote across valid frames.
    valid = (
        np.isfinite(left_sign)
        & np.isfinite(right_sign)
        & (np.abs(ax) + np.abs(ay) > 1e-6)  # axis has non-zero length
    )

    if valid.sum() < 10:
        return {
            "is_swapped": np.zeros(n, dtype=bool),
            "n_swapped": 0,
            "pct_swapped": 0.0,
            "left_sign": left_sign,
        }

    # Majority of frames: left ear should be on one consistent side
    majority_left_positive = np.sum(left_sign[valid] > 0) > valid.sum() / 2

    if majority_left_positive:
        # Expected: left_sign > 0, right_sign < 0
        is_swapped = valid & (left_sign < 0) & (right_sign > 0)
    else:
        # Expected: left_sign < 0, right_sign > 0
        is_swapped = valid & (left_sign > 0) & (right_sign < 0)

    n_swapped = int(is_swapped.sum())
    pct_swapped = n_swapped / valid.sum() if valid.sum() > 0 else 0.0

    return {
        "is_swapped": is_swapped,
        "n_swapped": n_swapped,
        "pct_swapped": float(pct_swapped),
        "left_sign": left_sign,
    }


def detect_point_in_triangle(
    tri_ax: npt.NDArray[np.floating],
    tri_ay: npt.NDArray[np.floating],
    tri_bx: npt.NDArray[np.floating],
    tri_by: npt.NDArray[np.floating],
    tri_cx: npt.NDArray[np.floating],
    tri_cy: npt.NDArray[np.floating],
    point_x: npt.NDArray[np.floating],
    point_y: npt.NDArray[np.floating],
    *,
    expect_inside: bool = True,
) -> dict:
    """Test whether a point is inside or outside a triangle per frame.

    Uses the sign-of-cross-product method. Flags frames where the point
    is on the wrong side (inside when ``expect_inside=False``, or outside
    when ``expect_inside=True``).

    Parameters
    ----------
    tri_ax, tri_ay, tri_bx, tri_by, tri_cx, tri_cy : (n_frames,) float
        The three triangle vertices.
    point_x, point_y : (n_frames,) float
        The point to test.
    expect_inside : bool
        If True (default), flag frames where the point is *outside*.
        If False, flag frames where the point is *inside*.

    Returns
    -------
    dict
        ``"is_flagged"`` — (n_frames,) bool.
        ``"n_flagged"`` — count.
        ``"pct_flagged"`` — fraction of valid frames.
    """

    def _cross(ox, oy, ax, ay, bx, by):
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)

    d1 = _cross(point_x, point_y, tri_ax, tri_ay, tri_bx, tri_by)
    d2 = _cross(point_x, point_y, tri_bx, tri_by, tri_cx, tri_cy)
    d3 = _cross(point_x, point_y, tri_cx, tri_cy, tri_ax, tri_ay)

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    is_outside = has_neg & has_pos  # mixed signs = outside

    valid = np.isfinite(tri_ax) & np.isfinite(tri_bx) & np.isfinite(tri_cx) & np.isfinite(point_x)

    if expect_inside:
        is_flagged = valid & is_outside
    else:
        is_flagged = valid & ~is_outside

    n_flagged = int(is_flagged.sum())
    n_valid = int(valid.sum())

    return {
        "is_flagged": is_flagged,
        "n_flagged": n_flagged,
        "pct_flagged": n_flagged / n_valid if n_valid > 0 else 0.0,
    }


def detect_head_midpoint_outside_triangle(
    nose_x: npt.NDArray[np.floating],
    nose_y: npt.NDArray[np.floating],
    left_ear_x: npt.NDArray[np.floating],
    left_ear_y: npt.NDArray[np.floating],
    right_ear_x: npt.NDArray[np.floating],
    right_ear_y: npt.NDArray[np.floating],
    midpoint_x: npt.NDArray[np.floating],
    midpoint_y: npt.NDArray[np.floating],
) -> dict:
    """Detect frames where head_midpoint is outside the nose-ears triangle.

    Wrapper around ``detect_point_in_triangle`` with ``expect_inside=True``.
    Returns dict with ``"is_outside"``, ``"n_outside"``, ``"pct_outside"``.
    """
    result = detect_point_in_triangle(
        nose_x,
        nose_y,
        left_ear_x,
        left_ear_y,
        right_ear_x,
        right_ear_y,
        midpoint_x,
        midpoint_y,
        expect_inside=True,
    )
    return {
        "is_outside": result["is_flagged"],
        "n_outside": result["n_flagged"],
        "pct_outside": result["pct_flagged"],
    }


def detect_neck_inside_triangle(
    nose_x: npt.NDArray[np.floating],
    nose_y: npt.NDArray[np.floating],
    left_ear_x: npt.NDArray[np.floating],
    left_ear_y: npt.NDArray[np.floating],
    right_ear_x: npt.NDArray[np.floating],
    right_ear_y: npt.NDArray[np.floating],
    neck_x: npt.NDArray[np.floating],
    neck_y: npt.NDArray[np.floating],
) -> dict:
    """Detect frames where neck is inside the nose-ears triangle.

    The neck should be posterior to the ears and therefore outside the
    triangle formed by nose_tip, left_ear, right_ear. If it falls
    inside, the neck label is likely misplaced (confused with
    head_midpoint).

    Returns dict with ``"is_inside"``, ``"n_inside"``, ``"pct_inside"``.
    """
    result = detect_point_in_triangle(
        nose_x,
        nose_y,
        left_ear_x,
        left_ear_y,
        right_ear_x,
        right_ear_y,
        neck_x,
        neck_y,
        expect_inside=False,
    )
    return {
        "is_inside": result["is_flagged"],
        "n_inside": result["n_flagged"],
        "pct_inside": result["pct_flagged"],
    }


def detect_anterior_posterior_violations(
    keypoints: dict[str, tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]],
    order: list[str] | None = None,
) -> dict:
    """Detect frames where body parts are out of anterior-posterior order.

    Projects each keypoint onto the body axis (first → last keypoint in
    ``order``) and checks that the projections are monotonically increasing
    from anterior to posterior.

    Parameters
    ----------
    keypoints : dict
        Mapping from body part name to ``(x, y)`` arrays.
    order : list[str]
        Expected anterior → posterior ordering.
        Default: ``["nose_tip", "head_midpoint", "neck", "mid_back",
        "mouse_center", "tail_base"]``.

    Returns
    -------
    dict
        ``"is_violated"`` — (n_frames,) bool, True where ordering is wrong.
        ``"n_violated"`` — count.
        ``"pct_violated"`` — fraction of valid frames.
        ``"violations_per_pair"`` — dict mapping ``"A>B"`` to count of frames
        where A was posterior to B (should be anterior).
    """
    if order is None:
        order = ["nose_tip", "head_midpoint", "neck", "mid_back", "mouse_center", "tail_base"]

    # Keep only keypoints that are available
    available = [bp for bp in order if bp in keypoints]
    if len(available) < 2:
        n = len(next(iter(keypoints.values()))[0]) if keypoints else 0
        return {
            "is_violated": np.zeros(n, dtype=bool),
            "n_violated": 0,
            "pct_violated": 0.0,
            "violations_per_pair": {},
        }

    # Body axis: first available → last available
    first_bp = available[0]
    last_bp = available[-1]
    ax_x = keypoints[last_bp][0] - keypoints[first_bp][0]
    ax_y = keypoints[last_bp][1] - keypoints[first_bp][1]
    ax_len = np.sqrt(ax_x**2 + ax_y**2)
    ax_len[ax_len < 1e-6] = np.nan  # avoid division by zero

    # Project each keypoint onto the axis
    projections = {}
    for bp in available:
        dx = keypoints[bp][0] - keypoints[first_bp][0]
        dy = keypoints[bp][1] - keypoints[first_bp][1]
        proj = (dx * ax_x + dy * ax_y) / (ax_len**2)
        projections[bp] = proj

    n = len(ax_x)
    is_violated = np.zeros(n, dtype=bool)
    violations_per_pair: dict[str, int] = {}
    valid_all = np.isfinite(ax_len)

    for i in range(len(available) - 1):
        bp_a = available[i]  # should be anterior (smaller projection)
        bp_b = available[i + 1]  # should be posterior (larger projection)
        pair_valid = valid_all & np.isfinite(projections[bp_a]) & np.isfinite(projections[bp_b])
        wrong = pair_valid & (projections[bp_a] > projections[bp_b])
        is_violated |= wrong
        n_wrong = int(wrong.sum())
        if n_wrong > 0:
            violations_per_pair[f"{bp_a}>{bp_b}"] = n_wrong

    n_valid = int(valid_all.sum())
    n_violated = int(is_violated.sum())

    return {
        "is_violated": is_violated,
        "n_violated": n_violated,
        "pct_violated": n_violated / n_valid if n_valid > 0 else 0.0,
        "violations_per_pair": violations_per_pair,
    }


def detect_ear_asymmetry(
    left_ear_x: npt.NDArray[np.floating],
    left_ear_y: npt.NDArray[np.floating],
    right_ear_x: npt.NDArray[np.floating],
    right_ear_y: npt.NDArray[np.floating],
    axis_x1: npt.NDArray[np.floating],
    axis_y1: npt.NDArray[np.floating],
    axis_x2: npt.NDArray[np.floating],
    axis_y2: npt.NDArray[np.floating],
    ratio_threshold: float = 3.0,
) -> dict:
    """Detect frames where ears are asymmetrically placed about the body axis.

    Computes the perpendicular distance of each ear from the body axis.
    If one ear is more than ``ratio_threshold`` times further from the
    axis than the other, the frame is flagged.

    Returns
    -------
    dict
        ``"is_asymmetric"`` — (n_frames,) bool.
        ``"n_asymmetric"`` — count.
        ``"ratio"`` — (n_frames,) float, max(d_left, d_right) / min(...).
    """
    ax = axis_x2 - axis_x1
    ay = axis_y2 - axis_y1
    ax_len = np.sqrt(ax**2 + ay**2)
    ax_len[ax_len < 1e-6] = np.nan

    # Perpendicular distance = |cross product| / axis length
    d_left = np.abs(ax * (left_ear_y - axis_y1) - ay * (left_ear_x - axis_x1)) / ax_len
    d_right = np.abs(ax * (right_ear_y - axis_y1) - ay * (right_ear_x - axis_x1)) / ax_len

    min_d = np.minimum(d_left, d_right)
    min_d[min_d < 1e-6] = np.nan
    ratio = np.maximum(d_left, d_right) / min_d

    valid = np.isfinite(ratio)
    is_asymmetric = valid & (ratio > ratio_threshold)

    return {
        "is_asymmetric": is_asymmetric,
        "n_asymmetric": int(is_asymmetric.sum()),
        "ratio": ratio,
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
                f"{bp_name}: only {lik_stats['pct_above_90'] * 100:.0f}% frames "
                f"above 0.9 confidence"
            )
        if jumps.sum() > n_frames * 0.01:
            issues.append(
                f"{bp_name}: {int(jumps.sum())} jump frames ({jumps.sum() / n_frames * 100:.1f}%)"
            )

    if problem_frames is None:
        problem_frames = np.zeros(0, dtype=bool)

    n = len(problem_frames)
    pct_good = float(1.0 - np.mean(problem_frames)) if n > 0 else 0.0

    # Score: 100 = perfect, 0 = terrible
    # Weighted combination of per-keypoint quality
    if per_kp:
        mean_lik = np.mean([v["likelihood"]["mean"] for v in per_kp.values()])
        mean_pct_good = np.mean([1.0 - v["pct_low_confidence"] for v in per_kp.values()])
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
    positions: npt.NDArray[np.floating] | None = None,
    min_position_dist: float = 50.0,
) -> npt.NDArray[np.intp]:
    """Select the worst-tracked frames for manual review / retraining.

    Picks frames with lowest mean likelihood, enforcing minimum spacing
    and pose diversity so frames aren't all from the same bad stretch
    or the same mouse pose.

    Parameters
    ----------
    likelihood : (n_frames, n_keypoints) or (n_frames,) float
        Per-frame confidence. If 2D, uses the mean across keypoints.
    n_frames : int
        Number of frames to select.
    min_spacing : int
        Minimum frame gap between selected frames.
    positions : (n_frames, n_keypoints*2) float, optional
        Bodypart positions for diversity filtering.
    min_position_dist : float
        Minimum centroid/shape distance (pixels) between selected frames.

    Returns
    -------
    indices : (n_selected,) int
        Frame indices sorted by ascending quality.
    """
    if likelihood.ndim == 2:
        mean_lik = np.nanmean(likelihood, axis=1)
    else:
        mean_lik = likelihood.copy()

    # Precompute centroids and shapes for diversity check
    _centroids = None
    _shapes = None
    if positions is not None:
        p = positions.reshape(len(mean_lik), -1).astype(np.float64)
        p = np.nan_to_num(p, nan=0.0)
        n_kp = p.shape[1] // 2
        xs = p[:, 0::2]
        ys = p[:, 1::2]
        cx = np.mean(xs, axis=1, keepdims=True)
        cy = np.mean(ys, axis=1, keepdims=True)
        _centroids = np.column_stack([cx.ravel(), cy.ravel()])
        _shapes = np.column_stack([xs - cx, ys - cy])

    # Sort by ascending likelihood (worst first)
    order = np.argsort(mean_lik)

    selected = []
    for idx in order:
        if len(selected) >= n_frames:
            break
        # Check spacing
        if not all(abs(int(idx) - int(s)) >= min_spacing for s in selected):
            continue
        # Check pose diversity
        if _centroids is not None:
            too_similar = False
            for s in selected:
                c_dist = np.sqrt(np.sum((_centroids[idx] - _centroids[s]) ** 2))
                s_dist = np.mean(np.abs(_shapes[idx] - _shapes[s]))
                if c_dist < min_position_dist and s_dist < min_position_dist:
                    too_similar = True
                    break
            if too_similar:
                continue
        selected.append(idx)

    return np.array(sorted(selected), dtype=np.intp)


def stratified_frame_selection(
    likelihood: npt.NDArray[np.floating],
    n_per_bin: int = 5,
    n_bins: int = 4,
    min_spacing: int = 30,
    positions: npt.NDArray[np.floating] | None = None,
    min_position_dist: float = 50.0,
) -> dict:
    """Select frames stratified across quality bins for retraining.

    Selects frames from different quality levels: worst, poor, moderate,
    and good — to ensure retraining data covers the full range.

    Similarity filtering uses two criteria:
    1. **Centroid distance** — the Euclidean distance between the mean
       body position (centroid of all keypoints). Frames with centroids
       closer than ``min_position_dist`` are candidates for dedup.
    2. **Pose shape distance** — the mean displacement of centroid-
       subtracted keypoints. Two frames with the same pose shape but
       in different arena locations will have high centroid distance
       (kept) but low shape distance. Two frames in the same location
       with the same pose will fail both (deduplicated).

    A frame is considered too similar only if BOTH centroid distance
    AND shape distance are below threshold.

    Parameters
    ----------
    likelihood : (n_frames, n_keypoints) or (n_frames,) float
    n_per_bin : int
        Frames to select per quality bin.
    n_bins : int
        Number of quality bins (e.g. 4 = worst/poor/moderate/good).
    min_spacing : int
        Minimum frames between selected frames.
    positions : (n_frames, n_keypoints, 2) or (n_frames, n_keypoints*2) float, optional
        Bodypart x/y positions per frame.
    min_position_dist : float
        Minimum centroid displacement (pixels) for the location check.
        Default 50px (~40mm). Also used as the shape distance threshold.

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

    # Use data-adaptive quantile edges so bins have roughly equal counts.
    # Fixed 0-1 edges fail when all confidences are in a narrow range
    # (e.g. DLC 3.0 PyTorch backend outputs 0.1-0.3 for all frames).
    # Precompute position data for similarity checking
    _pos_flat = None  # (n_frames, n_keypoints*2) raw positions
    _centroids = None  # (n_frames, 2) mean x, mean y
    _shapes = None  # (n_frames, n_keypoints*2) centroid-subtracted
    if positions is not None:
        p = positions.reshape(len(mean_lik), -1).astype(np.float64)
        _pos_flat = np.nan_to_num(p, nan=0.0)
        # Centroid: mean of x coords and mean of y coords
        n_kp = p.shape[1] // 2
        xs = _pos_flat[:, 0::2]  # (n_frames, n_keypoints)
        ys = _pos_flat[:, 1::2]
        cx = np.mean(xs, axis=1, keepdims=True)  # (n_frames, 1)
        cy = np.mean(ys, axis=1, keepdims=True)
        _centroids = np.column_stack([cx.ravel(), cy.ravel()])
        # Shape: centroid-subtracted positions
        xs_c = xs - cx
        ys_c = ys - cy
        _shapes = np.column_stack([xs_c, ys_c])  # (n_frames, n_keypoints*2)

    def _is_too_similar(idx: int, selected_set: set) -> bool:
        """Check if frame is too similar in both location and pose shape."""
        if _centroids is None:
            return False
        for s in selected_set:
            # Centroid (location) distance
            c_dist = np.sqrt(np.sum((_centroids[idx] - _centroids[s]) ** 2))
            # Pose shape distance (centroid-subtracted)
            s_dist = np.mean(np.abs(_shapes[idx] - _shapes[s]))
            # Only reject if BOTH location and shape are similar
            if c_dist < min_position_dist and s_dist < min_position_dist:
                return True
        return False

    valid_lik = mean_lik[np.isfinite(mean_lik)]
    if len(valid_lik) > 0:
        bin_edges = np.quantile(valid_lik, np.linspace(0, 1, n_bins + 1))
        # Ensure edges are strictly increasing (can happen with very uniform data)
        for j in range(1, len(bin_edges)):
            if bin_edges[j] <= bin_edges[j - 1]:
                bin_edges[j] = bin_edges[j - 1] + 1e-6
    else:
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
            if idx in all_selected:
                continue
            if not all(abs(int(idx) - int(s)) >= min_spacing for s in selected):
                continue
            if _is_too_similar(idx, all_selected):
                continue
            selected.append(idx)
            all_selected.add(idx)

        bins_result.append((bin_labels[i], np.array(sorted(selected), dtype=np.intp)))

    # Second pass: if we have fewer than n_per_bin * n_bins, fill from all
    # remaining candidates across all bins (worst-first globally).
    target = n_per_bin * n_bins
    if len(all_selected) < target:
        remaining_order = np.argsort(mean_lik)
        for idx in remaining_order:
            if len(all_selected) >= target:
                break
            if idx in all_selected:
                continue
            if not all(abs(int(idx) - int(s)) >= min_spacing for s in all_selected):
                continue
            if _is_too_similar(idx, all_selected):
                continue
            all_selected.add(idx)

    all_indices = np.array(sorted(all_selected), dtype=np.intp)
    return {
        "indices": all_indices,
        "bins": bins_result,
        "total_selected": len(all_indices),
    }
