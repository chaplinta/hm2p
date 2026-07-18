"""Additional tests for hm2p.pose.quality functions not covered elsewhere.

Covers the anatomical-constraint validators (ear swaps, point-in-triangle,
anterior-posterior ordering, ear asymmetry) and the positions-based
diversity paths of the frame selectors.
"""

from __future__ import annotations

import numpy as np

from hm2p.pose.quality import (
    detect_anterior_posterior_violations,
    detect_ear_asymmetry,
    detect_ear_swaps,
    detect_head_midpoint_outside_triangle,
    detect_neck_inside_triangle,
    detect_point_in_triangle,
    stratified_frame_selection,
    worst_frames,
)

# ── detect_ear_swaps ────────────────────────────────────────────────


def _straight_axis(n: int):
    """Body axis pointing +x (anterior x1=0 → posterior x2=10)."""
    axis_x1 = np.zeros(n)
    axis_y1 = np.zeros(n)
    axis_x2 = np.full(n, 10.0)
    axis_y2 = np.zeros(n)
    return axis_x1, axis_y1, axis_x2, axis_y2


def test_detect_ear_swaps_consistent_no_swaps() -> None:
    """Ears consistently on opposite sides of the axis → none swapped."""
    n = 20
    ax1, ay1, ax2, ay2 = _straight_axis(n)
    # For axis pointing +x, left_sign = ax*le_dy → positive when ear above.
    left_ear_x = np.full(n, 5.0)
    left_ear_y = np.full(n, 3.0)  # above axis → left_sign > 0
    right_ear_x = np.full(n, 5.0)
    right_ear_y = np.full(n, -3.0)  # below axis → right_sign < 0
    res = detect_ear_swaps(
        left_ear_x, left_ear_y, right_ear_x, right_ear_y, ax1, ay1, ax2, ay2
    )
    assert res["n_swapped"] == 0
    assert res["pct_swapped"] == 0.0
    assert np.all(res["left_sign"][:5] > 0)


def test_detect_ear_swaps_detects_swapped_frames() -> None:
    """A minority of frames with ears on the wrong side are flagged."""
    n = 20
    ax1, ay1, ax2, ay2 = _straight_axis(n)
    left_ear_y = np.full(n, 3.0)
    right_ear_y = np.full(n, -3.0)
    # Swap frames 0 and 1: left goes below, right goes above.
    left_ear_y[:2] = -3.0
    right_ear_y[:2] = 3.0
    left_ear_x = np.full(n, 5.0)
    right_ear_x = np.full(n, 5.0)
    res = detect_ear_swaps(
        left_ear_x, left_ear_y, right_ear_x, right_ear_y, ax1, ay1, ax2, ay2
    )
    assert res["is_swapped"][0]
    assert res["is_swapped"][1]
    assert res["n_swapped"] == 2


def test_detect_ear_swaps_majority_negative_branch() -> None:
    """When the canonical side is negative, the else-branch is used."""
    n = 20
    ax1, ay1, ax2, ay2 = _straight_axis(n)
    # Majority: left ear BELOW axis (left_sign < 0).
    left_ear_y = np.full(n, -3.0)
    right_ear_y = np.full(n, 3.0)
    # Flip two frames to trigger swap detection in the negative branch.
    left_ear_y[:2] = 3.0
    right_ear_y[:2] = -3.0
    left_ear_x = np.full(n, 5.0)
    right_ear_x = np.full(n, 5.0)
    res = detect_ear_swaps(
        left_ear_x, left_ear_y, right_ear_x, right_ear_y, ax1, ay1, ax2, ay2
    )
    assert res["n_swapped"] == 2


def test_detect_ear_swaps_too_few_valid() -> None:
    """Fewer than 10 valid frames returns a zeroed result."""
    n = 5
    ax1, ay1, ax2, ay2 = _straight_axis(n)
    left_ear_x = np.full(n, 5.0)
    left_ear_y = np.full(n, 3.0)
    right_ear_x = np.full(n, 5.0)
    right_ear_y = np.full(n, -3.0)
    res = detect_ear_swaps(
        left_ear_x, left_ear_y, right_ear_x, right_ear_y, ax1, ay1, ax2, ay2
    )
    assert res["n_swapped"] == 0
    assert res["pct_swapped"] == 0.0
    assert not res["is_swapped"].any()


# ── detect_point_in_triangle ────────────────────────────────────────


def _triangle(n: int):
    """A fixed triangle (0,0), (10,0), (0,10) per frame."""
    ax = np.zeros(n)
    ay = np.zeros(n)
    bx = np.full(n, 10.0)
    by = np.zeros(n)
    cx = np.zeros(n)
    cy = np.full(n, 10.0)
    return ax, ay, bx, by, cx, cy


def test_point_in_triangle_inside_expected_inside() -> None:
    """A point inside the triangle is not flagged when expect_inside=True."""
    n = 4
    ax, ay, bx, by, cx, cy = _triangle(n)
    px = np.full(n, 2.0)
    py = np.full(n, 2.0)
    res = detect_point_in_triangle(ax, ay, bx, by, cx, cy, px, py, expect_inside=True)
    assert res["n_flagged"] == 0
    assert res["pct_flagged"] == 0.0


def test_point_in_triangle_outside_expected_inside() -> None:
    """A point outside is flagged when expect_inside=True."""
    n = 4
    ax, ay, bx, by, cx, cy = _triangle(n)
    px = np.full(n, 20.0)  # clearly outside
    py = np.full(n, 20.0)
    res = detect_point_in_triangle(ax, ay, bx, by, cx, cy, px, py, expect_inside=True)
    assert res["n_flagged"] == n
    assert res["pct_flagged"] == 1.0


def test_point_in_triangle_inside_expected_outside() -> None:
    """expect_inside=False flags points that are inside."""
    n = 4
    ax, ay, bx, by, cx, cy = _triangle(n)
    px = np.full(n, 2.0)
    py = np.full(n, 2.0)
    res = detect_point_in_triangle(ax, ay, bx, by, cx, cy, px, py, expect_inside=False)
    assert res["n_flagged"] == n


def test_point_in_triangle_all_nan_valid_zero() -> None:
    """All-NaN vertices → no valid frames, pct_flagged=0."""
    n = 3
    nan = np.full(n, np.nan)
    res = detect_point_in_triangle(nan, nan, nan, nan, nan, nan, nan, nan)
    assert res["n_flagged"] == 0
    assert res["pct_flagged"] == 0.0


def test_detect_head_midpoint_outside_triangle_wrapper() -> None:
    """Wrapper flags a head_midpoint far outside the nose-ears triangle."""
    n = 12
    nose_x = np.zeros(n)
    nose_y = np.zeros(n)
    le_x = np.full(n, 10.0)
    le_y = np.zeros(n)
    re_x = np.zeros(n)
    re_y = np.full(n, 10.0)
    mid_x = np.full(n, 50.0)  # far outside
    mid_y = np.full(n, 50.0)
    res = detect_head_midpoint_outside_triangle(
        nose_x, nose_y, le_x, le_y, re_x, re_y, mid_x, mid_y
    )
    assert res["n_outside"] == n
    assert res["pct_outside"] == 1.0
    assert res["is_outside"].all()


def test_detect_neck_inside_triangle_wrapper() -> None:
    """Wrapper flags a neck that falls inside the nose-ears triangle."""
    n = 12
    nose_x = np.zeros(n)
    nose_y = np.zeros(n)
    le_x = np.full(n, 10.0)
    le_y = np.zeros(n)
    re_x = np.zeros(n)
    re_y = np.full(n, 10.0)
    neck_x = np.full(n, 2.0)  # inside
    neck_y = np.full(n, 2.0)
    res = detect_neck_inside_triangle(
        nose_x, nose_y, le_x, le_y, re_x, re_y, neck_x, neck_y
    )
    assert res["n_inside"] == n
    assert res["is_inside"].all()


# ── detect_anterior_posterior_violations ────────────────────────────


def test_ap_violations_correct_order_none() -> None:
    """Keypoints in correct anterior→posterior order → no violations."""
    n = 15
    kps = {
        "nose_tip": (np.full(n, 0.0), np.zeros(n)),
        "neck": (np.full(n, 5.0), np.zeros(n)),
        "tail_base": (np.full(n, 10.0), np.zeros(n)),
    }
    res = detect_anterior_posterior_violations(kps)
    assert res["n_violated"] == 0
    assert res["violations_per_pair"] == {}


def test_ap_violations_swapped_order_flagged() -> None:
    """A keypoint posterior to its successor is flagged."""
    n = 15
    kps = {
        "nose_tip": (np.full(n, 0.0), np.zeros(n)),
        # neck placed behind tail_base (projection larger than tail).
        "neck": (np.full(n, 12.0), np.zeros(n)),
        "tail_base": (np.full(n, 10.0), np.zeros(n)),
    }
    res = detect_anterior_posterior_violations(kps)
    assert res["n_violated"] > 0
    assert "neck>tail_base" in res["violations_per_pair"]


def test_ap_violations_custom_order() -> None:
    """A custom order list is honoured."""
    n = 15
    kps = {
        "a": (np.full(n, 0.0), np.zeros(n)),
        "b": (np.full(n, 10.0), np.zeros(n)),
    }
    res = detect_anterior_posterior_violations(kps, order=["a", "b"])
    assert res["n_violated"] == 0


def test_ap_violations_too_few_available() -> None:
    """Fewer than 2 keypoints in the order → zeroed result."""
    n = 8
    kps = {"unknown_part": (np.zeros(n), np.zeros(n))}
    res = detect_anterior_posterior_violations(kps)
    assert res["n_violated"] == 0
    assert res["is_violated"].shape == (n,)
    assert res["pct_violated"] == 0.0


def test_ap_violations_empty_keypoints() -> None:
    """Empty keypoint dict → zero-length result."""
    res = detect_anterior_posterior_violations({})
    assert res["n_violated"] == 0
    assert res["is_violated"].shape == (0,)


# ── detect_ear_asymmetry ────────────────────────────────────────────


def test_ear_asymmetry_symmetric_none() -> None:
    """Ears equidistant from the axis → not asymmetric."""
    n = 12
    ax1, ay1, ax2, ay2 = _straight_axis(n)
    le_x = np.full(n, 5.0)
    le_y = np.full(n, 3.0)
    re_x = np.full(n, 5.0)
    re_y = np.full(n, -3.0)
    res = detect_ear_asymmetry(le_x, le_y, re_x, re_y, ax1, ay1, ax2, ay2)
    assert res["n_asymmetric"] == 0
    assert np.allclose(res["ratio"], 1.0)


def test_ear_asymmetry_flags_lopsided() -> None:
    """One ear much further from the axis than the other → flagged."""
    n = 12
    ax1, ay1, ax2, ay2 = _straight_axis(n)
    le_x = np.full(n, 5.0)
    le_y = np.full(n, 9.0)  # far from axis
    re_x = np.full(n, 5.0)
    re_y = np.full(n, -1.0)  # near axis → ratio ~9
    res = detect_ear_asymmetry(
        le_x, le_y, re_x, re_y, ax1, ay1, ax2, ay2, ratio_threshold=3.0
    )
    assert res["n_asymmetric"] == n
    assert res["is_asymmetric"].all()


# ── worst_frames / stratified_frame_selection (positions paths) ─────


def test_worst_frames_with_positions_diversity() -> None:
    """Positions enable pose-diversity filtering (2D likelihood input)."""
    n = 200
    rng = np.random.default_rng(0)
    lik = rng.uniform(0, 1, size=(n, 3))
    # 3 keypoints → 6 position columns; give each frame a distinct centroid.
    positions = rng.uniform(0, 500, size=(n, 6))
    idx = worst_frames(lik, n_frames=10, min_spacing=5, positions=positions)
    assert idx.ndim == 1
    assert len(idx) <= 10
    assert np.all(np.diff(idx) >= 5)


def test_worst_frames_positions_reject_similar() -> None:
    """Frames sharing centroid AND shape are treated as too similar."""
    n = 60
    # All frames identical position (same centroid + shape) → after the
    # first pick, every other candidate is 'too similar'.
    lik = np.linspace(0, 1, n)
    positions = np.zeros((n, 4))  # 2 keypoints, all at origin
    idx = worst_frames(
        lik, n_frames=10, min_spacing=1, positions=positions, min_position_dist=50.0
    )
    # Only one frame survives the diversity filter.
    assert len(idx) == 1


def test_stratified_selection_with_positions() -> None:
    """Stratified selection returns bins and honours the positions path."""
    n = 400
    rng = np.random.default_rng(1)
    lik = rng.uniform(0, 1, size=(n, 4))
    positions = rng.uniform(0, 1000, size=(n, 8))
    res = stratified_frame_selection(
        lik, n_per_bin=3, n_bins=4, min_spacing=5, positions=positions
    )
    assert res["total_selected"] == len(res["indices"])
    assert len(res["bins"]) == 4
    labels = [lbl for lbl, _ in res["bins"]]
    assert labels == ["worst", "poor", "moderate", "good"]


def test_stratified_selection_uniform_likelihood() -> None:
    """Near-uniform confidence still yields strictly increasing bin edges."""
    n = 100
    lik = np.full(n, 0.2)  # degenerate — all equal
    res = stratified_frame_selection(lik, n_per_bin=2, n_bins=4, min_spacing=1)
    assert res["total_selected"] >= 0
    assert len(res["bins"]) == 4


def test_stratified_selection_second_pass_fill() -> None:
    """Second pass fills up to the target when bins under-select."""
    n = 300
    rng = np.random.default_rng(2)
    lik = rng.uniform(0, 1, size=n)
    res = stratified_frame_selection(lik, n_per_bin=5, n_bins=4, min_spacing=2)
    # With plenty of well-spaced frames, the fill pass should reach target.
    assert res["total_selected"] <= 20
    assert res["total_selected"] > 0
