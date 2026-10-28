"""Tests for hm2p.qc.frames (synthetic keypoints and images only)."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.kinematics.compute import _rotate_xy, _vector_angle_deg
from hm2p.qc.frames import (
    SKELETON,
    ear_swap_mask,
    encode_jpeg,
    frame_keypoints,
    hd_to_video_vector,
    likelihood_rank,
    max_displacement,
    mm_to_video_px,
    select_review_frames,
)
from tests.qc.test_tracking import _mouse

PARTS = [
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]


def test_skeleton_uses_pipeline_keypoints():
    assert {b for e in SKELETON for b in e} == set(PARTS)


def test_likelihood_rank_orders_frames_and_equalises_scales():
    kp = {
        "a": {"likelihood": np.array([0.1, 0.2, 0.3, 0.4])},
        "b": {"likelihood": np.array([10.0, 20.0, 30.0, 40.0])},  # other scale, same order
    }
    r = likelihood_rank(kp, ["a", "b"])
    np.testing.assert_allclose(r, [0, 1 / 3, 2 / 3, 1])


def test_likelihood_rank_nan_is_worst():
    kp = {"a": {"likelihood": np.array([0.5, np.nan, 0.1])}}
    assert np.argmin(likelihood_rank(kp, ["a"])) == 1


def test_max_displacement_picks_largest_keypoint():
    kp = {
        "a": {"x": np.array([0.0, 1.0, 2.0]), "y": np.zeros(3)},
        "b": {"x": np.array([0.0, 0.0, 10.0]), "y": np.zeros(3)},
    }
    d, which = max_displacement(kp, ["a", "b"])
    assert np.isnan(d[0]) and which[0] == -1
    assert d[1] == 1.0 and which[1] == 0
    assert d[2] == 10.0 and which[2] == 1


def test_max_displacement_all_nan_frame():
    kp = {"a": {"x": np.array([0.0, np.nan, 1.0]), "y": np.zeros(3)}}
    d, which = max_displacement(kp, ["a"])
    assert np.isnan(d[1]) and np.isnan(d[2]) and (which[1:] == -1).all()


def test_ear_swap_mask_finds_swapped_frames():
    kp = _mouse(n=300)
    for f in range(100, 110):  # swap the ears for 10 frames
        for c in ("x", "y"):
            kp["left_ear"][c][f], kp["right_ear"][c][f] = (
                kp["right_ear"][c][f],
                kp["left_ear"][c][f],
            )
    m = ear_swap_mask(kp)
    assert m is not None and m[100:110].all() and m.sum() == 10


def test_ear_swap_mask_needs_ears():
    kp = _mouse(n=50)
    del kp["left_ear"]
    assert ear_swap_mask(kp) is None


def test_select_review_frames_categories_and_spacing():
    kp = _mouse(n=3000)
    for f in range(1000, 1012):
        for c in ("x", "y"):
            kp["left_ear"][c][f], kp["right_ear"][c][f] = (
                kp["right_ear"][c][f],
                kp["left_ear"][c][f],
            )
    kp["tail_base"]["x"][2000] += 200.0  # a single-frame jump (off the ear-swap axis)
    kp["tail_base"]["likelihood"][500:800] = 0.0
    picks = select_review_frames(kp, PARTS, fps=30.0, jump_px=50.0, worst_bin_starts_min=[
        500 / 30 / 60], bin_s=10.0, n_random=3, seed=1)  # fmt: skip
    reasons = [p["reason"] for p in picks]
    assert reasons.count("low likelihood") == 1
    assert reasons.count("ear swap") == 1
    assert reasons.count("reference") == 3
    jumps = [p for p in picks if p["reason"] == "jump"]
    assert jumps and jumps[0]["frame"] in (2000, 2001) and "tail_base" in jumps[0]["detail"]
    low = next(p for p in picks if p["reason"] == "low likelihood")
    assert 500 <= low["frame"] < 800 and low["bin_t_min"] == pytest.approx(500 / 1800)
    swap = next(p for p in picks if p["reason"] == "ear swap")
    assert 1000 <= swap["frame"] < 1012
    frames = [p["frame"] for p in picks]
    assert frames == sorted(frames)
    assert min(np.diff(frames)) >= 30  # at least 1 s apart
    # reference frames avoid flagged frames
    for p in picks:
        if p["reason"] == "reference":
            assert not 1000 <= p["frame"] < 1012 and p["frame"] not in (2000, 2001)


def test_select_review_frames_is_reproducible_and_skips_out_of_range_bins():
    kp = _mouse(n=600)
    a = select_review_frames(kp, PARTS, 30.0, 50.0, [99.0], 10.0, seed=3)
    b = select_review_frames(kp, PARTS, 30.0, 50.0, [99.0], 10.0, seed=3)
    assert a == b
    assert not any(p["reason"] == "low likelihood" for p in a)


def test_select_review_frames_empty_session():
    kp = {b: {"x": np.array([]), "y": np.array([]), "likelihood": np.array([])} for b in PARTS}
    assert select_review_frames(kp, PARTS, 30.0, 50.0, [], 10.0) == []


def test_frame_keypoints_rounds_and_maps_nan_to_none():
    kp = {"a": {"x": np.array([1.23456]), "y": np.array([np.nan]), "likelihood": np.array([0.5])}}
    assert frame_keypoints(kp, ["a"], 0) == {"a": [1.23, None, 0.5]}


@settings(max_examples=60, deadline=None)
@given(
    angle=st.floats(-360, 360),
    x=st.floats(0, 800),
    y=st.floats(0, 600),
    scale=st.floats(0.2, 2.0),
)
def test_mm_to_video_px_inverts_stage3_rotation(angle, x, y, scale):
    cx, cy = 400.0, 300.0
    xr, yr = _rotate_xy(np.array([x]), np.array([y]), angle, cx, cy)
    bx, by = mm_to_video_px(xr * scale, yr * scale, scale, angle, (cx, cy))
    np.testing.assert_allclose([bx[0], by[0]], [x, y], atol=1e-6)


@settings(max_examples=60, deadline=None)
@given(angle=st.floats(-360, 360), heading=st.floats(0, 2 * np.pi))
def test_hd_to_video_vector_points_from_neck_to_nose(angle, heading):
    # neck and nose in video pixels; stage 3 rotates them, then measures neck -> nose
    neck = np.array([300.0, 200.0])
    nose = neck + 20.0 * np.array([np.cos(heading), np.sin(heading)])
    pts_x, pts_y = _rotate_xy(np.array([neck[0], nose[0]]), np.array([neck[1], nose[1]]), angle,
                              350.0, 250.0)  # fmt: skip
    hd = _vector_angle_deg(pts_x[:1], pts_y[:1], pts_x[1:], pts_y[1:])
    vx, vy = hd_to_video_vector(hd, angle)
    np.testing.assert_allclose([vx[0], vy[0]], [np.cos(heading), np.sin(heading)], atol=1e-6)


def test_encode_jpeg_grey_and_colour_roundtrip():
    import cv2

    grey = np.tile(np.arange(64, dtype=np.uint8) * 4, (48, 1))
    b = base64.b64decode(encode_jpeg(grey, quality=90))
    back = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    assert back.shape == (48, 64) and np.abs(back.astype(int) - grey).max() < 12
    colour = np.zeros((10, 12, 3), np.uint8)
    assert base64.b64decode(encode_jpeg(colour))[:2] == b"\xff\xd8"
