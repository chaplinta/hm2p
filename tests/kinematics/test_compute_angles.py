"""Tests for kinematics/compute.py angle, position and speed helpers.

Covers confidence-weighted head-centre, head/body posture angles, the
4-estimate HD fusion, weighted head/body positions and multipoint speeds.
All tests build small synthetic movement-style xarray Datasets inline.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from hm2p.kinematics.compute import (
    compute_body_position,
    compute_body_position_unweighted,
    compute_hd_multi,
    compute_head_body_angle,
    compute_head_centre,
    compute_head_direction,
    compute_head_position,
    compute_head_speed,
    compute_locomotion_speed,
    compute_neck_angle,
)

ALL_KEYPOINTS = [
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]

# Base (pixel) layout of a mouse facing +x, roughly along the x-axis.
_BASE = {
    "nose_tip": (10.0, 0.0),
    "left_ear": (5.0, 3.0),
    "right_ear": (5.0, -3.0),
    "head_midpoint": (5.0, 0.0),
    "neck": (2.0, 0.0),
    "mid_back": (-5.0, 0.0),
    "mouse_center": (-8.0, 0.0),
    "tail_base": (-12.0, 0.0),
}


def _make_ds(
    keypoints: list[str] | None = None,
    n_frames: int = 40,
    with_conf: bool = True,
    drift: tuple[float, float] = (0.5, 0.2),
) -> xr.Dataset:
    """Build a movement-style Dataset with the given keypoints.

    Each keypoint sits at its base layout plus a per-frame linear drift so
    that speed-based functions see genuine translation.
    """
    if keypoints is None:
        keypoints = ALL_KEYPOINTS
    n_kp = len(keypoints)
    pos = np.zeros((n_frames, 2, n_kp, 1), dtype=np.float64)
    t = np.arange(n_frames, dtype=np.float64)
    for k, name in enumerate(keypoints):
        bx, by = _BASE[name]
        pos[:, 0, k, 0] = bx + drift[0] * t
        pos[:, 1, k, 0] = by + drift[1] * t

    position = xr.DataArray(
        pos,
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": t,
            "space": ["x", "y"],
            "keypoints": keypoints,
            "individuals": ["mouse"],
        },
    )
    data = {"position": position}
    if with_conf:
        conf = np.full((n_frames, n_kp, 1), 0.9, dtype=np.float64)
        data["confidence"] = xr.DataArray(
            conf,
            dims=["time", "keypoints", "individuals"],
            coords={
                "time": t,
                "keypoints": keypoints,
                "individuals": ["mouse"],
            },
        )
    return xr.Dataset(data)


# ── compute_head_centre ─────────────────────────────────────────────


def test_head_centre_full_keypoints() -> None:
    ds = _make_ds()
    cx, cy = compute_head_centre(ds)
    assert cx.shape == (40,)
    assert cx.dtype == np.float32
    assert np.all(np.isfinite(cx))
    assert np.all(np.isfinite(cy))


def test_head_centre_without_confidence() -> None:
    """Presence-based weighting when no confidence array is present."""
    ds = _make_ds(with_conf=False)
    cx, cy = compute_head_centre(ds)
    assert np.all(np.isfinite(cx))


def test_head_centre_no_head_keypoints_all_nan() -> None:
    """Only body keypoints present → head centre is all NaN."""
    ds = _make_ds(keypoints=["mid_back", "mouse_center", "tail_base"])
    cx, cy = compute_head_centre(ds)
    assert np.all(np.isnan(cx))
    assert np.all(np.isnan(cy))


def test_head_centre_ear_midpoint_fallback() -> None:
    """With only ears available, centre is their midpoint (y≈0)."""
    ds = _make_ds(keypoints=["left_ear", "right_ear"])
    cx, cy = compute_head_centre(ds)
    # At t=0 ears are (5, ±3) → midpoint (5, 0).
    assert cx[0] == np.float32(5.0)
    assert abs(cy[0]) < 1e-4


# ── compute_head_body_angle ─────────────────────────────────────────


def test_head_body_angle_full_in_range() -> None:
    """A full-keypoint dataset yields finite angles wrapped to (-180, 180]."""
    ds = _make_ds()
    ang = compute_head_body_angle(ds)
    assert ang.shape == (40,)
    assert ang.dtype == np.float32
    assert np.all(np.isfinite(ang))
    assert np.all(ang <= 180.0)
    assert np.all(ang > -180.0)


def test_head_body_angle_missing_ears_nan() -> None:
    ds = _make_ds(keypoints=["mid_back", "tail_base"])
    ang = compute_head_body_angle(ds)
    assert np.all(np.isnan(ang))


def test_head_body_angle_missing_body_nan() -> None:
    """Ears present but no tail/back → NaN."""
    ds = _make_ds(keypoints=["left_ear", "right_ear", "nose_tip"])
    ang = compute_head_body_angle(ds)
    assert np.all(np.isnan(ang))


# ── compute_neck_angle ──────────────────────────────────────────────


def test_neck_angle_full_finite() -> None:
    ds = _make_ds()
    ang = compute_neck_angle(ds)
    assert ang.dtype == np.float32
    assert np.all(np.isfinite(ang))


def test_neck_angle_ear_fallback_when_no_implant() -> None:
    """No head_midpoint but ears present → ear-midpoint head end used."""
    kps = ["left_ear", "right_ear", "neck", "mid_back"]
    ds = _make_ds(keypoints=kps)
    ang = compute_neck_angle(ds)
    assert np.all(np.isfinite(ang))


def test_neck_angle_missing_neck_nan() -> None:
    ds = _make_ds(keypoints=["head_midpoint", "mid_back", "left_ear", "right_ear"])
    ang = compute_neck_angle(ds)
    assert np.all(np.isnan(ang))


def test_neck_angle_no_head_end_nan() -> None:
    """Neck + back present but no implant and not both ears → NaN."""
    ds = _make_ds(keypoints=["neck", "mid_back", "left_ear"])
    ang = compute_neck_angle(ds)
    assert np.all(np.isnan(ang))


# ── compute_hd_multi ────────────────────────────────────────────────


def test_hd_multi_full_returns_all_estimates() -> None:
    ds = _make_ds()
    out = compute_hd_multi(ds, scale_mm_per_px=0.5)
    for key in (
        "hd_deg",
        "hd_ears",
        "hd_nose_head",
        "hd_nose_neck",
        "hd_head_neck",
        "hd_confidence",
    ):
        assert key in out
        assert out[key].shape == (40,)
    assert np.all(np.isfinite(out["hd_deg"]))
    assert np.all(out["hd_confidence"] > 0)


def test_hd_multi_only_ears_other_estimates_nan() -> None:
    """With just the two ears, the three axis estimates are all NaN."""
    ds = _make_ds(keypoints=["left_ear", "right_ear"])
    out = compute_hd_multi(ds, scale_mm_per_px=0.5)
    assert np.all(np.isnan(out["hd_nose_head"]))
    assert np.all(np.isnan(out["hd_nose_neck"]))
    assert np.all(np.isnan(out["hd_head_neck"]))
    assert np.all(np.isfinite(out["hd_ears"]))


def test_hd_multi_missing_ears_raises() -> None:
    ds = _make_ds(keypoints=["nose_tip", "neck", "mid_back"])
    try:
        compute_hd_multi(ds, scale_mm_per_px=0.5)
    except ValueError as exc:
        assert "required for HD" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("expected ValueError for missing ears")


def test_hd_multi_without_confidence() -> None:
    """Presence-based weighting path when confidence is absent."""
    ds = _make_ds(with_conf=False)
    out = compute_hd_multi(ds, scale_mm_per_px=0.5)
    assert np.all(np.isfinite(out["hd_deg"]))


# ── compute_head_position / compute_body_position ───────────────────


def test_head_position_full_mm() -> None:
    ds = _make_ds()
    x, y = compute_head_position(ds, scale_mm_per_px=0.5)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))


def test_head_position_ear_nose_estimate_only() -> None:
    """No head_midpoint/neck → only the nose+ears centroid estimate."""
    ds = _make_ds(keypoints=["nose_tip", "left_ear", "right_ear"])
    x, y = compute_head_position(ds, scale_mm_per_px=1.0)
    assert np.all(np.isfinite(x))


def test_head_position_without_confidence() -> None:
    ds = _make_ds(with_conf=False)
    x, y = compute_head_position(ds, scale_mm_per_px=0.5)
    assert np.all(np.isfinite(x))


def test_body_position_full_mm() -> None:
    ds = _make_ds()
    x, y = compute_body_position(ds, scale_mm_per_px=0.5)
    assert x.shape == (40,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))


# ── compute_locomotion_speed / compute_head_speed ───────────────────


def test_locomotion_speed_positive_for_moving_mouse() -> None:
    ds = _make_ds(drift=(1.0, 0.0))  # steady translation along x
    frame_times = np.arange(40, dtype=np.float64) / 10.0  # 10 fps
    speed = compute_locomotion_speed(ds, frame_times, scale_mm_per_px=1.0)
    assert speed.shape == (40,)
    assert speed.dtype == np.float32
    assert np.nanmedian(speed) > 0


def test_head_speed_zero_for_static_mouse() -> None:
    ds = _make_ds(drift=(0.0, 0.0))  # no motion
    frame_times = np.arange(40, dtype=np.float64) / 10.0
    speed = compute_head_speed(ds, frame_times, scale_mm_per_px=1.0)
    assert np.allclose(np.nan_to_num(speed), 0.0, atol=1e-3)


# ── compute_head_direction (5-estimate fusion) ──────────────────────


def test_head_direction_full_finite() -> None:
    ds = _make_ds()
    hd = compute_head_direction(ds)
    assert hd.shape == (40,)
    assert hd.dtype == np.float32
    assert np.all(np.isfinite(hd))


def test_head_direction_ears_only() -> None:
    """Only ears → ear-perpendicular estimate is used (legacy path)."""
    ds = _make_ds(keypoints=["left_ear", "right_ear"])
    hd = compute_head_direction(ds)
    assert np.all(np.isfinite(hd))


def test_head_direction_missing_ears_raises() -> None:
    ds = _make_ds(keypoints=["nose_tip", "neck", "mid_back"])
    try:
        compute_head_direction(ds)
    except ValueError as exc:
        assert "Ears required for HD" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("expected ValueError for missing ears")


def test_head_direction_without_confidence() -> None:
    ds = _make_ds(with_conf=False)
    hd = compute_head_direction(ds)
    assert np.all(np.isfinite(hd))


# ── compute_body_position_unweighted ────────────────────────────────


def test_body_position_unweighted_full() -> None:
    ds = _make_ds()
    x, y = compute_body_position_unweighted(ds, scale_mm_per_px=0.5)
    assert x.shape == (40,)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))


def test_body_position_unweighted_no_body_all_nan() -> None:
    """No body keypoints present → result is all NaN."""
    ds = _make_ds(keypoints=["nose_tip", "left_ear", "right_ear"])
    x, y = compute_body_position_unweighted(ds, scale_mm_per_px=0.5)
    assert np.all(np.isnan(x))
    assert np.all(np.isnan(y))
