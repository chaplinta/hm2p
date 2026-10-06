"""Tests for hm2p.qc.tracking (synthetic keypoints only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hm2p.qc.tracking import (
    DEFAULT_JUMP_PX,
    frozen_mask,
    jump_threshold_px,
    keypoints_from_dataset,
    normalise_names,
    overview_row,
    summarise_tracking,
)


def _mouse(n: int = 900, seed: int = 0) -> dict:
    """A mouse walking on a circle, facing along its path, with jitter."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 4 * np.pi, n)
    cx, cy = 300 + 100 * np.cos(th), 300 + 100 * np.sin(th)
    hx, hy = -np.sin(th), np.cos(th)  # heading unit vector
    px, py = -hy, hx  # left-pointing perpendicular
    offs = {
        "nose_tip": (20, 0),
        "left_ear": (10, 6),
        "right_ear": (10, -6),
        "head_midpoint": (10, 0),
        "neck": (4, 0),
        "mid_back": (-8, 0),
        "mouse_center": (-12, 0),
        "tail_base": (-30, 0),
    }
    kp = {}
    for b, (a, s) in offs.items():
        kp[b] = {
            "x": cx + a * hx + s * px + rng.normal(0, 0.8, n),
            "y": cy + a * hy + s * py + rng.normal(0, 0.8, n),
            "likelihood": rng.uniform(0.2, 0.6, n),
        }
    return kp


def test_keypoints_from_dataset_picks_best_individual():
    import xarray as xr

    T = 4
    pos = np.zeros((T, 2, 2, 2))  # time, individuals, keypoints, space
    pos[:, 0] = 1.0
    pos[:, 1] = 2.0
    conf = np.full((T, 2, 2), 0.1)
    conf[2:, 1] = 0.9  # individual 1 better on the last two frames
    ds = xr.Dataset(
        {
            "position": (("time", "individuals", "keypoints", "space"), pos),
            "confidence": (("time", "individuals", "keypoints"), conf),
        },
        coords={"keypoints": ["nose_tip", "neck"], "space": ["x", "y"]},
    )
    kp = keypoints_from_dataset(ds)
    assert kp["neck"]["x"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert kp["nose_tip"]["likelihood"].tolist() == [0.1, 0.1, 0.9, 0.9]
    one = keypoints_from_dataset(ds.isel(individuals=[0]))
    assert one["neck"]["y"].tolist() == [1.0] * 4


def test_normalise_names_maps_legacy():
    kp = normalise_names({"implant_base_rear": {"x": [1]}, "nose": {"x": [2]}})
    assert set(kp) == {"head_midpoint", "nose_tip"}


def test_jump_threshold_scales():
    assert jump_threshold_px(30, None) == DEFAULT_JUMP_PX
    assert jump_threshold_px(30, 0.5) == pytest.approx(100.0)


def test_frozen_mask_detects_still_segment():
    x = np.arange(100.0)
    y = np.zeros(100)
    x[40:60] = 40.0
    m = frozen_mask(x, y, window=10)
    assert m[45] and not m[5] and not m[90]
    assert not frozen_mask(x[:5], y[:5], window=10).any()


def test_frozen_mask_ignores_nan_windows():
    x = np.full(50, np.nan)
    assert not frozen_mask(x, x, window=10).any()


def test_summary_clean_mouse():
    kp = _mouse()
    s = summarise_tracking(kp, fps=30.0, mm_per_px=0.5, light_on=np.arange(900) < 450)
    json.dumps(s)
    assert s["unit"] == "mm" and s["n_frames"] == 900
    assert s["ear"]["swap_frac"] == 0.0
    assert s["ear"]["median"] == pytest.approx(6.0, abs=0.5)  # 12 px * 0.5
    assert s["ap_order"]["frac"] == 0.0
    assert s["any_jump_frac"] == 0.0
    assert s["per_bp"]["nose_tip"]["lik_median_light"] is not None
    assert len(s["timeline"]["t_min"]) == 3  # 30 s / 10 s bins
    assert len(s["worst_bins"]) == 3
    row = overview_row(s)
    assert 0.2 < row["median_lik"] < 0.6


def test_summary_detects_swaps_and_jumps():
    kp = _mouse()
    le, r = kp["left_ear"], kp["right_ear"]
    sw = slice(100, 200)
    le["x"][sw], r["x"][sw] = r["x"][sw].copy(), le["x"][sw].copy()
    le["y"][sw], r["y"][sw] = r["y"][sw].copy(), le["y"][sw].copy()
    kp["nose_tip"]["x"][500] += 400
    s = summarise_tracking(kp, fps=30.0, mm_per_px=0.5)
    assert s["ear"]["swap_frac"] == pytest.approx(100 / 900, abs=0.01)
    assert s["any_jump_frac"] > 0
    assert "light" not in s["timeline"]


def test_extra_keypoints_listed_not_scored():
    kp = _mouse()
    rng = np.random.default_rng(3)
    kp["tail_end"] = {
        "x": rng.uniform(0, 600, 900),
        "y": rng.uniform(0, 600, 900),
        "likelihood": np.full(900, 0.01),
    }
    s = summarise_tracking(kp, fps=30.0, mm_per_px=0.5)
    assert s["extra_keypoints"] == ["tail_end"]
    assert "tail_end" not in s["per_bp"]
    assert s["any_jump_frac"] == 0.0  # the jumping tail point is not scored


def test_frozen_counted_only_while_body_moves():
    kp = _mouse()
    kp["nose_tip"]["x"] = np.full(900, 100.0)  # stuck nose while the mouse walks
    kp["nose_tip"]["y"] = np.full(900, 100.0)
    s = summarise_tracking(kp, fps=30.0, mm_per_px=0.5)
    assert s["per_bp"]["nose_tip"]["frozen_moving_frac"] > 0.9
    assert s["per_bp"]["neck"]["frozen_moving_frac"] == 0.0
    still = {
        b: {"x": np.full(900, 50.0), "y": np.full(900, 50.0), "likelihood": np.ones(900)}
        for b in kp
    }
    s2 = summarise_tracking(still, fps=30.0, mm_per_px=0.5)
    assert s2["per_bp"]["nose_tip"]["frozen_moving_frac"] == 0.0  # still mouse: not flagged
    assert s2["body_moving_frac"] == 0.0


def test_summary_pixels_without_scale_and_errors():
    s = summarise_tracking(_mouse(), fps=30.0)
    assert s["unit"] == "px" and s["jump_threshold"] == DEFAULT_JUMP_PX
    with pytest.raises(ValueError):
        summarise_tracking({}, fps=30.0)
