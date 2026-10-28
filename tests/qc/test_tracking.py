"""Tests for hm2p.qc.tracking (synthetic keypoints only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hm2p.qc.common import decode_i16
from hm2p.qc.tracking import (
    DEFAULT_JUMP_PX,
    _bin_max,
    displacement_trace,
    frozen_mask,
    jump_threshold_px,
    keypoints_from_dataset,
    light_steps,
    normalise_names,
    overview_row,
    summarise_illumination,
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


# --- displacement trace and illumination ---------------------------------------


def test_displacement_trace_bins_maxima():
    kp = {
        "a": {"x": np.array([0.0, 1.0, 2.0, 12.0, 13.0, np.nan]), "y": np.zeros(6)},
        "b": {"x": np.zeros(6), "y": np.array([0.0, 0.0, 5.0, 5.0, 5.0, 5.0])},
    }
    out = displacement_trace(kp, ["a", "b"], per_bin=3, scale=2.0)
    np.testing.assert_allclose(decode_i16(out["max"]["a"]), [2.0, 20.0], atol=1e-3)
    np.testing.assert_allclose(decode_i16(out["max"]["b"]), [10.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(decode_i16(out["any"]), [10.0, 20.0], atol=1e-3)
    with pytest.raises(ValueError):
        displacement_trace(kp, ["a"], per_bin=0, scale=1.0)


def test_bin_max_all_nan_bin_and_partial_bin():
    r = _bin_max(np.array([np.nan, np.nan, 1.0, 3.0, 2.0]), 2)
    assert np.isnan(r[0]) and r[1] == 3.0 and r[2] == 2.0


def test_summary_has_displacement_trace():
    s = summarise_tracking(_mouse(n=900), fps=30.0, mm_per_px=0.5)
    assert s["disp"]["bin_s"] == 2.0 and s["disp"]["max"]["nose_tip"]["n"] == 15
    json.dumps(s)


def test_light_steps_ignore_drift():
    t = np.arange(0.0, 200.0, 1.0)
    v = 100.0 - 0.2 * t  # slow decay
    v[(t >= 60) & (t < 120)] += 5.0  # light on 60-120 s
    on = light_steps(t, v, [60.0], win_s=10.0)
    off = light_steps(t, v, [120.0], win_s=10.0)
    # the step, less the decay between the two window medians (~10.5 s apart)
    assert on[0] == pytest.approx(5.0 - 0.2 * 10.5, abs=0.3)
    assert off[0] == pytest.approx(-5.0 - 0.2 * 10.5, abs=0.3)
    assert np.isnan(light_steps(t, v, [500.0])[0])


def test_summarise_illumination_and_overview():
    t = np.arange(0.0, 300.0, 1.0)
    light = (t // 60) % 2 == 1
    d = {
        "frame_times": t,
        "intensities": 100.0 + 2.0 * light,
        "contrasts": np.full(t.size, 50.0),
        "is_light_on": light,
        "light_on_times": np.array([60.0, 180.0]),
        "light_off_times": np.array([120.0, 240.0]),
    }
    attrs = {"decay_pct": np.float64(3.5), "sample_every": np.int64(100), "exp_id": "x"}
    s = summarise_illumination(d, attrs)
    assert s["on_step_median"] == pytest.approx(2.0)
    assert s["off_step_median"] == pytest.approx(-2.0)
    assert s["step_rel"] == pytest.approx(0.02, abs=1e-3)
    assert s["attrs"] == {"decay_pct": 3.5, "sample_every": 100}
    assert decode_i16(s["light"]).sum() == light.sum() and "contrast" in s
    json.dumps(s)
    tr = summarise_tracking(_mouse(n=300), fps=30.0)
    tr["illumination"] = s
    assert overview_row(tr)["illum_step_rel"] == s["step_rel"]


def test_summarise_illumination_without_switches():
    s = summarise_illumination({"frame_times": np.arange(5.0), "intensities": np.ones(5)})
    assert s["on_step_median"] is None and s["step_rel"] is None and "light" not in s
