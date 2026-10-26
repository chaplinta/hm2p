"""Tests for hm2p.qc.movement (synthetic kinematics only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hm2p.qc.common import decode_i16
from hm2p.qc.movement import (
    bout_lengths,
    clean_attrs,
    overview_row,
    summarise_movement,
    trajectory_asset,
)


def _kin(n: int = 6000, fps: float = 30.0, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    hd = np.cumsum(rng.normal(0, 2, n))
    speed = np.abs(rng.normal(5, 3, n))
    x = np.cumsum(rng.normal(0, 1, n))
    y = np.cumsum(rng.normal(0, 1, n))
    light = (t // 60).astype(int) % 2 == 0
    return {
        "frame_times": t,
        "hd_deg": hd,
        "hd_ears": np.mod(hd, 360),
        "hd_nose_head": np.mod(hd + rng.normal(0, 5, n), 360),
        "hd_nose_neck": np.mod(hd + rng.normal(0, 10, n), 360),
        "hd_head_neck": np.mod(hd + 180, 360),  # deliberately opposite
        "hd_confidence": rng.uniform(0.3, 1.0, n),
        "speed_cm_s": speed,
        "speed_head_cm_s": speed * 1.1,
        "speed_body_cm_s": speed,
        "ahv_deg_s": np.gradient(hd) * fps,
        "x_mm": x,
        "y_mm": y,
        "x_maze": x / 50,
        "y_maze": y / 50,
        "x_maze_raw": x / 50 + 0.01,
        "y_maze_raw": y / 50,
        "active": speed > 0.5,
        "light_on": light,
        "bad_behav": np.zeros(n, bool),
    }


def test_clean_attrs():
    a = clean_attrs(
        {
            "s": b"x",
            "f": np.float32(0.25),
            "i": np.int64(3),
            "b": np.bool_(True),
            "arr": np.zeros(3),
        }
    )
    assert a == {"s": "x", "f": 0.25, "i": 3, "b": True}


def test_summary_basic_fields():
    d = _kin()
    s = summarise_movement(
        d, {"dlc_champion_id": "dlc-x-snap1", "speed_active_threshold_cm_s": 0.5}
    )
    json.dumps(s)
    assert s["fps"] == pytest.approx(30.0)
    assert s["timing"]["n_gaps"] == 0 and s["timing"]["non_monotonic"] == 0
    assert s["frac_light"] == pytest.approx(0.6, abs=0.01)
    agr = s["hd_agreement"]
    assert agr["hd_nose_head"]["median_abs"] < agr["hd_nose_neck"]["median_abs"]
    assert agr["hd_head_neck"]["frac_gt90"] == pytest.approx(1.0)
    assert s["raw_vs_filtered"]["q"]["q50"] == pytest.approx(0.01, abs=1e-6)
    assert s["active"]["threshold_cm_s"] == 0.5
    assert len(s["occupancy"]["counts"]) == 40
    row = overview_row(s)
    assert row["speed_implausible"] == 0.0


def test_excerpt_decodes_and_spans_two_minutes():
    s = summarise_movement(_kin())
    ex = s["excerpt"]
    sp = decode_i16(ex["speed_cm_s"])
    assert sp.size == 1800  # 120 s at 15 Hz
    li = decode_i16(ex["light"])
    assert set(np.unique(np.round(li))) == {0.0, 1.0}  # excerpt crosses a light switch


def test_gap_and_bad_behav_detection():
    d = _kin()
    d["frame_times"] = d["frame_times"].copy()
    d["frame_times"][3000:] += 1.0
    d["bad_behav"] = np.zeros(6000, bool)
    d["bad_behav"][100:200] = True
    s = summarise_movement(d)
    assert s["timing"]["n_gaps"] == 1
    assert s["frac_bad_behav"] == pytest.approx(100 / 6000, abs=1e-4)
    assert len(s["bad_intervals_s"]) == 1


def test_hd_confidence_histogram_spans_four_estimators():
    d = _kin()
    d["hd_confidence"] = np.full(6000, 3.2)
    h = summarise_movement(d)["hd_confidence"]["hist"]["light"]
    assert h["hi"] == 4.0 and h["above"] == 0


def test_missing_optional_keys_and_errors():
    d = {k: v for k, v in _kin().items() if k in ("frame_times", "speed_cm_s")}
    s = summarise_movement(d)
    assert "hd_agreement" not in s and s["raw_vs_filtered"] == {}
    with pytest.raises(ValueError):
        summarise_movement({"frame_times": np.zeros(2)})


# --- bout lengths and trajectory asset ---------------------------------------


def test_bout_lengths_censoring_and_durations():
    # light on for 40 frames then off for 40; bouts of 5 active / 5 inactive frames
    a = np.tile(np.r_[np.ones(5), np.zeros(5)], 8).astype(bool)
    light = np.r_[np.ones(40), np.zeros(40)].astype(bool)
    out = bout_lengths(a, light, np.zeros(80, bool), fps=10.0)
    al = out["active"]["light"]
    # the first active bout touches the session start
    assert al["n"] == 3 and al["n_censored"] == 1 and al["median_s"] == pytest.approx(0.5)
    il = out["inactive"]["light"]
    assert il["n"] == 3 and il["n_censored"] == 1  # the light switch at frame 40 cuts one
    ad = out["active"]["dark"]
    assert ad["n"] == 3 and ad["n_censored"] == 1  # ... and the bout after it
    idk = out["inactive"]["dark"]
    assert idk["n"] == 3 and idk["n_censored"] == 1  # the last bout touches the session end
    assert sum(idk["hist"]["counts"]) == 3
    assert ad["frac_single_frame"] == 0.0


def test_bout_lengths_bad_behav_censors_neighbours():
    a = np.r_[np.zeros(3), np.ones(4), np.zeros(3), np.ones(4), np.zeros(3)].astype(bool)
    bad = np.zeros(17, bool)
    bad[10:14] = True  # the second active bout is bad_behav
    out = bout_lengths(a, np.ones(17, bool), bad, fps=1.0)
    assert out["active"]["light"]["n"] == 1  # only the first active bout is uncensored
    assert out["inactive"]["light"]["n"] == 0  # all three inactive bouts are censored
    assert out["inactive"]["light"]["median_s"] is None


def test_bout_lengths_errors():
    with pytest.raises(ValueError):
        bout_lengths(np.ones(3, bool), np.ones(2, bool), np.zeros(3, bool), 30.0)
    with pytest.raises(ValueError):
        bout_lengths(np.ones(3, bool), np.ones(3, bool), np.zeros(3, bool), 0.0)


def test_summary_includes_bouts():
    s = summarise_movement(_kin(n=3000))
    assert set(s["bouts"]) >= {"active", "inactive", "log10_range"}
    json.dumps(s)


def test_trajectory_asset_subsamples_and_encodes():
    d = _kin(n=300)
    d["bad_behav"][30:60] = True
    a = trajectory_asset(d, step=3)
    assert a["step"] == 3 and a["t_s"]["n"] == 100
    np.testing.assert_allclose(decode_i16(a["t_s"]), d["frame_times"][::3], atol=1e-3)
    np.testing.assert_allclose(decode_i16(a["x_maze"]), d["x_maze"][::3], atol=1e-3)
    hd = decode_i16(a["hd_deg"])
    assert hd.min() >= 0 and hd.max() < 360.01
    bad = decode_i16(a["bad"]) > 0.5
    assert bad[10:20].all() and not bad[:10].any() and not bad[20:].any()
    assert "x_maze_raw" in a and "speed_cm_s" in a
    json.dumps(a)


def test_trajectory_asset_nan_and_errors():
    d = _kin(n=30)
    d["x_maze"][3] = np.nan
    assert np.isnan(decode_i16(trajectory_asset(d)["x_maze"])[1])
    with pytest.raises(ValueError):
        trajectory_asset({"x_maze": np.zeros(3)})
    with pytest.raises(ValueError):
        trajectory_asset(d, step=0)
