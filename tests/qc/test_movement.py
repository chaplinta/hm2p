"""Tests for hm2p.qc.movement (synthetic kinematics only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hm2p.qc.common import decode_i16
from hm2p.qc.movement import clean_attrs, overview_row, summarise_movement


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


def test_missing_optional_keys_and_errors():
    d = {k: v for k, v in _kin().items() if k in ("frame_times", "speed_cm_s")}
    s = summarise_movement(d)
    assert "hd_agreement" not in s and s["raw_vs_filtered"] == {}
    with pytest.raises(ValueError):
        summarise_movement({"frame_times": np.zeros(2)})
