"""Tests for hm2p.qc.spikes (synthetic traces only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hm2p.qc.common import decode_i16
from hm2p.qc.spikes import (
    busiest_window,
    event_count,
    jaccard,
    noise_sd,
    overview_row,
    summarise_spikes,
)


def test_noise_sd_recovers_white_noise():
    x = np.random.default_rng(0).normal(0, 0.2, 20000)
    assert noise_sd(x) == pytest.approx(0.2, rel=0.05)
    assert np.isnan(noise_sd([1.0]))


def test_noise_sd_ignores_slow_drift():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, 20000) + np.linspace(0, 50, 20000)
    assert noise_sd(x) == pytest.approx(0.1, rel=0.05)


def test_jaccard_and_event_count():
    a = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([1, 0, 0, 1, 1], bool)
    assert jaccard(a, b) == pytest.approx(2 / 4)
    assert np.isnan(jaccard(np.zeros(3, bool), np.zeros(3, bool)))
    assert event_count(a) == 2


def test_busiest_window():
    m = np.zeros(100, bool)
    m[70:80] = True
    assert 60 <= busiest_window(m, 20) <= 70
    assert busiest_window(np.zeros(100, bool), 20) == 40
    assert busiest_window(m, 200) == 0


def _ca(n_rois: int = 3, T: int = 3000, fps: float = 10.0) -> dict:
    rng = np.random.default_rng(2)
    dff = rng.normal(0, 0.05, (n_rois, T))
    vh = np.zeros((n_rois, T), bool)
    spk = np.zeros((n_rois, T))
    for i in range(n_rois):
        for on in range(100, T - 50, 400):
            dff[i, on : on + 20] += np.exp(-np.arange(20) / 8.0)
            vh[i, on : on + 20] = True
            spk[i, on] = 5.0
    F0 = np.full((n_rois, T), 100.0) * np.linspace(1.0, 0.8, T)
    return {
        "dff": dff,
        "event_masks": vh,
        "event_masks_sd": vh.copy(),
        "spikes": spk,
        "F0_rolling": F0,
        "F_corr": F0 * (1 + dff),
        "roi_types": np.array([0, 1, 2][:n_rois], dtype=np.uint8),
        "frame_times": np.arange(T) / fps,
        "roi_qc/snr_event": np.array([5.0, 1.0, 2.0][:n_rois]),
        "roi_qc/decay_tau_s": np.array([1.0, 1.0, 9.0][:n_rois]),
        "roi_qc/curated_label": np.array([b"", b"", b""][:n_rois]),
    }


def test_summary_metrics():
    s = summarise_spikes(
        _ca(),
        {"fps_imaging": 10.0, "f0_method": b"rolling", "spikes_model": "Global_EXC"},
        light_on=np.ones(3000),
    )
    json.dumps(s)
    r0 = s["rois"][0]
    assert r0["type"] == "soma"
    assert r0["rate_vh"] == pytest.approx(8 / 5.0, rel=0.01)  # 8 events in 5 min
    assert r0["jaccard_vh_sd"] == 1.0
    assert r0["spk_in_events"] == 1.0 and r0["vh_with_spikes"] == 1.0
    assert r0["f0_end_over_start"] == pytest.approx(0.8, abs=0.02)
    assert r0["qc_snr_event"] == 5.0 and "qc_curated_label" not in r0
    assert "trace" in r0 and "trace" not in s["rois"][2]  # artefact gets no trace
    assert decode_i16(r0["trace"]["dff"]).size == 900
    assert s["counts"] == {"soma": 1, "dend": 1, "artefact": 1}
    assert s["qc_fail"]["snr_event"] == 0.0
    assert "light" in s["population"]
    assert overview_row(s)["n_soma"] == 1


def test_summary_without_optional_arrays_uses_frame_times():
    ca = {
        "dff": np.random.default_rng(0).normal(size=(2, 500)),
        "frame_times": np.arange(500) / 9.6,
    }
    s = summarise_spikes(ca)
    assert s["fps"] == pytest.approx(9.6)
    assert s["has"] == {"spikes": False, "vh": False, "sd": False, "f0": False}
    assert "rate_vh" not in s["rois"][0]


def test_cascade_output_treated_as_expected_spikes_per_frame():
    # CASCADE gives expected spikes per frame: 0.1 in one frame is 0.1 spikes (< 0.5)
    ca = _ca(n_rois=1)
    ca["spikes"] = np.where(ca["spikes"] > 0, 0.1, 0.0)
    r = summarise_spikes(ca, {"fps_imaging": 10.0})["rois"][0]
    assert r["vh_with_spikes"] == 0.0
    # mean rate in Hz = mean expected spikes per frame x frame rate
    assert r["spk_rate"] == pytest.approx(ca["spikes"][0].mean() * 10.0, abs=1e-4)


def test_qc_fail_counts_nan_as_pass():
    ca = _ca(n_rois=1)
    ca["roi_qc/decay_tau_s"] = np.array([np.nan])
    s = summarise_spikes(ca, {"fps_imaging": 10.0})
    assert s["qc_fail"]["decay_tau_s"] == 0.0


def test_light_tolerates_one_frame_offset():
    s = summarise_spikes(_ca(n_rois=1), {"fps_imaging": 10.0}, light_on=np.ones(2999))
    assert "light" in s["population"]


def test_drift_ratio_and_raw_metric():
    from hm2p.qc.spikes import drift_ratio

    assert drift_ratio(np.linspace(100, 50, 1000)) == pytest.approx(0.5, abs=0.03)
    assert np.isnan(drift_ratio([1.0]))
    assert np.isnan(drift_ratio(np.r_[np.zeros(100), np.ones(100)]))
    ca = _ca(n_rois=1)
    ca["F_raw"] = np.linspace(200, 100, 3000)[None, :]
    s = summarise_spikes(ca, {"fps_imaging": 10.0, "dff_denominator": "F0 of F_raw"})
    assert s["rois"][0]["raw_end_over_start"] == pytest.approx(0.5, abs=0.03)
    assert s["dff_denominator"] == "F0 of F_raw"
    assert overview_row(s)["raw_end_over_start"] == pytest.approx(0.5, abs=0.03)


def test_summary_errors():
    with pytest.raises(ValueError):
        summarise_spikes({"dff": np.zeros(5)})
    with pytest.raises(ValueError):
        summarise_spikes({"dff": np.zeros((2, 5))})
