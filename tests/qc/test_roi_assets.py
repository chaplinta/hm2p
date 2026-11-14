"""Tests for hm2p.qc.roi_assets (synthetic Suite2p outputs only)."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.qc import roi_assets as ra
from hm2p.qc.common import decode_i16


def _roi(cy: int, cx: int, r: int, **extra: object) -> dict:
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    m = yy**2 + xx**2 <= r**2
    lam = np.exp(-(yy[m] ** 2 + xx[m] ** 2) / (r**2))
    d = {"ypix": yy[m] + cy, "xpix": xx[m] + cx, "lam": lam, "npix": int(m.sum()), "med": [cy, cx]}
    d.update(extra)
    return d


def _ops(ly: int = 40, lx: int = 50) -> dict:
    rng = np.random.default_rng(1)
    return {
        "Ly": ly,
        "Lx": lx,
        "yrange": [3, ly - 2],
        "xrange": [4, lx - 5],
        "meanImg": rng.normal(size=(ly, lx)),
        "meanImgE": rng.normal(size=(ly, lx)),
        "max_proj": rng.normal(size=(ly - 5, lx - 9)),
        "Vcorr": rng.normal(size=(ly - 5, lx - 9)),
        "xoff": rng.integers(-3, 4, 100),
        "yoff": rng.integers(-2, 3, 100),
        "corrXY": rng.uniform(0.5, 0.9, 100),
    }


def _u(b64: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


def test_pad_to_frame_full_and_cropped():
    full = np.arange(12.0).reshape(3, 4)
    np.testing.assert_array_equal(ra.pad_to_frame(full, 3, 4), full)
    crop = np.ones((2, 2))
    out = ra.pad_to_frame(crop, 4, 5, yrange=[1, 3], xrange=[2, 4])
    assert out.shape == (4, 5)
    assert np.all(out[1:3, 2:4] == 1) and np.isnan(out[0, 0]) and np.isnan(out[3, 4])
    assert np.isfinite(out).sum() == 4


def test_pad_to_frame_rejects_mismatch():
    with pytest.raises(ValueError):
        ra.pad_to_frame(np.ones((2, 2)), 4, 5)
    with pytest.raises(ValueError):
        ra.pad_to_frame(np.ones((3, 2)), 4, 5, yrange=[1, 3], xrange=[2, 4])
    with pytest.raises(ValueError):
        ra.pad_to_frame(np.ones((2, 2)), 4, 5, yrange=[3, 5], xrange=[2, 4])


def test_frame_images_pads_cropped_and_skips_bad():
    ops = _ops()
    ops["refImg"] = np.zeros((5, 5))  # not an IMAGE_KEY: ignored
    ops["meanImgE"] = np.zeros((7, 7))  # wrong shape: skipped
    imgs = ra.frame_images(ops)
    assert set(imgs) == {"meanImg", "max_proj", "Vcorr"}
    assert all(v.shape == (40, 50) for v in imgs.values())
    # the crop sits at yrange/xrange
    np.testing.assert_allclose(imgs["max_proj"][3:38, 4:45], ops["max_proj"])
    assert np.isnan(imgs["Vcorr"][0]).all()


def test_encode_footprints_roundtrip():
    stat = [_roi(10, 10, 2), _roi(20, 30, 3), {"ypix": [], "xpix": [], "lam": []}]
    fp = ra.encode_footprints(stat)
    start = _u(fp["start"], "<u4")
    y, x, w = _u(fp["y"], "<u2"), _u(fp["x"], "<u2"), _u(fp["w"], "u1")
    assert fp["n"] == 3 and start.tolist()[0] == 0 and start[-1] == y.size == x.size == w.size
    for i, s in enumerate(stat):
        sl = slice(start[i], start[i + 1])
        np.testing.assert_array_equal(y[sl], s["ypix"])
        np.testing.assert_array_equal(x[sl], s["xpix"])
        if len(s["lam"]):
            assert w[sl].max() == 255
            np.testing.assert_allclose(w[sl] / 255, s["lam"] / s["lam"].max(), atol=1 / 255)
    assert start[2] == start[3]  # empty ROI


def test_encode_footprints_missing_or_bad_lam():
    fp = ra.encode_footprints([{"ypix": [1, 2], "xpix": [3, 4], "lam": [1.0]}])
    assert _u(fp["w"], "u1").tolist() == [255, 255]
    assert ra.encode_footprints([])["n"] == 0


def test_stat_fields_and_iscell():
    stat = [_roi(5, 5, 2, radius=2.5, compact=1.1, skew=np.float32(3.0)), {"ypix": [1, 2, 3]}]
    iscell = np.array([[1, 0.93], [0, 0.12]])
    f = ra.stat_fields(stat, iscell)
    assert f[0]["radius"] == 2.5 and f[0]["skew"] == 3.0 and f[0]["aspect_ratio"] is None
    assert f[0]["iscell_prob"] == 0.93 and f[1]["npix"] == 3 and f[1]["iscell_prob"] == 0.12
    assert set(ra.STAT_FIELDS) <= set(f[0])
    assert "iscell_prob" not in ra.stat_fields(stat)[0]
    assert ra.stat_fields(stat, np.array([0.5, 0.2]))[1]["iscell_prob"] == 0.2
    json.dumps(f)


def test_choose_bin_meets_budget():
    assert ra.choose_bin(1000, 2, 5, budget_bytes=10**9) == 1
    k = ra.choose_bin(20000, 60, 5, budget_bytes=2_000_000)
    assert k > 1
    assert 60 * 5 * np.ceil(20000 / k) * 2 * 4 / 3 <= 2_000_000 * 1.01
    assert 60 * 5 * np.ceil(20000 / (k - 1)) * 2 * 4 / 3 > 2_000_000
    assert ra.choose_bin(0, 5, 5) == 1 and ra.choose_bin(10, 0, 5) == 1


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(1, 200),
    rois=st.integers(1, 50),
    traces=st.integers(1, 6),
    budget=st.integers(1000, 10**7),
)
def test_choose_bin_property(n, rois, traces, budget):
    k = ra.choose_bin(n, rois, traces, budget)
    assert k >= 1
    if k > 1:
        assert n * rois * traces * 2 * 4 / 3 / k <= budget


def test_encode_traces_bins_and_selects():
    a = np.arange(40, dtype=float).reshape(4, 10)
    enc = ra.encode_traces({"F": a, "none": None, "flat": np.zeros(5)}, [1, 3, 9], n_per_bin=2)
    assert set(enc) == {"F"}
    assert enc["F"][2] is None
    np.testing.assert_allclose(decode_i16(enc["F"][0]), a[1].reshape(5, 2).mean(axis=1), atol=1e-3)
    np.testing.assert_allclose(
        decode_i16(ra.encode_traces({"F": a}, [3])["F"][0]), a[3], atol=1e-3
    )


def test_binned_dark_intervals():
    light = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    assert ra.binned_dark_intervals(light) == [[2, 6], [8, 10]]
    assert ra.binned_dark_intervals(light, 2) == [[1, 3], [4, 5]]
    assert ra.binned_dark_intervals([]) == []


def test_registration_encodes_offsets():
    ops = _ops()
    r = ra.registration(ops)
    np.testing.assert_allclose(decode_i16(r["xoff"]), ops["xoff"], atol=1e-3)
    assert r["xoff_abs_max"] == np.abs(ops["xoff"]).max()
    assert r["corrXY_min"] == pytest.approx(ops["corrXY"].min(), abs=1e-4)
    assert ra.registration({}) == {}


def test_roi_qc_thresholds_match_calcium_qc():
    from hm2p.calcium import qc

    t = ra.roi_qc_thresholds()
    assert t["snr_event"] == [qc.SNR_MIN, None]
    assert t["decay_tau_s"] == [qc.TAU_MIN_S, qc.TAU_MAX_S]
    assert t["fneu_dff_corr"][1] == qc.FNEU_CORR_MAX
    assert set(t) == {
        "snr_event",
        "decay_tau_s",
        "fneu_dff_corr",
        "bleach_slope",
        "active_fraction",
    }


def test_session_asset_complete_and_json():
    ops = _ops()
    stat = [_roi(10, 10, 2), _roi(20, 30, 3), _roi(30, 15, 2)]
    rng = np.random.default_rng(0)
    T = 300
    tr = {k: rng.normal(size=(3, T)) for k in ra.TRACE_KEYS}
    tr["spikes"] = None
    light = np.repeat([1, 0, 1], T // 3).astype(bool)
    a = ra.session_asset(ops, stat, tr, [0, 2], fps=9.6, light_on=light, budget_bytes=2000)
    assert set(a["images"]) == set(ra.IMAGE_KEYS)
    assert a["images"]["max_proj"]["width"] == 50 and a["image_shape"] == [40, 50]
    assert a["trace_rois"] == [0, 2] and set(a["traces"]) == {"F", "Fneu", "F_corr", "dff"}
    assert a["bin"] > 1 and a["n_frames"] == T
    assert len(decode_i16(a["traces"]["dff"][1])) == int(np.ceil(T / a["bin"]))
    assert a["dark"] and a["footprints"]["n"] == 3 and "xoff" in a["registration"]
    json.dumps(a)
    b = ra.session_asset(ops, stat, {}, [], fps=9.6)
    assert b["traces"] == {} and b["bin"] == 1 and "dark" not in b
