"""Tests for hm2p.qc.common."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hm2p.qc.common import (
    bin_reduce,
    circ_diff_deg,
    decode_i16,
    encode_i16,
    finite,
    fnum,
    hist,
    mask_intervals,
    quantiles,
    resample_bool,
    rounded,
    run_lengths,
    span_fps,
)


def test_finite_drops_nan_and_inf():
    assert finite([1.0, np.nan, np.inf, 2.0]).tolist() == [1.0, 2.0]


def test_fnum_handles_none_nan_and_rounding():
    assert fnum(None) is None
    assert fnum(np.nan) is None
    assert fnum(1.234567, 2) == 1.23


def test_rounded_nan_to_none():
    assert rounded([1.23456, np.nan], 2) == [1.23, None]


def test_hist_counts_off_scale_separately():
    h = hist([-1.0, 0.1, 0.5, 0.9, 2.0, np.nan], 0.0, 1.0, 2)
    assert h["counts"] == [1, 2]
    assert h["below"] == 1 and h["above"] == 1 and h["n"] == 5


def test_quantiles_empty_gives_none():
    q = quantiles([np.nan])
    assert set(q) == {"q05", "q25", "q50", "q75", "q95"}
    assert all(v is None for v in q.values())


def test_quantiles_median():
    assert quantiles(np.arange(101.0))["q50"] == 50.0


@given(
    arrays(np.float64, 20, elements=st.floats(-1e4, 1e4)),
    arrays(np.float64, 20, elements=st.floats(-1e4, 1e4)),
)
def test_circ_diff_in_range_and_consistent(a, b):
    d = circ_diff_deg(a, b)
    assert np.all(d >= -180.0) and np.all(d < 180.0)
    r = np.mod(a - d - b, 360.0)  # a - d must equal b modulo 360
    assert np.all(np.minimum(r, 360.0 - r) < 1e-6)


def test_span_fps_unbiased_for_subsampled_camera_times():
    # 100 Hz camera times subsampled to 30 fps with rounded indices: steps 30/30/40 ms
    t = np.round(np.linspace(0, 17999, 5400)).astype(int) / 100.0
    assert 1.0 / np.median(np.diff(t)) == pytest.approx(33.33, abs=0.01)  # the biased estimate
    assert span_fps(t) == pytest.approx(30.0, abs=0.01)
    assert np.isnan(span_fps([1.0])) and np.isnan(span_fps([2.0, 1.0]))


def test_bin_reduce_mean_and_partial_bin():
    np.testing.assert_allclose(bin_reduce([1, 2, 3, 4, 5], 2), [1.5, 3.5, 5.0])


def test_bin_reduce_median_all_nan_bin():
    out = bin_reduce([np.nan, np.nan, 1.0, 3.0], 2, "median")
    assert np.isnan(out[0]) and out[1] == 2.0


def test_bin_reduce_bad_args():
    with pytest.raises(ValueError):
        bin_reduce([1.0], 0)
    with pytest.raises(ValueError):
        bin_reduce([1.0], 1, "max")


@settings(max_examples=50)
@given(arrays(np.float64, st.integers(1, 200), elements=st.floats(-1e3, 1e3) | st.just(np.nan)))
def test_encode_decode_roundtrip(x):
    enc = encode_i16(x)
    json.dumps(enc)
    y = decode_i16(enc)
    ok = np.isfinite(x)
    assert np.array_equal(np.isnan(y), ~ok)
    if ok.any():
        span = x[ok].max() - x[ok].min()
        assert np.max(np.abs(y[ok] - x[ok])) <= max(span / 65534.0, 1e-9) * 0.51 + 1e-9


def test_encode_constant_trace():
    np.testing.assert_allclose(decode_i16(encode_i16(np.full(5, 3.0))), 3.0)


@given(st.lists(st.integers(0, 3), max_size=60))
def test_run_lengths_reconstructs(seq):
    v, s, n = run_lengths(seq)
    rebuilt = np.repeat(v, n) if len(seq) else np.array([])
    assert rebuilt.tolist() == list(seq)
    if len(seq):
        assert s[0] == 0 and np.all(np.diff(s) == n[:-1])


def test_mask_intervals():
    assert mask_intervals([0, 1, 1, 0, 1]) == [[1, 3], [4, 5]]
    assert mask_intervals([]) == []


def test_resample_bool_same_and_different_length():
    m = np.array([True, False, True, False])
    assert resample_bool(m, 4).tolist() == m.tolist()
    assert resample_bool(m, 8).tolist() == [True, True, False, False, True, True, False, False]
    assert resample_bool([], 3).tolist() == [False] * 3
