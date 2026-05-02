"""Tests for calcium/qc.py — per-ROI quality control metrics.

All tests use small synthetic arrays with known ground-truth metric values.
No real data files are read (per CLAUDE.md policy).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from hm2p.calcium.events import EventResult
from hm2p.calcium.qc import (
    _fit_exponential_decay,
    compute_active_fraction,
    compute_bleach_slope,
    compute_decay_tau,
    compute_fneu_dff_corr,
    compute_roi_qc,
    flag_roi_qc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event_result(
    onsets: list[int],
    offsets: list[int],
    n_frames: int,
    amplitudes: np.ndarray | None = None,
) -> EventResult:
    """Create a minimal EventResult for testing."""
    event_mask = np.zeros(n_frames, dtype=np.float32)
    for on, off in zip(onsets, offsets, strict=False):
        event_mask[on:off] = 1.0
    if amplitudes is None:
        amplitudes = np.ones(len(onsets), dtype=np.float32)
    noise_prob = np.ones(n_frames, dtype=np.float64)
    for on, off in zip(onsets, offsets, strict=False):
        noise_prob[on:off] = 0.05  # below typical alpha threshold
    return EventResult(
        onsets=np.array(onsets, dtype=int),
        offsets=np.array(offsets, dtype=int),
        amplitudes=amplitudes,
        event_mask=event_mask,
        noise_prob=noise_prob,
    )


def _exponential_decay_trace(
    n_frames: int,
    fps: float,
    tau_s: float,
    amplitude: float = 2.0,
    baseline: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a clean exponential decay starting at frame 0."""
    t = np.arange(n_frames, dtype=float) / fps
    trace = amplitude * np.exp(-t / tau_s) + baseline
    if rng is not None:
        trace += rng.normal(0, 0.01, n_frames)
    return trace.astype(np.float32)


# ---------------------------------------------------------------------------
# _fit_exponential_decay
# ---------------------------------------------------------------------------


class TestFitExponentialDecay:
    def test_clean_decay_recovers_tau(self):
        """Fit on a clean decay segment recovers tau within 10%."""
        fps = 9.6
        tau_true = 1.2
        n = 80
        segment = _exponential_decay_trace(n, fps, tau_true, amplitude=3.0)
        tau_fit = _fit_exponential_decay(segment, fps)
        assert np.isfinite(tau_fit)
        assert abs(tau_fit - tau_true) / tau_true < 0.10

    def test_too_short_returns_nan(self):
        """Segments shorter than 4 frames return NaN."""
        segment = np.array([2.0, 1.5, 1.0], dtype=np.float32)
        result = _fit_exponential_decay(segment, fps=9.6)
        assert np.isnan(result)

    def test_empty_returns_nan(self):
        result = _fit_exponential_decay(np.array([]), fps=9.6)
        assert np.isnan(result)

    def test_flat_trace_returns_valid_or_nan(self):
        """Flat trace (no decay) should not raise; may return NaN."""
        segment = np.ones(30, dtype=np.float32) * 1.5
        result = _fit_exponential_decay(segment, fps=9.6)
        # Either NaN (no decay detectable) or a very large tau — both acceptable.
        assert np.isnan(result) or result > 0

    def test_non_positive_peak_returns_nan(self):
        """Segment with non-positive peak → NaN (log-space fitting guard)."""
        segment = np.full(20, -1.0, dtype=np.float32)
        result = _fit_exponential_decay(segment, fps=9.6)
        assert np.isnan(result)

    def test_tau_positive(self):
        """Fitted tau must be positive (not zero or negative)."""
        fps = 9.6
        segment = _exponential_decay_trace(60, fps, tau_s=0.5, amplitude=2.0)
        tau_fit = _fit_exponential_decay(segment, fps)
        if np.isfinite(tau_fit):
            assert tau_fit > 0


# ---------------------------------------------------------------------------
# compute_decay_tau
# ---------------------------------------------------------------------------


class TestComputeDecayTau:
    def _make_trace_with_events(
        self,
        fps: float,
        n_frames: int,
        tau_s: float,
        n_events: int,
        event_len: int = 40,
    ) -> tuple[np.ndarray, EventResult]:
        """Build a trace with n_events identical exponential events."""
        trace = np.zeros(n_frames, dtype=np.float64)
        onsets, offsets = [], []
        spacing = n_frames // (n_events + 1)
        for k in range(n_events):
            start = spacing * (k + 1)
            end = min(start + event_len, n_frames)
            seg = _exponential_decay_trace(end - start, fps, tau_s, amplitude=3.0)
            trace[start:end] = seg
            onsets.append(start)
            offsets.append(end)
        er = _make_event_result(onsets, offsets, n_frames)
        return trace.astype(np.float32), er

    def test_median_tau_close_to_true(self):
        """Median tau across events matches the true decay constant within 15%."""
        fps = 9.6
        tau_true = 1.0
        trace, er = self._make_trace_with_events(fps, 600, tau_s=tau_true, n_events=5)
        tau_est = compute_decay_tau(trace, er, fps)
        assert np.isfinite(tau_est)
        assert abs(tau_est - tau_true) / tau_true < 0.15

    def test_fewer_than_min_events_returns_nan(self):
        """Fewer than MIN_EVENTS_FOR_TAU (3) events → NaN."""
        fps = 9.6
        trace, er = self._make_trace_with_events(fps, 200, tau_s=1.0, n_events=2)
        result = compute_decay_tau(trace, er, fps)
        assert np.isnan(result)

    def test_zero_events_returns_nan(self):
        er = _make_event_result([], [], 200)
        trace = np.zeros(200, dtype=np.float32)
        result = compute_decay_tau(trace, er, fps=9.6)
        assert np.isnan(result)

    def test_returns_float(self):
        fps = 9.6
        trace, er = self._make_trace_with_events(fps, 600, tau_s=1.0, n_events=5)
        result = compute_decay_tau(trace, er, fps)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_fneu_dff_corr
# ---------------------------------------------------------------------------


class TestComputeFneuDffCorr:
    def test_perfectly_correlated_returns_1(self):
        """Identical traces → Spearman r = 1.0."""
        rng = np.random.default_rng(1)
        trace = rng.standard_normal(200).astype(np.float64)
        r = compute_fneu_dff_corr(trace, trace)
        assert np.isfinite(r)
        assert abs(r - 1.0) < 1e-6

    def test_anticorrelated_returns_negative(self):
        rng = np.random.default_rng(2)
        trace = rng.standard_normal(200).astype(np.float64)
        r = compute_fneu_dff_corr(trace, -trace)
        assert np.isfinite(r)
        assert r < 0

    def test_uncorrelated_close_to_zero(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal(500).astype(np.float64)
        b = rng.standard_normal(500).astype(np.float64)
        r = compute_fneu_dff_corr(a, b)
        # Uncorrelated → |r| < 0.15 with high probability at n=500
        assert np.isfinite(r)
        assert abs(r) < 0.15

    def test_known_spearman_correlation(self):
        """Result matches scipy.stats.spearmanr directly."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(200).astype(np.float64)
        b = a * 0.8 + rng.standard_normal(200) * 0.2
        expected = float(stats.spearmanr(a, b).statistic)
        r = compute_fneu_dff_corr(a, b)
        assert abs(r - expected) < 1e-6

    def test_too_few_frames_returns_nan(self):
        a = np.arange(5, dtype=np.float64)
        b = np.arange(5, dtype=np.float64)
        result = compute_fneu_dff_corr(a, b)
        assert np.isnan(result)

    def test_mismatched_lengths_returns_nan(self):
        a = np.zeros(100, dtype=np.float64)
        b = np.zeros(50, dtype=np.float64)
        result = compute_fneu_dff_corr(a, b)
        assert np.isnan(result)

    def test_constant_trace_returns_nan(self):
        """Constant traces → std=0 → NaN (not an error)."""
        a = np.ones(100, dtype=np.float64)
        b = np.ones(100, dtype=np.float64)
        result = compute_fneu_dff_corr(a, b)
        assert np.isnan(result)

    def test_returns_float(self):
        rng = np.random.default_rng(99)
        a = rng.standard_normal(100).astype(np.float64)
        b = rng.standard_normal(100).astype(np.float64)
        result = compute_fneu_dff_corr(a, b)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_bleach_slope
# ---------------------------------------------------------------------------


class TestComputeBleachSlope:
    def test_no_bleach_near_zero(self):
        """Flat trace → bleach slope close to 0."""
        F = np.ones(200, dtype=np.float32) * 500.0
        slope = compute_bleach_slope(F)
        assert abs(slope) < 0.01

    def test_strong_bleach_negative(self):
        """50% drop in fluorescence → slope ≈ -0.5."""
        n = 200
        # Linear drop from 1000 to 500
        F = np.linspace(1000, 500, n, dtype=np.float32)
        slope = compute_bleach_slope(F)
        # First 10% mean ≈ 975, last 10% mean ≈ 525; slope ≈ (525-975)/975 ≈ -0.46
        assert slope < -0.3
        assert slope > -0.6

    def test_gain_positive_slope(self):
        """Fluorescence gain → positive slope."""
        n = 200
        F = np.linspace(500, 1000, n, dtype=np.float32)
        slope = compute_bleach_slope(F)
        assert slope > 0.3

    def test_zero_mean_start_returns_nan(self):
        """F_start mean of zero → NaN (avoid division by zero)."""
        F = np.zeros(200, dtype=np.float32)
        result = compute_bleach_slope(F)
        assert np.isnan(result)

    def test_negative_start_returns_nan(self):
        """Negative F_start → NaN (physically invalid fluorescence)."""
        F = np.full(200, -100.0, dtype=np.float32)
        result = compute_bleach_slope(F)
        assert np.isnan(result)

    def test_short_trace(self):
        """Very short trace (1 frame) should not crash."""
        F = np.array([500.0], dtype=np.float32)
        result = compute_bleach_slope(F)
        # Either 0.0 (start == end) or NaN — just must not raise
        assert isinstance(result, float)

    def test_returns_float(self):
        F = np.linspace(1000, 800, 200, dtype=np.float32)
        result = compute_bleach_slope(F)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_active_fraction
# ---------------------------------------------------------------------------


class TestComputeActiveFraction:
    def test_all_active_returns_1(self):
        """Event mask all-1 → active_fraction = 1.0."""
        trace = np.ones(100, dtype=np.float64)
        mask = np.ones(100, dtype=np.float32)
        result = compute_active_fraction(trace, mask)
        assert result == pytest.approx(1.0)

    def test_all_inactive_returns_0(self):
        """Event mask all-0 → active_fraction = 0.0."""
        trace = np.zeros(100, dtype=np.float64)
        mask = np.zeros(100, dtype=np.float32)
        result = compute_active_fraction(trace, mask)
        assert result == pytest.approx(0.0)

    def test_half_active(self):
        """50% active frames → active_fraction = 0.5."""
        trace = np.zeros(100, dtype=np.float64)
        mask = np.zeros(100, dtype=np.float32)
        mask[:50] = 1.0
        result = compute_active_fraction(trace, mask)
        assert result == pytest.approx(0.5)

    def test_none_mask_uses_mad_proxy(self):
        """With no event mask, falls back to 3*MAD threshold."""
        # Large spikes above baseline in 20% of frames
        rng = np.random.default_rng(5)
        trace = rng.standard_normal(500) * 0.1  # baseline noise
        n_active = 100
        trace[:n_active] += 3.0  # large excursion in first 100 frames
        result = compute_active_fraction(trace, None)
        assert 0.0 <= result <= 1.0

    def test_flat_trace_with_no_mask(self):
        """Flat trace with no event mask → near-zero active fraction."""
        trace = np.ones(200, dtype=np.float64) * 0.05
        result = compute_active_fraction(trace, None)
        # MAD of constant trace is 0 → special case → returns 0.0
        assert result == pytest.approx(0.0)

    def test_empty_trace_returns_nan(self):
        result = compute_active_fraction(np.array([]), None)
        assert np.isnan(result)

    def test_returns_float(self):
        trace = np.random.default_rng(7).standard_normal(100)
        mask = np.zeros(100, dtype=np.float32)
        result = compute_active_fraction(trace, mask)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_roi_qc — integration: schema and dtypes
# ---------------------------------------------------------------------------


class TestComputeRoiQc:
    """Integration tests for compute_roi_qc output structure."""

    def _make_inputs(
        self,
        n_rois: int = 5,
        n_frames: int = 300,
        fps: float = 9.6,
        rng: np.random.Generator | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng(42)
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32) * 0.1
        F_raw = (rng.uniform(100, 500, (n_rois, n_frames)) + dff * 100).astype(np.float32)
        Fneu_raw = rng.uniform(50, 200, (n_rois, n_frames)).astype(np.float32)

        # Add a few events per ROI so event metrics aren't all NaN
        event_results = []
        event_masks = np.zeros((n_rois, n_frames), dtype=np.float32)
        for i in range(n_rois):
            # 5 events per ROI spread across the trace
            onsets = [30, 80, 130, 180, 230]
            offsets = [50, 100, 150, 200, 250]
            for on, off in zip(onsets, offsets, strict=False):
                dff[i, on:off] += 0.5  # small transient
                event_masks[i, on:off] = 1.0
            er = _make_event_result(
                onsets,
                offsets,
                n_frames,
                amplitudes=np.ones(5, dtype=np.float32) * 0.5,
            )
            event_results.append(er)

        return dff, F_raw, Fneu_raw, event_results, event_masks, fps

    def test_output_keys_present(self):
        dff, F_raw, Fneu, ers, masks, fps = self._make_inputs()
        qc = compute_roi_qc(dff, F_raw, Fneu, ers, masks, fps)
        expected = {
            "roi_qc/roi_index",
            "roi_qc/snr_event",
            "roi_qc/decay_tau_s",
            "roi_qc/fneu_dff_corr",
            "roi_qc/bleach_slope",
            "roi_qc/active_fraction",
        }
        assert set(qc.keys()) == expected

    def test_all_arrays_have_length_n_rois(self):
        n_rois = 7
        dff, F_raw, Fneu, ers, masks, fps = self._make_inputs(n_rois=n_rois)
        qc = compute_roi_qc(dff, F_raw, Fneu, ers, masks, fps)
        for key, arr in qc.items():
            assert len(arr) == n_rois, f"{key} has length {len(arr)}, expected {n_rois}"

    def test_roi_index_dtype_int32(self):
        dff, F_raw, Fneu, ers, masks, fps = self._make_inputs()
        qc = compute_roi_qc(dff, F_raw, Fneu, ers, masks, fps)
        assert qc["roi_qc/roi_index"].dtype == np.dtype("int32")

    def test_float_arrays_dtype_float32(self):
        dff, F_raw, Fneu, ers, masks, fps = self._make_inputs()
        qc = compute_roi_qc(dff, F_raw, Fneu, ers, masks, fps)
        float_keys = [
            "roi_qc/snr_event",
            "roi_qc/decay_tau_s",
            "roi_qc/fneu_dff_corr",
            "roi_qc/bleach_slope",
            "roi_qc/active_fraction",
        ]
        for key in float_keys:
            assert qc[key].dtype == np.dtype("float32"), f"{key} dtype={qc[key].dtype}"

    def test_roi_index_is_arange(self):
        n_rois = 6
        dff, F_raw, Fneu, ers, masks, fps = self._make_inputs(n_rois=n_rois)
        qc = compute_roi_qc(dff, F_raw, Fneu, ers, masks, fps)
        np.testing.assert_array_equal(qc["roi_qc/roi_index"], np.arange(n_rois, dtype=np.int32))

    def test_active_fraction_in_range(self):
        dff, F_raw, Fneu, ers, masks, fps = self._make_inputs()
        qc = compute_roi_qc(dff, F_raw, Fneu, ers, masks, fps)
        af = qc["roi_qc/active_fraction"]
        valid = af[np.isfinite(af)]
        assert np.all(valid >= 0.0) and np.all(valid <= 1.0)

    def test_none_event_results_produces_nan_snr(self):
        """With event_results=None, snr_event should be NaN for all ROIs."""
        dff, F_raw, Fneu, _ers, masks, fps = self._make_inputs()
        qc = compute_roi_qc(dff, F_raw, Fneu, event_results=None, event_masks=masks, fps=fps)
        assert np.all(np.isnan(qc["roi_qc/snr_event"]))

    def test_bleach_slope_plausible_for_flat_raw_f(self):
        """Flat F_raw → bleach_slope near 0."""
        n_rois, n_frames = 3, 300
        rng = np.random.default_rng(10)
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32) * 0.1
        F_raw = np.full((n_rois, n_frames), 400.0, dtype=np.float32)
        Fneu = rng.uniform(50, 150, (n_rois, n_frames)).astype(np.float32)
        qc = compute_roi_qc(dff, F_raw, Fneu, event_results=None, event_masks=None, fps=9.6)
        bs = qc["roi_qc/bleach_slope"]
        np.testing.assert_allclose(bs, 0.0, atol=0.01)

    def test_validate_ca_h5_accepts_roi_qc(self):
        """validate_ca_h5 passes when roi_qc/* arrays have correct shape/dtype."""
        from hm2p.io.hdf5 import validate_ca_h5

        n_rois, n_frames = 5, 200
        dff, F_raw, Fneu, ers, masks, fps = self._make_inputs(n_rois=n_rois, n_frames=n_frames)
        qc = compute_roi_qc(dff, F_raw, Fneu, ers, masks, fps)

        arrays: dict[str, np.ndarray] = {
            "frame_times": np.linspace(0, n_frames / fps, n_frames, dtype=np.float64),
            "dff": dff,
        }
        arrays.update(qc)
        # Should not raise
        validate_ca_h5(arrays)

    def test_validate_ca_h5_rejects_wrong_roi_qc_length(self):
        """validate_ca_h5 raises SchemaError if roi_qc/* length != n_rois."""
        from pandera.errors import SchemaError

        from hm2p.io.hdf5 import validate_ca_h5

        n_rois, n_frames = 5, 200
        fps = 9.6
        dff = np.zeros((n_rois, n_frames), dtype=np.float32)
        arrays: dict[str, np.ndarray] = {
            "frame_times": np.linspace(0, n_frames / fps, n_frames, dtype=np.float64),
            "dff": dff,
            # Wrong length: 4 instead of 5
            "roi_qc/roi_index": np.arange(4, dtype=np.int32),
        }
        with pytest.raises(SchemaError):
            validate_ca_h5(arrays)


# ---------------------------------------------------------------------------
# flag_roi_qc
# ---------------------------------------------------------------------------


class TestFlagRoiQc:
    def _make_qc_dict(self, n: int = 5) -> dict[str, np.ndarray]:
        """QC dict with all-passing values."""
        return {
            "roi_qc/roi_index": np.arange(n, dtype=np.int32),
            "roi_qc/snr_event": np.full(n, 5.0, dtype=np.float32),
            "roi_qc/decay_tau_s": np.full(n, 1.0, dtype=np.float32),
            "roi_qc/fneu_dff_corr": np.full(n, 0.2, dtype=np.float32),
            "roi_qc/bleach_slope": np.full(n, -0.05, dtype=np.float32),
            "roi_qc/active_fraction": np.full(n, 0.15, dtype=np.float32),
        }

    def test_all_pass_returns_all_false(self):
        qc = self._make_qc_dict(5)
        flagged = flag_roi_qc(qc)
        assert not np.any(flagged)
        assert flagged.dtype == bool

    def test_low_snr_flags_roi(self):
        qc = self._make_qc_dict(5)
        qc["roi_qc/snr_event"][2] = 1.0  # below SNR_MIN=3.0
        flagged = flag_roi_qc(qc)
        assert flagged[2]
        assert not flagged[0]

    def test_tau_out_of_range_low_flags(self):
        qc = self._make_qc_dict(5)
        qc["roi_qc/decay_tau_s"][1] = 0.05  # below TAU_MIN_S=0.2
        flagged = flag_roi_qc(qc)
        assert flagged[1]

    def test_tau_out_of_range_high_flags(self):
        qc = self._make_qc_dict(5)
        qc["roi_qc/decay_tau_s"][3] = 10.0  # above TAU_MAX_S=4.0
        flagged = flag_roi_qc(qc)
        assert flagged[3]

    def test_high_fneu_corr_flags(self):
        qc = self._make_qc_dict(5)
        qc["roi_qc/fneu_dff_corr"][0] = 0.9  # above FNEU_CORR_MAX=0.6
        flagged = flag_roi_qc(qc)
        assert flagged[0]

    def test_strong_bleach_flags(self):
        qc = self._make_qc_dict(5)
        qc["roi_qc/bleach_slope"][4] = -0.8  # below BLEACH_MAX_LOSS=-0.4
        flagged = flag_roi_qc(qc)
        assert flagged[4]

    def test_low_active_fraction_flags(self):
        qc = self._make_qc_dict(5)
        qc["roi_qc/active_fraction"][2] = 0.01  # below ACTIVE_FRAC_MIN=0.05
        flagged = flag_roi_qc(qc)
        assert flagged[2]

    def test_nan_metric_does_not_flag(self):
        """NaN metrics are unknown, not bad — should not flag the ROI."""
        qc = self._make_qc_dict(3)
        qc["roi_qc/snr_event"][1] = np.nan
        qc["roi_qc/decay_tau_s"][1] = np.nan
        flagged = flag_roi_qc(qc)
        assert not flagged[1]

    def test_multiple_failures_still_flags_once(self):
        """Multiple failing metrics on one ROI → still True (boolean OR)."""
        qc = self._make_qc_dict(3)
        qc["roi_qc/snr_event"][0] = 0.0
        qc["roi_qc/fneu_dff_corr"][0] = 1.0
        flagged = flag_roi_qc(qc)
        assert flagged[0]

    def test_custom_thresholds(self):
        """Custom threshold arguments override the module defaults."""
        qc = self._make_qc_dict(3)
        # With snr_min=10.0, all ROIs with snr=5.0 fail.
        flagged = flag_roi_qc(qc, snr_min=10.0)
        assert np.all(flagged)

    def test_output_shape_matches_input(self):
        n = 8
        qc = self._make_qc_dict(n)
        flagged = flag_roi_qc(qc)
        assert flagged.shape == (n,)
        assert flagged.dtype == bool


# ---------------------------------------------------------------------------
# HDF5 round-trip: roi_qc group readable from h5py
# ---------------------------------------------------------------------------


class TestRoiQcH5RoundTrip:
    def test_write_and_read_roi_qc_group(self, tmp_path):
        """roi_qc/* slash-keyed arrays create readable h5py group on disk."""
        import h5py

        from hm2p.io.hdf5 import write_h5

        n_rois = 4
        arrays = {
            "roi_qc/roi_index": np.arange(n_rois, dtype=np.int32),
            "roi_qc/snr_event": np.array([5.0, 2.0, 8.0, 1.0], dtype=np.float32),
            "roi_qc/decay_tau_s": np.array([1.0, 0.5, 2.0, np.nan], dtype=np.float32),
            "roi_qc/fneu_dff_corr": np.array([0.1, 0.7, 0.3, np.nan], dtype=np.float32),
            "roi_qc/bleach_slope": np.array([-0.05, -0.5, -0.02, np.nan], dtype=np.float32),
            "roi_qc/active_fraction": np.array([0.15, 0.03, 0.20, 0.10], dtype=np.float32),
        }

        path = tmp_path / "test_ca.h5"
        write_h5(path, arrays)

        with h5py.File(path, "r") as f:
            assert "roi_qc" in f, "roi_qc group must exist in HDF5 file"
            grp = f["roi_qc"]
            assert "roi_index" in grp
            assert "snr_event" in grp
            assert "decay_tau_s" in grp
            assert "fneu_dff_corr" in grp
            assert "bleach_slope" in grp
            assert "active_fraction" in grp

            np.testing.assert_array_equal(grp["roi_index"][:], np.arange(n_rois, dtype=np.int32))
            np.testing.assert_allclose(grp["snr_event"][:], [5.0, 2.0, 8.0, 1.0], rtol=1e-5)
            # NaN round-trips correctly
            assert np.isnan(grp["decay_tau_s"][3])

    def test_validate_ca_h5_without_roi_qc_still_passes(self):
        """Absence of roi_qc is allowed — it's an optional group."""
        from hm2p.io.hdf5 import validate_ca_h5

        n_frames, n_rois = 100, 5
        fps = 9.6
        arrays = {
            "frame_times": np.linspace(0, n_frames / fps, n_frames, dtype=np.float64),
            "dff": np.zeros((n_rois, n_frames), dtype=np.float32),
        }
        validate_ca_h5(arrays)  # Should not raise
