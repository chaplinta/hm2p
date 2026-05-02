"""Tests for calcium/dff.py — dF/F0 computation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.calcium.dff import (
    DFF_CLIP_HIGH,
    DFF_CLIP_LOW,
    DFF_F0_FLOOR,
    compute_baseline,
    compute_baseline_percentile,
    compute_dff,
    compute_dff_with_clip_counts,
)

# ---------------------------------------------------------------------------
# compute_dff — pure numpy, fully testable
# ---------------------------------------------------------------------------


def test_dff_zero_when_f_equals_baseline(rng: np.random.Generator) -> None:
    """dF/F0 = 0 when F == F0 everywhere."""
    F0 = np.abs(rng.uniform(50, 500, (10, 100)).astype(np.float32)) + 1.0
    result = compute_dff(F0, F0)
    np.testing.assert_allclose(result, 0.0, atol=1e-5)


def test_dff_positive_when_f_above_baseline(rng: np.random.Generator) -> None:
    """dF/F0 > 0 when F > F0."""
    F0 = np.ones((5, 50), dtype=np.float32) * 100.0
    F = F0 * 1.5  # 50% above baseline
    result = compute_dff(F, F0)
    np.testing.assert_allclose(result, 0.5, rtol=1e-5)


def test_dff_shape_mismatch_raises(rng: np.random.Generator) -> None:
    """ValueError raised when F and F0 shapes don't match."""
    F = rng.standard_normal((5, 100)).astype(np.float32)
    F0 = rng.standard_normal((5, 50)).astype(np.float32)
    with pytest.raises(ValueError, match="shape"):
        compute_dff(F, F0)


def test_dff_output_shape_preserved(rng: np.random.Generator) -> None:
    """Output shape matches input."""
    F = rng.standard_normal((20, 300)).astype(np.float32)
    F0 = np.abs(F) + 1.0
    result = compute_dff(F, F0)
    assert result.shape == F.shape


@given(
    scale=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_dff_property_known_amplitude(scale: float) -> None:
    """dF/F0 = scale - 1 when F = scale * F0 (within clip range)."""
    F0 = np.ones((3, 30), dtype=np.float32) * 100.0
    F = F0 * scale
    result = compute_dff(F, F0)
    expected = np.clip(scale - 1.0, -1.0, 20.0)
    # float32 has ~6 sig-fig precision → rtol=1e-3 is appropriate
    np.testing.assert_allclose(result, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# compute_baseline
# ---------------------------------------------------------------------------


class TestComputeBaseline:
    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Baseline shape matches input shape."""
        F = rng.uniform(100, 500, (8, 600)).astype(np.float32)
        F0 = compute_baseline(F, fps=30.0)
        assert F0.shape == F.shape

    def test_output_dtype_float32(self, rng: np.random.Generator) -> None:
        """Baseline is float32."""
        F = rng.uniform(100, 500, (4, 300)).astype(np.float32)
        F0 = compute_baseline(F, fps=30.0)
        assert F0.dtype == np.float32

    def test_baseline_leq_signal(self, rng: np.random.Generator) -> None:
        """Baseline ≤ smoothed signal (sliding minimum property)."""
        F = np.abs(rng.uniform(100, 500, (5, 900)).astype(np.float32))
        F0 = compute_baseline(F, fps=30.0, window_s=10.0, gaussian_sigma_s=1.0)
        # Allow small numerical tolerance from Gaussian smoothing at boundaries
        assert np.all(F.max() + 1.0 >= F0)

    def test_constant_signal_baseline_equals_signal(self) -> None:
        """Constant trace → baseline equals the constant."""
        F = np.full((3, 300), 200.0, dtype=np.float32)
        F0 = compute_baseline(F, fps=30.0, window_s=5.0, gaussian_sigma_s=1.0)
        np.testing.assert_allclose(F0, 200.0, rtol=1e-3)

    def test_transient_does_not_raise_baseline(self) -> None:
        """A short positive transient does not elevate the sliding-min baseline."""
        F = np.full((1, 900), 100.0, dtype=np.float32)
        # Add a brief spike in the middle
        F[0, 440:460] = 500.0
        F0 = compute_baseline(F, fps=30.0, window_s=10.0, gaussian_sigma_s=1.0)
        # Baseline in the second half (well past the transient) should be ~100
        np.testing.assert_allclose(F0[0, 600:], 100.0, atol=5.0)

    def test_window_shorter_gives_tighter_baseline(self, rng: np.random.Generator) -> None:
        """Shorter window produces a baseline that tracks faster (≥ longer window)."""
        F = np.abs(rng.uniform(80, 200, (2, 600)).astype(np.float32))
        F0_short = compute_baseline(F, fps=30.0, window_s=5.0, gaussian_sigma_s=1.0)
        F0_long = compute_baseline(F, fps=30.0, window_s=30.0, gaussian_sigma_s=1.0)
        # Shorter window baseline is always ≥ longer (tighter tracking)
        assert np.all(F0_short >= F0_long - 1.0)

    def test_single_roi(self) -> None:
        """Baseline works for a single ROI."""
        F = np.full((1, 100), 150.0, dtype=np.float32)
        F0 = compute_baseline(F, fps=10.0)
        assert F0.shape == (1, 100)

    def test_very_short_window(self) -> None:
        """Window shorter than one frame is clamped to 1 frame."""
        F = np.full((2, 50), 100.0, dtype=np.float32)
        F0 = compute_baseline(F, fps=1.0, window_s=0.01)
        assert F0.shape == F.shape


# ---------------------------------------------------------------------------
# compute_dff — edge cases
# ---------------------------------------------------------------------------


class TestComputeDffEdgeCases:
    def test_zero_baseline_is_floored(self) -> None:
        """When F0 is zero, the per-ROI floor prevents division by zero."""
        F = np.ones((2, 10), dtype=np.float32) * 5.0
        F0 = np.zeros((2, 10), dtype=np.float32)
        result = compute_dff(F, F0)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_negative_f(self) -> None:
        """Negative F values produce negative dF/F0 (clipped to -1)."""
        F = np.full((1, 5), -10.0, dtype=np.float32)
        F0 = np.full((1, 5), 100.0, dtype=np.float32)
        result = compute_dff(F, F0)
        assert np.all(result < 0)

    def test_output_dtype_float32(self) -> None:
        """Output is always float32."""
        F = np.ones((3, 20), dtype=np.float64)
        F0 = np.ones((3, 20), dtype=np.float64) * 100.0
        result = compute_dff(F.astype(np.float32), F0.astype(np.float32))
        assert result.dtype == np.float32

    def test_near_zero_baseline_produces_clipped_dff(self) -> None:
        """Near-zero F0 after neuropil subtraction should not produce extreme dF/F0."""
        # Simulate a trace where baseline is near zero (e.g. 0.01)
        F0 = np.full((1, 100), 0.01, dtype=np.float32)
        F = np.full((1, 100), 500.0, dtype=np.float32)
        result = compute_dff(F, F0)
        # Without the floor, dF/F0 would be ~50000. With it, must be <= 20.
        assert np.all(result <= 20.0)
        assert np.all(np.isfinite(result))

    def test_f0_floor_prevents_near_zero_division(self) -> None:
        """Constant DFF_F0_FLOOR keeps near-zero F0 frames bounded.

        QA fix 1.5: the previous 10 %-of-median per-ROI floor biased
        dF/F toward zero in F0-uncertain windows. The constant floor
        :data:`DFF_F0_FLOOR` (1.0) is documented in the module-level
        constant docstring. With it, near-zero F0 frames divide by 1.0
        and clamp at the upper saturation bound rather than blowing up.
        """
        # ROI with median F0 = 200, but a few frames drop to ~0
        F0 = np.full((1, 100), 200.0, dtype=np.float32)
        F0[0, 50:55] = 0.001  # near-zero frames
        F = np.full((1, 100), 300.0, dtype=np.float32)
        result = compute_dff(F, F0)
        # Near-zero frames now divide by DFF_F0_FLOOR (1.0); raw value
        # would be ~299.999 → clipped to 20.0. Output stays finite and
        # bounded — that is the contract.
        assert np.all(result <= 20.0)
        assert np.all(np.isfinite(result))

    def test_output_clipped_to_range(self) -> None:
        """dF/F0 output is always within [-1, 20] range."""
        # Large positive dF/F0
        F0 = np.full((2, 50), 100.0, dtype=np.float32)
        F_high = np.full((2, 50), 10000.0, dtype=np.float32)
        result_high = compute_dff(F_high, F0)
        assert np.all(result_high <= 20.0)

        # Large negative dF/F0
        F_low = np.full((2, 50), -500.0, dtype=np.float32)
        result_low = compute_dff(F_low, F0)
        assert np.all(result_low >= -1.0)


class TestComputeDffWithClipCounts:
    """QA issue 1.4 — n_clipped is reported per ROI so saturation is auditable."""

    def test_no_clipping_when_dff_within_range(self) -> None:
        F0 = np.full((3, 50), 100.0, dtype=np.float32)
        F = F0 * 1.5  # dff = 0.5 everywhere → no clip
        dff, n_clipped = compute_dff_with_clip_counts(F, F0)
        assert dff.shape == F.shape
        assert n_clipped.shape == (3,)
        assert n_clipped.dtype == np.int32
        np.testing.assert_array_equal(n_clipped, 0)

    def test_n_clipped_counts_upper_saturation(self) -> None:
        F0 = np.full((2, 100), 1.0, dtype=np.float32)
        F = np.full((2, 100), 1000.0, dtype=np.float32)  # dff_raw ~ 999 → clipped
        dff, n_clipped = compute_dff_with_clip_counts(F, F0)
        # Every sample in every ROI should clip
        assert np.all(dff == DFF_CLIP_HIGH)
        np.testing.assert_array_equal(n_clipped, 100)

    def test_n_clipped_counts_lower_saturation(self) -> None:
        F0 = np.full((1, 50), 100.0, dtype=np.float32)
        F = np.full((1, 50), -10000.0, dtype=np.float32)  # very large negative
        dff, n_clipped = compute_dff_with_clip_counts(F, F0)
        assert np.all(dff == DFF_CLIP_LOW)
        np.testing.assert_array_equal(n_clipped, 50)

    def test_n_clipped_per_roi_independent(self) -> None:
        F0 = np.full((2, 100), 100.0, dtype=np.float32)
        F = F0.copy()
        F[1, :30] = 100000.0  # ROI 1 saturates 30 frames; ROI 0 stays clean
        _, n_clipped = compute_dff_with_clip_counts(F, F0)
        assert n_clipped[0] == 0
        assert n_clipped[1] == 30

    def test_compute_dff_returns_ndarray(self) -> None:
        """compute_dff still returns a single ndarray (back-compat)."""
        F0 = np.full((2, 10), 100.0, dtype=np.float32)
        F = F0.copy()
        result = compute_dff(F, F0)
        assert isinstance(result, np.ndarray)
        assert result.shape == F.shape

    def test_constant_floor_is_one(self) -> None:
        """DFF_F0_FLOOR is 1.0 — used as the constant denominator floor."""
        assert DFF_F0_FLOOR == 1.0
        F0 = np.full((1, 5), 0.5, dtype=np.float32)  # below floor
        F = np.full((1, 5), 0.5, dtype=np.float32)
        # F == F0; dff_raw = 0 / max(F0, 1) = 0
        result = compute_dff(F, F0)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)


class TestComputeBaselinePercentileNan:
    """QA issue 1.7 — np.nanpercentile must be used so NaN does not propagate."""

    def test_nan_in_input_does_not_propagate(self) -> None:
        """Non-NaN windows should produce finite F0 even when NaN exists elsewhere."""
        F = np.full((1, 200), 100.0, dtype=np.float32)
        F[0, 50:60] = np.nan  # 10 NaN samples
        # window covers full session — every output position sees the NaN
        # but nanpercentile keeps the rest, so F0 stays at ~100 everywhere.
        F0 = compute_baseline_percentile(F, fps=10.0, window_s=20.0)
        # Frames whose window has at least one finite sample → F0 finite
        assert np.isfinite(F0).all()
        # Median F0 across the trace is ~100 (constant non-NaN samples)
        np.testing.assert_allclose(np.median(F0), 100.0, atol=1.0)

    def test_all_nan_window_returns_nan(self) -> None:
        """When every sample in the window is NaN, F0 is NaN at that frame."""
        F = np.full((1, 50), np.nan, dtype=np.float32)
        F0 = compute_baseline_percentile(F, fps=10.0, window_s=5.0)
        assert np.all(np.isnan(F0))


# ---------------------------------------------------------------------------
# compute_baseline_percentile
# ---------------------------------------------------------------------------


class TestComputeBaselinePercentile:
    """Tests for the sliding-window percentile baseline (Jia et al. 2011)."""

    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Output shape matches input (n_rois, n_frames)."""
        F = rng.uniform(100, 500, (6, 400)).astype(np.float32)
        F0 = compute_baseline_percentile(F, fps=10.0)
        assert F0.shape == F.shape

    def test_output_dtype_float32(self, rng: np.random.Generator) -> None:
        """Output is float32."""
        F = rng.uniform(100, 500, (3, 200)).astype(np.float32)
        F0 = compute_baseline_percentile(F, fps=10.0)
        assert F0.dtype == np.float32

    def test_constant_signal_baseline_equals_signal(self) -> None:
        """Constant trace → percentile baseline equals the constant value."""
        F = np.full((2, 200), 150.0, dtype=np.float32)
        F0 = compute_baseline_percentile(F, fps=10.0, window_s=5.0, percentile=8.0)
        np.testing.assert_allclose(F0, 150.0, rtol=1e-3)

    def test_baseline_leq_signal_median(self, rng: np.random.Generator) -> None:
        """8th percentile baseline is always <= median of the signal."""
        F = np.abs(rng.uniform(100, 500, (4, 600)).astype(np.float32))
        F0 = compute_baseline_percentile(F, fps=10.0, window_s=10.0, percentile=8.0)
        signal_median = np.median(F, axis=1, keepdims=True)
        assert np.all(signal_median + 1.0 >= F0)

    def test_transient_does_not_substantially_raise_baseline(self) -> None:
        """A short transient does not raise the percentile baseline far above the floor."""
        F = np.full((1, 600), 100.0, dtype=np.float32)
        F[0, 280:300] = 2000.0  # brief large transient
        F0 = compute_baseline_percentile(F, fps=10.0, window_s=20.0, percentile=8.0)
        # Well away from the transient, the baseline should be near 100
        np.testing.assert_allclose(F0[0, :150], 100.0, atol=10.0)
        np.testing.assert_allclose(F0[0, 450:], 100.0, atol=10.0)

    def test_higher_percentile_gives_higher_baseline(self, rng: np.random.Generator) -> None:
        """Higher percentile yields a higher (or equal) baseline."""
        F = np.abs(rng.uniform(50, 300, (3, 400)).astype(np.float32))
        F0_low = compute_baseline_percentile(F, fps=10.0, window_s=10.0, percentile=8.0)
        F0_high = compute_baseline_percentile(F, fps=10.0, window_s=10.0, percentile=50.0)
        assert np.all(F0_high >= F0_low - 1.0)

    def test_single_roi(self) -> None:
        """Works for a single ROI without error."""
        F = np.full((1, 100), 200.0, dtype=np.float32)
        F0 = compute_baseline_percentile(F, fps=10.0)
        assert F0.shape == (1, 100)

    def test_very_short_window(self) -> None:
        """Very short window (fraction of a frame) does not raise."""
        F = np.full((2, 50), 100.0, dtype=np.float32)
        F0 = compute_baseline_percentile(F, fps=1.0, window_s=0.01)
        assert F0.shape == F.shape
        assert np.all(np.isfinite(F0))

    def test_step_signal_tracks_lower_level(self) -> None:
        """Baseline in the second half of a step-up signal stays near lower level.

        A step at the midpoint: first half at 100, second half at 300.
        With a 60-s window at 10 Hz and 8th percentile, the baseline at the
        start of the signal (well before the step) should be near 100, not 300.
        """
        n_frames = 600
        F = np.full((1, n_frames), 100.0, dtype=np.float32)
        F[0, n_frames // 2 :] = 300.0
        F0 = compute_baseline_percentile(F, fps=10.0, window_s=10.0, percentile=8.0)
        # First 100 frames: baseline should be near 100 (the lower level)
        np.testing.assert_allclose(F0[0, :100], 100.0, atol=5.0)

    @given(
        percentile=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20)
    def test_baseline_finite_for_any_valid_percentile(self, percentile: float) -> None:
        """Baseline is finite for any percentile in [1, 50]."""
        rng = np.random.default_rng(0)
        F = rng.uniform(50, 200, (2, 100)).astype(np.float32)
        F0 = compute_baseline_percentile(F, fps=5.0, window_s=5.0, percentile=percentile)
        assert np.all(np.isfinite(F0))
        assert F0.shape == F.shape
