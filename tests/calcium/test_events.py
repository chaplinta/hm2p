"""Tests for calcium/events.py — Voigts & Harnett event detection."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.calcium.events import detect_events


def test_events_module_importable() -> None:
    """events module can be imported without CASCADE."""
    from hm2p.calcium import events  # noqa: F401


# ---------------------------------------------------------------------------
# detect_events
# ---------------------------------------------------------------------------


class TestDetectEvents:
    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Output shape matches input (n_rois, n_frames)."""
        dff = rng.standard_normal((10, 500)).astype(np.float32)
        result = detect_events(dff, fps=30.0)
        assert result.shape == dff.shape

    def test_output_dtype_float32(self, rng: np.random.Generator) -> None:
        """Output is float32."""
        dff = rng.standard_normal((5, 200)).astype(np.float32)
        result = detect_events(dff, fps=30.0)
        assert result.dtype == np.float32

    def test_output_binary(self, rng: np.random.Generator) -> None:
        """Output contains only 0.0 and 1.0."""
        dff = rng.standard_normal((8, 300)).astype(np.float32)
        result = detect_events(dff, fps=30.0)
        unique = np.unique(result)
        assert set(unique).issubset({0.0, 1.0})

    def test_flat_signal_no_events(self) -> None:
        """Constant signal produces no events (z-score is 0)."""
        dff = np.zeros((3, 300), dtype=np.float32)
        result = detect_events(dff, fps=30.0, threshold_z=2.0)
        assert result.sum() == 0.0

    def test_large_transient_detected(self) -> None:
        """A large, sustained transient is detected as an event."""
        fps = 30.0
        n_frames = 300
        dff = np.zeros((1, n_frames), dtype=np.float32)
        dff[0, 100:130] = 10.0  # well above any threshold
        result = detect_events(dff, fps=fps, threshold_z=2.0, min_duration_s=0.1)
        assert result[0, 100:130].sum() > 0

    def test_min_duration_filters_short_events(self) -> None:
        """Events shorter than min_duration_s are suppressed."""
        fps = 30.0
        n_frames = 300
        dff = np.zeros((1, n_frames), dtype=np.float32)
        # 2-frame spike: 2/30 ≈ 0.067 s < min_duration_s=0.5 s
        dff[0, 100:102] = 10.0
        result = detect_events(dff, fps=fps, threshold_z=2.0, min_duration_s=0.5)
        assert result[0, 100:102].sum() == 0.0

    def test_min_duration_keeps_long_events(self) -> None:
        """Events longer than min_duration_s are kept."""
        fps = 30.0
        n_frames = 600
        dff = np.zeros((1, n_frames), dtype=np.float32)
        # 60-frame transient: 2.0 s >> min_duration_s=0.1 s
        dff[0, 200:260] = 8.0
        result = detect_events(dff, fps=fps, threshold_z=2.0, min_duration_s=0.1)
        assert result[0, 200:260].sum() > 0

    def test_high_threshold_fewer_events(self, rng: np.random.Generator) -> None:
        """Higher threshold_z detects ≤ events compared to lower threshold."""
        dff = rng.standard_normal((5, 500)).astype(np.float32)
        dff[:, 100:130] += 5.0
        dff[:, 300:330] += 5.0
        result_low = detect_events(dff, fps=30.0, threshold_z=1.5)
        result_high = detect_events(dff, fps=30.0, threshold_z=6.0)
        assert result_low.sum() >= result_high.sum()

    def test_no_events_in_pure_noise_high_threshold(self) -> None:
        """Pure low-amplitude noise does not trigger events at threshold_z=4."""
        rng = np.random.default_rng(0)
        dff = rng.normal(0, 0.05, (10, 1000)).astype(np.float32)
        result = detect_events(dff, fps=30.0, threshold_z=4.0)
        assert result.sum() == 0.0

    @given(
        n_rois=st.integers(min_value=1, max_value=20),
        n_frames=st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=30)
    def test_shape_property(self, n_rois: int, n_frames: int) -> None:
        """Output shape always equals input shape."""
        rng = np.random.default_rng(1)
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        result = detect_events(dff, fps=10.0)
        assert result.shape == (n_rois, n_frames)
