"""Tests for calcium/spikes.py — CASCADE spike inference helpers."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hm2p.calcium.spikes import (
    _FPS_MISMATCH_THRESHOLD_HZ,
    _parse_model_fps,
    compute_mean_spike_rate,
    predict_spike_rates,
)


def test_mean_spike_rate_no_bad_frames() -> None:
    """Mean spike rate (spikes/min) matches expected value without bad frames."""
    # 1 spike/s constant → 60 spikes/min
    spikes = np.ones((5, 1000), dtype=np.float32)
    result = compute_mean_spike_rate(spikes, fps=10.0, bad_frames=None)
    np.testing.assert_allclose(result, 60.0, rtol=1e-5)


def test_mean_spike_rate_with_bad_frames() -> None:
    """Bad frames are excluded from mean spike rate computation."""
    # ROI 0 has spike rate 1 spikes/s in good frames, 0 in bad frames
    n_frames = 100
    spikes = np.zeros((1, n_frames), dtype=np.float32)
    bad_frames = np.zeros(n_frames, dtype=bool)
    bad_frames[50:] = True  # last 50 frames are bad
    spikes[0, :50] = 1.0  # only good frames have spikes

    result = compute_mean_spike_rate(spikes, fps=1.0, bad_frames=bad_frames)
    # Mean over 50 good frames: 1.0 spikes/s * 60 = 60 spikes/min
    np.testing.assert_allclose(result, [60.0], rtol=1e-5)


def test_mean_spike_rate_shape(rng: np.random.Generator) -> None:
    """Output shape is (n_rois,)."""
    spikes = rng.uniform(0, 5, (12, 500)).astype(np.float32)
    result = compute_mean_spike_rate(spikes, fps=30.0)
    assert result.shape == (12,)


def test_mean_spike_rate_zero_signal() -> None:
    """Zero spike rates → zero mean."""
    spikes = np.zeros((4, 200), dtype=np.float32)
    result = compute_mean_spike_rate(spikes, fps=30.0)
    np.testing.assert_allclose(result, 0.0, atol=1e-8)


def test_mean_spike_rate_all_bad_frames() -> None:
    """All frames bad → output shape is (n_rois,)."""
    spikes = np.ones((2, 50), dtype=np.float32)
    bad_frames = np.ones(50, dtype=bool)
    result = compute_mean_spike_rate(spikes, fps=30.0, bad_frames=bad_frames)
    assert result.shape == (2,)


# ---------------------------------------------------------------------------
# predict_spike_rates — CASCADE (cloud/conda only)
# ---------------------------------------------------------------------------


def test_predict_spike_rates_raises_importerror_without_cascade() -> None:
    """predict_spike_rates raises ImportError if cascade2p is not installed."""
    try:
        import cascade2p  # noqa: F401

        pytest.skip("cascade2p is installed; skipping ImportError test")
    except ImportError:
        pass

    with pytest.raises(ImportError, match="cascade2p"):
        predict_spike_rates(
            dff=np.zeros((3, 100), dtype=np.float32),
            model_name="Global_EXC_7.5Hz_smoothing200ms",
            fps=30.0,
        )


def test_mean_spike_rate_all_bad_nan() -> None:
    """All frames bad -> output is NaN for all ROIs."""
    spikes = np.ones((3, 50), dtype=np.float32)
    bad_frames = np.ones(50, dtype=bool)
    result = compute_mean_spike_rate(spikes, fps=30.0, bad_frames=bad_frames)
    assert result.shape == (3,)
    assert np.all(np.isnan(result))


def test_mean_spike_rate_dtype() -> None:
    """Output dtype is float32."""
    spikes = np.ones((2, 100), dtype=np.float32)
    result = compute_mean_spike_rate(spikes, fps=30.0)
    assert result.dtype == np.float32


def test_mean_spike_rate_varying_rates() -> None:
    """Different ROIs should have different mean rates."""
    spikes = np.zeros((2, 100), dtype=np.float32)
    spikes[0, :] = 1.0  # 1 spike/s -> 60/min
    spikes[1, :] = 2.0  # 2 spikes/s -> 120/min
    result = compute_mean_spike_rate(spikes, fps=10.0)
    np.testing.assert_allclose(result[0], 60.0, rtol=1e-5)
    np.testing.assert_allclose(result[1], 120.0, rtol=1e-5)


def test_predict_spike_rates_with_mock_cascade() -> None:
    """predict_spike_rates returns correct shape when cascade2p is available (mocked)."""
    n_rois, n_frames = 5, 200
    dff = np.random.default_rng(42).uniform(0, 1, (n_rois, n_frames)).astype(np.float32)
    fake_output = np.random.default_rng(0).uniform(0, 2, (n_rois, n_frames)).astype(np.float32)

    # `from cascade2p import cascade` looks up sys.modules["cascade2p"].cascade
    mock_cascade_module = MagicMock()
    mock_cascade_module.predict.return_value = fake_output
    mock_cascade2p = MagicMock()
    mock_cascade2p.cascade = mock_cascade_module

    with patch.dict(
        "sys.modules",
        {"cascade2p": mock_cascade2p, "cascade2p.cascade": mock_cascade_module},
    ):
        result = predict_spike_rates(
            dff=dff,
            model_name="Global_EXC_10Hz_smoothing200ms",
            fps=9.6,
        )

    assert result.shape == (n_rois, n_frames)
    assert result.dtype == np.float32
    mock_cascade_module.predict.assert_called_once_with("Global_EXC_10Hz_smoothing200ms", dff)
    np.testing.assert_array_equal(result, fake_output)


# ---------------------------------------------------------------------------
# _parse_model_fps — model-rate extraction from model name
# ---------------------------------------------------------------------------


class TestParseModelFps:
    def test_10hz_model(self):
        assert _parse_model_fps("Global_EXC_10Hz_smoothing200ms") == pytest.approx(10.0)

    def test_7p5hz_model(self):
        assert _parse_model_fps("Global_EXC_7.5Hz_smoothing200ms") == pytest.approx(7.5)

    def test_30hz_model(self):
        assert _parse_model_fps("Global_EXC_30Hz_smoothing50ms") == pytest.approx(30.0)

    def test_no_hz_pattern_returns_none(self):
        assert _parse_model_fps("SomeModel_noHz_here") is None

    def test_empty_string_returns_none(self):
        assert _parse_model_fps("") is None

    def test_causal_kernel_model(self):
        assert _parse_model_fps("Global_EXC_10Hz_smoothing200ms_causalkernel") == pytest.approx(
            10.0
        )

    def test_orice_zf_model(self):
        assert _parse_model_fps("OGB_zf_pDp_7.5Hz_smoothing200ms") == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# predict_spike_rates — model-fps mismatch warning
# ---------------------------------------------------------------------------


def _make_mock_cascade(fake_output: np.ndarray):
    """Return a sys.modules patch dict for cascade2p."""
    mock_cascade_module = MagicMock()
    mock_cascade_module.predict.return_value = fake_output
    mock_cascade2p = MagicMock()
    mock_cascade2p.cascade = mock_cascade_module
    return {"cascade2p": mock_cascade2p, "cascade2p.cascade": mock_cascade_module}


class TestPredictSpikeRatesMismatchWarning:
    def _fake_dff(self, n_rois: int = 3, n_frames: int = 100) -> np.ndarray:
        return np.zeros((n_rois, n_frames), dtype=np.float32)

    def test_no_warning_when_fps_matches(self):
        """No UserWarning when session fps matches model rate within threshold."""
        dff = self._fake_dff()
        fake_out = np.zeros_like(dff)

        with patch.dict("sys.modules", _make_mock_cascade(fake_out)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                predict_spike_rates(dff, "Global_EXC_10Hz_smoothing200ms", fps=9.6)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0, "Expected no UserWarning for fps=9.6 vs 10Hz model"

    def test_no_warning_exactly_at_threshold(self):
        """No warning when |model_fps - fps| == threshold exactly."""
        dff = self._fake_dff()
        fake_out = np.zeros_like(dff)
        # threshold is 1.5 Hz; 10 Hz model + 10 - 1.5 = 8.5 Hz session → delta == threshold
        fps_at_threshold = 10.0 - _FPS_MISMATCH_THRESHOLD_HZ

        with patch.dict("sys.modules", _make_mock_cascade(fake_out)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                predict_spike_rates(dff, "Global_EXC_10Hz_smoothing200ms", fps=fps_at_threshold)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            # Exactly at threshold is NOT > threshold → no warning
            assert len(user_warnings) == 0

    def test_warning_fires_when_fps_mismatched(self):
        """UserWarning raised when |model_fps - session_fps| > threshold."""
        dff = self._fake_dff()
        fake_out = np.zeros_like(dff)

        with patch.dict("sys.modules", _make_mock_cascade(fake_out)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # 10 Hz model, 30 Hz session → delta = 20 Hz >> threshold
                predict_spike_rates(dff, "Global_EXC_10Hz_smoothing200ms", fps=30.0)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 1
            assert "10.0 Hz" in str(user_warnings[0].message) or "10Hz" in str(
                user_warnings[0].message
            )
            assert "30.00 Hz" in str(user_warnings[0].message) or "30.0" in str(
                user_warnings[0].message
            )

    def test_warning_message_contains_model_name(self):
        """Warning text includes the model name."""
        dff = self._fake_dff()
        fake_out = np.zeros_like(dff)

        with patch.dict("sys.modules", _make_mock_cascade(fake_out)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                predict_spike_rates(dff, "Global_EXC_7.5Hz_smoothing200ms", fps=30.0)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 1
            assert "Global_EXC_7.5Hz_smoothing200ms" in str(user_warnings[0].message)

    def test_no_warning_for_unparseable_model_name(self):
        """No warning (not even error) when model name contains no Hz pattern."""
        dff = self._fake_dff()
        fake_out = np.zeros_like(dff)

        with patch.dict("sys.modules", _make_mock_cascade(fake_out)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                predict_spike_rates(dff, "my_custom_model_no_rate", fps=30.0)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0
