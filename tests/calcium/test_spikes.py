"""Tests for calcium/spikes.py — CASCADE spike inference helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hm2p.calcium.spikes import compute_mean_spike_rate, predict_spike_rates


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
            model_name="Global_EXC_7.5Hz_smoothing200ms",
            fps=30.0,
        )

    assert result.shape == (n_rois, n_frames)
    assert result.dtype == np.float32
    mock_cascade_module.predict.assert_called_once_with("Global_EXC_7.5Hz_smoothing200ms", dff)
    np.testing.assert_array_equal(result, fake_output)
