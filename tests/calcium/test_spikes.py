"""Tests for calcium/spikes.py — CASCADE spike inference helpers."""

from __future__ import annotations

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
