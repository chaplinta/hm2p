"""Tests for calcium/spikes.py — CASCADE spike inference helpers."""

from __future__ import annotations

import numpy as np

from hm2p.calcium.spikes import compute_mean_spike_rate


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
