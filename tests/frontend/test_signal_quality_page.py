"""Tests for signal quality page logic — F0 baseline estimation."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.calcium.dff import compute_baseline, compute_dff


class TestF0BaselineProxy:
    """Test the F0 baseline proxy computation used in the signal quality page."""

    def test_f0_proxy_shape(self):
        """F0 proxy should have same shape as input dff."""
        rng = np.random.default_rng(42)
        n_rois, n_frames = 5, 1000
        dff = rng.normal(0, 0.5, (n_rois, n_frames)).astype(np.float32)
        f_proxy = 1.0 + dff
        f0_proxy = compute_baseline(f_proxy.astype(np.float32), fps=10.0)
        assert f0_proxy.shape == (n_rois, n_frames)

    def test_f0_proxy_positive(self):
        """F0 proxy should be positive (since 1 + dff is mostly positive)."""
        rng = np.random.default_rng(42)
        n_rois, n_frames = 3, 2000
        # Realistic dff: mostly near 0 with occasional transients
        dff = np.clip(rng.normal(0, 0.2, (n_rois, n_frames)), -0.5, 5.0).astype(np.float32)
        f_proxy = 1.0 + dff
        f0_proxy = compute_baseline(f_proxy, fps=10.0)
        # Baseline should be positive
        assert np.all(f0_proxy > 0)

    def test_f0_normalised_mean_near_one(self):
        """Normalised F0 (F0/mean(F0)) should have mean ~1."""
        rng = np.random.default_rng(42)
        n_rois, n_frames = 5, 5000
        dff = np.clip(rng.normal(0, 0.1, (n_rois, n_frames)), -0.3, 3.0).astype(np.float32)
        f_proxy = 1.0 + dff
        f0_proxy = compute_baseline(f_proxy, fps=10.0)

        for i in range(n_rois):
            mean_f0 = np.nanmean(f0_proxy[i])
            normed = f0_proxy[i] / mean_f0
            assert np.nanmean(normed) == pytest.approx(1.0, abs=0.01)

    def test_f0_proxy_smooth(self):
        """F0 proxy should be smoother than the input (lower variance)."""
        rng = np.random.default_rng(42)
        n_frames = 5000
        dff = np.clip(rng.normal(0, 0.3, (1, n_frames)), -0.5, 5.0).astype(np.float32)
        f_proxy = 1.0 + dff
        f0_proxy = compute_baseline(f_proxy, fps=10.0)
        # F0 should be much smoother
        assert np.std(f0_proxy[0]) < np.std(f_proxy[0])

    def test_f0_drift_metric(self):
        """Drift metric should detect a declining baseline."""
        n_frames = 5000
        fps = 10.0
        # Simulate photobleaching: slow exponential decay
        t = np.arange(n_frames) / fps
        decay = np.exp(-t / 300.0)  # 5-minute time constant
        dff = (decay - 1.0).reshape(1, -1).astype(np.float32)
        f_proxy = 1.0 + dff
        f0_proxy = compute_baseline(f_proxy, fps=fps)

        mean_normed = f0_proxy[0] / np.nanmean(f0_proxy[0])
        # Last 30s vs first 30s
        n30 = int(fps * 30)
        drift = np.nanmean(mean_normed[-n30:]) - np.nanmean(mean_normed[:n30])
        # Should be negative (decaying)
        assert drift < 0


class TestComputeBaseline:
    """Test the underlying compute_baseline function."""

    def test_constant_input(self):
        """Constant input should give constant baseline."""
        F = np.ones((2, 1000), dtype=np.float32) * 100.0
        F0 = compute_baseline(F, fps=10.0)
        # Should be close to 100 everywhere
        np.testing.assert_allclose(F0, 100.0, atol=1.0)

    def test_baseline_below_signal(self):
        """Baseline should generally be at or below the signal."""
        rng = np.random.default_rng(42)
        n_frames = 5000
        # Signal with positive transients on top of baseline
        F = np.ones((1, n_frames), dtype=np.float32) * 100.0
        # Add positive transients
        for _ in range(10):
            start = rng.integers(0, n_frames - 100)
            F[0, start:start + 50] += rng.uniform(50, 200)

        F0 = compute_baseline(F, fps=10.0)
        # Baseline should be <= signal for most frames
        frac_below = np.mean(F0[0] <= F[0] + 10.0)
        assert frac_below > 0.75

    def test_compute_dff_roundtrip(self):
        """dff = (F - F0) / F0 should be near 0 for constant signal."""
        F = np.ones((1, 2000), dtype=np.float32) * 100.0
        F0 = compute_baseline(F, fps=10.0)
        dff = compute_dff(F, F0)
        # Should be close to 0
        np.testing.assert_allclose(dff, 0.0, atol=0.01)
