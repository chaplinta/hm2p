"""Tests for HD tuning explorer page logic."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
    peak_to_trough_ratio,
    preferred_direction,
    tuning_width_fwhm,
)


def _generate_synthetic_hd_cell(
    n_frames=5000, preferred_deg=180.0, concentration=2.0,
    noise_level=0.3, baseline=0.1, seed=42,
):
    """Reproduce the synthetic generator from hd_tuning_page.py."""
    rng = np.random.default_rng(seed)
    hd_steps = rng.normal(0, 5, n_frames)
    hd_deg = np.cumsum(hd_steps) % 360.0
    theta = np.deg2rad(hd_deg)
    pref_rad = np.deg2rad(preferred_deg)
    signal = baseline + np.exp(concentration * np.cos(theta - pref_rad))
    signal /= np.max(signal)
    signal += rng.normal(0, noise_level, n_frames)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n_frames, dtype=bool)
    return signal, hd_deg, mask


class TestSyntheticHDCell:
    """Test synthetic HD cell generation for page demos."""

    def test_output_shapes(self):
        signal, hd, mask = _generate_synthetic_hd_cell()
        assert signal.shape == (5000,)
        assert hd.shape == (5000,)
        assert mask.shape == (5000,)

    def test_signal_non_negative(self):
        signal, _, _ = _generate_synthetic_hd_cell()
        assert np.all(signal >= 0)

    def test_hd_in_range(self):
        _, hd, _ = _generate_synthetic_hd_cell()
        assert np.all(hd >= 0)
        assert np.all(hd < 360)

    def test_preferred_direction_recovered(self):
        """Tuning curve should peak near the input preferred direction."""
        for pref in [0, 90, 180, 270]:
            signal, hd, mask = _generate_synthetic_hd_cell(
                preferred_deg=pref, concentration=5.0, noise_level=0.1,
            )
            tc, bc = compute_hd_tuning_curve(signal, hd, mask, n_bins=36)
            pd_deg = preferred_direction(tc, bc)
            # Should be within 30° of true preferred direction
            diff = min(abs(pd_deg - pref), 360 - abs(pd_deg - pref))
            assert diff < 30, f"pref={pref}, recovered={pd_deg:.1f}"

    def test_mvl_increases_with_concentration(self):
        """Higher κ should give higher MVL."""
        mvls = []
        for kappa in [0.5, 2.0, 5.0]:
            signal, hd, mask = _generate_synthetic_hd_cell(
                concentration=kappa, noise_level=0.1, seed=42,
            )
            tc, bc = compute_hd_tuning_curve(signal, hd, mask)
            mvls.append(mean_vector_length(tc, bc))
        assert mvls[2] > mvls[0]

    def test_fwhm_decreases_with_concentration(self):
        """Higher κ should give narrower tuning (smaller FWHM)."""
        fwhms = []
        for kappa in [0.5, 2.0, 5.0]:
            signal, hd, mask = _generate_synthetic_hd_cell(
                concentration=kappa, noise_level=0.1, seed=42,
            )
            tc, bc = compute_hd_tuning_curve(signal, hd, mask)
            fwhms.append(tuning_width_fwhm(tc, bc))
        assert fwhms[2] < fwhms[0]

    def test_noise_reduces_mvl(self):
        """Higher noise should reduce MVL."""
        mvl_low = mean_vector_length(*compute_hd_tuning_curve(
            *_generate_synthetic_hd_cell(noise_level=0.05, seed=42),
        ))
        mvl_high = mean_vector_length(*compute_hd_tuning_curve(
            *_generate_synthetic_hd_cell(noise_level=0.8, seed=42),
        ))
        assert mvl_low > mvl_high


class TestPopulationDisplay:
    """Test population-level computations for the page."""

    def test_population_uniform_coverage(self):
        """Population with uniform preferred directions should cover 360°."""
        n_cells = 20
        prefs = np.linspace(0, 360, n_cells, endpoint=False)
        recovered = []
        for i, pref in enumerate(prefs):
            signal, hd, mask = _generate_synthetic_hd_cell(
                preferred_deg=pref, concentration=3.0, noise_level=0.1, seed=i,
            )
            tc, bc = compute_hd_tuning_curve(signal, hd, mask)
            recovered.append(preferred_direction(tc, bc))
        # Should span ~360° range
        recovered = np.array(recovered)
        assert np.ptp(recovered) > 200  # At least 200° spread

    def test_tuning_curve_heatmap_shape(self):
        """Normalised tuning curve matrix should have correct shape."""
        n_cells, n_bins = 10, 36
        tc_matrix = np.zeros((n_cells, n_bins))
        for i in range(n_cells):
            signal, hd, mask = _generate_synthetic_hd_cell(
                preferred_deg=i * 36, seed=i,
            )
            tc, _ = compute_hd_tuning_curve(signal, hd, mask, n_bins=n_bins)
            tc_matrix[i] = np.where(np.isnan(tc), 0, tc)
        # Normalise
        row_max = tc_matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        tc_norm = tc_matrix / row_max
        assert tc_norm.shape == (n_cells, n_bins)
        assert np.all(tc_norm >= 0)
        assert np.all(tc_norm <= 1.0 + 1e-10)


class TestSignificancePage:
    """Test significance testing display logic."""

    def test_significant_cell_detected(self):
        """Strongly tuned cell should be significant."""
        from hm2p.analysis.significance import hd_tuning_significance

        signal, hd, mask = _generate_synthetic_hd_cell(
            concentration=5.0, noise_level=0.1, seed=42,
        )
        result = hd_tuning_significance(
            signal, hd, mask, n_shuffles=100,
            rng=np.random.default_rng(42),
        )
        assert result["p_value"] < 0.05

    def test_untuned_cell_not_significant(self):
        """Untuned cell (κ≈0) should not be significant."""
        from hm2p.analysis.significance import hd_tuning_significance

        signal, hd, mask = _generate_synthetic_hd_cell(
            concentration=0.01, noise_level=0.5, seed=42,
        )
        result = hd_tuning_significance(
            signal, hd, mask, n_shuffles=100,
            rng=np.random.default_rng(42),
        )
        assert result["p_value"] > 0.05

    def test_shuffle_distribution_shape(self):
        """Shuffle distribution should have n_shuffles entries."""
        from hm2p.analysis.significance import hd_tuning_significance

        signal, hd, mask = _generate_synthetic_hd_cell(seed=42)
        result = hd_tuning_significance(
            signal, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(42),
        )
        assert len(result["shuffle_distribution"]) == 50
        assert result["tuning_curve"].shape == (36,)
        assert result["bin_centers"].shape == (36,)


class TestParameterSweep:
    """Test parameter sweep logic used in the page."""

    def test_kappa_sweep(self):
        """Sweeping κ should produce monotonically increasing MVL."""
        kappas = [0.5, 1.0, 2.0, 4.0, 8.0]
        mvls = []
        for k in kappas:
            s, h, m = _generate_synthetic_hd_cell(concentration=k, noise_level=0.1, seed=42)
            tc, bc = compute_hd_tuning_curve(s, h, m)
            mvls.append(mean_vector_length(tc, bc))
        # Should be generally increasing (allow small noise)
        assert mvls[-1] > mvls[0]
        # Each step should not decrease by more than 10%
        for i in range(1, len(mvls)):
            assert mvls[i] >= mvls[i - 1] * 0.9

    def test_nbins_sweep_mvl_stable(self):
        """MVL should be relatively stable across bin counts."""
        s, h, m = _generate_synthetic_hd_cell(concentration=3.0, seed=42)
        mvls = []
        for nb in [12, 24, 36, 72]:
            tc, bc = compute_hd_tuning_curve(s, h, m, n_bins=nb)
            mvls.append(mean_vector_length(tc, bc))
        # All MVLs should be within 20% of each other
        assert max(mvls) - min(mvls) < 0.2
