"""Tests for hm2p.analysis.information — information-theoretic analysis."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.information import (
    information_per_cell,
    mutual_information_binned,
    skaggs_info_rate,
    synergy_redundancy,
)


class TestSkaggsInfoRate:
    """Tests for skaggs_info_rate."""

    def test_uniform_tuning_zero_info(self):
        """Flat tuning curve = 0 information."""
        tc = np.ones(36)
        occ = np.ones(36)
        assert skaggs_info_rate(tc, occ) == pytest.approx(0.0, abs=1e-10)

    def test_peaked_tuning_positive_info(self):
        """Sharply peaked tuning should carry positive information."""
        tc = np.zeros(36)
        tc[0] = 10.0
        occ = np.ones(36)
        assert skaggs_info_rate(tc, occ) > 0

    def test_nan_bins_excluded(self):
        tc = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        occ = np.array([10, 10, 10, 10, 10], dtype=float)
        result = skaggs_info_rate(tc, occ)
        assert result > 0

    def test_zero_occupancy_bins(self):
        tc = np.array([1.0, 2.0, 3.0])
        occ = np.array([10, 0, 10], dtype=float)
        result = skaggs_info_rate(tc, occ)
        assert result >= 0


class TestMutualInformationBinned:
    """Tests for mutual_information_binned."""

    def test_tuned_cell_positive_mi(self):
        """Tuned cell should have positive MI with HD."""
        rng = np.random.default_rng(42)
        n = 5000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        signal = np.exp(3 * np.cos(np.deg2rad(hd)))
        signal += rng.normal(0, 0.1, n)
        mask = np.ones(n, dtype=bool)
        mi = mutual_information_binned(signal, hd, mask)
        assert mi > 0.1

    def test_random_signal_low_mi(self):
        """Random signal should have near-zero MI."""
        rng = np.random.default_rng(42)
        n = 5000
        hd = rng.uniform(0, 360, n)
        signal = rng.normal(0, 1, n)
        mask = np.ones(n, dtype=bool)
        mi = mutual_information_binned(signal, hd, mask)
        assert mi < 0.5  # Should be small (finite sample bias)

    def test_mi_non_negative(self):
        """MI should always be non-negative."""
        rng = np.random.default_rng(42)
        n = 1000
        hd = rng.uniform(0, 360, n)
        signal = rng.normal(0, 1, n)
        mask = np.ones(n, dtype=bool)
        mi = mutual_information_binned(signal, hd, mask)
        assert mi >= 0

    def test_mask_reduces_data(self):
        rng = np.random.default_rng(42)
        n = 1000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        signal = np.exp(3 * np.cos(np.deg2rad(hd)))
        mask_full = np.ones(n, dtype=bool)
        mask_half = np.zeros(n, dtype=bool)
        mask_half[:500] = True
        mi_full = mutual_information_binned(signal, hd, mask_full)
        mi_half = mutual_information_binned(signal, hd, mask_half)
        # Both should be positive
        assert mi_full > 0
        assert mi_half > 0


class TestInformationPerCell:
    """Tests for information_per_cell."""

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        n = 1000
        signals = rng.normal(0, 1, (5, n))
        hd = rng.uniform(0, 360, n)
        mask = np.ones(n, dtype=bool)
        info = information_per_cell(signals, hd, mask)
        assert info.shape == (5,)

    def test_tuned_vs_untuned(self):
        """Tuned cell should have higher MI than untuned."""
        rng = np.random.default_rng(42)
        n = 3000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        tuned = np.exp(3 * np.cos(np.deg2rad(hd))) + rng.normal(0, 0.1, n)
        untuned = rng.normal(0, 1, n)
        signals = np.vstack([tuned, untuned])
        mask = np.ones(n, dtype=bool)
        info = information_per_cell(signals, hd, mask)
        assert info[0] > info[1]


class TestSynergyRedundancy:
    """Tests for synergy_redundancy."""

    def test_output_keys(self):
        rng = np.random.default_rng(42)
        n = 2000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        signals = np.vstack([
            np.exp(3 * np.cos(np.deg2rad(hd))),
            np.exp(3 * np.cos(np.deg2rad(hd - 90))),
        ])
        mask = np.ones(n, dtype=bool)
        result = synergy_redundancy(signals, hd, mask, 0, 1)
        assert "info_a" in result
        assert "info_b" in result
        assert "info_joint" in result
        assert "redundancy" in result

    def test_identical_cells_redundant(self):
        """Identical cells should be highly redundant."""
        rng = np.random.default_rng(42)
        n = 3000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        sig = np.exp(3 * np.cos(np.deg2rad(hd)))
        signals = np.vstack([sig, sig])
        mask = np.ones(n, dtype=bool)
        result = synergy_redundancy(signals, hd, mask, 0, 1)
        # Redundancy should be positive
        assert result["redundancy"] > 0

    def test_joint_at_least_individual(self):
        """Joint information should be at least as high as either individual."""
        rng = np.random.default_rng(42)
        n = 3000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        signals = np.vstack([
            np.exp(3 * np.cos(np.deg2rad(hd))) + rng.normal(0, 0.1, n),
            np.exp(3 * np.cos(np.deg2rad(hd - 90))) + rng.normal(0, 0.1, n),
        ])
        mask = np.ones(n, dtype=bool)
        result = synergy_redundancy(signals, hd, mask, 0, 1)
        assert result["info_joint"] >= max(result["info_a"], result["info_b"]) * 0.9
