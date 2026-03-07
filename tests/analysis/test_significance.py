"""Tests for hm2p.analysis.significance."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.significance import (
    circular_shift_significance,
    hd_tuning_significance,
    place_tuning_significance,
)


class TestCircularShiftSignificance:
    """Tests for the generic circular_shift_significance function."""

    def test_constant_signal_not_significant(self) -> None:
        """A constant signal should produce p ~ 1.0 (not significant)."""
        rng = np.random.default_rng(42)
        signal = np.ones(200, dtype=np.float64)

        # Tuning function: just return the mean (always 1.0 for constant)
        def tuning_fn(s: np.ndarray) -> float:
            return float(np.mean(s))

        result = circular_shift_significance(
            signal, tuning_fn, observed_metric=1.0, n_shuffles=50, rng=rng
        )
        # All shuffles produce the same metric as observed
        assert result["p_value"] >= 0.9

    def test_tuned_signal_significant(self) -> None:
        """A clearly structured signal should yield low p-value."""
        rng = np.random.default_rng(123)
        n = 500
        # Signal with strong structure: high in first quarter, low elsewhere
        signal = np.zeros(n, dtype=np.float64)
        signal[:125] = 10.0

        # "Tuning function": peak value in first quarter minus rest
        def tuning_fn(s: np.ndarray) -> float:
            return float(np.mean(s[:125]) - np.mean(s[125:]))

        observed = tuning_fn(signal)

        result = circular_shift_significance(
            signal, tuning_fn, observed_metric=observed, n_shuffles=50, rng=rng
        )
        # The observed metric is the maximum possible contrast; shuffles reduce it
        assert result["p_value"] < 0.1

    def test_p_value_in_valid_range(self) -> None:
        """p-value must always be in [0, 1]."""
        rng = np.random.default_rng(7)
        signal = rng.standard_normal(100).astype(np.float64)

        result = circular_shift_significance(
            signal,
            lambda s: float(np.max(s)),
            observed_metric=float(np.max(signal)),
            n_shuffles=50,
            rng=rng,
        )
        assert 0.0 <= result["p_value"] <= 1.0

    def test_shuffle_distribution_length(self) -> None:
        """shuffle_distribution should have length n_shuffles."""
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(80).astype(np.float64)

        result = circular_shift_significance(
            signal,
            lambda s: float(np.std(s)),
            observed_metric=float(np.std(signal)),
            n_shuffles=30,
            rng=rng,
        )
        assert len(result["shuffle_distribution"]) == 30
        assert result["n_shuffles"] == 30

    def test_result_keys(self) -> None:
        """Result dict must contain all expected keys."""
        rng = np.random.default_rng(1)
        signal = np.ones(50, dtype=np.float64)

        result = circular_shift_significance(
            signal,
            lambda s: float(np.mean(s)),
            observed_metric=1.0,
            n_shuffles=10,
            rng=rng,
        )
        assert set(result.keys()) == {
            "observed",
            "p_value",
            "shuffle_distribution",
            "n_shuffles",
        }

    def test_reproducibility_with_same_rng_seed(self) -> None:
        """Same rng seed should produce identical results."""
        signal = np.random.default_rng(99).standard_normal(100).astype(np.float64)

        r1 = circular_shift_significance(
            signal,
            lambda s: float(np.mean(s)),
            observed_metric=float(np.mean(signal)),
            n_shuffles=20,
            rng=np.random.default_rng(42),
        )
        r2 = circular_shift_significance(
            signal,
            lambda s: float(np.mean(s)),
            observed_metric=float(np.mean(signal)),
            n_shuffles=20,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(
            r1["shuffle_distribution"], r2["shuffle_distribution"]
        )


class TestHdTuningSignificance:
    """Tests for hd_tuning_significance."""

    def test_returns_tuning_curve_and_bin_centers(self) -> None:
        """Result should include tuning_curve and bin_centers."""
        rng = np.random.default_rng(10)
        n = 300
        signal = rng.standard_normal(n).astype(np.float64)
        hd_deg = rng.uniform(0, 360, n).astype(np.float64)
        mask = np.ones(n, dtype=bool)

        result = hd_tuning_significance(
            signal, hd_deg, mask, n_shuffles=10, n_bins=12, rng=rng
        )
        assert "tuning_curve" in result
        assert "bin_centers" in result
        assert len(result["tuning_curve"]) == 12
        assert len(result["bin_centers"]) == 12

    def test_untuned_signal_high_p(self) -> None:
        """Random signal with random HD should not be significant."""
        rng = np.random.default_rng(55)
        n = 500
        signal = rng.standard_normal(n).astype(np.float64)
        hd_deg = rng.uniform(0, 360, n).astype(np.float64)
        mask = np.ones(n, dtype=bool)

        result = hd_tuning_significance(
            signal, hd_deg, mask, n_shuffles=50, rng=rng
        )
        # Should not be significant (p > 0.05 typically)
        assert result["p_value"] > 0.01

    def test_invalid_metric_raises(self) -> None:
        """Invalid metric name should raise ValueError."""
        with pytest.raises(ValueError, match="metric must be"):
            hd_tuning_significance(
                np.ones(10, dtype=np.float64),
                np.linspace(0, 360, 10, dtype=np.float64),
                np.ones(10, dtype=bool),
                metric="invalid",
            )

    def test_peak_to_trough_metric(self) -> None:
        """Should work with peak_to_trough metric."""
        rng = np.random.default_rng(33)
        n = 200
        signal = rng.standard_normal(n).astype(np.float64) + 5.0  # positive
        hd_deg = rng.uniform(0, 360, n).astype(np.float64)
        mask = np.ones(n, dtype=bool)

        result = hd_tuning_significance(
            signal, hd_deg, mask, n_shuffles=10, metric="peak_to_trough", rng=rng
        )
        assert "observed" in result
        assert 0.0 <= result["p_value"] <= 1.0


class TestPlaceTuningSignificance:
    """Tests for place_tuning_significance."""

    def test_returns_rate_and_occupancy_maps(self) -> None:
        """Result should include rate_map and occupancy_map."""
        rng = np.random.default_rng(77)
        n = 500
        signal = rng.standard_normal(n).astype(np.float64)
        x = rng.uniform(0, 50, n).astype(np.float64)
        y = rng.uniform(0, 50, n).astype(np.float64)
        mask = np.ones(n, dtype=bool)

        result = place_tuning_significance(
            signal,
            x,
            y,
            mask,
            n_shuffles=10,
            bin_size=10.0,
            min_occupancy_s=0.01,
            fps=10.0,
            rng=rng,
        )
        assert "rate_map" in result
        assert "occupancy_map" in result
        assert result["rate_map"].ndim == 2
        assert result["occupancy_map"].ndim == 2

    def test_p_value_in_range(self) -> None:
        """p-value should be in [0, 1]."""
        rng = np.random.default_rng(88)
        n = 300
        signal = rng.standard_normal(n).astype(np.float64)
        x = rng.uniform(0, 30, n).astype(np.float64)
        y = rng.uniform(0, 30, n).astype(np.float64)
        mask = np.ones(n, dtype=bool)

        result = place_tuning_significance(
            signal,
            x,
            y,
            mask,
            n_shuffles=10,
            bin_size=10.0,
            min_occupancy_s=0.01,
            fps=10.0,
            rng=rng,
        )
        assert 0.0 <= result["p_value"] <= 1.0
