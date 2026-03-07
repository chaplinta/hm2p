"""Tests for hm2p.analysis.comparison."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.comparison import (
    mvl_ratio,
    preferred_direction_shift,
    rate_map_correlation,
    si_ratio,
    tuning_curve_correlation,
)


class TestTuningCurveCorrelation:
    """Tests for tuning_curve_correlation."""

    def test_identical_curves_return_one(self) -> None:
        """Correlation of a curve with itself should be 1.0."""
        curve = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        assert tuning_curve_correlation(curve, curve) == pytest.approx(1.0)

    def test_inverted_curves_return_negative_one(self) -> None:
        """Correlation of a curve with its negation should be -1.0."""
        curve = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        inverted = -curve
        assert tuning_curve_correlation(curve, inverted) == pytest.approx(-1.0)

    def test_nan_handling(self) -> None:
        """NaN bins should be excluded from correlation."""
        a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype=np.float64)
        b = np.array([1.0, 2.0, 99.0, 4.0, 5.0, 6.0], dtype=np.float64)
        # NaN in a masks bin 2; remaining bins are identical
        assert tuning_curve_correlation(a, b) == pytest.approx(1.0)

    def test_fewer_than_3_valid_returns_nan(self) -> None:
        """Should return NaN if fewer than 3 non-NaN bins overlap."""
        a = np.array([1.0, np.nan, np.nan, np.nan], dtype=np.float64)
        b = np.array([1.0, 2.0, np.nan, np.nan], dtype=np.float64)
        assert np.isnan(tuning_curve_correlation(a, b))

    def test_constant_curve_returns_nan(self) -> None:
        """Constant curve has zero std, correlation is undefined."""
        a = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        assert np.isnan(tuning_curve_correlation(a, b))


class TestPreferredDirectionShift:
    """Tests for preferred_direction_shift."""

    def test_identical_curves_zero_shift(self) -> None:
        """Identical curves should have zero shift."""
        bin_centers = np.linspace(5, 355, 36, dtype=np.float64)
        # Create a cosine-shaped tuning curve peaked at 90 deg
        curve = np.cos(np.deg2rad(bin_centers - 90.0)) + 1.0
        shift = preferred_direction_shift(curve, curve, bin_centers)
        assert shift == pytest.approx(0.0, abs=0.5)

    def test_known_shift(self) -> None:
        """Shifting a curve by 90 degrees should give ~90 degree shift."""
        bin_centers = np.linspace(5, 355, 36, dtype=np.float64)
        curve_a = np.cos(np.deg2rad(bin_centers - 0.0)) + 1.0
        curve_b = np.cos(np.deg2rad(bin_centers - 90.0)) + 1.0
        shift = preferred_direction_shift(curve_a, curve_b, bin_centers)
        assert shift == pytest.approx(90.0, abs=2.0)

    def test_shift_wrapping(self) -> None:
        """Result should be in [-180, 180]."""
        bin_centers = np.linspace(5, 355, 36, dtype=np.float64)
        curve_a = np.cos(np.deg2rad(bin_centers - 10.0)) + 1.0
        curve_b = np.cos(np.deg2rad(bin_centers - 350.0)) + 1.0
        shift = preferred_direction_shift(curve_a, curve_b, bin_centers)
        assert -180.0 <= shift <= 180.0
        # 350 - 10 = 340, wrapped = -20
        assert shift == pytest.approx(-20.0, abs=2.0)


class TestRateMapCorrelation:
    """Tests for rate_map_correlation."""

    def test_identical_maps_return_one(self) -> None:
        """Correlation of a map with itself should be 1.0."""
        rate_map = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        assert rate_map_correlation(rate_map, rate_map) == pytest.approx(1.0)

    def test_nan_handling(self) -> None:
        """NaN bins in either map should be excluded."""
        a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        b = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]], dtype=np.float64)
        # Valid bins: (0,0)=1/1, (0,2)=3/3, (1,0)=4/4, (1,2)=6/6 -> identical
        r = rate_map_correlation(a, b)
        assert r == pytest.approx(1.0)

    def test_all_nan_returns_nan(self) -> None:
        """All-NaN maps should return NaN."""
        a = np.full((3, 3), np.nan)
        b = np.full((3, 3), np.nan)
        assert np.isnan(rate_map_correlation(a, b))

    def test_fewer_than_3_valid_returns_nan(self) -> None:
        """Should return NaN if fewer than 3 valid bins."""
        a = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float64)
        b = np.array([[2.0, np.nan], [np.nan, np.nan]], dtype=np.float64)
        assert np.isnan(rate_map_correlation(a, b))


class TestMvlRatio:
    """Tests for mvl_ratio."""

    def test_identical_curves_ratio_one(self) -> None:
        """MVL ratio of identical curves should be 1.0."""
        bin_centers = np.linspace(5, 355, 36, dtype=np.float64)
        curve = np.cos(np.deg2rad(bin_centers)) + 1.0
        ratio = mvl_ratio(curve, curve, bin_centers)
        assert ratio == pytest.approx(1.0)

    def test_zero_mvl_a_returns_nan(self) -> None:
        """If MVL of curve_a is zero, should return NaN."""
        bin_centers = np.linspace(5, 355, 36, dtype=np.float64)
        flat = np.zeros(36, dtype=np.float64)
        tuned = np.cos(np.deg2rad(bin_centers)) + 1.0
        assert np.isnan(mvl_ratio(flat, tuned, bin_centers))

    def test_broader_curve_lower_mvl(self) -> None:
        """A broader curve should have lower MVL, giving ratio < 1."""
        bin_centers = np.linspace(5, 355, 36, dtype=np.float64)
        sharp = np.cos(np.deg2rad(bin_centers)) + 1.0
        broad = 0.1 * np.cos(np.deg2rad(bin_centers)) + 1.0
        ratio = mvl_ratio(sharp, broad, bin_centers)
        assert ratio < 1.0


class TestSiRatio:
    """Tests for si_ratio."""

    def test_identical_maps_ratio_one(self) -> None:
        """SI ratio of identical maps should be 1.0."""
        rng = np.random.default_rng(42)
        rate_map = rng.uniform(0.5, 5.0, (5, 5)).astype(np.float64)
        occ = np.ones((5, 5), dtype=np.float64)
        ratio = si_ratio(rate_map, occ, rate_map, occ)
        assert ratio == pytest.approx(1.0)

    def test_zero_si_a_returns_nan(self) -> None:
        """If SI of map_a is zero (uniform map), should return NaN."""
        uniform = np.ones((4, 4), dtype=np.float64) * 2.0
        occ = np.ones((4, 4), dtype=np.float64)
        tuned = np.array(
            [[5.0, 1.0, 1.0, 1.0]] * 4, dtype=np.float64
        )
        # Uniform map has SI = 0
        ratio = si_ratio(uniform, occ, tuned, occ)
        assert np.isnan(ratio)
