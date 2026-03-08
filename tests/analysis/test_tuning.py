"""Tests for hm2p.analysis.tuning — HD and place tuning curve functions."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    compute_place_rate_map,
    mean_vector_length,
    peak_to_trough_ratio,
    preferred_direction,
    spatial_coherence,
    spatial_information,
    spatial_sparsity,
    tuning_width_fwhm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_peaked_signal(peak_deg: float, n_frames: int = 3600, n_bins: int = 36):
    """Create a synthetic signal that peaks at *peak_deg*.

    Returns signal, hd_deg, mask with uniform angular sampling and a
    cosine-shaped signal peaking at *peak_deg*.
    """
    rng = np.random.default_rng(42)
    hd_deg = np.linspace(0, 360, n_frames, endpoint=False)
    rng.shuffle(hd_deg)
    # cosine signal peaking at peak_deg
    diff = np.deg2rad(hd_deg - peak_deg)
    signal = np.maximum(0, np.cos(diff))  # [0, 1]
    mask = np.ones(n_frames, dtype=bool)
    return signal, hd_deg, mask


# ---------------------------------------------------------------------------
# HD tuning curve
# ---------------------------------------------------------------------------


class TestComputeHdTuningCurve:
    def test_peak_at_180(self):
        signal, hd_deg, mask = _make_peaked_signal(180.0)
        tc, centers = compute_hd_tuning_curve(
            signal, hd_deg, mask, n_bins=36, smoothing_sigma_deg=0.0
        )
        assert tc.shape == (36,)
        assert centers.shape == (36,)
        # Peak should be near 180 deg
        peak_center = centers[np.nanargmax(tc)]
        assert abs(peak_center - 180.0) < 360.0 / 36

    def test_uniform_signal_flat_curve(self):
        n = 7200
        hd_deg = np.linspace(0, 360, n, endpoint=False)
        signal = np.ones(n)
        mask = np.ones(n, dtype=bool)
        tc, _ = compute_hd_tuning_curve(signal, hd_deg, mask, smoothing_sigma_deg=0.0)
        # All bins should be ~1.0
        assert np.allclose(tc[~np.isnan(tc)], 1.0, atol=0.05)

    def test_unoccupied_bins_are_nan(self):
        # Signal only in [0, 90) degrees
        hd_deg = np.linspace(0, 89, 1000)
        signal = np.ones(1000)
        mask = np.ones(1000, dtype=bool)
        tc, _ = compute_hd_tuning_curve(
            signal, hd_deg, mask, n_bins=36, smoothing_sigma_deg=0.0
        )
        # Bins outside [0, 90) should be NaN
        assert np.any(np.isnan(tc))

    def test_mask_excludes_frames(self):
        n = 3600
        hd_deg = np.linspace(0, 360, n, endpoint=False)
        signal = np.ones(n)
        mask = np.zeros(n, dtype=bool)
        mask[:900] = True  # only first quarter
        tc, _ = compute_hd_tuning_curve(
            signal, hd_deg, mask, n_bins=36, smoothing_sigma_deg=0.0
        )
        # Bins in the masked-out region should be NaN
        assert np.sum(np.isnan(tc)) > 0

    def test_smoothing_reduces_noise(self):
        rng = np.random.default_rng(0)
        n = 36000
        hd_deg = np.linspace(0, 360, n, endpoint=False)
        signal = np.maximum(0, np.cos(np.deg2rad(hd_deg - 180)))
        signal += rng.normal(0, 0.3, n)
        mask = np.ones(n, dtype=bool)

        tc_raw, _ = compute_hd_tuning_curve(
            signal, hd_deg, mask, smoothing_sigma_deg=0.0
        )
        tc_smooth, _ = compute_hd_tuning_curve(
            signal, hd_deg, mask, smoothing_sigma_deg=15.0
        )
        # Smoothed curve should have lower variance
        assert np.nanstd(tc_smooth) <= np.nanstd(tc_raw) + 0.01

    def test_unwrapped_hd(self):
        """HD values outside [0,360) should be wrapped internally."""
        n = 3600
        hd_deg = np.linspace(0, 720, n, endpoint=False)  # two full turns
        signal = np.maximum(0, np.cos(np.deg2rad(hd_deg - 180)))
        mask = np.ones(n, dtype=bool)
        tc, centers = compute_hd_tuning_curve(
            signal, hd_deg, mask, n_bins=36, smoothing_sigma_deg=0.0
        )
        peak = centers[np.nanargmax(tc)]
        assert abs(peak - 180.0) < 20.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_hd_tuning_curve(
                np.ones(10), np.ones(11), np.ones(10, dtype=bool)
            )


# ---------------------------------------------------------------------------
# Mean vector length
# ---------------------------------------------------------------------------


class TestMeanVectorLength:
    def test_perfect_tuning(self):
        """A delta-function tuning curve should give MVL close to 1."""
        n_bins = 36
        centers = np.linspace(5, 355, n_bins)
        tc = np.zeros(n_bins)
        tc[18] = 1.0  # single peak at 185 deg
        mvl = mean_vector_length(tc, centers)
        assert mvl == pytest.approx(1.0, abs=1e-6)

    def test_flat_tuning(self):
        """A flat curve should give MVL close to 0."""
        n_bins = 360
        centers = np.linspace(0.5, 359.5, n_bins)
        tc = np.ones(n_bins)
        mvl = mean_vector_length(tc, centers)
        assert mvl < 0.02

    def test_all_nan_returns_zero(self):
        tc = np.full(36, np.nan)
        centers = np.linspace(5, 355, 36)
        assert mean_vector_length(tc, centers) == 0.0

    def test_mvl_range(self):
        """MVL should be in [0, 1]."""
        signal, hd_deg, mask = _make_peaked_signal(90.0)
        tc, centers = compute_hd_tuning_curve(signal, hd_deg, mask)
        mvl = mean_vector_length(tc, centers)
        assert 0.0 <= mvl <= 1.0


# ---------------------------------------------------------------------------
# Preferred direction
# ---------------------------------------------------------------------------


class TestPreferredDirection:
    @pytest.mark.parametrize("peak_deg", [0, 45, 90, 180, 270, 350])
    def test_matches_known_peak(self, peak_deg):
        signal, hd_deg, mask = _make_peaked_signal(float(peak_deg), n_frames=36000)
        tc, centers = compute_hd_tuning_curve(
            signal, hd_deg, mask, n_bins=360, smoothing_sigma_deg=0.0
        )
        pd = preferred_direction(tc, centers)
        # Allow tolerance of a few degrees
        diff = min(abs(pd - peak_deg), 360 - abs(pd - peak_deg))
        assert diff < 5.0, f"Expected ~{peak_deg}, got {pd}"

    def test_range_0_360(self):
        tc = np.array([0.0, 1.0, 0.0, 0.0])
        centers = np.array([45.0, 135.0, 225.0, 315.0])
        pd = preferred_direction(tc, centers)
        assert 0.0 <= pd < 360.0


# ---------------------------------------------------------------------------
# Tuning width (FWHM)
# ---------------------------------------------------------------------------


class TestTuningWidthFwhm:
    def test_narrow_peak(self):
        n = 360
        centers = np.linspace(0.5, 359.5, n)
        tc = np.exp(-0.5 * ((centers - 180) / 10) ** 2)
        fwhm = tuning_width_fwhm(tc, centers)
        # Gaussian FWHM = 2.355 * sigma ≈ 23.5 deg
        assert 20 < fwhm < 30

    def test_all_nan(self):
        assert np.isnan(tuning_width_fwhm(np.full(10, np.nan), np.arange(10) * 36.0))


# ---------------------------------------------------------------------------
# Peak to trough ratio
# ---------------------------------------------------------------------------


class TestPeakToTroughRatio:
    def test_basic(self):
        tc = np.array([1.0, 2.0, 3.0, 4.0])
        assert peak_to_trough_ratio(tc) == pytest.approx(4.0)

    def test_zero_min_returns_nan(self):
        tc = np.array([0.0, 1.0, 2.0])
        assert np.isnan(peak_to_trough_ratio(tc))

    def test_ignores_nan(self):
        tc = np.array([np.nan, 2.0, 4.0])
        assert peak_to_trough_ratio(tc) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Place rate map
# ---------------------------------------------------------------------------


class TestComputePlaceRateMap:
    def test_concentrated_signal(self):
        """Signal at one location should produce a localized rate map."""
        rng = np.random.default_rng(7)
        n = 10000
        x = rng.uniform(0, 50, n)
        y = rng.uniform(0, 50, n)
        mask = np.ones(n, dtype=bool)
        # Signal peaks near (25, 25)
        signal = np.exp(-0.5 * (((x - 25) ** 2 + (y - 25) ** 2) / 5**2))
        rate_map, occ, bx, by = compute_place_rate_map(
            signal, x, y, mask,
            bin_size=5.0, smoothing_sigma=0.0, min_occupancy_s=0.0, fps=30.0,
        )
        # Peak should be near the centre
        peak_iy, peak_ix = np.unravel_index(np.nanargmax(rate_map), rate_map.shape)
        peak_x = 0.5 * (bx[peak_ix] + bx[peak_ix + 1])
        peak_y = 0.5 * (by[peak_iy] + by[peak_iy + 1])
        assert abs(peak_x - 25) < 10
        assert abs(peak_y - 25) < 10

    def test_min_occupancy_threshold(self):
        n = 500
        # All positions clustered
        x = np.full(n, 10.0)
        y = np.full(n, 10.0)
        # Add one outlier
        x = np.append(x, 90.0)
        y = np.append(y, 90.0)
        signal = np.ones(n + 1)
        mask = np.ones(n + 1, dtype=bool)
        rate_map, occ, _, _ = compute_place_rate_map(
            signal, x, y, mask,
            bin_size=5.0, smoothing_sigma=0.0, min_occupancy_s=1.0, fps=10.0,
        )
        # The outlier bin has 1 frame = 0.1s < 1.0s threshold → NaN
        assert np.any(np.isnan(rate_map))

    def test_occupancy_in_seconds(self):
        n = 100
        x = np.zeros(n)
        y = np.zeros(n)
        signal = np.ones(n)
        mask = np.ones(n, dtype=bool)
        _, occ, _, _ = compute_place_rate_map(
            signal, x, y, mask,
            bin_size=5.0, smoothing_sigma=0.0, min_occupancy_s=0.0, fps=10.0,
        )
        # All 100 frames in one bin at 10 fps = 10 seconds
        assert np.nanmax(occ) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Spatial information
# ---------------------------------------------------------------------------


class TestSpatialInformation:
    def test_localized_higher_than_uniform(self):
        """A localized rate map should have higher SI than a uniform one."""
        rng = np.random.default_rng(3)
        n = 50000
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        mask = np.ones(n, dtype=bool)

        # Localized signal
        sig_loc = np.exp(-0.5 * (((x - 50) ** 2 + (y - 50) ** 2) / 10**2))
        rm_loc, occ_loc, _, _ = compute_place_rate_map(
            sig_loc, x, y, mask,
            bin_size=5.0, smoothing_sigma=0.0, min_occupancy_s=0.0, fps=30.0,
        )
        si_loc = spatial_information(rm_loc, occ_loc)

        # Uniform signal
        sig_unif = np.ones(n)
        rm_unif, occ_unif, _, _ = compute_place_rate_map(
            sig_unif, x, y, mask,
            bin_size=5.0, smoothing_sigma=0.0, min_occupancy_s=0.0, fps=30.0,
        )
        si_unif = spatial_information(rm_unif, occ_unif)

        assert si_loc > si_unif

    def test_uniform_rate_map_zero_si(self):
        rate_map = np.ones((5, 5))
        occ = np.ones((5, 5))
        assert spatial_information(rate_map, occ) == pytest.approx(0.0, abs=1e-10)

    def test_all_nan_returns_zero(self):
        rate_map = np.full((3, 3), np.nan)
        occ = np.ones((3, 3))
        assert spatial_information(rate_map, occ) == 0.0


# ---------------------------------------------------------------------------
# Spatial coherence
# ---------------------------------------------------------------------------


class TestSpatialCoherence:
    def test_smooth_map_high_coherence(self):
        """A smooth Gaussian bump should have high coherence."""
        yy, xx = np.mgrid[0:20, 0:20]
        rate_map = np.exp(-0.5 * (((xx - 10) ** 2 + (yy - 10) ** 2) / 3**2))
        coh = spatial_coherence(rate_map)
        assert coh > 0.9

    def test_random_map_low_coherence(self):
        rng = np.random.default_rng(1)
        rate_map = rng.uniform(0, 1, (20, 20))
        coh = spatial_coherence(rate_map)
        assert coh < 0.3

    def test_all_nan_returns_nan(self):
        assert np.isnan(spatial_coherence(np.full((3, 3), np.nan)))


# ---------------------------------------------------------------------------
# Spatial sparsity
# ---------------------------------------------------------------------------


class TestSpatialSparsity:
    def test_uniform_gives_one(self):
        rate_map = np.ones((5, 5)) * 3.0
        occ = np.ones((5, 5))
        assert spatial_sparsity(rate_map, occ) == pytest.approx(1.0)

    def test_sparse_less_than_one(self):
        rate_map = np.zeros((5, 5))
        rate_map[2, 2] = 10.0
        occ = np.ones((5, 5))
        sp = spatial_sparsity(rate_map, occ)
        assert 0.0 < sp < 1.0


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


class TestHypothesisMVL:
    @given(
        tc=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=4, max_value=100),
            elements=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        ),
    )
    @settings(max_examples=100)
    def test_mvl_in_unit_interval(self, tc):
        n = len(tc)
        centers = np.linspace(360.0 / (2 * n), 360.0 - 360.0 / (2 * n), n)
        mvl = mean_vector_length(tc, centers)
        assert 0.0 <= mvl <= 1.0 + 1e-10

    @given(
        tc=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=4, max_value=100),
            elements=st.one_of(
                st.floats(min_value=0.0, max_value=100.0),
                st.just(float("nan")),
            ),
        ),
    )
    @settings(max_examples=50)
    def test_mvl_handles_nan(self, tc):
        n = len(tc)
        centers = np.linspace(360.0 / (2 * n), 360.0 - 360.0 / (2 * n), n)
        mvl = mean_vector_length(tc, centers)
        assert np.isfinite(mvl)
        assert 0.0 <= mvl <= 1.0 + 1e-10


class TestHypothesisSI:
    @given(
        rate=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=2, max_value=10),
                st.integers(min_value=2, max_value=10),
            ),
            elements=st.floats(min_value=0.0, max_value=50.0, allow_nan=False),
        ),
    )
    @settings(max_examples=100)
    def test_si_non_negative(self, rate):
        occ = np.ones_like(rate)
        si = spatial_information(rate, occ)
        assert si >= -1e-9, f"SI was negative: {si}"
