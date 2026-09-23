"""Tests for hm2p.analysis.coupling — population coupling and noise correlations."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis.coupling import (
    _hd_bin_index,
    _safe_pearson,
    _zscore_rows,
    correlation_vs_distance,
    coupling_summary,
    lagged_population_crosscorr,
    noise_correlations,
    pairwise_distances_um,
    population_coupling,
    population_coupling_by_condition,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _independent_population(n_rois=12, n_frames=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n_rois, n_frames))


def _hd_population(n_rois=8, n_frames=400, kappa=3.0, seed=1, shared_noise=0.0):
    """von Mises HD-tuned population with optional shared (noise-correlated) drive."""
    rng = np.random.default_rng(seed)
    hd = np.mod(np.linspace(0.0, 4 * 360.0, n_frames), 360.0)
    theta = np.deg2rad(hd)
    prefs = np.deg2rad(np.linspace(0.0, 360.0, n_rois, endpoint=False))
    shared = rng.normal(0.0, 1.0, n_frames)
    signals = np.empty((n_rois, n_frames))
    for i in range(n_rois):
        tuned = np.exp(kappa * np.cos(theta - prefs[i]))
        tuned /= tuned.max()
        signals[i] = tuned + shared_noise * shared + 0.05 * rng.normal(0.0, 1.0, n_frames)
    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, mask


# ---------------------------------------------------------------------------
# _safe_pearson
# ---------------------------------------------------------------------------


def test_safe_pearson_perfect_positive():
    a = np.arange(10.0)
    assert _safe_pearson(a, 2.0 * a + 1.0) == pytest.approx(1.0)


def test_safe_pearson_perfect_negative():
    a = np.arange(10.0)
    assert _safe_pearson(a, -a) == pytest.approx(-1.0)


def test_safe_pearson_constant_input_is_nan_without_warning():
    a = np.arange(10.0)
    with np.errstate(all="raise"):
        result = _safe_pearson(a, np.ones(10))
    assert np.isnan(result)


def test_safe_pearson_length_mismatch_and_too_short():
    assert np.isnan(_safe_pearson(np.arange(5.0), np.arange(4.0)))
    assert np.isnan(_safe_pearson(np.array([1.0]), np.array([2.0])))


def test_safe_pearson_drops_non_finite_pairs():
    a = np.array([0.0, 1.0, 2.0, np.nan])
    b = np.array([0.0, 1.0, 2.0, 100.0])
    assert _safe_pearson(a, b) == pytest.approx(1.0)
    assert np.isnan(_safe_pearson(np.full(4, np.nan), b))


# ---------------------------------------------------------------------------
# _zscore_rows
# ---------------------------------------------------------------------------


def test_zscore_rows_unit_variance():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, 10.0, 0.0]])
    z = _zscore_rows(x)
    assert np.allclose(z.mean(axis=1), 0.0)
    assert np.allclose(z.std(axis=1), 1.0)


def test_zscore_rows_constant_row_is_zero():
    x = np.vstack([np.ones(5), np.arange(5.0)])
    z = _zscore_rows(x)
    assert np.all(z[0] == 0.0)
    assert np.isfinite(z).all()


# ---------------------------------------------------------------------------
# _hd_bin_index
# ---------------------------------------------------------------------------


def test_hd_bin_index_values():
    hd = np.array([0.0, 91.0, 181.0, 271.0, 360.0, -1.0])
    idx = _hd_bin_index(hd, n_bins=4)
    assert idx.tolist() == [0, 1, 2, 3, 0, 3]


def test_hd_bin_index_range_and_error():
    rng = np.random.default_rng(3)
    idx = _hd_bin_index(rng.uniform(-720, 720, 500), n_bins=36)
    assert idx.min() >= 0
    assert idx.max() <= 35
    with pytest.raises(ValueError, match="n_bins"):
        _hd_bin_index(np.zeros(3), n_bins=0)


# ---------------------------------------------------------------------------
# population_coupling
# ---------------------------------------------------------------------------


def test_population_coupling_chorister_is_near_one():
    base = _independent_population(n_rois=10, n_frames=500, seed=2)
    mean_of_others = base.mean(axis=0)
    signals = np.vstack([mean_of_others, base])
    coupling = population_coupling(signals)
    assert coupling[0] > 0.99


def test_population_coupling_independent_is_near_zero():
    signals = _independent_population(n_rois=20, n_frames=1000, seed=4)
    coupling = population_coupling(signals)
    assert np.all(np.abs(coupling) < 0.2)
    assert abs(float(np.mean(coupling))) < 0.1


def test_population_coupling_exclude_self_false_inflates_coupling():
    signals = _independent_population(n_rois=4, n_frames=300, seed=5)
    with_self = population_coupling(signals, exclude_self=False)
    without_self = population_coupling(signals, exclude_self=True)
    assert np.all(with_self > without_self)


def test_population_coupling_constant_trace_is_nan():
    signals = _independent_population(n_rois=5, n_frames=200, seed=6)
    signals[2] = 3.0
    coupling = population_coupling(signals)
    assert np.isnan(coupling[2])
    assert np.isfinite(coupling[[0, 1, 3, 4]]).all()


def test_population_coupling_too_few_rois_or_frames():
    assert np.isnan(population_coupling(np.zeros((1, 100)))).all()
    assert np.isnan(population_coupling(np.zeros((5, 1)))).all()


def test_population_coupling_requires_2d():
    with pytest.raises(ValueError, match="2-D"):
        population_coupling(np.zeros(10))


# ---------------------------------------------------------------------------
# population_coupling_by_condition
# ---------------------------------------------------------------------------


def test_population_coupling_by_condition_matches_direct_call():
    signals = _independent_population(n_rois=6, n_frames=300, seed=7)
    mask = np.zeros(300, dtype=bool)
    mask[:150] = True
    out = population_coupling_by_condition(signals, {"light": mask, "dark": ~mask})
    assert set(out) == {"light", "dark"}
    expected = population_coupling(signals[:, mask])
    assert np.allclose(out["light"], expected, equal_nan=True)


def test_population_coupling_by_condition_empty_condition_is_nan():
    signals = _independent_population(n_rois=3, n_frames=50, seed=8)
    out = population_coupling_by_condition(signals, {"none": np.zeros(50, dtype=bool)})
    assert np.isnan(out["none"]).all()


def test_population_coupling_by_condition_validates_shapes():
    signals = _independent_population(n_rois=3, n_frames=50, seed=9)
    with pytest.raises(ValueError, match="condition mask"):
        population_coupling_by_condition(signals, {"bad": np.ones(10, dtype=bool)})
    with pytest.raises(ValueError, match="2-D"):
        population_coupling_by_condition(np.zeros(5), {"a": np.ones(5, dtype=bool)})


# ---------------------------------------------------------------------------
# noise_correlations
# ---------------------------------------------------------------------------


def test_noise_correlations_shape_symmetry_and_nan_diagonal():
    signals, hd, mask = _hd_population(n_rois=6, n_frames=300, seed=10)
    noise = noise_correlations(signals, hd, mask)
    assert noise.shape == (6, 6)
    assert np.isnan(np.diag(noise)).all()
    assert np.allclose(noise, noise.T, equal_nan=True)


def test_noise_correlations_detect_shared_residual_drive():
    shared_sig, hd, mask = _hd_population(n_rois=6, n_frames=400, seed=11, shared_noise=0.5)
    indep_sig, _, _ = _hd_population(n_rois=6, n_frames=400, seed=11, shared_noise=0.0)
    triu = np.triu(np.ones((6, 6), dtype=bool), k=1)
    shared_mean = float(np.nanmean(noise_correlations(shared_sig, hd, mask)[triu]))
    indep_mean = float(np.nanmean(noise_correlations(indep_sig, hd, mask)[triu]))
    assert shared_mean > 0.5
    assert shared_mean > indep_mean


def test_noise_correlations_constant_cell_is_nan():
    signals, hd, mask = _hd_population(n_rois=4, n_frames=300, seed=12, shared_noise=0.4)
    signals[1] = 2.0
    noise = noise_correlations(signals, hd, mask)
    assert np.isnan(noise[1]).all()
    assert np.isnan(noise[:, 1]).all()
    assert np.isfinite(noise[0, 2])


def test_noise_correlations_degenerate_inputs():
    signals, hd, mask = _hd_population(n_rois=1, n_frames=100, seed=13)
    assert np.isnan(noise_correlations(signals, hd, mask)).all()
    signals, hd, mask = _hd_population(n_rois=3, n_frames=100, seed=13)
    assert np.isnan(noise_correlations(signals, hd, np.zeros(100, dtype=bool))).all()


def test_noise_correlations_all_but_one_cell_constant_is_all_nan():
    signals, hd, mask = _hd_population(n_rois=3, n_frames=200, seed=22)
    signals[0] = 1.0
    signals[1] = 2.0
    assert np.isnan(noise_correlations(signals, hd, mask)).all()


def test_noise_correlations_validates_shapes():
    signals, hd, mask = _hd_population(n_rois=3, n_frames=100, seed=14)
    with pytest.raises(ValueError, match="2-D"):
        noise_correlations(signals[0], hd, mask)
    with pytest.raises(ValueError, match="hd_deg and mask"):
        noise_correlations(signals, hd[:50], mask)


# ---------------------------------------------------------------------------
# pairwise_distances_um
# ---------------------------------------------------------------------------


def test_pairwise_distances_um_known_values():
    centroids = np.array([[0.0, 0.0], [3.0, 4.0]])
    dist = pairwise_distances_um(centroids, um_per_pixel=2.0)
    assert dist[0, 1] == pytest.approx(10.0)
    assert dist[0, 0] == 0.0


def test_pairwise_distances_um_validates_inputs():
    with pytest.raises(ValueError, match="centroids_xy"):
        pairwise_distances_um(np.zeros((4, 3)), 1.0)
    with pytest.raises(ValueError, match="um_per_pixel"):
        pairwise_distances_um(np.zeros((4, 2)), 0.0)
    with pytest.raises(ValueError, match="um_per_pixel"):
        pairwise_distances_um(np.zeros((4, 2)), np.nan)


@settings(max_examples=30, deadline=None)
@given(
    coords=st.lists(
        st.tuples(
            st.floats(-500, 500, allow_nan=False, allow_infinity=False),
            st.floats(-500, 500, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=12,
    ),
    scale=st.floats(0.05, 20.0, allow_nan=False, allow_infinity=False),
)
def test_pairwise_distances_um_symmetric_with_zero_diagonal(coords, scale):
    dist = pairwise_distances_um(np.array(coords, dtype=float), scale)
    assert dist.shape == (len(coords), len(coords))
    assert np.array_equal(dist, dist.T)
    assert np.all(np.diag(dist) == 0.0)
    assert np.all(dist >= 0.0)


# ---------------------------------------------------------------------------
# correlation_vs_distance
# ---------------------------------------------------------------------------


def _distance_dependent_corr(n_rois=15, seed=15):
    rng = np.random.default_rng(seed)
    centroids = rng.uniform(0.0, 500.0, size=(n_rois, 2))
    dist = pairwise_distances_um(centroids, um_per_pixel=1.0)
    corr = np.exp(-dist / 200.0) + rng.normal(0.0, 0.01, size=dist.shape)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, np.nan)
    return corr, dist


def test_correlation_vs_distance_recovers_negative_relationship():
    corr, dist = _distance_dependent_corr()
    result = correlation_vs_distance(corr, dist, np.array([0.0, 100.0, 300.0, 1000.0]))
    assert result["spearman_rho"] < -0.8
    assert result["spearman_p"] < 0.05
    assert result["n_pairs"] == 15 * 14 // 2
    assert result["n_pairs_per_bin"].sum() == result["n_pairs"]
    finite = np.isfinite(result["mean_corr"])
    assert np.all(np.diff(result["mean_corr"][finite]) < 0)
    assert result["bin_centers_um"].tolist() == [50.0, 200.0, 650.0]


def test_correlation_vs_distance_empty_bins_are_nan():
    corr, dist = _distance_dependent_corr(n_rois=4, seed=16)
    result = correlation_vs_distance(corr, dist, np.array([0.0, 1.0, 2.0, 5000.0]))
    assert np.isnan(result["mean_corr"][0])
    assert np.isnan(result["median_corr"][0])
    assert result["n_pairs_per_bin"][0] == 0


def test_correlation_vs_distance_degenerate_pairs_give_nan_rho():
    corr = np.full((2, 2), np.nan)
    dist = np.zeros((2, 2))
    result = correlation_vs_distance(corr, dist, np.array([0.0, 10.0]))
    assert result["n_pairs"] == 0
    assert np.isnan(result["spearman_rho"])
    assert np.isnan(result["spearman_p"])


def test_correlation_vs_distance_validates_inputs():
    with pytest.raises(ValueError, match="square"):
        correlation_vs_distance(np.zeros((2, 3)), np.zeros((2, 3)), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="dist_matrix shape"):
        correlation_vs_distance(np.zeros((2, 2)), np.zeros((3, 3)), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="at least two edges"):
        correlation_vs_distance(np.zeros((2, 2)), np.zeros((2, 2)), np.array([0.0]))


# ---------------------------------------------------------------------------
# lagged_population_crosscorr
# ---------------------------------------------------------------------------


def _leading_population(n_rois=6, n_frames=600, delay=3, seed=17):
    """Population in which ROI 0 leads every other ROI by ``delay`` frames."""
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(seed)
    base = gaussian_filter1d(rng.normal(0.0, 1.0, n_frames), sigma=4.0)
    signals = np.empty((n_rois, n_frames))
    signals[0] = base
    for i in range(1, n_rois):
        signals[i] = np.roll(base, delay) + 0.05 * rng.normal(0.0, 1.0, n_frames)
    return signals


def test_lagged_crosscorr_synchronous_population_peaks_at_zero():
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(18)
    base = gaussian_filter1d(rng.normal(0.0, 1.0, 600), sigma=4.0)
    signals = base[None, :] + 0.05 * rng.normal(0.0, 1.0, size=(5, 600))
    result = lagged_population_crosscorr(signals, max_lag_frames=8)
    assert result["lags"].tolist() == list(range(-8, 9))
    assert result["xcorr"].shape == (5, 17)
    assert np.all(result["peak_lag"] == 0.0)
    assert np.all(np.abs(result["asymmetry"]) < 0.5)


def test_lagged_crosscorr_leading_cell_has_positive_lag_and_asymmetry():
    signals = _leading_population(delay=3)
    result = lagged_population_crosscorr(signals, max_lag_frames=8)
    assert result["peak_lag"][0] == pytest.approx(3.0)
    assert result["asymmetry"][0] > 0
    # The lagging cells follow the leader, so their asymmetry is negative.
    assert np.all(result["asymmetry"][1:] < 0)


def test_lagged_crosscorr_degenerate_inputs():
    single = lagged_population_crosscorr(np.zeros((1, 100)), max_lag_frames=2)
    assert np.isnan(single["xcorr"]).all()
    assert np.isnan(single["peak_lag"]).all()
    assert np.isnan(single["asymmetry"]).all()
    assert np.isnan(lagged_population_crosscorr(np.zeros((3, 1)))["xcorr"]).all()


def test_lagged_crosscorr_constant_cell_is_nan():
    signals = _leading_population(n_rois=4)
    signals[2] = 1.0
    result = lagged_population_crosscorr(signals, max_lag_frames=4)
    assert np.isnan(result["xcorr"][2]).all()
    assert np.isnan(result["peak_lag"][2])
    assert np.isnan(result["asymmetry"][2])


def test_lagged_crosscorr_validates_inputs():
    with pytest.raises(ValueError, match="2-D"):
        lagged_population_crosscorr(np.zeros(10))
    with pytest.raises(ValueError, match="max_lag_frames"):
        lagged_population_crosscorr(np.zeros((3, 10)), max_lag_frames=-1)


# ---------------------------------------------------------------------------
# coupling_summary
# ---------------------------------------------------------------------------


def test_coupling_summary_columns_and_shape():
    signals, hd, mask = _hd_population(n_rois=6, n_frames=400, seed=19, shared_noise=0.3)
    light_on = np.zeros(400, dtype=bool)
    light_on[:200] = True
    frame = coupling_summary(signals, hd, mask, light_on, max_lag_frames=4)
    assert list(frame.columns) == [
        "pop_coupling_all",
        "pop_coupling_light",
        "pop_coupling_dark",
        "mean_noise_corr",
        "xcorr_peak_lag",
        "xcorr_asymmetry",
    ]
    assert len(frame) == 6
    assert frame.index.name == "roi"
    assert np.isfinite(frame["pop_coupling_all"]).all()
    assert np.isfinite(frame["mean_noise_corr"]).all()


def test_coupling_summary_all_dark_session_has_nan_light_column():
    signals, hd, mask = _hd_population(n_rois=4, n_frames=200, seed=20)
    frame = coupling_summary(signals, hd, mask, np.zeros(200, dtype=bool), max_lag_frames=3)
    assert frame["pop_coupling_light"].isna().all()
    assert frame["pop_coupling_dark"].notna().all()


def test_coupling_summary_validates_shapes():
    signals, hd, mask = _hd_population(n_rois=3, n_frames=100, seed=21)
    with pytest.raises(ValueError, match="2-D"):
        coupling_summary(signals[0], hd, mask, np.ones(100, dtype=bool))
    with pytest.raises(ValueError, match="light_on must have shape"):
        coupling_summary(signals, hd, mask, np.ones(50, dtype=bool))
