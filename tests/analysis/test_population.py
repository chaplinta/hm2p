"""Tests for hm2p.analysis.population — population-level analysis."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.population import (
    ensemble_coherence,
    pairwise_correlations,
    population_pca,
    population_vector_correlation,
)


def _make_population(n_cells=10, n_frames=2000, seed=42):
    """Generate synthetic population with correlated activity."""
    rng = np.random.default_rng(seed)
    # Shared signal + noise
    shared = rng.normal(0, 1, n_frames)
    signals = np.zeros((n_cells, n_frames))
    for i in range(n_cells):
        signals[i] = 0.3 * shared + rng.normal(0, 1, n_frames)
    return signals


def _make_hd_population(n_cells=10, n_frames=3000, kappa=3.0, seed=42):
    """Generate HD-tuned population."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    theta = np.deg2rad(hd)
    prefs = np.linspace(0, 360, n_cells, endpoint=False)
    signals = np.zeros((n_cells, n_frames))
    for i in range(n_cells):
        signals[i] = 0.1 + np.exp(kappa * np.cos(theta - np.deg2rad(prefs[i])))
        signals[i] /= signals[i].max()
        signals[i] += rng.normal(0, 0.1, n_frames)
        signals[i] = np.clip(signals[i], 0, None)
    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, mask


class TestPopulationPCA:
    """Tests for population_pca."""

    def test_output_shapes(self):
        signals = _make_population(n_cells=5, n_frames=200)
        result = population_pca(signals)
        assert result["components"].shape[0] == 5
        assert len(result["explained_variance_ratio"]) == 5
        assert len(result["cumulative_variance"]) == 5

    def test_variance_sums_to_one(self):
        signals = _make_population()
        result = population_pca(signals)
        assert result["cumulative_variance"][-1] == pytest.approx(1.0, abs=1e-10)

    def test_variance_monotonically_decreasing(self):
        signals = _make_population()
        result = population_pca(signals)
        evr = result["explained_variance_ratio"]
        assert np.all(evr[:-1] >= evr[1:])

    def test_n_components_limit(self):
        signals = _make_population(n_cells=10, n_frames=500)
        result = population_pca(signals, n_components=3)
        assert result["components"].shape[0] == 3
        assert result["loadings"].shape == (10, 3)

    def test_n_components_95(self):
        signals = _make_population(n_cells=10)
        result = population_pca(signals)
        assert 1 <= result["n_components_95"] <= 10

    def test_correlated_pop_low_dimensionality(self):
        """Highly correlated population should have low dimensionality."""
        rng = np.random.default_rng(42)
        shared = rng.normal(0, 1, 1000)
        signals = np.array([shared + rng.normal(0, 0.1, 1000) for _ in range(10)])
        result = population_pca(signals)
        # First component should explain most variance
        assert result["explained_variance_ratio"][0] > 0.8

    def test_loadings_shape(self):
        signals = _make_population(n_cells=8, n_frames=300)
        result = population_pca(signals)
        assert result["loadings"].shape == (8, 8)


class TestPairwiseCorrelations:
    """Tests for pairwise_correlations."""

    def test_diagonal_is_one(self):
        signals = _make_population()
        corr = pairwise_correlations(signals)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_symmetric(self):
        signals = _make_population()
        corr = pairwise_correlations(signals)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_shape(self):
        signals = _make_population(n_cells=7)
        corr = pairwise_correlations(signals)
        assert corr.shape == (7, 7)

    def test_range(self):
        signals = _make_population()
        corr = pairwise_correlations(signals)
        assert np.all(corr >= -1.0 - 1e-10)
        assert np.all(corr <= 1.0 + 1e-10)


class TestPopulationVectorCorrelation:
    """Tests for population_vector_correlation."""

    def test_shape(self):
        signals, hd, mask = _make_hd_population()
        pvc = population_vector_correlation(signals, hd, mask, n_bins=36)
        assert pvc.shape == (36, 36)

    def test_diagonal_is_one(self):
        signals, hd, mask = _make_hd_population()
        pvc = population_vector_correlation(signals, hd, mask)
        diag = np.diag(pvc)
        valid_diag = diag[~np.isnan(diag)]
        np.testing.assert_allclose(valid_diag, 1.0, atol=1e-10)

    def test_circular_structure(self):
        """Adjacent HD bins should be more correlated than opposite bins."""
        signals, hd, mask = _make_hd_population(n_cells=20, kappa=3.0)
        pvc = population_vector_correlation(signals, hd, mask, n_bins=36)
        # Adjacent bins (offset 1) should have higher correlation than opposite (offset 18)
        adj_corrs = []
        opp_corrs = []
        for i in range(36):
            j_adj = (i + 1) % 36
            j_opp = (i + 18) % 36
            if not np.isnan(pvc[i, j_adj]):
                adj_corrs.append(pvc[i, j_adj])
            if not np.isnan(pvc[i, j_opp]):
                opp_corrs.append(pvc[i, j_opp])
        if adj_corrs and opp_corrs:
            assert np.mean(adj_corrs) > np.mean(opp_corrs)


class TestEnsembleCoherence:
    """Tests for ensemble_coherence."""

    def test_output_shapes(self):
        signals = _make_population(n_cells=5, n_frames=500)
        centers, coh = ensemble_coherence(signals, window_frames=100)
        assert len(centers) == len(coh)
        assert len(centers) > 0

    def test_correlated_pop_high_coherence(self):
        rng = np.random.default_rng(42)
        shared = rng.normal(0, 1, 500)
        signals = np.array([shared + rng.normal(0, 0.1, 500) for _ in range(5)])
        _, coh = ensemble_coherence(signals, window_frames=100)
        assert np.mean(coh) > 0.5

    def test_independent_pop_low_coherence(self):
        rng = np.random.default_rng(42)
        signals = rng.normal(0, 1, (5, 500))
        _, coh = ensemble_coherence(signals, window_frames=100)
        assert np.mean(coh) < 0.3
