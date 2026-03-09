"""Tests for Population Dynamics page logic."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.population import (
    ensemble_coherence,
    pairwise_correlations,
    population_pca,
    population_vector_correlation,
)


def _make_population(n_cells=10, n_frames=3000, kappa=3.0, noise=0.15, seed=42):
    """Generate HD-tuned population."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    theta = np.deg2rad(hd)
    prefs = np.linspace(0, 360, n_cells, endpoint=False)
    signals = np.zeros((n_cells, n_frames))
    for i in range(n_cells):
        k = np.clip(rng.normal(kappa, 0.5), 0.5, 10.0)
        signals[i] = 0.1 + np.exp(k * np.cos(theta - np.deg2rad(prefs[i])))
        signals[i] /= signals[i].max()
        signals[i] += rng.normal(0, noise, n_frames)
        signals[i] = np.clip(signals[i], 0, None)
    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, mask


class TestPCAPageWorkflow:
    """Test PCA as used in pop dynamics page."""

    def test_pca_hd_ring(self):
        """PCA of HD population should show ring structure in PC1/PC2."""
        signals, hd, mask = _make_population()
        result = population_pca(signals[:, mask])
        assert result["explained_variance_ratio"].shape[0] > 0
        # First 2 PCs should capture significant variance for HD population
        assert result["explained_variance_ratio"][:2].sum() > 0.3

    def test_n_components_95(self):
        signals, hd, mask = _make_population()
        result = population_pca(signals[:, mask])
        assert 1 <= result["n_components_95"] <= 10


class TestCorrelationsPageWorkflow:
    """Test correlation analysis as used in page."""

    def test_correlation_matrix_shape(self):
        signals, _, mask = _make_population(n_cells=8)
        result = pairwise_correlations(signals[:, mask])
        assert result.shape == (8, 8)

    def test_correlation_diagonal_is_one(self):
        signals, _, mask = _make_population()
        result = pairwise_correlations(signals[:, mask])
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)


class TestPopVectorCorrelation:
    """Test population vector correlation."""

    def test_pv_matrix_symmetric(self):
        signals, hd, mask = _make_population()
        result = population_vector_correlation(signals, hd, mask)
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_pv_diagonal_is_one(self):
        signals, hd, mask = _make_population()
        result = population_vector_correlation(signals, hd, mask)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=0.01)


class TestEnsembleCoherence:
    """Test ensemble coherence computation."""

    def test_coherence_shape(self):
        signals, _, mask = _make_population()
        centers, coherence = ensemble_coherence(signals[:, mask], window_frames=500)
        assert len(coherence) > 0
        assert len(centers) == len(coherence)

    def test_coherence_bounded(self):
        signals, _, mask = _make_population()
        _, coherence = ensemble_coherence(signals[:, mask], window_frames=500)
        assert np.all(coherence >= -1)
        assert np.all(coherence <= 1)
