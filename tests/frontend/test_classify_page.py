"""Tests for Cell Classification page logic and module integration."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.classify import (
    classification_summary_table,
    classify_population,
    classify_single_cell,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length


def _make_population(n_hd=4, n_noise=4, n_frames=3000, kappa=3.0,
                     noise=0.2, seed=42):
    """Reproduce the page's synthetic population generator."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    theta = np.deg2rad(hd)
    n_total = n_hd + n_noise
    signals = np.zeros((n_total, n_frames))
    prefs = np.linspace(0, 360, n_hd, endpoint=False)
    for i in range(n_hd):
        kappas_i = np.clip(rng.normal(kappa, 0.8), 0.5, 10.0)
        signals[i] = 0.1 + np.exp(kappas_i * np.cos(theta - np.deg2rad(prefs[i])))
        signals[i] /= signals[i].max()
        signals[i] += rng.normal(0, noise, n_frames)
        signals[i] = np.clip(signals[i], 0, None)
    for i in range(n_hd, n_total):
        signals[i] = np.abs(rng.normal(1, 0.5, n_frames))
    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, mask


class TestPagePopulationGenerator:
    """Test that page's synthetic data works correctly."""

    def test_shape(self):
        signals, hd, mask = _make_population()
        assert signals.shape == (8, 3000)
        assert hd.shape == (3000,)
        assert mask.shape == (3000,)

    def test_hd_range(self):
        _, hd, _ = _make_population()
        assert np.all(hd >= 0)
        assert np.all(hd < 360)

    def test_signals_non_negative(self):
        signals, _, _ = _make_population()
        # HD cells are clipped to 0
        for i in range(4):
            assert np.all(signals[i] >= 0)

    def test_tuned_cells_have_higher_mvl(self):
        signals, hd, mask = _make_population(kappa=4.0)
        tuned_mvls = []
        noise_mvls = []
        for i in range(8):
            tc, bc = compute_hd_tuning_curve(signals[i], hd, mask)
            mvl = mean_vector_length(tc, bc)
            if i < 4:
                tuned_mvls.append(mvl)
            else:
                noise_mvls.append(mvl)
        assert np.mean(tuned_mvls) > np.mean(noise_mvls)


class TestClassificationIntegration:
    """Integration tests for classification pipeline as used by page."""

    def test_full_pipeline(self):
        """Full classify -> table pipeline mirrors page logic."""
        signals, hd, mask = _make_population(n_hd=3, n_noise=3, kappa=4.0, noise=0.1)
        pop = classify_population(
            signals, hd, mask, n_shuffles=100,
            rng=np.random.default_rng(42),
        )
        table = classification_summary_table(pop)

        assert len(table) == 6
        assert pop["n_hd"] + pop["n_non_hd"] == 6
        # At least some should be HD with strong tuning
        assert pop["n_hd"] >= 1

    def test_threshold_sensitivity(self):
        """Stricter thresholds should yield fewer HD cells."""
        signals, hd, mask = _make_population(kappa=2.0, noise=0.2)

        pop_lenient = classify_population(
            signals, hd, mask, mvl_threshold=0.05, p_threshold=0.1,
            reliability_threshold=0.2, n_shuffles=100,
            rng=np.random.default_rng(42),
        )
        pop_strict = classify_population(
            signals, hd, mask, mvl_threshold=0.4, p_threshold=0.01,
            reliability_threshold=0.8, n_shuffles=100,
            rng=np.random.default_rng(42),
        )
        assert pop_lenient["n_hd"] >= pop_strict["n_hd"]

    def test_grade_distribution(self):
        """Grades should be valid characters."""
        signals, hd, mask = _make_population()
        pop = classify_population(
            signals, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(42),
        )
        table = classification_summary_table(pop)
        for row in table:
            assert row["grade"] in ("A", "B", "C", "D")

    def test_scatter_data_consistency(self):
        """MVL, p-value, reliability arrays match cell count (page scatter plot)."""
        signals, hd, mask = _make_population()
        pop = classify_population(
            signals, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(42),
        )
        cells = pop["cells"]
        mvls = [c["mvl"] for c in cells]
        pvals = [c["p_value"] for c in cells]
        reliabilities = [c["reliability"] for c in cells]
        assert len(mvls) == 8
        assert all(0 <= p <= 1 for p in pvals)
        assert all(m >= 0 for m in mvls)

    def test_tuning_curves_for_gallery(self):
        """Tuning curves can be computed for all classified cells."""
        signals, hd, mask = _make_population(n_hd=3, n_noise=2)
        pop = classify_population(
            signals, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(42),
        )
        for idx in range(5):
            tc, bc = compute_hd_tuning_curve(signals[idx], hd, mask, n_bins=36)
            assert len(tc) == 36
            assert len(bc) == 36
            assert np.all(np.isfinite(tc))
