"""Tests for hm2p.analysis.classify — cell classification."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.classify import (
    classification_summary_table,
    classify_population,
    classify_single_cell,
)


def _make_hd_cell(n=3000, kappa=3.0, noise=0.1, seed=42):
    """Strongly HD-tuned synthetic cell."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
    signal = np.exp(kappa * np.cos(np.deg2rad(hd)))
    signal /= signal.max()
    signal += rng.normal(0, noise, n)
    signal = np.clip(signal, 0, None)
    mask = np.ones(n, dtype=bool)
    return signal, hd, mask


def _make_random_cell(n=3000, seed=99):
    """Non-tuned random cell."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
    signal = np.abs(rng.normal(1, 0.5, n))
    mask = np.ones(n, dtype=bool)
    return signal, hd, mask


class TestClassifySingleCell:
    """Tests for classify_single_cell."""

    def test_hd_cell_classified_correctly(self):
        """Strongly tuned cell should be classified as HD."""
        signal, hd, mask = _make_hd_cell(kappa=4.0, noise=0.05)
        result = classify_single_cell(
            signal, hd, mask, n_shuffles=200, rng=np.random.default_rng(0),
        )
        assert result["is_hd"] is True
        assert result["mvl"] > 0.15
        assert result["p_value"] < 0.05

    def test_random_cell_not_hd(self):
        """Random noise cell should not be classified as HD."""
        signal, hd, mask = _make_random_cell()
        result = classify_single_cell(
            signal, hd, mask, n_shuffles=200, rng=np.random.default_rng(0),
        )
        assert result["is_hd"] is False

    def test_output_keys(self):
        """Result dict should have all expected keys."""
        signal, hd, mask = _make_hd_cell()
        result = classify_single_cell(
            signal, hd, mask, n_shuffles=50, rng=np.random.default_rng(0),
        )
        assert "is_hd" in result
        assert "mvl" in result
        assert "p_value" in result
        assert "reliability" in result
        assert "mi" in result
        assert "preferred_direction" in result
        assert "criteria_passed" in result

    def test_criteria_passed_keys(self):
        """Criteria dict should have mvl, significance, reliability."""
        signal, hd, mask = _make_hd_cell()
        result = classify_single_cell(
            signal, hd, mask, n_shuffles=50, rng=np.random.default_rng(0),
        )
        criteria = result["criteria_passed"]
        assert "mvl" in criteria
        assert "significance" in criteria
        assert "reliability" in criteria

    def test_pd_in_valid_range(self):
        """Preferred direction should be in [0, 360)."""
        signal, hd, mask = _make_hd_cell()
        result = classify_single_cell(
            signal, hd, mask, n_shuffles=50, rng=np.random.default_rng(0),
        )
        assert 0 <= result["preferred_direction"] < 360

    def test_mi_positive_for_tuned(self):
        """Tuned cell should have positive MI."""
        signal, hd, mask = _make_hd_cell(kappa=4.0)
        result = classify_single_cell(
            signal, hd, mask, n_shuffles=50, rng=np.random.default_rng(0),
        )
        assert result["mi"] > 0

    def test_strict_threshold_rejects_weak(self):
        """High MVL threshold should reject weakly tuned cell."""
        signal, hd, mask = _make_hd_cell(kappa=1.0, noise=0.3)
        result = classify_single_cell(
            signal, hd, mask,
            mvl_threshold=0.6,  # Very strict
            n_shuffles=50,
            rng=np.random.default_rng(0),
        )
        assert result["is_hd"] is False
        assert result["criteria_passed"]["mvl"] is False

    def test_lenient_threshold_accepts_moderate(self):
        """Lenient threshold should accept moderately tuned cell."""
        signal, hd, mask = _make_hd_cell(kappa=2.0, noise=0.15)
        result = classify_single_cell(
            signal, hd, mask,
            mvl_threshold=0.05,
            reliability_threshold=0.2,
            n_shuffles=200,
            rng=np.random.default_rng(0),
        )
        assert result["mvl"] > 0.05


class TestClassifyPopulation:
    """Tests for classify_population."""

    def test_mixed_population(self):
        """Population with tuned and untuned cells."""
        rng = np.random.default_rng(42)
        n = 3000
        hd = np.cumsum(rng.normal(0, 5, n)) % 360.0
        mask = np.ones(n, dtype=bool)

        # 2 tuned, 1 random
        tuned1 = np.exp(4.0 * np.cos(np.deg2rad(hd))) / np.exp(4.0)
        tuned1 += rng.normal(0, 0.05, n)
        tuned1 = np.clip(tuned1, 0, None)

        tuned2 = np.exp(3.5 * np.cos(np.deg2rad(hd - 120))) / np.exp(3.5)
        tuned2 += rng.normal(0, 0.05, n)
        tuned2 = np.clip(tuned2, 0, None)

        random = np.abs(rng.normal(1, 0.5, n))

        signals = np.vstack([tuned1, tuned2, random])
        result = classify_population(
            signals, hd, mask, n_shuffles=200,
            rng=np.random.default_rng(0),
        )
        assert result["n_hd"] >= 1  # At least one tuned cell detected
        assert result["n_non_hd"] >= 1  # At least one non-HD
        assert result["n_hd"] + result["n_non_hd"] == 3
        assert len(result["cells"]) == 3
        assert len(result["hd_indices"]) == result["n_hd"]

    def test_fraction_hd(self):
        """fraction_hd should be between 0 and 1."""
        signal, hd, mask = _make_hd_cell()
        signals = np.vstack([signal, signal])
        result = classify_population(
            signals, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(0),
        )
        assert 0.0 <= result["fraction_hd"] <= 1.0

    def test_empty_population(self):
        """Single-cell population should work."""
        signal, hd, mask = _make_hd_cell()
        signals = signal[np.newaxis, :]
        result = classify_population(
            signals, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(0),
        )
        assert len(result["cells"]) == 1


class TestClassificationSummaryTable:
    """Tests for classification_summary_table."""

    def test_table_length(self):
        """Table should have one row per cell."""
        signal, hd, mask = _make_hd_cell()
        signals = np.vstack([signal, signal])
        pop = classify_population(
            signals, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(0),
        )
        table = classification_summary_table(pop)
        assert len(table) == 2

    def test_table_columns(self):
        """Each row should have expected columns."""
        signal, hd, mask = _make_hd_cell()
        signals = signal[np.newaxis, :]
        pop = classify_population(
            signals, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(0),
        )
        table = classification_summary_table(pop)
        row = table[0]
        assert "cell" in row
        assert "is_hd" in row
        assert "mvl" in row
        assert "grade" in row

    def test_grade_assignment(self):
        """HD cells should get grade A/B/C, non-HD should get D."""
        rng_gen = np.random.default_rng(42)
        n = 3000
        hd = np.cumsum(rng_gen.normal(0, 5, n)) % 360.0
        mask = np.ones(n, dtype=bool)
        random = np.abs(rng_gen.normal(1, 0.5, n))
        signals = random[np.newaxis, :]
        pop = classify_population(
            signals, hd, mask, n_shuffles=100,
            rng=np.random.default_rng(0),
        )
        table = classification_summary_table(pop)
        # Random cell should be grade D
        assert table[0]["grade"] == "D"
