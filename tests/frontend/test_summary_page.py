"""Tests for Session Summary page workflow."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.classify import classify_population, classification_summary_table
from hm2p.analysis.gain import population_gain_modulation
from hm2p.analysis.speed import speed_modulation_index
from hm2p.analysis.stability import drift_per_epoch


def _make_session(n_cells=6, n_frames=6000, seed=42):
    """Simplified version of summary page's synthetic session."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    speed = np.abs(rng.normal(10, 5, n_frames))
    theta = np.deg2rad(hd)
    prefs = np.linspace(0, 360, n_cells, endpoint=False)

    light_on = np.zeros(n_frames, dtype=bool)
    for start in range(0, n_frames, 3600):
        light_on[start:min(start + 1800, n_frames)] = True

    signals = np.zeros((n_cells, n_frames))
    n_hd = n_cells * 2 // 3

    for i in range(n_cells):
        if i < n_hd:
            k = np.clip(rng.normal(3.0, 0.5), 0.5, 10.0)
            signals[i] = 0.1 + np.exp(k * np.cos(theta - np.deg2rad(prefs[i])))
            signals[i] /= signals[i].max()
        else:
            signals[i] = np.abs(rng.normal(1, 0.5, n_frames))
        signals[i] += rng.normal(0, 0.15, n_frames)
        signals[i] = np.clip(signals[i], 0, None)

    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, speed, mask, light_on


class TestSummaryPageWorkflow:
    """Test the full summary page analysis pipeline."""

    def test_classification_runs(self):
        signals, hd, _, mask, _ = _make_session()
        pop = classify_population(
            signals, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(42),
        )
        assert pop["n_hd"] + pop["n_non_hd"] == 6
        table = classification_summary_table(pop)
        assert len(table) == 6

    def test_gain_analysis_runs(self):
        signals, hd, _, mask, light_on = _make_session()
        gains = population_gain_modulation(signals, hd, mask, light_on)
        assert len(gains) == 6
        gmis = [g["gain_index"] for g in gains]
        assert all(-1 <= g <= 1 for g in gmis)

    def test_speed_analysis_runs(self):
        signals, _, speed, mask, _ = _make_session()
        smis = []
        for i in range(6):
            r = speed_modulation_index(signals[i], speed, mask)
            smis.append(r["speed_modulation_index"])
        assert len(smis) == 6
        assert all(-1 <= s <= 1 for s in smis)

    def test_drift_analysis_runs(self):
        signals, hd, _, mask, light_on = _make_session()
        dr = drift_per_epoch(signals[0], hd, mask, light_on)
        assert dr["n_epochs"] > 0

    def test_full_pipeline_completes(self):
        """All analyses run end-to-end without error."""
        signals, hd, speed, mask, light_on = _make_session()

        # Classification
        pop = classify_population(
            signals, hd, mask, n_shuffles=50,
            rng=np.random.default_rng(42),
        )
        table = classification_summary_table(pop)

        # Gain
        gains = population_gain_modulation(signals, hd, mask, light_on)
        mean_gmi = np.mean([g["gain_index"] for g in gains])

        # Speed
        smis = [speed_modulation_index(signals[i], speed, mask)["speed_modulation_index"]
                for i in range(6)]
        mean_smi = np.mean(smis)

        # Drift
        if pop["hd_indices"]:
            dr = drift_per_epoch(signals[pop["hd_indices"][0]], hd, mask, light_on)

        # Verify all results are finite
        assert np.isfinite(mean_gmi)
        assert np.isfinite(mean_smi)
        assert len(table) == 6
