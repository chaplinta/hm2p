"""Tests for PVA HD decoder page logic."""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.decoder import (
    build_decoder,
    cross_validated_decode,
    decode_error,
    decode_hd,
)
from hm2p.analysis.tuning import compute_hd_tuning_curve


def _make_population(n_cells=8, n_frames=3000, kappa=3.0, noise=0.15, seed=42):
    """Generate HD-tuned population for decoder page tests."""
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


class TestDecoderPagePipeline:
    """Test the decoder workflow as used in the page."""

    def test_build_decode_error_pipeline(self):
        """Full pipeline: build -> decode -> error."""
        signals, hd, mask = _make_population()
        model = build_decoder(signals, hd, mask)
        decoded, confidence = decode_hd(signals, model)
        err = decode_error(decoded, hd)

        assert decoded.shape == (3000,)
        assert confidence.shape == (3000,)
        assert "mean_abs_error" in err
        assert err["mean_abs_error"] > 0

    def test_cv_decode(self):
        """Cross-validated decoding."""
        signals, hd, mask = _make_population()
        result = cross_validated_decode(signals, hd, mask, n_folds=3)
        assert "errors" in result
        assert "decoded_deg" in result
        assert "actual_deg" in result
        assert "confidence" in result
        assert result["errors"]["mean_abs_error"] > 0

    def test_confidence_in_range(self):
        """Each frame's confidence should be in [0, 1]."""
        signals, hd, mask = _make_population()
        model = build_decoder(signals, hd, mask)
        _, confidence = decode_hd(signals, model)
        assert np.all(confidence >= 0)
        assert np.all(confidence <= 1.0 + 1e-10)

    def test_good_population_low_error(self):
        """Well-tuned population should decode with low error."""
        signals, hd, mask = _make_population(n_cells=12, kappa=4.0, noise=0.1)
        model = build_decoder(signals, hd, mask)
        decoded, _ = decode_hd(signals, model)
        err = decode_error(decoded, hd)
        assert err["mean_abs_error"] < 60  # Should be fairly accurate

    def test_varying_population_size(self):
        """Decoder should work with different population sizes."""
        for n_cells in [3, 6, 12]:
            signals, hd, mask = _make_population(n_cells=n_cells)
            model = build_decoder(signals, hd, mask)
            decoded, _ = decode_hd(signals, model)
            assert decoded.shape == (3000,)

    def test_build_decoder_has_pds_and_mvl(self):
        """Decoder dict should include preferred_directions and mvl."""
        signals, hd, mask = _make_population()
        model = build_decoder(signals, hd, mask)
        assert "preferred_directions" in model
        assert "mvl" in model
        assert model["preferred_directions"].shape == (8,)
        assert model["mvl"].shape == (8,)
