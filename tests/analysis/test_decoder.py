"""Tests for hm2p.analysis.decoder — Population Vector Average HD decoder.

References
----------
Georgopoulos et al. 1986. "Neuronal population coding of movement direction."
    Science. doi:10.1126/science.3749885
Peyrache et al. 2015. "Internally organized mechanisms of the head direction
    sense." Nature Neuroscience. doi:10.1038/nn.3968
"""

from __future__ import annotations

import numpy as np
import pytest

from hm2p.analysis.decoder import (
    build_decoder,
    cross_validated_decode,
    decode_error,
    decode_hd,
)


def _make_population(
    n_cells=10, n_frames=3000, kappa=3.0, noise=0.1, seed=42,
):
    """Generate synthetic population of HD cells."""
    rng = np.random.default_rng(seed)
    # Random walk HD trajectory
    hd_deg = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    theta = np.deg2rad(hd_deg)
    mask = np.ones(n_frames, dtype=bool)

    # Uniformly spaced preferred directions
    prefs = np.linspace(0, 360, n_cells, endpoint=False)
    signals = np.zeros((n_cells, n_frames), dtype=np.float64)
    for i in range(n_cells):
        pref_rad = np.deg2rad(prefs[i])
        rate = 0.1 + np.exp(kappa * np.cos(theta - pref_rad))
        rate /= rate.max()
        rate += rng.normal(0, noise, n_frames)
        signals[i] = np.clip(rate, 0, None)

    return signals, hd_deg, mask


class TestBuildDecoder:
    """Tests for build_decoder."""

    def test_output_shapes(self):
        signals, hd, mask = _make_population(n_cells=5, n_frames=1000)
        dec = build_decoder(signals, hd, mask, n_bins=36)
        assert dec["tuning_curves"].shape == (5, 36)
        assert dec["bin_centers"].shape == (36,)
        assert dec["preferred_directions"].shape == (5,)
        assert dec["mvl"].shape == (5,)
        assert dec["n_cells"] == 5
        assert dec["n_bins"] == 36

    def test_tuning_curves_finite(self):
        """Tuning curves should contain finite values."""
        signals, hd, mask = _make_population()
        dec = build_decoder(signals, hd, mask)
        assert np.all(np.isfinite(dec["tuning_curves"]))

    def test_bin_centers_cover_360(self):
        signals, hd, mask = _make_population()
        dec = build_decoder(signals, hd, mask, n_bins=36)
        bc = dec["bin_centers"]
        assert bc[0] > 0
        assert bc[-1] < 360
        assert len(bc) == 36

    def test_preferred_directions_in_range(self):
        """PDs should be in [0, 360)."""
        signals, hd, mask = _make_population()
        dec = build_decoder(signals, hd, mask)
        assert np.all(dec["preferred_directions"] >= 0)
        assert np.all(dec["preferred_directions"] < 360)

    def test_mvl_in_range(self):
        """MVL should be in [0, 1]."""
        signals, hd, mask = _make_population()
        dec = build_decoder(signals, hd, mask)
        assert np.all(dec["mvl"] >= 0)
        assert np.all(dec["mvl"] <= 1.0 + 1e-10)

    def test_preferred_directions_recover_true_pds(self):
        """With strong tuning, recovered PDs should be near true PDs."""
        n_cells = 8
        signals, hd, mask = _make_population(
            n_cells=n_cells, n_frames=5000, kappa=5.0, noise=0.05,
        )
        dec = build_decoder(signals, hd, mask)
        true_pds = np.linspace(0, 360, n_cells, endpoint=False)
        for i in range(n_cells):
            diff = abs(dec["preferred_directions"][i] - true_pds[i])
            diff = min(diff, 360 - diff)
            assert diff < 20, f"Cell {i}: PD error {diff:.1f} deg"


class TestDecodeHD:
    """Tests for decode_hd."""

    def test_output_shapes(self):
        signals, hd, mask = _make_population(n_cells=10, n_frames=500)
        dec = build_decoder(signals, hd, mask)
        decoded, confidence = decode_hd(signals, dec)
        assert decoded.shape == (500,)
        assert confidence.shape == (500,)

    def test_confidence_in_range(self):
        """Confidence (resultant vector length) should be in [0, 1]."""
        signals, hd, mask = _make_population(n_cells=10, n_frames=200)
        dec = build_decoder(signals, hd, mask)
        _, confidence = decode_hd(signals, dec)
        assert np.all(confidence >= 0)
        assert np.all(confidence <= 1.0 + 1e-10)

    def test_decoded_in_range(self):
        signals, hd, mask = _make_population()
        dec = build_decoder(signals, hd, mask)
        decoded, _ = decode_hd(signals, dec)
        assert np.all(decoded >= 0)
        assert np.all(decoded < 360)

    def test_time_binning(self):
        signals, hd, mask = _make_population(n_frames=100)
        dec = build_decoder(signals, hd, mask)
        decoded, confidence = decode_hd(signals, dec, time_bins=5)
        assert decoded.shape == (20,)
        assert confidence.shape == (20,)

    def test_good_population_decodes_well(self):
        """With strong tuning, decoding error should be modest."""
        signals, hd, mask = _make_population(n_cells=20, kappa=5.0, noise=0.05)
        dec = build_decoder(signals, hd, mask)
        decoded, _ = decode_hd(signals, dec)
        errs = decode_error(decoded, hd % 360.0)
        # Mean absolute error should be < 45 deg with 20 well-tuned cells
        assert errs["mean_abs_error"] < 45.0

    def test_uniform_activity_low_confidence(self):
        """When all cells fire equally, confidence should be low."""
        rng = np.random.default_rng(123)
        n_cells, n_frames = 10, 100
        # All cells have the same constant activity + tiny noise
        signals = np.ones((n_cells, n_frames)) + rng.normal(0, 0.001, (n_cells, n_frames))
        hd_deg = np.linspace(0, 360, n_frames, endpoint=False)
        mask = np.ones(n_frames, dtype=bool)
        dec = build_decoder(signals, hd_deg, mask)
        _, confidence = decode_hd(signals, dec)
        # With near-uniform activity, mean confidence should be moderate-to-low
        assert np.mean(confidence) < 0.8


class TestDecodeError:
    """Tests for decode_error."""

    def test_perfect_decode(self):
        actual = np.array([0, 90, 180, 270], dtype=float)
        errs = decode_error(actual, actual)
        assert errs["mean_abs_error"] == pytest.approx(0.0)
        np.testing.assert_array_equal(errs["errors_deg"], 0.0)

    def test_known_errors(self):
        decoded = np.array([10.0, 100.0, 190.0])
        actual = np.array([0.0, 90.0, 180.0])
        errs = decode_error(decoded, actual)
        np.testing.assert_allclose(errs["errors_deg"], [10, 10, 10])
        assert errs["mean_abs_error"] == pytest.approx(10.0)

    def test_wrapping(self):
        """Error should wrap correctly across 0/360 boundary."""
        decoded = np.array([350.0])
        actual = np.array([10.0])
        errs = decode_error(decoded, actual)
        # 350 - 10 = 340, wrapped to -20
        assert errs["errors_deg"][0] == pytest.approx(-20.0)
        assert errs["abs_errors_deg"][0] == pytest.approx(20.0)

    def test_circular_statistics(self):
        errs = decode_error(np.array([10.0, 350.0]), np.array([0.0, 0.0]))
        # errors: 10, -10 -> circular mean should be ~0
        assert abs(errs["circular_mean_error"]) < 5


class TestCrossValidatedDecode:
    """Tests for cross_validated_decode."""

    def test_output_shapes(self):
        signals, hd, mask = _make_population(n_cells=8, n_frames=500)
        result = cross_validated_decode(
            signals, hd, mask, n_folds=5, rng=np.random.default_rng(42),
        )
        assert len(result["decoded_deg"]) == np.sum(mask)
        assert len(result["actual_deg"]) == np.sum(mask)
        assert len(result["confidence"]) == np.sum(mask)
        assert result["n_folds"] == 5

    def test_cv_error_keys(self):
        signals, hd, mask = _make_population(n_cells=5, n_frames=300)
        result = cross_validated_decode(
            signals, hd, mask, n_folds=3, rng=np.random.default_rng(42),
        )
        errs = result["errors"]
        assert "mean_abs_error" in errs
        assert "median_abs_error" in errs
        assert "circular_std_error" in errs

    def test_cv_has_confidence(self):
        """CV result should include confidence array."""
        signals, hd, mask = _make_population(n_cells=5, n_frames=300)
        result = cross_validated_decode(
            signals, hd, mask, n_folds=3, rng=np.random.default_rng(42),
        )
        conf = result["confidence"]
        assert conf.shape == (np.sum(mask),)
        assert np.all(conf >= 0)
        assert np.all(conf <= 1.0 + 1e-10)

    def test_cv_better_than_chance(self):
        """CV decoding with tuned population should beat chance (90 deg)."""
        signals, hd, mask = _make_population(n_cells=15, kappa=4.0, noise=0.1)
        result = cross_validated_decode(
            signals, hd, mask, n_folds=5, rng=np.random.default_rng(42),
        )
        # Chance level for uniform distribution is ~90 deg MAE
        assert result["errors"]["mean_abs_error"] < 90.0

    def test_cv_reproducible(self):
        """Same RNG seed should give same results."""
        signals, hd, mask = _make_population(n_cells=5, n_frames=200)
        r1 = cross_validated_decode(signals, hd, mask, n_folds=3,
                                     rng=np.random.default_rng(42))
        r2 = cross_validated_decode(signals, hd, mask, n_folds=3,
                                     rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1["decoded_deg"], r2["decoded_deg"])
        np.testing.assert_array_equal(r1["confidence"], r2["confidence"])
