"""Tests for calcium/neuropil.py — neuropil subtraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.calcium.neuropil import (
    estimate_neuropil_coefficient,
    subtract_estimated_coefficient,
    subtract_fixed_coefficient,
)

# ---------------------------------------------------------------------------
# subtract_fixed_coefficient — pure numpy, fully testable
# ---------------------------------------------------------------------------


def test_fixed_coefficient_default(rng: np.random.Generator) -> None:
    """Default coefficient (0.7) is applied correctly."""
    n_rois, n_frames = 10, 500
    F = rng.uniform(100, 1000, (n_rois, n_frames)).astype(np.float32)
    Fneu = rng.uniform(50, 500, (n_rois, n_frames)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu)
    np.testing.assert_allclose(result, F - 0.7 * Fneu, rtol=1e-5)


def test_fixed_coefficient_custom(rng: np.random.Generator) -> None:
    """Custom coefficient is applied correctly."""
    F = rng.uniform(0, 1, (5, 100)).astype(np.float32)
    Fneu = rng.uniform(0, 1, (5, 100)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=0.5)
    np.testing.assert_allclose(result, F - 0.5 * Fneu, rtol=1e-5)


def test_fixed_coefficient_output_shape(rng: np.random.Generator) -> None:
    """Output shape matches input shape."""
    F = rng.standard_normal((20, 300)).astype(np.float32)
    Fneu = rng.standard_normal((20, 300)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu)
    assert result.shape == F.shape


@given(
    coefficient=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_fixed_coefficient_property_range(coefficient: float) -> None:
    """subtract_fixed_coefficient produces finite outputs for any valid coefficient."""
    rng = np.random.default_rng(0)
    F = rng.uniform(0, 1000, (5, 50)).astype(np.float32)
    Fneu = rng.uniform(0, 500, (5, 50)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=coefficient)
    assert np.all(np.isfinite(result))


def test_fixed_coefficient_zero(rng: np.random.Generator) -> None:
    """Coefficient of 0 returns F unchanged."""
    F = rng.uniform(100, 500, (5, 100)).astype(np.float32)
    Fneu = rng.uniform(50, 200, (5, 100)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=0.0)
    np.testing.assert_array_equal(result, F)


def test_fixed_coefficient_one(rng: np.random.Generator) -> None:
    """Coefficient of 1 subtracts full neuropil."""
    F = rng.uniform(100, 500, (5, 100)).astype(np.float32)
    Fneu = rng.uniform(50, 200, (5, 100)).astype(np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=1.0)
    np.testing.assert_allclose(result, F - Fneu, rtol=1e-5)


def test_fixed_coefficient_can_go_negative() -> None:
    """Result can be negative when Fneu > F."""
    F = np.full((1, 10), 100.0, dtype=np.float32)
    Fneu = np.full((1, 10), 200.0, dtype=np.float32)
    result = subtract_fixed_coefficient(F, Fneu, coefficient=0.7)
    assert np.all(result < 0)


# ---------------------------------------------------------------------------
# estimate_neuropil_coefficient
# ---------------------------------------------------------------------------


def test_estimate_coefficient_recovers_known_alpha() -> None:
    """Estimator recovers a known contamination coefficient from synthetic data.

    Construct F = alpha * Fneu + signal, where signal is sparse. The estimator
    should recover alpha from the lower-envelope frames where signal ≈ 0.
    """
    rng = np.random.default_rng(42)
    alpha_true = 0.75
    n_rois, n_frames = 3, 2000

    Fneu = rng.uniform(200, 400, (n_rois, n_frames)).astype(np.float32)
    # Sparse signal: non-zero in only 10% of frames
    signal = np.zeros((n_rois, n_frames), dtype=np.float32)
    active = rng.random((n_rois, n_frames)) < 0.10
    signal[active] = rng.uniform(50, 300, active.sum()).astype(np.float32)

    F = (alpha_true * Fneu + signal).astype(np.float32)

    coeffs = estimate_neuropil_coefficient(F, Fneu, percentile=20.0)

    # Allow ±0.05 tolerance (estimator uses lower-envelope regression, not ground truth)
    np.testing.assert_allclose(coeffs, alpha_true, atol=0.08)


def test_estimate_coefficient_shape(rng: np.random.Generator) -> None:
    """Output shape is (n_rois,)."""
    n_rois, n_frames = 8, 300
    F = rng.uniform(100, 500, (n_rois, n_frames)).astype(np.float32)
    Fneu = rng.uniform(50, 200, (n_rois, n_frames)).astype(np.float32)
    coeffs = estimate_neuropil_coefficient(F, Fneu)
    assert coeffs.shape == (n_rois,)
    assert coeffs.dtype == np.float32


def test_estimate_coefficient_clipped_to_unit_interval(rng: np.random.Generator) -> None:
    """Estimated coefficients are always in [0, 1]."""
    F = rng.uniform(0, 1000, (10, 500)).astype(np.float32)
    Fneu = rng.uniform(0, 500, (10, 500)).astype(np.float32)
    coeffs = estimate_neuropil_coefficient(F, Fneu)
    assert np.all(coeffs >= 0.0)
    assert np.all(coeffs <= 1.0)


def test_estimate_coefficient_raises_on_wrong_ndim() -> None:
    """ValueError raised when F or Fneu are not 2-D."""
    with pytest.raises(ValueError, match="2-D"):
        estimate_neuropil_coefficient(
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        )


def test_estimate_coefficient_raises_on_shape_mismatch(rng: np.random.Generator) -> None:
    """ValueError raised when F and Fneu shapes differ."""
    F = rng.uniform(0, 1, (5, 100)).astype(np.float32)
    Fneu = rng.uniform(0, 1, (5, 50)).astype(np.float32)
    with pytest.raises(ValueError, match="shape"):
        estimate_neuropil_coefficient(F, Fneu)


def test_estimate_coefficient_raises_on_bad_percentile(rng: np.random.Generator) -> None:
    """ValueError raised when percentile is out of (0, 100)."""
    F = rng.uniform(0, 1, (2, 100)).astype(np.float32)
    Fneu = rng.uniform(0, 1, (2, 100)).astype(np.float32)
    with pytest.raises(ValueError, match="percentile"):
        estimate_neuropil_coefficient(F, Fneu, percentile=0.0)
    with pytest.raises(ValueError, match="percentile"):
        estimate_neuropil_coefficient(F, Fneu, percentile=100.0)


def test_estimate_coefficient_fallback_on_near_zero_neuropil() -> None:
    """Falls back to 0.7 when neuropil variance is near zero (warns, no crash)."""
    F = np.full((1, 200), 300.0, dtype=np.float32)
    Fneu = np.zeros((1, 200), dtype=np.float32)  # zero neuropil
    coeffs = estimate_neuropil_coefficient(F, Fneu, percentile=20.0)
    # Should return default 0.7 without raising
    assert coeffs.shape == (1,)
    assert np.isfinite(coeffs[0])


@given(
    alpha_true=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
    signal_scale=st.floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30)
def test_estimate_coefficient_property_recovery(alpha_true: float, signal_scale: float) -> None:
    """Property test: estimator returns values in [0, 1] for any alpha in [0.1, 0.9]."""
    rng = np.random.default_rng(seed=int(alpha_true * 1000))
    n_rois, n_frames = 2, 1000
    Fneu = rng.uniform(100, 300, (n_rois, n_frames)).astype(np.float32)
    signal = np.zeros((n_rois, n_frames), dtype=np.float32)
    active = rng.random((n_rois, n_frames)) < 0.10
    signal[active] = float(signal_scale)
    F = (alpha_true * Fneu + signal).astype(np.float32)
    coeffs = estimate_neuropil_coefficient(F, Fneu, percentile=20.0)
    assert np.all(coeffs >= 0.0)
    assert np.all(coeffs <= 1.0)
    assert np.all(np.isfinite(coeffs))


# ---------------------------------------------------------------------------
# subtract_estimated_coefficient
# ---------------------------------------------------------------------------


def test_subtract_estimated_returns_corrected_and_coefficients(rng: np.random.Generator) -> None:
    """subtract_estimated_coefficient returns (F_corr, coefficients) tuple."""
    n_rois, n_frames = 5, 200
    F = rng.uniform(200, 600, (n_rois, n_frames)).astype(np.float32)
    Fneu = rng.uniform(100, 300, (n_rois, n_frames)).astype(np.float32)
    F_corr, coeffs = subtract_estimated_coefficient(F, Fneu)
    assert F_corr.shape == (n_rois, n_frames)
    assert F_corr.dtype == np.float32
    assert coeffs.shape == (n_rois,)
    assert coeffs.dtype == np.float32


def test_subtract_estimated_applies_per_roi_coefficient(rng: np.random.Generator) -> None:
    """Corrected trace equals F - coeff[i] * Fneu for each ROI."""
    n_rois, n_frames = 4, 150
    F = rng.uniform(200, 600, (n_rois, n_frames)).astype(np.float32)
    Fneu = rng.uniform(100, 300, (n_rois, n_frames)).astype(np.float32)
    F_corr, coeffs = subtract_estimated_coefficient(F, Fneu)
    expected = F - coeffs[:, np.newaxis] * Fneu
    np.testing.assert_allclose(F_corr, expected.astype(np.float32), rtol=1e-5)


def test_subtract_estimated_output_dtype(rng: np.random.Generator) -> None:
    """Output is always float32."""
    F = rng.uniform(100, 400, (3, 100)).astype(np.float32)
    Fneu = rng.uniform(50, 200, (3, 100)).astype(np.float32)
    F_corr, _ = subtract_estimated_coefficient(F, Fneu)
    assert F_corr.dtype == np.float32


# ---------------------------------------------------------------------------
# subtract_fissa — tested with mock to avoid requiring real TIFFs
# ---------------------------------------------------------------------------


def _make_fissa_mock(n_rois: int, n_frames: int) -> MagicMock:
    """Build a mock fissa.Experiment that returns plausible separated results."""
    # Each ROI's result: exp.result[i][0] is a (n_frames,) array
    mock_exp = MagicMock()
    mock_exp.result = {
        i: {0: np.full(n_frames, float(i + 1), dtype=np.float64)} for i in range(n_rois)
    }
    return mock_exp


def test_fissa_returns_correct_shape(tmp_path: Path, rng: np.random.Generator) -> None:
    """subtract_fissa returns (n_rois, n_frames) float32 when FISSA succeeds."""
    n_rois, n_frames = 5, 300
    tiff_paths = [tmp_path / "frame.tif"]
    roi_masks = [np.zeros((64, 64), dtype=bool) for _ in range(n_rois)]
    roi_masks[0][10:20, 10:20] = True  # at least one non-empty mask

    mock_exp = _make_fissa_mock(n_rois, n_frames)

    with patch.dict(
        "sys.modules", {"fissa": MagicMock(Experiment=MagicMock(return_value=mock_exp))}
    ):
        import importlib

        import hm2p.calcium.neuropil as neuropil_mod

        importlib.reload(neuropil_mod)

        from hm2p.calcium.neuropil import subtract_fissa as subtract_fissa_reloaded

        result = subtract_fissa_reloaded(
            tiff_paths=tiff_paths,
            roi_masks=roi_masks,
            output_dir=tmp_path / "fissa_cache",
        )

    assert result.shape == (n_rois, n_frames)
    assert result.dtype == np.float32


def test_fissa_output_matches_exp_result(tmp_path: Path) -> None:
    """subtract_fissa extracts exp.result[i][0] for each ROI correctly."""
    n_rois, n_frames = 3, 100
    tiff_paths = [tmp_path / "t.tif"]
    roi_masks = [np.zeros((32, 32), dtype=bool) for _ in range(n_rois)]

    # Distinctive values per ROI to verify correct extraction
    mock_exp = _make_fissa_mock(n_rois, n_frames)

    with patch.dict(
        "sys.modules", {"fissa": MagicMock(Experiment=MagicMock(return_value=mock_exp))}
    ):
        import importlib

        import hm2p.calcium.neuropil as neuropil_mod

        importlib.reload(neuropil_mod)

        from hm2p.calcium.neuropil import subtract_fissa as sf

        result = sf(tiff_paths=tiff_paths, roi_masks=roi_masks, output_dir=tmp_path / "cache")

    for i in range(n_rois):
        np.testing.assert_allclose(result[i], float(i + 1), rtol=1e-5)


def test_fissa_fallback_when_fissa_fails(tmp_path: Path, rng: np.random.Generator) -> None:
    """When FISSA raises, falls back to estimated subtraction if fallback provided."""
    n_rois, n_frames = 4, 200
    F = rng.uniform(200, 600, (n_rois, n_frames)).astype(np.float32)
    Fneu = rng.uniform(100, 300, (n_rois, n_frames)).astype(np.float32)

    # Mock fissa.Experiment to raise on separate()
    crashing_exp = MagicMock()
    crashing_exp.separate.side_effect = RuntimeError("FISSA internal error")

    with patch.dict(
        "sys.modules", {"fissa": MagicMock(Experiment=MagicMock(return_value=crashing_exp))}
    ):
        import importlib

        import hm2p.calcium.neuropil as neuropil_mod

        importlib.reload(neuropil_mod)

        from hm2p.calcium.neuropil import subtract_fissa as sf

        result = sf(
            tiff_paths=[tmp_path / "t.tif"],
            roi_masks=[np.zeros((32, 32), dtype=bool)] * n_rois,
            output_dir=tmp_path / "cache",
            F_fallback=F,
            Fneu_fallback=Fneu,
        )

    # Fallback should produce a valid (n_rois, n_frames) float32 array
    assert result.shape == (n_rois, n_frames)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


def test_fissa_raises_without_fallback_on_failure(tmp_path: Path) -> None:
    """When FISSA fails and no fallback provided, RuntimeError is raised."""
    crashing_exp = MagicMock()
    crashing_exp.separate.side_effect = RuntimeError("boom")

    with patch.dict(
        "sys.modules", {"fissa": MagicMock(Experiment=MagicMock(return_value=crashing_exp))}
    ):
        import importlib

        import hm2p.calcium.neuropil as neuropil_mod

        importlib.reload(neuropil_mod)

        from hm2p.calcium.neuropil import subtract_fissa as sf

        with pytest.raises(RuntimeError, match="FISSA neuropil subtraction failed"):
            sf(
                tiff_paths=[tmp_path / "t.tif"],
                roi_masks=[np.zeros((32, 32), dtype=bool)],
                output_dir=tmp_path / "cache",
            )


def test_fissa_import_error_when_not_installed(tmp_path: Path) -> None:
    """ImportError is raised clearly when fissa is not installed."""
    with patch.dict("sys.modules", {"fissa": None}):
        import importlib

        import hm2p.calcium.neuropil as neuropil_mod

        importlib.reload(neuropil_mod)

        from hm2p.calcium.neuropil import subtract_fissa as sf

        with pytest.raises((ImportError, TypeError)):
            sf(
                tiff_paths=[tmp_path / "t.tif"],
                roi_masks=[np.zeros((32, 32), dtype=bool)],
                output_dir=tmp_path / "cache",
            )


def test_fissa_handles_2d_result_shape(tmp_path: Path) -> None:
    """subtract_fissa handles the case where exp.result[i][0] is (1, n_frames)."""
    n_rois, n_frames = 2, 80
    # Simulate FISSA returning a (1, n_frames) array instead of (n_frames,)
    mock_exp = MagicMock()
    mock_exp.result = {
        i: {0: np.full((1, n_frames), float(i + 10), dtype=np.float64)} for i in range(n_rois)
    }

    with patch.dict(
        "sys.modules", {"fissa": MagicMock(Experiment=MagicMock(return_value=mock_exp))}
    ):
        import importlib

        import hm2p.calcium.neuropil as neuropil_mod

        importlib.reload(neuropil_mod)

        from hm2p.calcium.neuropil import subtract_fissa as sf

        result = sf(
            tiff_paths=[tmp_path / "t.tif"],
            roi_masks=[np.zeros((32, 32), dtype=bool)] * n_rois,
            output_dir=tmp_path / "cache",
        )

    assert result.shape == (n_rois, n_frames)
    assert result.dtype == np.float32
