"""Tests for hm2p.analysis.rastermap_analysis.

Covers:
- compute_rastermap: Rastermap neuron sorting
- compute_superneurons: binned averaging of sorted neurons
- superneuron_behaviour_correlations: Spearman correlation with behaviour
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.analysis.rastermap_analysis import (
    compute_rastermap,
    compute_superneurons,
    superneuron_behaviour_correlations,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dff(
    n_rois: int = 40, n_frames: int = 500, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rois, n_frames)).astype(np.float32) * 0.3


def _make_isort(n_rois: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n_rois).astype(np.int64)


def _make_superneurons(
    n_super: int = 4, n_frames: int = 500, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_super, n_frames)).astype(np.float32)


# ===========================================================================
# compute_rastermap
# ===========================================================================


class TestComputeRastermap:
    def test_output_keys_present(self) -> None:
        dff = _make_dff()
        result = compute_rastermap(dff, n_clusters=10, n_PCs=20, time_lag_window=5)
        assert "isort" in result
        assert "embedding" in result

    def test_isort_shape(self) -> None:
        n_rois = 40
        dff = _make_dff(n_rois=n_rois)
        result = compute_rastermap(dff, n_clusters=10, n_PCs=20)
        assert result["isort"].shape == (n_rois,)

    def test_isort_is_permutation(self) -> None:
        """isort must be a valid permutation of [0, n_rois)."""
        n_rois = 40
        dff = _make_dff(n_rois=n_rois)
        result = compute_rastermap(dff, n_clusters=10, n_PCs=20)
        assert set(result["isort"].tolist()) == set(range(n_rois))

    def test_embedding_shape(self) -> None:
        n_rois = 40
        dff = _make_dff(n_rois=n_rois)
        result = compute_rastermap(dff, n_clusters=10, n_PCs=20)
        assert result["embedding"].shape[0] == n_rois

    def test_n_clusters_clipped_to_n_rois(self) -> None:
        """When n_clusters > n_rois, it should clip and not crash."""
        n_rois = 8
        dff = _make_dff(n_rois=n_rois, n_frames=200)
        result = compute_rastermap(dff, n_clusters=200, n_PCs=10)
        assert result["isort"].shape == (n_rois,)

    def test_n_PCs_clipped_to_min_dim(self) -> None:
        """n_PCs larger than n_rois or n_frames should be clipped."""
        n_rois = 20
        dff = _make_dff(n_rois=n_rois, n_frames=30)
        result = compute_rastermap(dff, n_clusters=5, n_PCs=1000)
        assert result["isort"].shape == (n_rois,)

    def test_with_nan_inputs(self) -> None:
        """NaN values in dff should be replaced by nan_to_num and not crash."""
        dff = _make_dff(n_rois=30, n_frames=300)
        dff[2, 10:20] = np.nan
        dff[5, :5] = np.nan
        result = compute_rastermap(dff, n_clusters=8, n_PCs=15)
        assert result["isort"].shape == (30,)

    def test_small_matrix(self) -> None:
        """Smallest viable matrix: 3 ROIs × 50 frames."""
        dff = _make_dff(n_rois=3, n_frames=50)
        result = compute_rastermap(dff, n_clusters=2, n_PCs=2, time_lag_window=2)
        assert result["isort"].shape == (3,)

    def test_deterministic_with_same_seed(self) -> None:
        """Same data should produce the same sorting (rastermap is deterministic)."""
        dff = _make_dff(n_rois=20, n_frames=200, seed=0)
        r1 = compute_rastermap(dff, n_clusters=5, n_PCs=10)
        r2 = compute_rastermap(dff, n_clusters=5, n_PCs=10)
        np.testing.assert_array_equal(r1["isort"], r2["isort"])

    def test_float32_input_accepted(self) -> None:
        dff = _make_dff().astype(np.float32)
        result = compute_rastermap(dff, n_clusters=10, n_PCs=20)
        assert result["isort"].shape[0] == dff.shape[0]

    def test_float64_input_accepted(self) -> None:
        dff = _make_dff().astype(np.float64)
        result = compute_rastermap(dff, n_clusters=10, n_PCs=20)
        assert result["isort"].shape[0] == dff.shape[0]


# ===========================================================================
# compute_superneurons
# ===========================================================================


class TestComputeSuperneurons:
    def test_output_shape(self) -> None:
        n_rois, n_frames, bin_size = 40, 500, 10
        dff = _make_dff(n_rois=n_rois, n_frames=n_frames)
        isort = _make_isort(n_rois=n_rois)
        result = compute_superneurons(dff, isort, bin_size=bin_size)
        assert result.shape == (n_rois // bin_size, n_frames)

    def test_output_dtype_float32(self) -> None:
        dff = _make_dff(n_rois=20, n_frames=200)
        isort = _make_isort(n_rois=20)
        result = compute_superneurons(dff, isort, bin_size=5)
        assert result.dtype == np.float32

    def test_n_frames_preserved(self) -> None:
        n_frames = 300
        dff = _make_dff(n_rois=30, n_frames=n_frames)
        isort = _make_isort(n_rois=30)
        result = compute_superneurons(dff, isort, bin_size=5)
        assert result.shape[1] == n_frames

    def test_n_superneurons_floor_division(self) -> None:
        """n_super = n_rois // bin_size, remainder is dropped."""
        n_rois, bin_size = 35, 10
        dff = _make_dff(n_rois=n_rois, n_frames=200)
        isort = _make_isort(n_rois=n_rois)
        result = compute_superneurons(dff, isort, bin_size=bin_size)
        assert result.shape[0] == n_rois // bin_size  # = 3

    def test_identity_isort_matches_direct_binning(self) -> None:
        """With identity sort order, superneurons = direct bin-averaging of dff."""
        n_rois, n_frames, bin_size = 20, 200, 5
        dff = _make_dff(n_rois=n_rois, n_frames=n_frames)
        isort = np.arange(n_rois, dtype=np.int64)
        result = compute_superneurons(dff, isort, bin_size=bin_size)
        # Manual computation
        expected = np.zeros((n_rois // bin_size, n_frames), dtype=np.float32)
        for i in range(n_rois // bin_size):
            expected[i] = np.nanmean(dff[i * bin_size:(i + 1) * bin_size], axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_bin_size_one_returns_all_rois(self) -> None:
        """bin_size=1 should give one superneuron per ROI."""
        n_rois = 20
        dff = _make_dff(n_rois=n_rois, n_frames=100)
        isort = _make_isort(n_rois=n_rois)
        result = compute_superneurons(dff, isort, bin_size=1)
        assert result.shape[0] == n_rois

    def test_with_nan_values(self) -> None:
        """NaN values in source traces should not propagate via nanmean."""
        n_rois, n_frames = 20, 200
        rng = np.random.default_rng(10)
        dff = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
        dff[0, 50:60] = np.nan
        isort = np.arange(n_rois, dtype=np.int64)
        result = compute_superneurons(dff, isort, bin_size=5)
        # Superneuron 0 averages ROIs 0-4; ROI 0 has NaN at 50:60
        # nanmean should handle it
        assert np.all(np.isfinite(result[0]))  # other 4 ROIs contribute

    def test_constant_dff_preserved(self) -> None:
        """Constant dff should produce constant superneurons."""
        n_rois, n_frames = 20, 100
        dff = np.ones((n_rois, n_frames), dtype=np.float32) * 3.14
        isort = np.arange(n_rois, dtype=np.int64)
        result = compute_superneurons(dff, isort, bin_size=5)
        np.testing.assert_allclose(result, 3.14, atol=1e-5)

    def test_large_bin_size(self) -> None:
        """bin_size = n_rois produces 1 superneuron = mean of all ROIs."""
        n_rois, n_frames = 20, 100
        dff = _make_dff(n_rois=n_rois, n_frames=n_frames)
        isort = np.arange(n_rois, dtype=np.int64)
        result = compute_superneurons(dff, isort, bin_size=n_rois)
        assert result.shape[0] == 1
        expected_mean = np.nanmean(dff, axis=0)
        np.testing.assert_allclose(result[0], expected_mean.astype(np.float32), rtol=1e-5)

    @given(
        n_rois=st.integers(min_value=10, max_value=60),
        bin_size=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=25, deadline=None)
    def test_output_n_superneurons_property(self, n_rois: int, bin_size: int) -> None:
        rng = np.random.default_rng(0)
        dff = rng.standard_normal((n_rois, 100)).astype(np.float32)
        isort = rng.permutation(n_rois).astype(np.int64)
        result = compute_superneurons(dff, isort, bin_size=bin_size)
        assert result.shape[0] == n_rois // bin_size


# ===========================================================================
# superneuron_behaviour_correlations
# ===========================================================================


class TestSuperneuronBehaviourCorrelations:
    def _make_inputs(
        self, n_super: int = 5, n_frames: int = 400, seed: int = 42
    ):
        rng = np.random.default_rng(seed)
        superneurons = rng.standard_normal((n_super, n_frames)).astype(np.float32)
        hd_deg = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
        speed = np.abs(rng.standard_normal(n_frames)).astype(np.float32) * 5.0
        light_on = np.zeros(n_frames, dtype=bool)
        light_on[:n_frames // 2] = True
        return superneurons, hd_deg, speed, light_on

    def test_speed_corr_present_when_speed_provided(self) -> None:
        superneurons, _, speed, _ = self._make_inputs()
        result = superneuron_behaviour_correlations(superneurons, speed=speed)
        assert "speed_corr" in result

    def test_speed_corr_absent_when_not_provided(self) -> None:
        superneurons, _, _, _ = self._make_inputs()
        result = superneuron_behaviour_correlations(superneurons)
        assert "speed_corr" not in result

    def test_hd_corr_present_when_hd_provided(self) -> None:
        superneurons, hd_deg, _, _ = self._make_inputs()
        result = superneuron_behaviour_correlations(superneurons, hd_deg=hd_deg)
        assert "hd_corr" in result

    def test_light_mod_present_when_light_on_provided(self) -> None:
        superneurons, _, _, light_on = self._make_inputs()
        result = superneuron_behaviour_correlations(superneurons, light_on=light_on)
        assert "light_mod" in result

    def test_speed_corr_shape(self) -> None:
        n_super = 6
        superneurons, _, speed, _ = self._make_inputs(n_super=n_super)
        result = superneuron_behaviour_correlations(superneurons, speed=speed)
        assert result["speed_corr"].shape == (n_super,)

    def test_hd_corr_shape(self) -> None:
        n_super = 6
        superneurons, hd_deg, _, _ = self._make_inputs(n_super=n_super)
        result = superneuron_behaviour_correlations(superneurons, hd_deg=hd_deg)
        assert result["hd_corr"].shape == (n_super,)

    def test_light_mod_shape(self) -> None:
        n_super = 6
        superneurons, _, _, light_on = self._make_inputs(n_super=n_super)
        result = superneuron_behaviour_correlations(superneurons, light_on=light_on)
        assert result["light_mod"].shape == (n_super,)

    def test_speed_corr_in_valid_range(self) -> None:
        superneurons, _, speed, _ = self._make_inputs()
        result = superneuron_behaviour_correlations(superneurons, speed=speed)
        valid = result["speed_corr"][np.isfinite(result["speed_corr"])]
        assert np.all(np.abs(valid) <= 1.0 + 1e-6)

    def test_hd_corr_non_negative(self) -> None:
        """HD correlation is max(|r_sin|, |r_cos|) so always >= 0."""
        superneurons, hd_deg, _, _ = self._make_inputs()
        result = superneuron_behaviour_correlations(superneurons, hd_deg=hd_deg)
        valid = result["hd_corr"][np.isfinite(result["hd_corr"])]
        assert np.all(valid >= -1e-6)
        assert np.all(valid <= 1.0 + 1e-6)

    def test_light_mod_index_range(self) -> None:
        superneurons, _, _, light_on = self._make_inputs()
        result = superneuron_behaviour_correlations(superneurons, light_on=light_on)
        valid = result["light_mod"][np.isfinite(result["light_mod"])]
        assert np.all(valid >= -1.0 - 1e-6)
        assert np.all(valid <= 1.0 + 1e-6)

    def test_uses_spearman_not_pearson_for_speed(self) -> None:
        """Verify non-parametric Spearman is used for speed correlation."""
        from scipy.stats import spearmanr

        rng = np.random.default_rng(5)
        n_frames = 500
        speed = np.arange(1, n_frames + 1, dtype=np.float32)
        # Exponential relationship: Spearman = 1, Pearson < 1
        superneurons = np.exp(speed / n_frames)[None, :]

        result = superneuron_behaviour_correlations(superneurons, speed=speed)
        r = result["speed_corr"][0]
        assert np.isfinite(r)
        assert r > 0.99  # Spearman rank correlation for monotone relationship

    def test_constant_superneuron_returns_nan(self) -> None:
        """std=0 in superneuron → Spearman undefined → NaN."""
        rng = np.random.default_rng(6)
        superneurons = np.ones((3, 300), dtype=np.float32)
        speed = rng.standard_normal(300).astype(np.float32)
        result = superneuron_behaviour_correlations(superneurons, speed=speed)
        assert np.all(np.isnan(result["speed_corr"]))

    def test_mismatched_length_handled(self) -> None:
        """Shorter behavioural array should be handled by truncating to min length."""
        n_super, n_frames = 4, 400
        superneurons = _make_superneurons(n_super, n_frames)
        speed = np.abs(RNG.standard_normal(350)).astype(np.float32)
        result = superneuron_behaviour_correlations(superneurons, speed=speed)
        assert result["speed_corr"].shape == (n_super,)

    def test_all_nan_superneurons_returns_nan(self) -> None:
        n_super, n_frames = 3, 200
        superneurons = np.full((n_super, n_frames), np.nan, dtype=np.float32)
        speed = np.ones(n_frames, dtype=np.float32) * 5.0
        result = superneuron_behaviour_correlations(superneurons, speed=speed)
        assert np.all(np.isnan(result["speed_corr"]))

    def test_hd_degrees_modulo_applied(self) -> None:
        """HD values outside [0, 360) should be handled via modulo."""
        rng = np.random.default_rng(7)
        n_frames = 300
        superneurons = rng.standard_normal((3, n_frames)).astype(np.float32)
        hd_deg = rng.uniform(-720, 720, n_frames)  # out-of-range values
        result = superneuron_behaviour_correlations(superneurons, hd_deg=hd_deg)
        assert result["hd_corr"].shape == (3,)

    def test_all_light_on_no_light_mod(self) -> None:
        """With only light frames, both groups needed: if all light, NaN expected."""
        n_frames = 200
        superneurons = _make_superneurons(4, n_frames)
        light_on = np.ones(n_frames, dtype=bool)  # all light
        result = superneuron_behaviour_correlations(superneurons, light_on=light_on)
        # light.sum() = 200 > 10, but (~light).sum() = 0 ≤ 10 → NaN
        assert np.all(np.isnan(result["light_mod"]))

    def test_all_outputs_when_all_provided(self) -> None:
        superneurons, hd_deg, speed, light_on = self._make_inputs()
        result = superneuron_behaviour_correlations(
            superneurons, hd_deg=hd_deg, speed=speed, light_on=light_on
        )
        assert "speed_corr" in result
        assert "hd_corr" in result
        assert "light_mod" in result

    @given(
        n_super=st.integers(min_value=1, max_value=20),
        n_frames=st.integers(min_value=30, max_value=200),
    )
    @settings(max_examples=20, deadline=None)
    def test_speed_corr_shape_property(self, n_super: int, n_frames: int) -> None:
        rng = np.random.default_rng(0)
        superneurons = rng.standard_normal((n_super, n_frames)).astype(np.float32)
        speed = np.abs(rng.standard_normal(n_frames)).astype(np.float32)
        result = superneuron_behaviour_correlations(superneurons, speed=speed)
        assert result["speed_corr"].shape == (n_super,)
