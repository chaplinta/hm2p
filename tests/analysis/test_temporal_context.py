"""Tests for hm2p.analysis.temporal_context (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from hm2p.analysis import temporal_context as tc

FPS = 10.0


def _drift_population(n_rois: int = 12, n_bins: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_bins)
    slopes = rng.choice([-1.0, 1.0], n_rois) * rng.uniform(0.5, 1.5, n_rois)
    return slopes[:, None] * t[None, :] + 0.1 * rng.standard_normal((n_rois, n_bins))


def _noise_population(n_rois: int = 12, n_bins: int = 60, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n_rois, n_bins))


def _epoch_labels(n_epochs: int = 8, bins_per_epoch: int = 6) -> tuple[np.ndarray, np.ndarray]:
    idx = np.repeat(np.arange(n_epochs), bins_per_epoch)
    return idx, idx % 2 == 0


def _epoch_specific_population(seed: int = 2, n_rois: int = 10) -> tuple[np.ndarray, ...]:
    """Each epoch has its own random population pattern (no monotonic drift)."""
    rng = np.random.default_rng(seed)
    idx, light = _epoch_labels()
    patterns = rng.standard_normal((n_rois, idx.max() + 1)) * 2.0
    x = patterns[:, idx] + 0.3 * rng.standard_normal((n_rois, idx.size))
    return x, idx, light


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_as_rng_variants(self) -> None:
        g = np.random.default_rng(0)
        assert tc._as_rng(g) is g
        assert isinstance(tc._as_rng(3), np.random.Generator)
        assert isinstance(tc._as_rng(None), np.random.Generator)

    def test_as_2d(self) -> None:
        assert tc._as_2d(np.zeros(4), "x").shape == (1, 4)
        assert tc._as_2d(np.zeros((2, 4)), "x").shape == (2, 4)
        with pytest.raises(ValueError):
            tc._as_2d(np.zeros((2, 2, 2)), "x")

    def test_perm_p_greater(self) -> None:
        assert tc._perm_p_greater(5.0, np.arange(4.0)) == pytest.approx(1 / 5)
        assert tc._perm_p_greater(0.0, np.arange(4.0)) == pytest.approx(5 / 5)
        assert np.isnan(tc._perm_p_greater(np.nan, np.arange(4.0)))
        assert np.isnan(tc._perm_p_greater(1.0, np.array([np.nan])))

    def test_zscore_rows_constant(self) -> None:
        z = tc._zscore_rows(np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]))
        assert np.all(z[0] == 0)
        assert z[1].mean() == pytest.approx(0)

    def test_spearman(self) -> None:
        assert tc._spearman(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)
        assert np.isnan(tc._spearman(np.ones(5), np.arange(5.0)))
        assert np.isnan(tc._spearman(np.arange(2.0), np.arange(2.0)))

    def test_shift_values_range(self) -> None:
        s = tc._shift_values(50, 100, np.random.default_rng(0))
        assert s.min() >= 5 and s.max() < 45
        small = tc._shift_values(2, 5, np.random.default_rng(0))
        assert np.all(small >= 1)

    def test_circular_partition_splits_wrapped_block(self) -> None:
        lab = np.repeat([0, 1, 2], 3)
        np.testing.assert_array_equal(tc._circular_partition(lab, 1), [-1, 0, 0, 0, 1, 1, 1, 2, 2])
        np.testing.assert_array_equal(tc._circular_partition(lab, 3), [2, 2, 2, 0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(tc._circular_partition(np.zeros(3), 1), [0, 0, 0])

    def test_subtract_state_median(self) -> None:
        x = np.array([[1.0, 3.0, 10.0, 14.0]])
        st = np.array([True, True, False, False])
        np.testing.assert_allclose(tc._subtract_state_median(x, st), [[-1, 1, -2, 2]])
        assert tc._subtract_state_median(x, None) is x

    def test_epoch_accuracy(self) -> None:
        z = np.array([[0.0], [0.1], [5.0], [5.1], [9.0]])
        acc, chance, n_used, n_cls = tc._epoch_accuracy(z, np.array([0, 0, 1, 1, 2]), np.zeros(5))
        assert acc == 1.0 and chance == 0.5 and n_used == 4 and n_cls == 2
        acc2, _, n2, _ = tc._epoch_accuracy(z, np.arange(5), np.zeros(5))
        assert np.isnan(acc2) and n2 == 0

    def test_kw_epsilon_sq(self) -> None:
        rng = np.random.default_rng(1)
        x = np.vstack([rng.standard_normal(12), np.ones(12)])
        lab = np.repeat(np.arange(3), 4)
        e = tc._kw_epsilon_sq(x, lab)
        h = stats.kruskal(*[x[0, lab == k] for k in range(3)]).statistic
        assert e[0] == pytest.approx(h / 11)
        assert np.isnan(e[1])

    def test_valid_columns(self) -> None:
        x = np.array([[1.0, np.nan], [1.0, 1.0]])
        np.testing.assert_array_equal(tc._valid_columns(x), [True, False])

    def test_fold_ids(self) -> None:
        f = tc._fold_ids(10, 5, True, np.random.default_rng(0))
        np.testing.assert_array_equal(f, np.repeat(np.arange(5), 2))
        g = tc._fold_ids(10, 5, False, np.random.default_rng(0))
        assert sorted(g.tolist()) == sorted(f.tolist())

    def test_ridge_cv_predict_skips_tiny_training_fold(self) -> None:
        x = np.arange(6.0)[:, None]
        pred = tc._ridge_cv_predict(x, np.arange(6.0), np.array([0, 0, 0, 0, 0, 1]))
        assert np.isnan(pred[:5]).all() and np.isfinite(pred[5])

    def test_rank_rows(self) -> None:
        r = tc._rank_rows(np.array([[3.0, 1.0, 2.0]]))
        np.testing.assert_array_equal(r, [[3.0, 1.0, 2.0]])

    def test_loo_centroid_perfect_separation(self) -> None:
        z = np.array([[0.0], [0.1], [10.0], [10.1]])
        assert tc._loo_centroid_correct(z, np.array([0, 0, 1, 1])).all()

    def test_kruskal_matches_scipy(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 30))
        x[0, :5] = 0.0  # ties
        lab = np.repeat(np.arange(3), 10)
        ranks = tc._rank_rows(x)
        _, c = np.unique(ranks[0], return_counts=True)
        tie = np.array([1.0 - (c**3 - c).sum() / (30**3 - 30)])
        h = tc._kruskal_h_rows(ranks, lab, tie)[0]
        expected = stats.kruskal(*[x[0, lab == k] for k in range(3)]).statistic
        assert h == pytest.approx(expected)


# ---------------------------------------------------------------------------
# binning and epochs
# ---------------------------------------------------------------------------


class TestBinActivity:
    def test_means_and_centres(self) -> None:
        sig = np.arange(40.0)[None, :]  # 4 s at 10 Hz
        binned, centres, valid = tc.bin_activity(sig, FPS, bin_s=1.0)
        np.testing.assert_allclose(binned[0], [4.5, 14.5, 24.5, 34.5])
        np.testing.assert_allclose(centres, [0.5, 1.5, 2.5, 3.5])
        assert valid.all()

    def test_partial_last_bin_invalid(self) -> None:
        sig = np.ones((2, 24))  # 2 full 1-s bins + 4 frames
        binned, _, valid = tc.bin_activity(sig, FPS, bin_s=1.0)
        np.testing.assert_array_equal(valid, [True, True, False])
        assert np.isnan(binned[:, 2]).all()

    def test_mask_below_half_invalidates(self) -> None:
        sig = np.ones(20)
        mask = np.ones(20, dtype=bool)
        mask[:6] = False  # bin 0 keeps 4/10 frames
        binned, _, valid = tc.bin_activity(sig, FPS, bin_s=1.0, mask=mask)
        np.testing.assert_array_equal(valid, [False, True])

    def test_nan_frames_excluded(self) -> None:
        sig = np.array([[1.0, np.nan, 3.0, 5.0]])
        binned, _, valid = tc.bin_activity(sig, 1.0, bin_s=4.0)
        assert valid[0]
        assert binned[0, 0] == pytest.approx(3.0)

    def test_empty(self) -> None:
        binned, centres, valid = tc.bin_activity(np.zeros((3, 0)), FPS)
        assert binned.shape == (3, 0) and centres.size == 0 and valid.size == 0

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            tc.bin_activity(np.ones(10), 0.0)
        with pytest.raises(ValueError):
            tc.bin_activity(np.ones(10), FPS, bin_s=-1)
        with pytest.raises(ValueError):
            tc.bin_activity(np.ones(10), FPS, mask=np.ones(3, dtype=bool))

    @settings(max_examples=30, deadline=None)
    @given(
        n_frames=st.integers(1, 200),
        bin_s=st.floats(0.2, 5.0),
        value=st.floats(-1e3, 1e3, allow_nan=False),
    )
    def test_constant_signal_property(self, n_frames: int, bin_s: float, value: float) -> None:
        binned, centres, valid = tc.bin_activity(np.full(n_frames, value), FPS, bin_s=bin_s)
        assert binned.shape[1] == centres.size == valid.size
        np.testing.assert_allclose(binned[0, valid], value, rtol=1e-9, atol=1e-9)
        assert np.all(np.diff(centres) > 0)


class TestEpochIndex:
    def test_epochs(self) -> None:
        light = np.repeat([True, False, True], 20)  # 2 s each at 10 Hz
        centres = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        idx, el = tc.epoch_index(light, FPS, centres)
        np.testing.assert_array_equal(idx, [0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(el, [True, True, False, False, True, True])

    def test_clips_beyond_end(self) -> None:
        idx, _ = tc.epoch_index(np.ones(10, dtype=bool), FPS, np.array([100.0]))
        assert idx[0] == 0

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            tc.epoch_index(np.zeros(0, dtype=bool), FPS, np.array([0.5]))
        with pytest.raises(ValueError):
            tc.epoch_index(np.ones(5, dtype=bool), 0.0, np.array([0.5]))

    def test_pure_epoch_bins(self) -> None:
        light = np.repeat([True, False], [15, 25])  # switch at 1.5 s
        centres = np.array([0.5, 1.5, 2.5, 3.5])
        pure = tc.pure_epoch_bins(light, FPS, centres, 1.0)
        np.testing.assert_array_equal(pure, [True, False, True, True])


# ---------------------------------------------------------------------------
# time decoding
# ---------------------------------------------------------------------------


class TestTimeDecoding:
    def test_drift_decodes_time(self) -> None:
        x = _drift_population()
        t = np.arange(x.shape[1]) * 10.0
        res = tc.time_decoding_null(x, t, n_perms=40, rng=0)
        assert res["rho"] > 0.8
        assert res["p_rho"] < 0.05
        assert res["p_error"] < 0.05

    def test_noise_does_not_decode(self) -> None:
        x = _noise_population()
        t = np.arange(x.shape[1]) * 10.0
        res = tc.time_decoding_null(x, t, n_perms=40, rng=0)
        assert res["rho"] < 0.5
        assert res["p_rho"] > 0.05

    def test_random_folds(self) -> None:
        x = _drift_population()
        res = tc.time_decoding(x, np.arange(x.shape[1]) * 1.0, block=False, rng=1)
        assert res["rho"] > 0.8
        assert res["predicted"].shape == res["true"].shape

    def test_nan_bins_dropped(self) -> None:
        x = _drift_population()
        x[:, 3] = np.nan
        res = tc.time_decoding(x, np.arange(x.shape[1]) * 1.0)
        assert res["n_bins"] == x.shape[1] - 1

    def test_too_few_bins(self) -> None:
        res = tc.time_decoding(np.ones((2, 5)), np.arange(5.0))
        assert np.isnan(res["rho"]) and res["predicted"].size == 0
        null = tc.time_decoding_null(np.ones((2, 5)), np.arange(5.0), n_perms=3, rng=0)
        assert np.isnan(null["p_rho"])

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            tc.time_decoding(np.ones((2, 20)), np.arange(19.0))
        with pytest.raises(ValueError):
            tc.time_decoding(np.ones((2, 20)), np.arange(20.0), n_folds=1)


class TestMatchedN:
    def test_shape_and_values(self) -> None:
        x = _drift_population(n_rois=10)
        res = tc.matched_n_decoding(x, np.arange(x.shape[1]) * 1.0, n_cells=4, n_draws=5, rng=0)
        assert res["errors"].shape == (5,) and res["rhos"].shape == (5,)
        assert res["n_cells"] == 4 and res["n_draws"] == 5
        assert res["median_rho"] > 0.5

    def test_errors(self) -> None:
        x = np.ones((3, 20))
        with pytest.raises(ValueError):
            tc.matched_n_decoding(x, np.arange(20.0), n_cells=4)
        with pytest.raises(ValueError):
            tc.matched_n_decoding(x, np.arange(20.0), n_cells=2, n_draws=0)


# ---------------------------------------------------------------------------
# epoch identity and temporal distance
# ---------------------------------------------------------------------------


class TestEpochIdentity:
    def test_epoch_specific_cells(self) -> None:
        idx, light = _epoch_labels(n_epochs=16)
        rng = np.random.default_rng(2)
        x = rng.standard_normal((12, 16))[:, idx] + rng.standard_normal((12, idx.size))
        res = tc.epoch_identity_decoding(x, idx, light, n_perms=50, rng=0)
        assert res["accuracy"] > res["chance"] + 0.2
        assert res["chance"] == pytest.approx(1 / 8)
        assert res["p_value"] < 0.05
        assert res["n_epochs"] == 16

    def test_noise_near_chance(self) -> None:
        idx, light = _epoch_labels()
        x = _noise_population(n_bins=idx.size)
        res = tc.epoch_identity_decoding(x, idx, light, n_perms=50, rng=0)
        assert res["accuracy"] < 0.6
        assert res["p_value"] > 0.05

    def test_noisy_drift_not_epoch_specific(self) -> None:
        # Per-cell shifts keep each cell's drift magnitude: moderate drift in
        # noise is not reported as epoch identity.
        idx, light = _epoch_labels(n_epochs=16)
        rng = np.random.default_rng(4)
        t = np.linspace(0, 1, idx.size)
        x = rng.choice([-1, 1], 12)[:, None] * t + 0.2 * rng.standard_normal((12, idx.size))
        res = tc.epoch_identity_decoding(x, idx, light, n_perms=50, rng=0)
        assert res["accuracy"] > res["chance"]
        assert res["p_value"] > 0.05

    def test_light_only_cells_not_epoch_specific(self) -> None:
        idx, light = _epoch_labels(n_epochs=16)
        rng = np.random.default_rng(3)
        x = 3.0 * light[None, :] + rng.standard_normal((12, idx.size))
        res = tc.epoch_identity_decoding(x, idx, light, n_perms=50, rng=0)
        assert res["p_value"] > 0.05

    def test_without_light(self) -> None:
        x, idx, _ = _epoch_specific_population()
        res = tc.epoch_identity_decoding(x, idx, None, n_perms=0)
        assert res["chance"] == pytest.approx(1 / 8)
        assert np.isnan(res["p_value"])

    def test_too_few_valid_bins(self) -> None:
        x = np.ones((2, 6))
        x[:, 3:] = np.nan
        res = tc.epoch_identity_decoding(x, np.repeat([0, 1], 3), None, n_perms=2)
        assert np.isnan(res["accuracy"]) and res["null_minus_chance"].shape == (2,)

    def test_unusable(self) -> None:
        res = tc.epoch_identity_decoding(np.ones((2, 4)), np.arange(4), None, n_perms=2)
        assert np.isnan(res["accuracy"]) and res["n_bins"] == 0

    def test_shape_error(self) -> None:
        with pytest.raises(ValueError):
            tc.epoch_identity_decoding(np.ones((2, 4)), np.arange(3))


class TestTemporalDistance:
    def test_drift_positive(self) -> None:
        idx, light = _epoch_labels(n_epochs=10)
        x = _drift_population(n_bins=idx.size)
        res = tc.temporal_distance(x, idx, light, n_perms=200, rng=0)
        assert res["rho"] > 0.5
        assert res["p_value"] < 0.05
        assert res["n_pairs"] == 2 * (5 * 4 // 2)

    def test_noise_null(self) -> None:
        idx, light = _epoch_labels(n_epochs=10)
        x = _noise_population(n_bins=idx.size, seed=5)
        res = tc.temporal_distance(x, idx, light, n_perms=200, rng=0)
        assert res["p_value"] > 0.05

    def test_too_few_pairs(self) -> None:
        idx = np.repeat(np.arange(2), 3)
        res = tc.temporal_distance(_noise_population(n_bins=6), idx, None, n_perms=5)
        assert np.isnan(res["rho"]) and res["n_pairs"] == 1
        empty = tc.temporal_distance(np.ones((2, 2)), np.arange(2), None)
        assert empty["n_pairs"] == 0

    def test_shape_error(self) -> None:
        with pytest.raises(ValueError):
            tc.temporal_distance(np.ones((2, 4)), np.arange(3))


# ---------------------------------------------------------------------------
# per-cell measures
# ---------------------------------------------------------------------------


class TestPerCell:
    def test_drift_sign(self) -> None:
        n = 60
        t = np.arange(n, dtype=float)
        rng = np.random.default_rng(0)
        x = np.vstack([t, -t, rng.standard_normal(n), np.ones(n)])
        res = tc.per_cell_drift(x, n_perms=200, rng=0)
        assert res["rho"][0] == pytest.approx(1.0)
        assert res["rho"][1] == pytest.approx(-1.0)
        assert res["p_value"][0] < 0.05 and res["p_value"][1] < 0.05
        assert res["p_value"][2] > 0.05
        assert np.isnan(res["rho"][3]) and np.isnan(res["p_value"][3])

    def test_drift_matches_scipy(self) -> None:
        x = np.random.default_rng(3).standard_normal((2, 30))
        res = tc.per_cell_drift(x, n_perms=5, rng=0)
        for i in range(2):
            assert res["rho"][i] == pytest.approx(stats.spearmanr(x[i], np.arange(30)).statistic)

    def test_drift_too_short(self) -> None:
        res = tc.per_cell_drift(np.ones((2, 2)))
        assert np.isnan(res["rho"]).all()

    def test_epoch_selectivity(self) -> None:
        rng = np.random.default_rng(2)
        idx = np.repeat(np.arange(8), 30)
        x = rng.standard_normal((3, 8))[:, idx] * 2 + 0.5 * rng.standard_normal((3, idx.size))
        x = np.vstack([x, np.ones(idx.size)])
        res = tc.per_cell_epoch_selectivity(x, idx, idx % 2 == 0, n_perms=100, rng=0)
        assert np.all(res["p_value"][:3] < 0.05)
        assert np.isnan(res["H"][3]) and np.isnan(res["p_value"][3])
        assert res["epsilon_sq"][0] == pytest.approx(res["H"][0] / (idx.size - 1))

    def test_epoch_selectivity_h_matches_scipy(self) -> None:
        x, idx, _ = _epoch_specific_population(n_rois=1)
        res = tc.per_cell_epoch_selectivity(x, idx, None, n_perms=0)
        expected = stats.kruskal(*[x[0, idx == k] for k in np.unique(idx)]).statistic
        assert res["H"][0] == pytest.approx(expected)
        assert np.isnan(res["p_value"][0])

    def test_epoch_selectivity_noise_and_light(self) -> None:
        rng = np.random.default_rng(7)
        idx = np.repeat(np.arange(8), 30)
        light = idx % 2 == 0
        noise = rng.standard_normal((20, idx.size))
        light_cells = 3.0 * light[None, :] + rng.standard_normal((20, idx.size))
        for x in (noise, light_cells):
            res = tc.per_cell_epoch_selectivity(x, idx, light, n_perms=100, rng=0)
            assert np.mean(res["p_value"] < 0.05) <= 0.15

    def test_epoch_selectivity_single_epoch(self) -> None:
        res = tc.per_cell_epoch_selectivity(np.ones((2, 5)), np.zeros(5, dtype=int))
        assert np.isnan(res["H"]).all() and np.isnan(res["epsilon_sq"]).all()
        with pytest.raises(ValueError):
            tc.per_cell_epoch_selectivity(np.ones((2, 5)), np.zeros(4, dtype=int))

    def test_first_vs_later_dark(self) -> None:
        fps = 10.0
        light = np.tile(np.repeat([True, False], 100), 3)
        sig = np.zeros((2, light.size))
        first_dark = np.flatnonzero(~light)[:100]
        sig[0, first_dark] = 1.0
        res = tc.first_vs_later_dark(sig, light, np.zeros(light.size, dtype=bool), fps)
        assert res["n_epochs"] == 3
        assert res["difference"][0] == pytest.approx(1.0)
        assert res["difference"][1] == pytest.approx(0.0)

    def test_first_vs_later_dark_all_bad(self) -> None:
        light = np.tile(np.repeat([True, False], 100), 2)
        res = tc.first_vs_later_dark(
            np.ones(light.size), light, np.ones(light.size, dtype=bool), 10.0
        )
        assert np.isnan(res["first_value"][0])
