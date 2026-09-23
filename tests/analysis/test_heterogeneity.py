"""Tests for hm2p.analysis.heterogeneity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hm2p.analysis import heterogeneity as het


def _two_group_data(
    n_animals_per_group: int = 4,
    n_cells_per_animal: int = 8,
    n_features: int = 4,
    separation: float = 0.0,
    spread_a: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic standardised feature matrix with animal-nested group labels.

    Group ``nonpenk`` is offset by *separation* and scaled by *spread_a*.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    animals: list[str] = []

    for group, offset, scale in (
        (het.GROUP_A, separation, spread_a),
        (het.GROUP_B, 0.0, 1.0),
    ):
        for a in range(n_animals_per_group):
            block = rng.standard_normal((n_cells_per_animal, n_features)) * scale + offset
            rows.append(block)
            labels.extend([group] * n_cells_per_animal)
            animals.extend([f"{group}_{a}"] * n_cells_per_animal)

    return np.vstack(rows), np.asarray(labels), np.asarray(animals)


# ---------------------------------------------------------------------------
# _as_rng
# ---------------------------------------------------------------------------


def test_as_rng_accepts_generator_seed_and_none() -> None:
    gen = np.random.default_rng(3)
    assert het._as_rng(gen) is gen
    assert isinstance(het._as_rng(7), np.random.Generator)
    assert isinstance(het._as_rng(None), np.random.Generator)


# ---------------------------------------------------------------------------
# prepare_feature_matrix
# ---------------------------------------------------------------------------


def _feature_df(n: int = 20, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "animal_id": [f"a{i % 4}" for i in range(n)],
            "celltype": ["penk" if i % 4 < 2 else "nonpenk" for i in range(n)],
            "roi_idx": np.arange(n),
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        }
    )


def test_prepare_feature_matrix_standardises_columns() -> None:
    df = _feature_df()
    x, meta, cols = het.prepare_feature_matrix(df, ["f1", "f2"])
    assert x.shape == (len(df), 2)
    assert cols == ["f1", "f2"]
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(x.std(axis=0), 1.0, atol=1e-12)
    assert list(meta.columns) == ["animal_id", "celltype", "roi_idx"]
    assert len(meta) == len(df)


def test_prepare_feature_matrix_drops_sparse_column() -> None:
    df = _feature_df()
    df["f3"] = np.nan
    df.loc[:2, "f3"] = 1.0
    _, _, cols = het.prepare_feature_matrix(df, ["f1", "f2", "f3"])
    assert "f3" not in cols


def test_prepare_feature_matrix_drops_rows_with_nan() -> None:
    df = _feature_df()
    df.loc[0, "f1"] = np.nan
    x, meta, _ = het.prepare_feature_matrix(df, ["f1", "f2"])
    assert x.shape[0] == len(df) - 1
    assert len(meta) == len(df) - 1


def test_prepare_feature_matrix_drops_zero_variance_column() -> None:
    df = _feature_df()
    df["f4"] = 3.0
    _, _, cols = het.prepare_feature_matrix(df, ["f1", "f4"])
    assert cols == ["f1"]


def test_prepare_feature_matrix_ignores_missing_columns() -> None:
    df = _feature_df()
    _, _, cols = het.prepare_feature_matrix(df, ["f1", "not_a_column"])
    assert cols == ["f1"]


def test_prepare_feature_matrix_raises_without_any_column() -> None:
    with pytest.raises(ValueError, match="none of the requested"):
        het.prepare_feature_matrix(_feature_df(), ["nope"])


def test_prepare_feature_matrix_raises_when_all_columns_sparse() -> None:
    df = _feature_df(n=10)
    df["f1"] = np.nan
    df["f2"] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        het.prepare_feature_matrix(df, ["f1", "f2"])


def test_prepare_feature_matrix_raises_when_no_row_complete() -> None:
    df = _feature_df(n=10)
    df.loc[:4, "f1"] = np.nan
    df.loc[5:, "f2"] = np.nan
    with pytest.raises(ValueError, match="no row has finite values"):
        het.prepare_feature_matrix(df, ["f1", "f2"], min_valid_fraction=0.4)


def test_prepare_feature_matrix_raises_when_all_zero_variance() -> None:
    df = _feature_df(n=10)
    df["f1"] = 1.0
    df["f2"] = 2.0
    with pytest.raises(ValueError, match="zero variance"):
        het.prepare_feature_matrix(df, ["f1", "f2"])


# ---------------------------------------------------------------------------
# group_dispersion
# ---------------------------------------------------------------------------


def test_group_dispersion_known_geometry() -> None:
    x = np.array([[-1.0, 0.0], [1.0, 0.0], [-3.0, 0.0], [3.0, 0.0]])
    labels = np.array(["penk", "penk", "nonpenk", "nonpenk"])
    out = het.group_dispersion(x, labels)
    assert out["penk_mean"] == pytest.approx(1.0)
    assert out["nonpenk_mean"] == pytest.approx(3.0)
    assert out["penk_median"] == pytest.approx(1.0)
    assert out["penk_n"] == 2.0


def test_group_dispersion_single_cell_group_is_zero() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    labels = np.array(["a", "b", "b"])
    out = het.group_dispersion(x, labels)
    assert out["a_mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _permute_labels_by_animal
# ---------------------------------------------------------------------------


def test_permute_labels_by_animal_keeps_animals_together() -> None:
    labels = np.array(["penk"] * 4 + ["nonpenk"] * 4)
    animals = np.array(["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        perm = het._permute_labels_by_animal(labels, animals, rng)
        for animal in np.unique(animals):
            assert len(set(perm[animals == animal])) == 1
        assert sorted(perm.tolist()) == sorted(labels.tolist())


def test_permute_labels_by_animal_can_change_assignment() -> None:
    labels = np.array(["penk"] * 3 + ["nonpenk"] * 3)
    animals = np.array(["a1", "a2", "a3", "b1", "b2", "b3"])
    rng = np.random.default_rng(1)
    perms = {tuple(het._permute_labels_by_animal(labels, animals, rng)) for _ in range(50)}
    assert len(perms) > 1


# ---------------------------------------------------------------------------
# p-value helpers
# ---------------------------------------------------------------------------


def test_two_sided_ratio_p_extreme_observation() -> None:
    null = np.ones(99)
    assert het._two_sided_ratio_p(10.0, null) == pytest.approx(1 / 100)


def test_two_sided_ratio_p_central_observation() -> None:
    null = np.exp(np.linspace(-2, 2, 99))
    assert het._two_sided_ratio_p(1.0, null) == pytest.approx(1.0)


def test_two_sided_ratio_p_invalid_observation_is_nan() -> None:
    assert np.isnan(het._two_sided_ratio_p(np.nan, np.ones(10)))
    assert np.isnan(het._two_sided_ratio_p(-1.0, np.ones(10)))


def test_two_sided_ratio_p_no_usable_null_is_nan() -> None:
    assert np.isnan(het._two_sided_ratio_p(2.0, np.full(5, np.nan)))


def test_one_sided_p_behaviour() -> None:
    null = np.arange(99, dtype=float)
    assert het._one_sided_p(1000.0, null) == pytest.approx(1 / 100)
    assert het._one_sided_p(-1.0, null) == pytest.approx(1.0)
    assert np.isnan(het._one_sided_p(np.nan, null))
    assert np.isnan(het._one_sided_p(1.0, np.full(4, np.nan)))


# ---------------------------------------------------------------------------
# _dispersion_ratio / dispersion_ratio_test
# ---------------------------------------------------------------------------


def test_dispersion_ratio_known_value() -> None:
    x = np.array([[-1.0], [1.0], [-3.0], [3.0]])
    labels = np.array(["penk", "penk", "nonpenk", "nonpenk"])
    assert het._dispersion_ratio(x, labels, "nonpenk", "penk") == pytest.approx(3.0)


def test_dispersion_ratio_missing_group_is_nan() -> None:
    x = np.array([[0.0], [1.0]])
    labels = np.array(["penk", "penk"])
    assert np.isnan(het._dispersion_ratio(x, labels, "nonpenk", "penk"))


def test_dispersion_ratio_zero_denominator_is_nan() -> None:
    x = np.array([[0.0], [0.0], [1.0], [-1.0]])
    labels = np.array(["penk", "penk", "nonpenk", "nonpenk"])
    assert np.isnan(het._dispersion_ratio(x, labels, "nonpenk", "penk"))


def test_dispersion_ratio_test_identical_groups_not_significant() -> None:
    x, labels, animals = _two_group_data(separation=0.0, spread_a=1.0, seed=10)
    out = het.dispersion_ratio_test(x, labels, animals, n_perms=199, rng=0)
    assert out["n_perms"] == 199
    assert out["null"].shape == (199,)
    assert out["p_value"] > 0.05
    assert out["observed"] == pytest.approx(1.0, abs=0.3)
    assert "nonpenk_mean" in out["dispersion"]


def test_dispersion_ratio_test_detects_larger_spread() -> None:
    x, labels, animals = _two_group_data(spread_a=4.0, seed=11)
    out = het.dispersion_ratio_test(x, labels, animals, n_perms=199, rng=0)
    assert out["observed"] > 2.0
    assert out["p_value"] < 0.05


def test_dispersion_ratio_test_raises_on_missing_group() -> None:
    x, labels, animals = _two_group_data(seed=12)
    labels = np.full(labels.shape, "penk")
    with pytest.raises(ValueError, match="not present"):
        het.dispersion_ratio_test(x, labels, animals, n_perms=5, rng=0)


# ---------------------------------------------------------------------------
# energy distance
# ---------------------------------------------------------------------------


def test_energy_distance_from_matrix_identical_groups_near_zero() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((40, 3))
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    mask_a = np.zeros(40, dtype=bool)
    mask_a[::2] = True
    value = het._energy_distance_from_matrix(dist, mask_a, ~mask_a)
    assert abs(value) < 0.5


def test_energy_distance_from_matrix_empty_group_is_nan() -> None:
    dist = np.zeros((4, 4))
    mask = np.zeros(4, dtype=bool)
    assert np.isnan(het._energy_distance_from_matrix(dist, mask, ~mask))


def test_energy_distance_from_matrix_increases_with_separation() -> None:
    a = np.zeros((10, 2))
    b = np.full((10, 2), 5.0)
    x = np.vstack([a, b])
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    mask_a = np.zeros(20, dtype=bool)
    mask_a[:10] = True
    assert het._energy_distance_from_matrix(dist, mask_a, ~mask_a) > 5.0


def test_energy_distance_test_identical_groups_not_significant() -> None:
    x, labels, animals = _two_group_data(separation=0.0, seed=13)
    out = het.energy_distance_test(x, labels, animals, n_perms=99, rng=0)
    assert out["p_value"] > 0.05
    assert out["null"].shape == (99,)
    assert out["n_perms"] == 99


def test_energy_distance_test_separated_groups_significant() -> None:
    # Six animals per group: with only four the smallest attainable
    # animal-level permutation p-value is bounded away from 0.05.
    x, labels, animals = _two_group_data(
        n_animals_per_group=6, n_cells_per_animal=6, separation=6.0, seed=14
    )
    out = het.energy_distance_test(x, labels, animals, n_perms=99, rng=0)
    assert out["observed"] > 1.0
    assert out["p_value"] < 0.05


def test_energy_distance_test_raises_on_missing_group() -> None:
    x, labels, animals = _two_group_data(seed=15)
    labels = np.full(labels.shape, "penk")
    with pytest.raises(ValueError, match="not present"):
        het.energy_distance_test(x, labels, animals, n_perms=5, rng=0)


# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------


def test_gmm_components_unimodal_prefers_one() -> None:
    rng = np.random.default_rng(16)
    x = rng.standard_normal((120, 2))
    out = het.gmm_components(x, max_components=4, rng=0)
    assert out["best_k"] == 1
    assert out["n_components_tried"] == [1, 2, 3, 4]
    assert len(out["bic"]) == 4
    assert out["n_cells"] == 120


def test_gmm_components_bimodal_prefers_more_than_one() -> None:
    rng = np.random.default_rng(17)
    x = np.vstack(
        [
            rng.standard_normal((80, 2)),
            rng.standard_normal((80, 2)) + 12.0,
        ]
    )
    assert het.gmm_components(x, max_components=4, rng=0)["best_k"] >= 2


def test_gmm_components_clips_to_number_of_cells() -> None:
    rng = np.random.default_rng(18)
    out = het.gmm_components(rng.standard_normal((2, 2)), max_components=6, rng=0)
    assert out["n_components_tried"] == [1, 2]


def test_gmm_components_rejects_zero_components() -> None:
    with pytest.raises(ValueError, match="max_components"):
        het.gmm_components(np.zeros((5, 2)), max_components=0)


def test_gmm_components_handles_failed_fits() -> None:
    # A single cell cannot support any mixture fit.
    out = het.gmm_components(np.zeros((1, 2)), max_components=3, rng=0)
    assert out["n_components_tried"] == [1]
    assert out["best_k"] in (1, -1)


def test_gmm_components_by_group_returns_one_entry_per_group() -> None:
    x, labels, _ = _two_group_data(seed=19)
    out = het.gmm_components_by_group(x, labels, max_components=3, rng=0)
    assert set(out) == {het.GROUP_A, het.GROUP_B}
    assert all("best_k" in v for v in out.values())


# ---------------------------------------------------------------------------
# classifiers
# ---------------------------------------------------------------------------


def test_make_classifier_returns_estimators() -> None:
    assert hasattr(het._make_classifier("logistic", 0), "fit")
    assert hasattr(het._make_classifier("gbm", 0), "fit")


def test_make_classifier_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="logistic"):
        het._make_classifier("svm", 0)


def test_balanced_accuracy_empty_is_nan() -> None:
    assert np.isnan(het._balanced_accuracy(np.asarray([]), np.asarray([])))


def test_balanced_accuracy_perfect_is_one() -> None:
    y = np.asarray(["a", "a", "b", "b"])
    assert het._balanced_accuracy(y, y) == pytest.approx(1.0)


def test_loao_predictions_skips_single_class_training_sets() -> None:
    x = np.arange(8, dtype=np.float64).reshape(8, 1)
    labels = np.asarray(["penk"] * 4 + ["nonpenk"] * 4)
    animals = np.asarray(["a"] * 4 + ["b"] * 4)
    out = het._loao_predictions(x, labels, animals, "logistic", 0, False, 1)
    assert out["n_folds"] == 0
    assert np.isnan(out["per_animal_accuracy"]["a"])
    assert np.isnan(out["importance"]).all()


def test_loao_classifier_separates_distinct_groups() -> None:
    x, labels, animals = _two_group_data(separation=6.0, seed=20)
    out = het.loao_classifier(x, labels, animals, rng=0, feature_names=["f0", "f1", "f2", "f3"])
    assert out["balanced_accuracy"] > 0.9
    assert out["n_folds"] == 8
    assert len(out["per_animal_accuracy"]) == 8
    assert out["importance"].shape == (4,)
    assert set(out["importance_by_feature"]) == {"f0", "f1", "f2", "f3"}
    assert out["model"] == "logistic"
    assert out["y_true"].shape == out["y_pred"].shape


def test_loao_classifier_at_chance_for_identical_groups() -> None:
    x, labels, animals = _two_group_data(separation=0.0, seed=21)
    out = het.loao_classifier(x, labels, animals, rng=0)
    assert 0.2 < out["balanced_accuracy"] < 0.8
    assert out["importance_by_feature"] == {}


def test_loao_classifier_gbm_model_runs() -> None:
    x, labels, animals = _two_group_data(
        n_animals_per_group=2, n_cells_per_animal=10, separation=6.0, seed=22
    )
    out = het.loao_classifier(x, labels, animals, model="gbm", rng=0, n_repeats=2)
    assert out["model"] == "gbm"
    assert np.isfinite(out["balanced_accuracy"])


def test_classifier_permutation_test_separated_groups_significant() -> None:
    x, labels, animals = _two_group_data(
        n_animals_per_group=6, n_cells_per_animal=6, separation=6.0, seed=23
    )
    out = het.classifier_permutation_test(x, labels, animals, n_perms=39, rng=0)
    assert out["observed"] > 0.9
    assert out["p_value"] < 0.05
    assert out["null"].shape == (39,)
    assert out["n_perms"] == 39


def test_classifier_permutation_test_identical_groups_not_significant() -> None:
    x, labels, animals = _two_group_data(separation=0.0, seed=24)
    out = het.classifier_permutation_test(x, labels, animals, n_perms=39, rng=0)
    assert out["p_value"] > 0.05


# ---------------------------------------------------------------------------
# penk_like_fraction
# ---------------------------------------------------------------------------


def test_penk_like_fraction_separated_groups_have_few_reference_neighbours() -> None:
    x, labels, _ = _two_group_data(separation=20.0, seed=25)
    out = het.penk_like_fraction(x, labels, k=5)
    assert out["median_fraction"] == pytest.approx(0.0)
    assert out["fraction_majority_reference"] == pytest.approx(0.0)
    assert out["k"] == 5
    assert out["n_target"] == 32
    assert out["baseline"] == pytest.approx(0.5)


def test_penk_like_fraction_intermingled_groups_match_baseline() -> None:
    x, labels, _ = _two_group_data(separation=0.0, seed=26)
    out = het.penk_like_fraction(x, labels, k=10)
    assert out["fractions"].shape == (32,)
    assert out["median_fraction"] == pytest.approx(0.5, abs=0.25)


def test_penk_like_fraction_clips_k_to_available_neighbours() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    labels = np.asarray(["nonpenk", "penk", "penk"])
    out = het.penk_like_fraction(x, labels, k=50)
    assert out["k"] == 2


def test_penk_like_fraction_rejects_small_k() -> None:
    x, labels, _ = _two_group_data(seed=27)
    with pytest.raises(ValueError, match="k must be"):
        het.penk_like_fraction(x, labels, k=0)


def test_penk_like_fraction_rejects_missing_group() -> None:
    x, labels, _ = _two_group_data(seed=28)
    with pytest.raises(ValueError, match="not present"):
        het.penk_like_fraction(x, np.full(labels.shape, "penk"))


def test_penk_like_fraction_rejects_single_cell() -> None:
    with pytest.raises(ValueError, match="at least two cells"):
        het.penk_like_fraction(
            np.zeros((1, 2)),
            np.asarray(["penk"]),
            k=1,
            target_group="penk",
            reference_group="penk",
        )


# ---------------------------------------------------------------------------
# variance_ratio_permutation
# ---------------------------------------------------------------------------


def test_variance_ratio_known_value() -> None:
    values = np.asarray([0.0, 2.0, 0.0, 4.0])
    labels = np.asarray(["nonpenk", "nonpenk", "penk", "penk"])
    assert het._variance_ratio(values, labels, "nonpenk", "penk") == pytest.approx(0.25)


def test_variance_ratio_too_few_cells_is_nan() -> None:
    values = np.asarray([0.0, 1.0, 2.0])
    labels = np.asarray(["nonpenk", "penk", "penk"])
    assert np.isnan(het._variance_ratio(values, labels, "nonpenk", "penk"))


def test_variance_ratio_zero_denominator_is_nan() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 2.0])
    labels = np.asarray(["nonpenk", "nonpenk", "penk", "penk"])
    assert np.isnan(het._variance_ratio(values, labels, "nonpenk", "penk"))


def test_variance_ratio_permutation_equal_variance_not_significant() -> None:
    x, labels, animals = _two_group_data(separation=0.0, seed=29)
    out = het.variance_ratio_permutation(x[:, 0], labels, animals, n_perms=199, rng=0)
    assert out["p_value"] > 0.05
    assert out["n_perms"] == 199
    assert out["n_cells"] == x.shape[0]
    assert np.isfinite(out["var_a"])
    assert np.isfinite(out["var_b"])


def test_variance_ratio_permutation_detects_unequal_variance() -> None:
    x, labels, animals = _two_group_data(spread_a=5.0, seed=30)
    out = het.variance_ratio_permutation(x[:, 0], labels, animals, n_perms=199, rng=0)
    assert out["observed"] > 4.0
    assert out["p_value"] < 0.05


def test_variance_ratio_permutation_drops_non_finite_values() -> None:
    x, labels, animals = _two_group_data(separation=0.0, seed=31)
    values = x[:, 0].copy()
    values[:3] = np.nan
    out = het.variance_ratio_permutation(values, labels, animals, n_perms=19, rng=0)
    assert out["n_cells"] == values.size - 3


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    hnp.arrays(
        np.float64,
        (24,),
        elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    )
)
def test_variance_ratio_permutation_p_value_in_unit_interval(values: np.ndarray) -> None:
    labels = np.asarray(["nonpenk"] * 12 + ["penk"] * 12)
    animals = np.asarray([f"n{i // 4}" for i in range(12)] + [f"p{i // 4}" for i in range(12)])
    out = het.variance_ratio_permutation(values, labels, animals, n_perms=9, rng=0)
    p = out["p_value"]
    assert np.isnan(p) or 0.0 < p <= 1.0
