"""Tests for the Poisson GLM encoding models. Synthetic arrays only."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hm2p.analysis.encoding import (
    DesignMatrix,
    FittedGLM,
    _check_xy,
    _cv_scores_for,
    _fit_intercept_only,
    _fit_nemos,
    _fold_indices,
    _full_and_contributions,
    _gaussian_bumps,
    _raised_cosine,
    _resolve_range,
    _valid_subset,
    _wilcoxon_greater,
    build_design_matrix,
    circular_basis,
    cross_validated_deviance,
    deviance_explained,
    drop_one_contributions,
    encoding_profile,
    fit_poisson_glm,
    forward_selection,
    linear_basis,
    onehot_basis,
    poisson_deviance,
    population_encoding_profiles,
    position_basis,
    predict_rate,
    valid_rows,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures — a small "session" of behaviour plus driven cells
# ---------------------------------------------------------------------------

N_FRAMES = 600


def _behaviour(seed: int = 0):
    """Random-walk head direction plus speed and AHV, N_FRAMES samples."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0.0, 25.0, N_FRAMES)) % 360.0
    speed = np.abs(rng.normal(5.0, 2.0, N_FRAMES))
    ahv = np.concatenate([[0.0], np.diff(np.unwrap(np.deg2rad(hd))) * (180.0 / np.pi) * 9.6])
    return hd, speed, ahv


def _hd_cell(hd, seed: int = 1, kappa: float = 2.0, peak: float = 90.0):
    """Poisson counts driven only by head direction."""
    rng = np.random.default_rng(seed)
    rate = 0.3 + 2.0 * np.exp(kappa * np.cos(np.deg2rad(hd - peak))) / np.exp(kappa)
    return rng.poisson(rate).astype(float)


def _speed_cell(speed, seed: int = 2):
    """Poisson counts driven only by speed."""
    rng = np.random.default_rng(seed)
    return rng.poisson(0.2 + 0.35 * speed).astype(float)


@pytest.fixture(scope="module")
def behaviour():
    return _behaviour()


@pytest.fixture(scope="module")
def design(behaviour):
    hd, speed, _ = behaviour
    return build_design_matrix(hd_deg=hd, speed_cm_s=speed, n_hd_basis=8, n_speed_basis=5)


@pytest.fixture(scope="module")
def hd_counts(behaviour):
    return _hd_cell(behaviour[0])


@pytest.fixture(scope="module")
def speed_counts(behaviour):
    return _speed_cell(behaviour[1])


# ---------------------------------------------------------------------------
# _raised_cosine
# ---------------------------------------------------------------------------


class TestRaisedCosine:
    def test_peak_and_support(self):
        out = _raised_cosine(np.array([0.0, 0.5, 1.0, 2.0, -1.0]), width=1.0)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.5)
        assert out[2] == 0.0  # zero at the edge of the support
        assert out[3] == 0.0
        assert out[4] == 0.0

    def test_values_in_unit_interval(self):
        out = _raised_cosine(np.linspace(-3, 3, 101), width=2.0)
        assert np.all((out >= 0) & (out <= 1))


# ---------------------------------------------------------------------------
# _resolve_range
# ---------------------------------------------------------------------------


class TestResolveRange:
    def test_explicit_range(self):
        assert _resolve_range(np.array([0.0, 1.0]), (-2.0, 3.0)) == (-2.0, 3.0)

    def test_data_range(self):
        assert _resolve_range(np.array([1.0, np.nan, 4.0]), None) == (1.0, 4.0)

    def test_constant_is_widened(self):
        lo, hi = _resolve_range(np.array([2.0, 2.0]), None)
        assert (lo, hi) == (1.5, 2.5)

    def test_all_non_finite_falls_back_to_unit(self):
        assert _resolve_range(np.array([np.nan, np.inf]), None) == (0.0, 1.0)

    @pytest.mark.parametrize("bad", [(1.0, 1.0), (2.0, 1.0), (np.nan, 1.0)])
    def test_bad_explicit_range_raises(self, bad):
        with pytest.raises(ValueError, match="value_range"):
            _resolve_range(np.array([0.0, 1.0]), bad)


# ---------------------------------------------------------------------------
# circular_basis
# ---------------------------------------------------------------------------


class TestCircularBasis:
    def test_shape_and_peak_column(self):
        basis = circular_basis(np.array([0.0, 90.0]), n_basis=4)
        assert basis.shape == (2, 4)
        assert basis[0, 0] == pytest.approx(1.0)
        assert basis[1, 1] == pytest.approx(1.0)

    def test_rows_sum_to_one(self):
        basis = circular_basis(np.linspace(0, 360, 73), n_basis=12)
        assert np.allclose(basis.sum(axis=1), 1.0)

    def test_continuous_across_wrap(self):
        basis = circular_basis(np.array([359.9, 0.1, 360.0, 0.0]), n_basis=12)
        assert np.allclose(basis[0], basis[1], atol=2e-3)
        assert np.allclose(basis[2], basis[3])

    def test_non_finite_rows_are_zero(self):
        basis = circular_basis(np.array([10.0, np.nan, np.inf]), n_basis=6)
        assert np.all(basis[1] == 0)
        assert np.all(basis[2] == 0)
        assert basis[0].sum() == pytest.approx(1.0)

    def test_custom_width(self):
        narrow = circular_basis(np.array([15.0]), n_basis=12, width_deg=10.0)
        assert narrow.sum() == 0.0  # 15 deg is outside every 10 deg half-width

    @pytest.mark.parametrize("n_basis", [0, 1])
    def test_too_few_basis_raises(self, n_basis):
        with pytest.raises(ValueError, match="n_basis"):
            circular_basis(np.array([0.0]), n_basis=n_basis)

    def test_non_positive_width_raises(self):
        with pytest.raises(ValueError, match="width_deg"):
            circular_basis(np.array([0.0]), n_basis=4, width_deg=0.0)

    @settings(max_examples=30, deadline=None)
    @given(
        arrays(np.float64, st.integers(1, 20), elements=st.floats(-1e4, 1e4)),
        st.integers(2, 20),
    )
    def test_rows_sum_to_one_property(self, hd, n_basis):
        basis = circular_basis(hd, n_basis=n_basis)
        assert np.allclose(basis.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# linear_basis
# ---------------------------------------------------------------------------


class TestLinearBasis:
    def test_shape_and_endpoints(self):
        basis = linear_basis(np.array([0.0, 1.0]), n_basis=3, value_range=(0.0, 1.0))
        assert basis.shape == (2, 3)
        assert basis[0, 0] == pytest.approx(1.0)
        assert basis[1, 2] == pytest.approx(1.0)

    def test_rows_sum_to_one(self):
        basis = linear_basis(np.linspace(-5, 5, 51), n_basis=6)
        assert np.allclose(basis.sum(axis=1), 1.0)

    def test_out_of_range_is_clipped(self):
        basis = linear_basis(np.array([-10.0, 10.0]), n_basis=4, value_range=(0.0, 1.0))
        assert basis[0, 0] == pytest.approx(1.0)
        assert basis[1, 3] == pytest.approx(1.0)

    def test_non_finite_rows_are_zero(self):
        basis = linear_basis(np.array([1.0, np.nan]), n_basis=4, value_range=(0.0, 2.0))
        assert np.all(basis[1] == 0)

    def test_constant_input(self):
        basis = linear_basis(np.full(4, 3.0), n_basis=5)
        assert np.allclose(basis.sum(axis=1), 1.0)

    def test_too_few_basis_raises(self):
        with pytest.raises(ValueError, match="n_basis"):
            linear_basis(np.array([0.0, 1.0]), n_basis=1)

    @settings(max_examples=30, deadline=None)
    @given(
        arrays(np.float64, st.integers(1, 20), elements=st.floats(-1e3, 1e3)),
        st.integers(2, 12),
    )
    def test_rows_sum_to_one_property(self, values, n_basis):
        basis = linear_basis(values, n_basis=n_basis)
        assert np.allclose(basis.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# _gaussian_bumps
# ---------------------------------------------------------------------------


class TestGaussianBumps:
    def test_peak_at_centre(self):
        bumps = _gaussian_bumps(np.array([0.0, 1.0]), 3, (0.0, 1.0))
        assert bumps[0, 0] == pytest.approx(1.0)
        assert bumps[1, 2] == pytest.approx(1.0)
        assert bumps[0, 2] < bumps[0, 1] < bumps[0, 0]

    def test_non_finite_rows_are_zero(self):
        bumps = _gaussian_bumps(np.array([np.nan, 0.5]), 3, (0.0, 1.0))
        assert np.all(bumps[0] == 0)

    def test_too_few_basis_raises(self):
        with pytest.raises(ValueError, match="n_basis"):
            _gaussian_bumps(np.array([0.0]), 1, None)


# ---------------------------------------------------------------------------
# position_basis
# ---------------------------------------------------------------------------


class TestPositionBasis:
    def test_shape_and_normalisation(self):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 20)
        basis = position_basis(x, y, n_x=4, n_y=3)
        assert basis.shape == (20, 12)
        assert np.allclose(basis.sum(axis=1), 1.0)

    def test_column_order_is_x_major(self):
        # A point at the (x_hi, y_lo) corner peaks at column (n_x - 1) * n_y.
        basis = position_basis(
            np.array([1.0]), np.array([0.0]), n_x=3, n_y=2, x_range=(0.0, 1.0), y_range=(0.0, 1.0)
        )
        assert int(np.argmax(basis[0])) == 2 * 2

    def test_non_finite_rows_are_zero(self):
        basis = position_basis(
            np.array([0.0, np.nan]),
            np.array([0.0, 0.0]),
            n_x=2,
            n_y=2,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
        )
        assert np.all(basis[1] == 0)
        assert basis[0].sum() == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            position_basis(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# onehot_basis
# ---------------------------------------------------------------------------


class TestOnehotBasis:
    def test_basic_encoding(self):
        basis = onehot_basis(np.array([0, 2, 1]))
        assert basis.shape == (3, 3)
        assert np.array_equal(basis, np.eye(3)[[0, 2, 1]])

    def test_ignore_value_gives_zero_row(self):
        basis = onehot_basis(np.array([0, -1, 1]), n_levels=2)
        assert np.all(basis[1] == 0)
        assert basis.sum() == 2.0

    def test_out_of_range_and_nan_give_zero_rows(self):
        basis = onehot_basis(np.array([0.0, 5.0, np.nan]), n_levels=2)
        assert basis.shape == (3, 2)
        assert np.all(basis[1] == 0)
        assert np.all(basis[2] == 0)

    def test_all_ignored_defaults_to_one_level(self):
        basis = onehot_basis(np.array([-1, -1]))
        assert basis.shape == (2, 1)
        assert basis.sum() == 0.0

    def test_custom_ignore_value(self):
        basis = onehot_basis(np.array([0, 9, 1]), n_levels=2, ignore=9)
        assert np.all(basis[1] == 0)

    def test_bad_n_levels_raises(self):
        with pytest.raises(ValueError, match="n_levels"):
            onehot_basis(np.array([0, 1]), n_levels=0)


# ---------------------------------------------------------------------------
# valid_rows
# ---------------------------------------------------------------------------


class TestValidRows:
    def test_combines_arrays(self):
        mask = valid_rows(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, np.inf]))
        assert list(mask) == [True, False, False]

    def test_none_entries_skipped_and_int_arrays_finite(self):
        mask = valid_rows(None, np.array([1, 2, 3]), np.array([True, False, True]))
        assert mask.all()

    def test_no_arrays_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            valid_rows(None)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            valid_rows(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# build_design_matrix / DesignMatrix
# ---------------------------------------------------------------------------


class TestBuildDesignMatrix:
    def test_blocks_and_group_indices(self, behaviour):
        hd, speed, ahv = behaviour
        d = build_design_matrix(
            hd_deg=hd,
            ahv_deg_s=ahv,
            speed_cm_s=speed,
            x_mm=np.linspace(0, 10, N_FRAMES),
            y_mm=np.linspace(0, 10, N_FRAMES),
            light_on=np.zeros(N_FRAMES, dtype=bool),
            syllable_id=np.zeros(N_FRAMES, dtype=int),
            n_hd_basis=6,
            n_ahv_basis=4,
            n_speed_basis=4,
            n_x=3,
            n_y=3,
            n_syllables=2,
        )
        assert d.variables == ["hd", "ahv", "speed", "position", "light", "syllable"]
        assert d.X.shape == (N_FRAMES, 6 + 4 + 4 + 9 + 1 + 2)
        assert list(d.groups["hd"]) == list(range(6))
        assert list(d.groups["light"]) == [23]
        assert d.n_frames == N_FRAMES
        assert d.valid.all()

    def test_light_column_is_zero_one(self):
        light = np.array([True, False, True])
        d = build_design_matrix(
            speed_cm_s=np.array([1.0, 2.0, 3.0]), light_on=light, n_speed_basis=2
        )
        assert list(d.X[:, d.groups["light"][0]]) == [1.0, 0.0, 1.0]

    def test_invalid_frames_zeroed(self):
        hd = np.array([0.0, np.nan, 180.0])
        d = build_design_matrix(hd_deg=hd, n_hd_basis=4)
        assert list(d.valid) == [True, False, True]
        assert np.all(d.X[1] == 0)

    def test_unassigned_syllable_keeps_frame_valid(self):
        d = build_design_matrix(
            hd_deg=np.array([0.0, 10.0]), syllable_id=np.array([-1, 0]), n_hd_basis=4
        )
        assert d.valid.all()
        assert d.X[0, d.groups["syllable"]].sum() == 0.0

    def test_syllable_only_design_is_all_valid(self):
        d = build_design_matrix(syllable_id=np.array([0, 1, 0]))
        assert d.valid.all()
        assert d.variables == ["syllable"]

    def test_no_variables_raises(self):
        with pytest.raises(ValueError, match="at least one variable"):
            build_design_matrix()

    def test_half_position_raises(self):
        with pytest.raises(ValueError, match="together"):
            build_design_matrix(x_mm=np.zeros(3))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            build_design_matrix(hd_deg=np.zeros(3), speed_cm_s=np.zeros(4))


class TestDesignMatrixSubset:
    def test_subset_returns_group_columns(self, design):
        sub = design.subset(["speed"])
        assert sub.shape == (N_FRAMES, design.groups["speed"].size)
        assert np.allclose(sub, design.X[:, design.groups["speed"]])

    def test_subset_keeps_group_order(self, design):
        sub = design.subset(["speed", "hd"])
        assert np.allclose(sub, design.X)

    def test_empty_subset(self, design):
        assert design.subset([]).shape == (N_FRAMES, 0)

    def test_unknown_name_raises(self, design):
        with pytest.raises(KeyError, match="unknown design variables"):
            design.subset(["nonsense"])

    def test_default_construction(self):
        d = DesignMatrix(X=np.zeros((2, 1)))
        assert d.groups == {}
        assert d.valid.size == 0


# ---------------------------------------------------------------------------
# _check_xy
# ---------------------------------------------------------------------------


class TestCheckXY:
    def test_coerces_to_float(self):
        X, y = _check_xy([[1, 2], [3, 4]], [0, 1])
        assert X.dtype == np.float64 and y.dtype == np.float64

    def test_one_dimensional_x_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            _check_xy(np.zeros(4), np.zeros(4))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows"):
            _check_xy(np.zeros((4, 2)), np.zeros(3))

    def test_non_finite_y_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            _check_xy(np.zeros((2, 1)), np.array([1.0, np.nan]))

    def test_negative_y_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _check_xy(np.zeros((2, 1)), np.array([1.0, -1.0]))


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


class TestFitIntercepOnly:
    def test_intercept_is_log_mean(self):
        model = _fit_intercept_only(np.array([1.0, 3.0]), "sklearn")
        assert np.exp(model.intercept) == pytest.approx(2.0)
        assert model.coef.shape == (0,)

    def test_zero_rate_is_floored_and_coef_padded(self):
        model = _fit_intercept_only(np.zeros(3), "sklearn", n_features=4)
        assert model.coef.shape == (4,)
        assert np.isfinite(model.intercept)
        assert np.exp(model.intercept) < 1e-6

    def test_empty_response(self):
        model = _fit_intercept_only(np.zeros(0), "sklearn")
        assert np.isfinite(model.intercept)


class TestFitPoissonGLM:
    def test_recovers_positive_slope(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 2, 500)
        y = rng.poisson(np.exp(-0.5 + 0.8 * x)).astype(float)
        model = fit_poisson_glm(x[:, None], y)
        assert model.backend == "sklearn"
        assert model.coef.shape == (1,)
        assert 0.4 < model.coef[0] < 1.2
        assert -1.2 < model.intercept < 0.1

    def test_zero_column_design_is_constant_rate(self):
        y = np.array([1.0, 2.0, 3.0])
        model = fit_poisson_glm(np.zeros((3, 0)), y)
        assert model.coef.shape == (0,)
        assert np.exp(model.intercept) == pytest.approx(2.0)

    def test_silent_cell_returns_intercept_only(self):
        model = fit_poisson_glm(np.ones((5, 2)), np.zeros(5))
        assert np.all(model.coef == 0)
        assert model.coef.shape == (2,)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            fit_poisson_glm(np.ones((3, 1)), np.array([1.0, 2.0, 3.0]), backend="jax")

    def test_nemos_missing_raises_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nemos", None)
        with pytest.raises(ImportError, match="nemos"):
            fit_poisson_glm(np.ones((3, 1)), np.array([1.0, 2.0, 3.0]), backend="nemos")


class _FakeNemosGLM:
    """Minimal stand-in for ``nemos.glm.GLM`` recording its constructor kwargs."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeNemosGLM.last_kwargs = kwargs
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        self.coef_ = np.zeros((X.shape[1], 1))
        self.coef_[0, 0] = 0.25
        self.intercept_ = np.array([-1.5])
        return self


def _fake_nemos_module():
    return SimpleNamespace(
        glm=SimpleNamespace(GLM=_FakeNemosGLM),
        observation_models=SimpleNamespace(PoissonObservations=lambda: "poisson"),
        regularizer=SimpleNamespace(Ridge=lambda: "ridge"),
    )


class TestNemosBackend:
    def test_fit_nemos_uses_the_package(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nemos", _fake_nemos_module())
        model = _fit_nemos(np.ones((4, 2)), np.array([1.0, 0.0, 2.0, 1.0]), alpha=0.5)
        assert model.backend == "nemos"
        assert model.coef.shape == (2,)
        assert model.coef[0] == pytest.approx(0.25)
        assert model.intercept == pytest.approx(-1.5)
        assert _FakeNemosGLM.last_kwargs["regularizer_strength"] == 0.5
        assert _FakeNemosGLM.last_kwargs["observation_model"] == "poisson"

    def test_fit_poisson_glm_routes_to_nemos(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nemos", _fake_nemos_module())
        model = fit_poisson_glm(np.ones((4, 2)), np.array([1.0, 0.0, 2.0, 1.0]), backend="nemos")
        assert model.backend == "nemos"

    def test_cross_validation_runs_on_nemos_backend(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nemos", _fake_nemos_module())
        rng = np.random.default_rng(5)
        X = rng.uniform(0, 1, (40, 2))
        y = rng.poisson(1.0, 40).astype(float)
        result = cross_validated_deviance(X, y, n_folds=3, backend="nemos")
        assert result["deviance_explained"].shape == (3,)
        assert np.all(np.isfinite(result["prediction"]))


class TestPredictRate:
    def test_exponential_link(self):
        model = FittedGLM(coef=np.array([0.5]), intercept=0.1)
        mu = predict_rate(model, np.array([[0.0], [2.0]]))
        assert mu[0] == pytest.approx(np.exp(0.1))
        assert mu[1] == pytest.approx(np.exp(1.1))

    def test_zero_column_design(self):
        model = FittedGLM(coef=np.zeros(0), intercept=0.0)
        mu = predict_rate(model, np.zeros((3, 0)))
        assert np.allclose(mu, 1.0)

    def test_extreme_linear_predictor_is_clipped(self):
        model = FittedGLM(coef=np.array([1000.0]), intercept=0.0)
        mu = predict_rate(model, np.array([[1.0], [-1.0]]))
        assert np.all(np.isfinite(mu))
        assert mu[1] > 0

    def test_column_mismatch_raises(self):
        model = FittedGLM(coef=np.ones(2), intercept=0.0)
        with pytest.raises(ValueError, match="coefficients"):
            predict_rate(model, np.ones((3, 3)))

    def test_one_dimensional_x_raises(self):
        model = FittedGLM(coef=np.ones(1), intercept=0.0)
        with pytest.raises(ValueError, match="2-D"):
            predict_rate(model, np.ones(3))


# ---------------------------------------------------------------------------
# deviance
# ---------------------------------------------------------------------------


class TestPoissonDeviance:
    def test_perfect_prediction_is_zero(self):
        y = np.array([0.0, 1.0, 4.0])
        assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_worse_prediction_is_larger(self):
        y = np.array([1.0, 2.0, 3.0])
        assert poisson_deviance(y, np.full(3, 2.0)) < poisson_deviance(y, np.full(3, 10.0))

    def test_non_finite_frames_dropped(self):
        y = np.array([1.0, np.nan, 2.0])
        mu = np.array([1.0, 1.0, 2.0])
        assert poisson_deviance(y, mu) == pytest.approx(0.0, abs=1e-9)

    def test_all_non_finite_returns_zero(self):
        assert poisson_deviance(np.array([np.nan]), np.array([np.nan])) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            poisson_deviance(np.zeros(3), np.zeros(4))

    @settings(max_examples=30, deadline=None)
    @given(
        arrays(np.float64, 8, elements=st.floats(0, 20)),
        arrays(np.float64, 8, elements=st.floats(0.01, 20)),
    )
    def test_deviance_is_non_negative(self, y, mu):
        assert poisson_deviance(y, mu) >= -1e-9


class TestDevianceExplained:
    def test_perfect_model_is_one(self):
        y = np.array([1.0, 2.0, 5.0, 0.0])
        assert deviance_explained(y, y) == pytest.approx(1.0)

    def test_constant_rate_cell_is_about_zero(self):
        rng = np.random.default_rng(7)
        y = rng.poisson(2.0, 400).astype(float)
        mu = np.full(400, y.mean())
        assert abs(deviance_explained(y, mu)) < 1e-9

    def test_explicit_baseline_rate(self):
        y = np.array([1.0, 3.0])
        assert deviance_explained(y, y, y_baseline_rate=2.0) == pytest.approx(1.0)

    def test_zero_null_deviance_returns_zero(self):
        # A constant response makes the null deviance exactly zero, so no
        # fraction of it can be explained.
        assert deviance_explained(np.full(5, 2.0), np.full(5, 2.0)) == 0.0

    def test_silent_cell_returns_zero(self):
        assert deviance_explained(np.zeros(5), np.zeros(5)) == 0.0

    def test_all_non_finite_returns_zero(self):
        assert deviance_explained(np.full(3, np.nan), np.ones(3)) == 0.0

    def test_worse_than_baseline_is_negative(self):
        rng = np.random.default_rng(8)
        y = rng.poisson(2.0, 200).astype(float)
        assert deviance_explained(y, np.full(200, 20.0)) < 0


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------


class TestFoldIndices:
    def test_block_folds_are_contiguous_and_complete(self):
        folds = _fold_indices(10, 3, block=True)
        assert len(folds) == 3
        for fold in folds:
            assert np.array_equal(fold, np.arange(fold[0], fold[-1] + 1))
        assert np.array_equal(np.sort(np.concatenate(folds)), np.arange(10))
        assert folds[0][-1] < folds[1][0] < folds[2][0]

    def test_shuffled_folds_partition_without_order(self):
        rng = np.random.default_rng(11)
        folds = _fold_indices(20, 4, block=False, rng=rng)
        assert np.array_equal(np.sort(np.concatenate(folds)), np.arange(20))
        assert not np.array_equal(folds[0], np.arange(5))

    def test_shuffled_without_rng(self):
        folds = _fold_indices(12, 3, block=False)
        assert np.array_equal(np.sort(np.concatenate(folds)), np.arange(12))

    def test_too_few_folds_raises(self):
        with pytest.raises(ValueError, match="n_folds must be"):
            _fold_indices(10, 1)

    def test_more_folds_than_frames_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            _fold_indices(3, 5)


class TestCrossValidatedDeviance:
    def test_keys_and_shapes(self, design, hd_counts):
        result = cross_validated_deviance(design.X, hd_counts, n_folds=3)
        assert set(result) == {
            "deviance_explained",
            "mean_deviance_explained",
            "prediction",
            "fold",
            "n_folds",
        }
        assert result["deviance_explained"].shape == (3,)
        assert result["prediction"].shape == (N_FRAMES,)
        assert np.all(np.isfinite(result["prediction"]))
        assert set(np.unique(result["fold"])) == {0, 1, 2}

    def test_hd_cell_is_predicted_above_chance(self, design, hd_counts):
        result = cross_validated_deviance(design.X, hd_counts, n_folds=3)
        assert result["mean_deviance_explained"] > 0.05

    def test_unrelated_regressor_does_not_help(self, design):
        rng = np.random.default_rng(13)
        y = rng.poisson(2.0, N_FRAMES).astype(float)
        result = cross_validated_deviance(design.X, y, n_folds=3)
        assert result["mean_deviance_explained"] < 0.05

    def test_shuffled_folds(self, design, hd_counts):
        rng = np.random.default_rng(14)
        result = cross_validated_deviance(design.X, hd_counts, n_folds=3, block=False, rng=rng)
        assert result["deviance_explained"].shape == (3,)


class TestCvScoresFor:
    def test_empty_variable_set_scores_about_zero(self, design, hd_counts):
        keep = design.valid
        scores = _cv_scores_for(
            design, hd_counts[keep], keep, [], 3, True, 1e-3, "sklearn", None, 500
        )
        assert scores.shape == (3,)
        assert np.all(np.abs(scores) < 1e-6)

    def test_hd_beats_speed_for_an_hd_cell(self, design, hd_counts):
        keep = design.valid
        hd_scores = _cv_scores_for(
            design, hd_counts[keep], keep, ["hd"], 3, True, 1e-3, "sklearn", None, 500
        )
        speed_scores = _cv_scores_for(
            design, hd_counts[keep], keep, ["speed"], 3, True, 1e-3, "sklearn", None, 500
        )
        assert hd_scores.mean() > speed_scores.mean()


class TestValidSubset:
    def test_drops_invalid_and_non_finite(self):
        d = build_design_matrix(hd_deg=np.array([0.0, np.nan, 90.0, 180.0]), n_hd_basis=4)
        keep, y_valid = _valid_subset(d, np.array([1.0, 1.0, np.nan, 2.0]))
        assert list(keep) == [True, False, False, True]
        assert list(y_valid) == [1.0, 2.0]

    def test_length_mismatch_raises(self, design):
        with pytest.raises(ValueError, match="frames"):
            _valid_subset(design, np.zeros(3))

    def test_no_valid_frames_raises(self):
        d = build_design_matrix(hd_deg=np.array([np.nan, np.nan]), n_hd_basis=4)
        with pytest.raises(ValueError, match="no valid frames"):
            _valid_subset(d, np.zeros(2))


# ---------------------------------------------------------------------------
# contributions and selection
# ---------------------------------------------------------------------------


class TestWilcoxonGreater:
    def test_identical_scores_give_one(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert _wilcoxon_greater(x, x) == 1.0

    def test_consistently_larger_is_significant(self):
        ref = np.array([0.10, 0.12, 0.09, 0.11, 0.13])
        cand = ref + np.array([0.05, 0.04, 0.06, 0.05, 0.07])
        assert _wilcoxon_greater(cand, ref) < 0.05

    def test_too_few_non_zero_differences(self):
        ref = np.zeros(5)
        cand = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert _wilcoxon_greater(cand, ref) == 1.0

    def test_smaller_candidate_is_not_significant(self):
        ref = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        assert _wilcoxon_greater(ref - 0.2, ref) > 0.5


class TestDropOneContributions:
    def test_hd_cell_hd_contributes_most(self, design, hd_counts):
        contrib = drop_one_contributions(design, hd_counts, n_folds=3)
        assert set(contrib) == {"hd", "speed"}
        assert contrib["hd"] > contrib["speed"]
        assert contrib["hd"] > 0.02

    def test_speed_cell_speed_contributes_most(self, design, speed_counts):
        contrib = drop_one_contributions(design, speed_counts, n_folds=3)
        assert contrib["speed"] > contrib["hd"]
        assert contrib["speed"] > 0.02


class TestFullAndContributions:
    def test_matches_drop_one(self, design, hd_counts):
        full, contrib = _full_and_contributions(
            design, hd_counts, 3, 1e-3, "sklearn", None, True, 500
        )
        assert full > 0.05
        assert contrib == drop_one_contributions(design, hd_counts, n_folds=3)


class TestForwardSelection:
    def test_hd_cell_selects_hd_first(self, design, hd_counts):
        result = forward_selection(design, hd_counts, n_folds=5)
        assert result["selected"][0] == "hd"
        assert result["steps"][0]["candidate"] == "hd"
        assert result["steps"][0]["accepted"] is True
        assert result["steps"][0]["p_value"] < 0.05
        assert result["deviance_explained"] > 0.05

    def test_unrelated_cell_selects_nothing(self, design):
        rng = np.random.default_rng(15)
        y = rng.poisson(2.0, N_FRAMES).astype(float)
        result = forward_selection(design, y, n_folds=5)
        assert result["selected"] == []
        assert result["steps"][0]["accepted"] is False

    def test_threshold_of_one_accepts_everything(self, design, hd_counts):
        result = forward_selection(design, hd_counts, n_folds=5, p_threshold=1.1)
        assert result["selected"] == ["hd", "speed"] or result["selected"] == ["speed", "hd"]
        assert len(result["steps"]) == 2


class TestEncodingProfile:
    def test_hd_cell_profile(self, design, hd_counts):
        result = encoding_profile(design, hd_counts, n_folds=5)
        assert set(result) == {
            "full_dev_expl",
            "contributions",
            "profile",
            "dominant",
            "selected",
            "steps",
        }
        assert result["dominant"] == "hd"
        assert result["profile"]["hd"] > result["profile"]["speed"]
        assert sum(result["profile"].values()) == pytest.approx(1.0)
        assert result["selected"][0] == "hd"

    def test_speed_cell_profile(self, design, speed_counts):
        result = encoding_profile(design, speed_counts, n_folds=3, run_selection=False)
        assert result["dominant"] == "speed"
        assert result["profile"]["speed"] > 0.5
        assert result["selected"] == []
        assert result["steps"] == []

    def test_silent_cell_has_empty_profile(self, design):
        result = encoding_profile(design, np.zeros(N_FRAMES), n_folds=3, run_selection=False)
        assert result["dominant"] is None
        assert all(v == 0.0 for v in result["profile"].values())
        assert result["full_dev_expl"] == pytest.approx(0.0)


class TestPopulationEncodingProfiles:
    def test_columns_and_rows(self, design, hd_counts, speed_counts):
        signals = np.vstack([hd_counts, speed_counts])
        table = population_encoding_profiles(signals, design, n_folds=3, run_selection=False)
        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == [
            "roi",
            "full_dev_expl",
            "part_hd",
            "part_speed",
            "dominant",
            "selected",
        ]
        assert len(table) == 2
        assert list(table["roi"]) == [0, 1]
        assert table.loc[0, "dominant"] == "hd"
        assert table.loc[1, "dominant"] == "speed"
        assert (table["selected"] == "").all()

    def test_selection_column_populated(self, design, hd_counts):
        table = population_encoding_profiles(hd_counts[None, :], design, n_folds=5)
        assert table.loc[0, "selected"].startswith("hd")

    def test_wrong_dimensionality_raises(self, design, hd_counts):
        with pytest.raises(ValueError, match="2-D"):
            population_encoding_profiles(hd_counts, design)

    def test_frame_mismatch_raises(self, design):
        with pytest.raises(ValueError, match="frames"):
            population_encoding_profiles(np.zeros((2, 10)), design)
