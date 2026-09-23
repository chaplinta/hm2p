"""Tests for hm2p.analysis.topology — ring structure of the HD population code."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hm2p.analysis.topology import (
    _empty_ring_result,
    _zscore_rows,
    cebra_embedding,
    circular_correlation,
    decoding_at_matched_n,
    persistence_diagram,
    population_embedding_pca,
    ring_score_pca,
    ring_score_persistence,
    topology_summary,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _hd_ring_population(n_rois=16, n_frames=360, kappa=2.0, noise=0.02, seed=0):
    """Population of von Mises HD cells sampled over full circular sweeps."""
    rng = np.random.default_rng(seed)
    hd = np.mod(np.linspace(0.0, 3 * 360.0, n_frames, endpoint=False), 360.0)
    theta = np.deg2rad(hd)
    prefs = np.deg2rad(np.linspace(0.0, 360.0, n_rois, endpoint=False))
    signals = np.empty((n_rois, n_frames))
    for i in range(n_rois):
        tuned = np.exp(kappa * np.cos(theta - prefs[i]))
        signals[i] = tuned / tuned.max() + noise * rng.normal(0.0, 1.0, n_frames)
    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, mask


def _noise_population(n_rois=16, n_frames=360, seed=1):
    rng = np.random.default_rng(seed)
    signals = rng.normal(0.0, 1.0, size=(n_rois, n_frames))
    hd = rng.uniform(0.0, 360.0, n_frames)
    return signals, hd, np.ones(n_frames, dtype=bool)


def _install_fake_ripser(monkeypatch, dgms, record):
    module = types.ModuleType("ripser")

    def _ripser(X, maxdim=1):
        record["n_points"] = int(np.asarray(X).shape[0])
        record["maxdim"] = maxdim
        return {"dgms": dgms}

    module.ripser = _ripser
    monkeypatch.setitem(sys.modules, "ripser", module)


def _install_fake_cebra(monkeypatch, record):
    module = types.ModuleType("cebra")

    class _CEBRA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            record["kwargs"] = kwargs

        def fit(self, neural, labels=None):
            record["neural_shape"] = neural.shape
            record["labels_shape"] = None if labels is None else labels.shape
            return self

        def transform(self, neural):
            return np.zeros((neural.shape[0], self.kwargs["output_dimension"]))

    module.CEBRA = _CEBRA
    monkeypatch.setitem(sys.modules, "cebra", module)


# ---------------------------------------------------------------------------
# _zscore_rows
# ---------------------------------------------------------------------------


def test_zscore_rows_normalises_and_handles_constant_rows():
    x = np.vstack([np.arange(6.0), np.full(6, 2.0)])
    z = _zscore_rows(x)
    assert z[0].mean() == pytest.approx(0.0)
    assert z[0].std() == pytest.approx(1.0)
    assert np.all(z[1] == 0.0)


# ---------------------------------------------------------------------------
# circular_correlation
# ---------------------------------------------------------------------------


def test_circular_correlation_identical_angles_is_one():
    angles = np.linspace(0.0, 350.0, 36)
    assert circular_correlation(angles, angles) == pytest.approx(1.0)
    assert circular_correlation(angles, np.mod(angles + 40.0, 360.0)) == pytest.approx(1.0)


def test_circular_correlation_reflected_angles_is_minus_one():
    angles = np.linspace(0.0, 350.0, 36)
    assert circular_correlation(angles, -angles) == pytest.approx(-1.0)


def test_circular_correlation_unrelated_angles_is_small():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 360, 500)
    b = rng.uniform(0, 360, 500)
    assert abs(circular_correlation(a, b)) < 0.2


def test_circular_correlation_degenerate_inputs():
    assert np.isnan(circular_correlation(np.array([10.0]), np.array([20.0])))
    assert np.isnan(circular_correlation(np.full(10, 30.0), np.linspace(0, 350, 10)))
    assert np.isnan(circular_correlation(np.array([np.nan, 1.0]), np.array([1.0, np.nan])))
    with pytest.raises(ValueError, match="equal length"):
        circular_correlation(np.zeros(4), np.zeros(5))


# ---------------------------------------------------------------------------
# _empty_ring_result
# ---------------------------------------------------------------------------


def test_empty_ring_result_is_all_nan():
    result = _empty_ring_result()
    assert result["n_points"] == 0
    assert np.isnan(result["radius_cv"])
    assert np.isnan(result["angle_hd_correlation"])
    assert np.isnan(result["angle_uniformity_p"])


# ---------------------------------------------------------------------------
# population_embedding_pca
# ---------------------------------------------------------------------------


def test_population_embedding_pca_shapes_and_variance():
    signals, hd, mask = _hd_ring_population()
    embedding, evr = population_embedding_pca(signals, mask, n_components=3)
    assert embedding.shape == (mask.sum(), 3)
    assert evr.shape == (3,)
    assert 0.0 < evr.sum() <= 1.0 + 1e-9
    # A ring is essentially two-dimensional, and its two components carry
    # equal variance (an arbitrary rotation within the plane).
    assert evr[:2].sum() > 0.8
    assert evr[0] == pytest.approx(evr[1], rel=0.1)


def test_population_embedding_pca_respects_mask_and_no_smoothing():
    signals, hd, mask = _hd_ring_population(n_frames=200)
    mask[:100] = False
    embedding, _ = population_embedding_pca(signals, mask, n_components=2, smooth_frames=1)
    assert embedding.shape == (100, 2)


def test_population_embedding_pca_limits_components_to_roi_count():
    signals, hd, mask = _hd_ring_population(n_rois=2, n_frames=50)
    embedding, evr = population_embedding_pca(signals, mask, n_components=5)
    assert embedding.shape[1] == 2
    assert evr.shape == (2,)


def test_population_embedding_pca_validates_inputs():
    signals, hd, mask = _hd_ring_population(n_rois=3, n_frames=40)
    with pytest.raises(ValueError, match="2-D"):
        population_embedding_pca(signals[0], mask)
    with pytest.raises(ValueError, match="mask must have shape"):
        population_embedding_pca(signals, mask[:10])
    with pytest.raises(ValueError, match="n_components"):
        population_embedding_pca(signals, mask, n_components=0)
    with pytest.raises(ValueError, match="at least 2 valid frames"):
        population_embedding_pca(signals, np.zeros(40, dtype=bool))


# ---------------------------------------------------------------------------
# ring_score_pca
# ---------------------------------------------------------------------------


def test_ring_score_pca_detects_hd_ring():
    signals, hd, mask = _hd_ring_population()
    embedding, _ = population_embedding_pca(signals, mask, n_components=3)
    result = ring_score_pca(embedding, hd)
    assert abs(result["angle_hd_correlation"]) > 0.9
    assert result["radius_cv"] < 0.2
    assert result["angle_uniformity_p"] > 0.05
    assert result["n_points"] == mask.sum()
    assert result["angle_deg"].min() >= 0.0
    assert result["angle_deg"].max() < 360.0
    assert result["mean_radius"] > 0.0


def test_ring_score_pca_noise_population_has_no_hd_ring():
    signals, hd, mask = _noise_population()
    embedding, _ = population_embedding_pca(signals, mask, n_components=3)
    ring = ring_score_pca(embedding, hd)
    hd_ring = ring_score_pca(*_hd_ring_embedding())
    assert abs(ring["angle_hd_correlation"]) < 0.3
    assert ring["radius_cv"] > hd_ring["radius_cv"]


def _hd_ring_embedding():
    signals, hd, mask = _hd_ring_population()
    embedding, _ = population_embedding_pca(signals, mask, n_components=3)
    return embedding, hd


def test_ring_score_pca_without_hd_gives_nan_correlation():
    embedding, _hd = _hd_ring_embedding()
    result = ring_score_pca(embedding)
    assert np.isnan(result["angle_hd_correlation"])


def test_ring_score_pca_zero_radius_gives_nan_cv():
    result = ring_score_pca(np.zeros((5, 2)))
    assert np.isnan(result["radius_cv"])
    assert result["mean_radius"] == 0.0


def test_ring_score_pca_validates_inputs():
    with pytest.raises(ValueError, match="n_points"):
        ring_score_pca(np.zeros((10, 1)))
    with pytest.raises(ValueError, match="hd_deg_valid"):
        ring_score_pca(np.zeros((10, 2)), np.zeros(5))


# ---------------------------------------------------------------------------
# persistence_diagram
# ---------------------------------------------------------------------------


def test_persistence_diagram_missing_ripser_raises_with_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "ripser", None)
    with pytest.raises(ImportError, match="uv pip install ripser"):
        persistence_diagram(np.zeros((10, 2)))


def test_persistence_diagram_with_stub_module(monkeypatch):
    record: dict = {}
    dgms = [np.array([[0.0, 1.0]]), np.array([[0.2, 1.5]])]
    _install_fake_ripser(monkeypatch, dgms, record)
    embedding, _ = _hd_ring_embedding()
    result = persistence_diagram(embedding, max_dim=1)
    assert record["n_points"] == embedding.shape[0]
    assert record["maxdim"] == 1
    assert result["n_points_used"] == embedding.shape[0]
    assert result["max_dim"] == 1
    assert len(result["dgms"]) == 2


def test_persistence_diagram_landmark_subsampling(monkeypatch):
    record: dict = {}
    _install_fake_ripser(monkeypatch, [np.zeros((0, 2)), np.zeros((0, 2))], record)
    embedding, _ = _hd_ring_embedding()
    result = persistence_diagram(embedding, n_landmarks=50, rng=np.random.default_rng(3))
    assert record["n_points"] == 50
    assert result["n_points_used"] == 50
    # n_landmarks >= n_points leaves the cloud untouched.
    full = persistence_diagram(embedding, n_landmarks=embedding.shape[0] + 10)
    assert full["n_points_used"] == embedding.shape[0]


def test_persistence_diagram_landmarks_without_rng(monkeypatch):
    record: dict = {}
    _install_fake_ripser(monkeypatch, [np.zeros((0, 2))], record)
    result = persistence_diagram(np.random.default_rng(4).normal(size=(30, 2)), n_landmarks=10)
    assert result["n_points_used"] == 10


def test_persistence_diagram_validates_inputs(monkeypatch):
    record: dict = {}
    _install_fake_ripser(monkeypatch, [np.zeros((0, 2))], record)
    with pytest.raises(ValueError, match="non-empty 2-D"):
        persistence_diagram(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="n_landmarks"):
        persistence_diagram(np.zeros((10, 2)), n_landmarks=0)


def test_persistence_diagram_with_real_ripser():
    pytest.importorskip("ripser")
    rng = np.random.default_rng(5)
    angles = rng.uniform(0.0, 2 * np.pi, 120)
    cloud = np.column_stack([np.cos(angles), np.sin(angles)])
    result = persistence_diagram(cloud, max_dim=1)
    assert len(result["dgms"]) == 2
    ring = ring_score_persistence(result["dgms"])
    assert ring["ring_score"] > 0.8


# ---------------------------------------------------------------------------
# ring_score_persistence
# ---------------------------------------------------------------------------


def test_ring_score_persistence_single_long_bar():
    dgms = [np.array([[0.0, np.inf]]), np.array([[0.05, 0.10], [0.1, 2.0]])]
    result = ring_score_persistence(dgms)
    assert result["h1_max_persistence"] == pytest.approx(1.9)
    assert result["h1_second_persistence"] == pytest.approx(0.05)
    assert result["ring_score"] > 0.97
    assert result["n_h1_features"] == 2
    assert result["n_h1_features_above"] == 1


def test_ring_score_persistence_two_equal_bars_scores_zero():
    dgms = [np.zeros((1, 2)), np.array([[0.0, 1.0], [0.5, 1.5]])]
    result = ring_score_persistence(dgms)
    assert result["ring_score"] == pytest.approx(0.0)
    assert result["n_h1_features_above"] == 2


def test_ring_score_persistence_single_feature_scores_one():
    result = ring_score_persistence([np.zeros((1, 2)), np.array([[0.0, 3.0]])])
    assert result["ring_score"] == pytest.approx(1.0)
    assert result["h1_second_persistence"] == 0.0


def test_ring_score_persistence_empty_cases():
    zero = {
        "h1_max_persistence": 0.0,
        "h1_second_persistence": 0.0,
        "ring_score": 0.0,
        "n_h1_features": 0,
        "n_h1_features_above": 0,
    }
    assert ring_score_persistence([]) == zero
    assert ring_score_persistence(None) == zero
    assert ring_score_persistence([np.zeros((1, 2)), np.zeros((0, 2))]) == zero
    # Only infinite H1 lifetimes are also treated as "no finite feature".
    assert ring_score_persistence([np.zeros((1, 2)), np.array([[0.0, np.inf]])]) == zero


def test_ring_score_persistence_zero_length_bar_scores_zero():
    result = ring_score_persistence([np.zeros((1, 2)), np.array([[1.0, 1.0]])])
    assert result["ring_score"] == 0.0
    assert result["n_h1_features"] == 1


def test_ring_score_persistence_validates_h1_shape():
    with pytest.raises(ValueError, match="H1 diagram"):
        ring_score_persistence([np.zeros((1, 2)), np.zeros((3, 3))])


# ---------------------------------------------------------------------------
# cebra_embedding
# ---------------------------------------------------------------------------


def test_cebra_embedding_missing_package_raises_with_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "cebra", None)
    signals, hd, mask = _hd_ring_population(n_rois=4, n_frames=40)
    with pytest.raises(ImportError, match="uv pip install cebra"):
        cebra_embedding(signals, mask, hd)


def test_cebra_embedding_behaviour_contrastive_with_stub(monkeypatch):
    record: dict = {}
    _install_fake_cebra(monkeypatch, record)
    signals, hd, mask = _hd_ring_population(n_rois=4, n_frames=40)
    embedding = cebra_embedding(
        signals, mask, hd, output_dim=3, max_iterations=5, rng=np.random.default_rng(6)
    )
    assert embedding.shape == (40, 3)
    assert record["neural_shape"] == (40, 4)
    assert record["labels_shape"] == (40, 2)
    assert record["kwargs"]["conditional"] == "time_delta"


def test_cebra_embedding_time_contrastive_with_stub(monkeypatch):
    record: dict = {}
    _install_fake_cebra(monkeypatch, record)
    signals, _hd, mask = _hd_ring_population(n_rois=4, n_frames=40)
    embedding = cebra_embedding(signals, mask, hd_deg=None, output_dim=2, max_iterations=5)
    assert embedding.shape == (40, 2)
    assert record["labels_shape"] is None
    assert record["kwargs"]["conditional"] == "time"


def test_cebra_embedding_validates_inputs(monkeypatch):
    record: dict = {}
    _install_fake_cebra(monkeypatch, record)
    signals, hd, mask = _hd_ring_population(n_rois=4, n_frames=40)
    with pytest.raises(ValueError, match="2-D"):
        cebra_embedding(signals[0], mask, hd)
    with pytest.raises(ValueError, match="mask must have shape"):
        cebra_embedding(signals, mask[:10], hd)
    with pytest.raises(ValueError, match="at least 2 valid frames"):
        cebra_embedding(signals, np.zeros(40, dtype=bool), hd)
    with pytest.raises(ValueError, match="hd_deg must have shape"):
        cebra_embedding(signals, mask, hd[:10])


# ---------------------------------------------------------------------------
# decoding_at_matched_n
# ---------------------------------------------------------------------------


def test_decoding_at_matched_n_returns_error_distribution():
    signals, hd, mask = _hd_ring_population(n_rois=12, n_frames=300, seed=7)
    result = decoding_at_matched_n(
        signals, hd, mask, n_cells=6, n_draws=4, rng=np.random.default_rng(8), n_folds=3
    )
    assert result["errors"].shape == (4,)
    assert np.isfinite(result["errors"]).all()
    assert result["median"] == pytest.approx(float(np.median(result["errors"])))
    assert result["iqr"] >= 0.0
    assert result["n_cells"] == 6
    assert result["n_draws"] == 4


def test_decoding_at_matched_n_uses_full_population_when_n_equals_rois():
    signals, hd, mask = _hd_ring_population(n_rois=8, n_frames=300, seed=9)
    result = decoding_at_matched_n(signals, hd, mask, n_cells=8, n_draws=2, n_folds=3)
    # With every ROI included each draw is identical.
    assert result["errors"][0] == pytest.approx(result["errors"][1])
    assert result["iqr"] == pytest.approx(0.0)


def test_decoding_at_matched_n_validates_inputs():
    signals, hd, mask = _hd_ring_population(n_rois=5, n_frames=100, seed=10)
    with pytest.raises(ValueError, match="2-D"):
        decoding_at_matched_n(signals[0], hd, mask, n_cells=2)
    with pytest.raises(ValueError, match=r"n_cells must be in \[1, 5\]"):
        decoding_at_matched_n(signals, hd, mask, n_cells=6)
    with pytest.raises(ValueError, match="n_draws"):
        decoding_at_matched_n(signals, hd, mask, n_cells=2, n_draws=0)


# ---------------------------------------------------------------------------
# topology_summary
# ---------------------------------------------------------------------------


def _light_dark(n_frames, block=13):
    """Alternating light/dark blocks short enough that both conditions
    sample the whole circle of head directions."""
    light = np.zeros(n_frames, dtype=bool)
    for start in range(0, n_frames, 2 * block):
        light[start : start + block] = True
    return light


def test_topology_summary_reports_ring_per_condition():
    signals, hd, mask = _hd_ring_population(n_rois=12, n_frames=360, seed=11)
    summary = topology_summary(signals, hd, mask, _light_dark(360))
    for key in ("pca_all", "pca_light", "pca_dark"):
        assert abs(summary[key]["angle_hd_correlation"]) > 0.8
        assert "angle_deg" not in summary[key]
        assert "radius" not in summary[key]
    assert summary["explained_variance_all"].shape == (3,)
    assert summary["matched_decoding"] is None


def test_topology_summary_without_ripser_marks_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "ripser", None)
    signals, hd, mask = _hd_ring_population(n_rois=8, n_frames=120, seed=12)
    summary = topology_summary(signals, hd, mask, _light_dark(120, block=20))
    assert summary["ripser_available"] is False
    assert np.isnan(summary["persistence"]["ring_score"])


def test_topology_summary_with_stub_ripser(monkeypatch):
    record: dict = {}
    _install_fake_ripser(
        monkeypatch, [np.zeros((1, 2)), np.array([[0.0, 2.0], [0.0, 0.1]])], record
    )
    signals, hd, mask = _hd_ring_population(n_rois=8, n_frames=120, seed=13)
    summary = topology_summary(signals, hd, mask, _light_dark(120, block=20), n_landmarks=None)
    assert summary["ripser_available"] is True
    assert summary["persistence"]["ring_score"] == pytest.approx(0.95)


def test_topology_summary_matched_decoding(monkeypatch):
    monkeypatch.setitem(sys.modules, "ripser", None)
    signals, hd, mask = _hd_ring_population(n_rois=10, n_frames=240, seed=14)
    summary = topology_summary(
        signals,
        hd,
        mask,
        _light_dark(240, block=40),
        n_cells_matched=4,
        rng=np.random.default_rng(15),
    )
    assert summary["matched_decoding"]["n_cells"] == 4
    assert np.isfinite(summary["matched_decoding"]["errors"]).all()


def test_topology_summary_single_roi_has_empty_ring_results(monkeypatch):
    monkeypatch.setitem(sys.modules, "ripser", None)
    signals, hd, mask = _hd_ring_population(n_rois=1, n_frames=100, seed=16)
    summary = topology_summary(signals, hd, mask, _light_dark(100, block=20))
    assert summary["pca_all"]["n_points"] == 0
    assert summary["ripser_available"] is False
    assert summary["explained_variance_all"].size == 0


def test_topology_summary_one_component_cannot_form_a_ring(monkeypatch):
    monkeypatch.setitem(sys.modules, "ripser", None)
    signals, hd, mask = _hd_ring_population(n_rois=6, n_frames=100, seed=17)
    summary = topology_summary(signals, hd, mask, _light_dark(100, block=20), n_components=1)
    assert summary["pca_all"]["n_points"] == 0
    assert np.isnan(summary["pca_light"]["radius_cv"])


def test_topology_summary_all_light_session_has_empty_dark_result():
    signals, hd, mask = _hd_ring_population(n_rois=6, n_frames=120, seed=18)
    summary = topology_summary(signals, hd, mask, np.ones(120, dtype=bool))
    assert summary["pca_dark"]["n_points"] == 0
    assert summary["pca_light"]["n_points"] == 120


def test_topology_summary_validates_shapes():
    signals, hd, mask = _hd_ring_population(n_rois=4, n_frames=80, seed=19)
    with pytest.raises(ValueError, match="2-D"):
        topology_summary(signals[0], hd, mask, np.ones(80, dtype=bool))
    with pytest.raises(ValueError, match="light_on must have shape"):
        topology_summary(signals, hd, mask, np.ones(40, dtype=bool))
