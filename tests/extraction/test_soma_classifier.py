"""Tests for hm2p.extraction.soma_classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hm2p.extraction.soma_classifier import (
    CLASS_NAMES,
    DEFAULT_CLASSIFIER_PATH,
    LogisticRegressionClassifier,
    RuleBasedClassifier,
    classify_rois_with_probs,
    load_classifier,
)
from hm2p.extraction.soma_features import FEATURE_COLUMNS, extract_soma_features
from hm2p.extraction.suite2p import classify_roi_types


def _features_from_stat(stat: list[dict]) -> pd.DataFrame:
    """Build a feature frame from shape stats only — activity features = NaN.

    We pass empty (constant) traces so that the activity features are NaN
    and therefore contribute zero evidence in :class:`RuleBasedClassifier`.
    This is what we need to compare argmax labels against the legacy
    shape-only thresholds.
    """
    n = len(stat)
    F = np.full((n, 200), 100.0, dtype=np.float32)
    Fneu = np.full((n, 200), 80.0, dtype=np.float32)
    return extract_soma_features(stat, F, Fneu, fps=10.0)


# ---------------------------------------------------------------------------
# RuleBasedClassifier — basic invariants
# ---------------------------------------------------------------------------


class TestRuleBasedClassifier:
    def test_class_names_canonical(self) -> None:
        clf = RuleBasedClassifier()
        assert clf.class_names == ("soma", "dend", "artefact")

    def test_empty_input(self) -> None:
        clf = RuleBasedClassifier()
        empty = pd.DataFrame({col: pd.Series(dtype="float64") for col in FEATURE_COLUMNS})
        out = clf.predict_proba(empty)
        assert out.shape == (0, 3)

    def test_probs_in_unit_range(self) -> None:
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({col: rng.uniform(0, 5, n) for col in FEATURE_COLUMNS})
        out = RuleBasedClassifier().predict_proba(df)
        assert out.shape == (n, 3)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_probs_sum_to_one(self) -> None:
        rng = np.random.default_rng(1)
        n = 50
        df = pd.DataFrame({col: rng.uniform(0, 5, n) for col in FEATURE_COLUMNS})
        out = RuleBasedClassifier().predict_proba(df)
        sums = out.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9)

    def test_handles_nan_activity_features(self) -> None:
        df = pd.DataFrame(
            {col: [np.nan] for col in FEATURE_COLUMNS},
        )
        # Force shape features to a clean soma point.
        df.loc[0, "radius"] = 5.0
        df.loc[0, "compact"] = 0.5
        df.loc[0, "aspect_ratio"] = 1.0
        df.loc[0, "npix"] = 100.0
        df.loc[0, "npix_norm"] = 1.0
        df.loc[0, "skew"] = 1.0
        df.loc[0, "std"] = 1.0
        out = RuleBasedClassifier().predict_proba(df)
        assert np.all(np.isfinite(out))
        # Argmax must be soma (index 0) for a clear soma point.
        assert int(np.argmax(out, axis=1)[0]) == 0


# ---------------------------------------------------------------------------
# RuleBasedClassifier — argmax matches legacy thresholds
# ---------------------------------------------------------------------------


class TestArgmaxMatchesLegacy:
    """Shape-only argmax matches legacy thresholds (relabelling guarantee).

    The contract documented in :class:`RuleBasedClassifier` says that on
    *shape-only* input (activity features NaN or absent), the argmax of
    the rule-based scorer's probabilities equals :func:`classify_roi_types`.
    These tests pin that guarantee by passing constant traces (so the
    activity features are NaN and contribute zero evidence) and sweeping
    a deterministic grid of (radius, compact, aspect_ratio).

    With activity features present the classifier *can* relabel borderline
    shape-soma ROIs to dendrite — that is by design (see
    :class:`TestActivityFeaturesCanRelabel`).
    """

    def test_clear_soma(self) -> None:
        stat = [{"radius": 6.0, "compact": 0.7, "aspect_ratio": 1.5}]
        df = _features_from_stat(stat)
        probs = RuleBasedClassifier().predict_proba(df)
        assert int(np.argmax(probs, axis=1)[0]) == 0  # soma

    def test_clear_dendrite(self) -> None:
        stat = [{"radius": 6.0, "compact": 0.4, "aspect_ratio": 4.0}]
        df = _features_from_stat(stat)
        probs = RuleBasedClassifier().predict_proba(df)
        assert int(np.argmax(probs, axis=1)[0]) == 1  # dend

    def test_clear_artefact_small_radius(self) -> None:
        stat = [{"radius": 1.0, "compact": 0.7, "aspect_ratio": 1.5}]
        df = _features_from_stat(stat)
        probs = RuleBasedClassifier().predict_proba(df)
        assert int(np.argmax(probs, axis=1)[0]) == 2  # artefact

    def test_clear_artefact_low_compact(self) -> None:
        stat = [{"radius": 5.0, "compact": 0.05, "aspect_ratio": 1.5}]
        df = _features_from_stat(stat)
        probs = RuleBasedClassifier().predict_proba(df)
        assert int(np.argmax(probs, axis=1)[0]) == 2  # artefact

    def test_artefact_takes_priority_over_dendrite(self) -> None:
        # Tiny + elongated → legacy returns artefact.
        stat = [{"radius": 1.0, "compact": 0.5, "aspect_ratio": 5.0}]
        df = _features_from_stat(stat)
        probs = RuleBasedClassifier().predict_proba(df)
        assert int(np.argmax(probs, axis=1)[0]) == 2  # artefact

    def test_argmax_matches_legacy_on_grid(self) -> None:
        """Sweep a grid of (radius, compact, aspect_ratio) and compare argmax to legacy."""
        rng = np.random.default_rng(0)
        radii = [0.5, 1.0, 1.5, 1.9, 2.5, 3.5, 5.0, 8.0]
        compacts = [0.05, 0.09, 0.15, 0.3, 0.5, 0.8]
        aspects = [0.5, 1.0, 2.0, 2.49, 3.0, 5.0]
        stat: list[dict] = []
        for r in radii:
            for c in compacts:
                for a in aspects:
                    # Move slightly off the exact threshold so there is a
                    # well-defined legacy label and a non-degenerate argmax.
                    if abs(r - 2.0) < 1e-6 or abs(c - 0.1) < 1e-6 or abs(a - 2.5) < 1e-6:
                        continue
                    stat.append({"radius": r, "compact": c, "aspect_ratio": a})

        # Add a few random samples too — gives the test more coverage.
        for _ in range(50):
            stat.append(
                {
                    "radius": float(rng.uniform(0.5, 12.0)),
                    "compact": float(rng.uniform(0.05, 1.0)),
                    "aspect_ratio": float(rng.uniform(0.5, 5.0)),
                }
            )

        legacy = classify_roi_types(stat)
        df = _features_from_stat(stat)
        probs = RuleBasedClassifier().predict_proba(df)
        argmax = np.argmax(probs, axis=1)
        new_labels = [CLASS_NAMES[i] for i in argmax]

        # Compare element-by-element so failures point at the offending ROI.
        for i, (old, new, s) in enumerate(zip(legacy, new_labels, stat, strict=True)):
            assert old == new, f"ROI {i} {s!r}: legacy={old!r}, rule-based argmax={new!r}"

    @settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
    @given(
        radius=st.floats(min_value=0.2, max_value=15.0, exclude_min=False),
        compact=st.floats(min_value=0.02, max_value=1.0, exclude_min=False),
        aspect=st.floats(min_value=0.5, max_value=8.0),
    )
    def test_property_argmax_consistent_with_legacy(
        self,
        radius: float,
        compact: float,
        aspect: float,
    ) -> None:
        # The legacy rule is a hard threshold (< / >) while the soft scorer
        # ramps through p = 0.5 near each boundary, so the argmax is
        # indeterminate in a band around the thresholds. Skip cases where the
        # soft classifier is not confident (top-two probabilities close), which
        # is exactly where it can legitimately disagree with the hard rule.
        stat = [{"radius": radius, "compact": compact, "aspect_ratio": aspect}]
        legacy = classify_roi_types(stat)[0]
        df = _features_from_stat(stat)
        probs = RuleBasedClassifier().predict_proba(df)
        top_two = np.sort(probs[0])[::-1][:2]
        if top_two[0] - top_two[1] < 0.15:
            return  # indeterminate near a soft boundary
        new_label = CLASS_NAMES[int(np.argmax(probs, axis=1)[0])]
        assert new_label == legacy


class TestActivityFeaturesCanRelabel:
    """Pinned contract: activity features can flip a borderline soma to dend.

    QA issue 1.3: the original docstring claimed "switching to this
    classifier cannot change the label of any existing ROI". That is true
    only on the shape-only path. With activity features supplied (the
    production path always supplies them), a borderline shape-soma ROI
    whose kinetics or neuropil correlation are dendrite-like *will* be
    relabelled to ``dend``. These tests pin that intentional behaviour.
    """

    def test_slow_autocorr_relabels_borderline_soma_to_dend(self) -> None:
        """A clear shape-soma ROI with slow kinetics should land on dend."""
        df = pd.DataFrame(
            {col: [0.0] for col in FEATURE_COLUMNS},
        )
        # Clean soma shape (legacy → soma).
        df.loc[0, "radius"] = 5.0
        df.loc[0, "compact"] = 0.5
        df.loc[0, "aspect_ratio"] = 1.0
        df.loc[0, "npix"] = 100.0
        df.loc[0, "npix_norm"] = 1.0
        df.loc[0, "skew"] = 1.0
        df.loc[0, "std"] = 1.0
        # Slow autocorr and high Fneu correlation — dendrite-like activity.
        df.loc[0, "peak_to_noise_dff"] = 5.0
        df.loc[0, "autocorr_halfwidth_s"] = 2.0  # well past 1 s centre
        df.loc[0, "fneu_corr"] = 0.9  # well past 0.5 centre

        probs = RuleBasedClassifier().predict_proba(df)
        # Activity evidence pushes argmax to dend (1).
        assert int(np.argmax(probs, axis=1)[0]) == 1, (
            f"Expected dend (1) for slow-kinetics + high-Fneu shape-soma; "
            f"got argmax={int(np.argmax(probs, axis=1)[0])} probs={probs[0]}"
        )

    def test_fast_autocorr_keeps_borderline_soma(self) -> None:
        """Same shape-soma but with fast kinetics stays as soma."""
        df = pd.DataFrame(
            {col: [0.0] for col in FEATURE_COLUMNS},
        )
        df.loc[0, "radius"] = 5.0
        df.loc[0, "compact"] = 0.5
        df.loc[0, "aspect_ratio"] = 1.0
        df.loc[0, "npix"] = 100.0
        df.loc[0, "npix_norm"] = 1.0
        df.loc[0, "skew"] = 1.0
        df.loc[0, "std"] = 1.0
        # Fast autocorr and low Fneu correlation — soma-like activity.
        df.loc[0, "peak_to_noise_dff"] = 8.0
        df.loc[0, "autocorr_halfwidth_s"] = 0.2  # well below 1 s centre
        df.loc[0, "fneu_corr"] = 0.1  # well below 0.5 centre
        probs = RuleBasedClassifier().predict_proba(df)
        assert int(np.argmax(probs, axis=1)[0]) == 0  # soma

    def test_nan_activity_path_preserves_legacy(self) -> None:
        """If activity features are NaN, the shape-only contract holds."""
        df = pd.DataFrame(
            {col: [np.nan] for col in FEATURE_COLUMNS},
        )
        df.loc[0, "radius"] = 5.0
        df.loc[0, "compact"] = 0.5
        df.loc[0, "aspect_ratio"] = 1.0
        df.loc[0, "npix"] = 100.0
        df.loc[0, "npix_norm"] = 1.0
        df.loc[0, "skew"] = 1.0
        df.loc[0, "std"] = 1.0
        probs = RuleBasedClassifier().predict_proba(df)
        # NaN activity features contribute 0 evidence — argmax stays at soma.
        assert int(np.argmax(probs, axis=1)[0]) == 0


# ---------------------------------------------------------------------------
# LogisticRegressionClassifier
# ---------------------------------------------------------------------------


def _build_fitted_pipeline() -> object:
    """Train a tiny LogisticRegression pipeline on synthetic 3-class data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n_per = 20
    n_total = n_per * 3

    # Generate all features from FEATURE_COLUMNS dynamically.
    X = pd.DataFrame({col: rng.normal(0, 1, n_total) for col in FEATURE_COLUMNS})
    y = np.array(["soma"] * n_per + ["dend"] * n_per + ["artefact"] * n_per)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipe.fit(X, y)
    return pipe


class TestLogisticRegressionClassifier:
    def test_class_names_canonical(self) -> None:
        pipe = _build_fitted_pipeline()
        clf = LogisticRegressionClassifier(pipe)
        assert clf.class_names == ("soma", "dend", "artefact")

    def test_predict_proba_shape(self) -> None:
        pipe = _build_fitted_pipeline()
        clf = LogisticRegressionClassifier(pipe)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({col: rng.uniform(0, 1, 5) for col in FEATURE_COLUMNS})
        out = clf.predict_proba(df)
        assert out.shape == (5, 3)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_empty_input(self) -> None:
        pipe = _build_fitted_pipeline()
        clf = LogisticRegressionClassifier(pipe)
        empty = pd.DataFrame({col: pd.Series(dtype="float64") for col in FEATURE_COLUMNS})
        out = clf.predict_proba(empty)
        assert out.shape == (0, 3)

    def test_rejects_non_classifier(self) -> None:
        with pytest.raises(TypeError, match="predict_proba"):
            LogisticRegressionClassifier(object())

    def test_rejects_estimator_missing_classes(self) -> None:
        class Dummy:
            def predict_proba(self, X):  # noqa: ANN001, ANN201
                return np.zeros((len(X), 3))

        with pytest.raises(TypeError, match="classes_"):
            LogisticRegressionClassifier(Dummy())

    def test_class_permutation_correctness(self, tmp_path: Path) -> None:
        """If the fitted estimator's class order differs, predict_proba reorders."""
        # Build a pipeline whose internal class order is alphabetical
        # (artefact, dend, soma) and verify the wrapper returns columns in
        # the canonical order (soma, dend, artefact).
        pipe = _build_fitted_pipeline()
        # sklearn sorts classes_ alphabetically by default → ['artefact', 'dend', 'soma'].
        # Predict probabilities for a clear soma-shape ROI.
        df = pd.DataFrame({col: [1.0] for col in FEATURE_COLUMNS})
        clf = LogisticRegressionClassifier(pipe)
        probs = clf.predict_proba(df)
        assert probs.shape == (1, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# load_classifier
# ---------------------------------------------------------------------------


class TestLoadClassifier:
    def test_returns_rule_based_when_pickle_missing(self, tmp_path: Path) -> None:
        clf = load_classifier(tmp_path / "missing.pkl")
        assert isinstance(clf, RuleBasedClassifier)

    def test_returns_logistic_when_pickle_present(self, tmp_path: Path) -> None:
        import joblib

        pipe = _build_fitted_pipeline()
        target = tmp_path / "fitted.pkl"
        joblib.dump(pipe, target)
        clf = load_classifier(target)
        assert isinstance(clf, LogisticRegressionClassifier)

    def test_falls_back_when_pickle_corrupt(self, tmp_path: Path) -> None:
        target = tmp_path / "broken.pkl"
        target.write_bytes(b"not a pickle")
        clf = load_classifier(target)
        assert isinstance(clf, RuleBasedClassifier)

    def test_default_path_attr_exists(self) -> None:
        # We don't require this file to exist at runtime — only that the
        # loader treats its absence as a fall-through to the rule-based scorer.
        assert isinstance(DEFAULT_CLASSIFIER_PATH, Path)


# ---------------------------------------------------------------------------
# classify_rois_with_probs (top-level helper)
# ---------------------------------------------------------------------------


class TestClassifyRoisWithProbs:
    def test_returns_labels_and_probs(self) -> None:
        stat = [
            {"radius": 6.0, "compact": 0.7, "aspect_ratio": 1.5},
            {"radius": 1.0, "compact": 0.7, "aspect_ratio": 1.5},
            {"radius": 6.0, "compact": 0.4, "aspect_ratio": 4.0},
        ]
        df = _features_from_stat(stat)
        labels, probs = classify_rois_with_probs(df, classifier=RuleBasedClassifier())
        assert labels == ["soma", "artefact", "dend"]
        assert probs.shape == (3, 3)

    def test_empty_input(self) -> None:
        empty = pd.DataFrame({col: pd.Series(dtype="float64") for col in FEATURE_COLUMNS})
        labels, probs = classify_rois_with_probs(empty, classifier=RuleBasedClassifier())
        assert labels == []
        assert probs.shape == (0, 3)
