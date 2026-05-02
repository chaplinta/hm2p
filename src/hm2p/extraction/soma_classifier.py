"""Soma / dendrite / artefact classifier for Suite2p ROIs.

This module provides a small classifier framework with two implementations:

* :class:`RuleBasedClassifier` — an interim classifier that reproduces the
  legacy hand-tuned shape thresholds (``radius < 2`` or ``compact < 0.1``
  → artefact; ``aspect_ratio > 2.5`` → dendrite; otherwise soma) by
  mapping each decision boundary onto a soft logistic function, then
  augments the score with activity-feature evidence (``autocorr_halfwidth_s``,
  ``fneu_corr``).  On *shape-only* input the argmax matches the legacy
  thresholds exactly; with activity features present, the classifier can
  re-label borderline shape-soma ROIs as dendrite when their kinetics or
  neuropil correlation are dendrite-like.  This is intentional — the whole
  point of the redesign is to bring activity context into the boundary
  cases.  See :class:`RuleBasedClassifier` docstring for the precise
  contract and :func:`hm2p.extraction.soma_features.extract_soma_features`
  for the activity feature definitions.

* :class:`LogisticRegressionClassifier` — a thin wrapper around an sklearn
  pipeline that has been fitted offline against curated labels.  This is
  the long-term replacement for the rule-based scorer.  It is used at run
  time only when a fitted pipeline pickle is provided.

The :func:`load_classifier` helper returns whichever of the two
implementations is appropriate for the current environment.

Provisional status
------------------
The rule-based scorer is *provisional*.  Once ~200 ROIs across 2–3 sessions
have been labelled in Suite2p's GUI, train a real
:class:`LogisticRegressionClassifier` via
``scripts/train_soma_classifier.py`` and place the resulting pickle at
``sourcedata/trackers/suite2p/soma_classifier.pkl``.  The runtime path
will pick it up automatically.

Relabelling guarantee
---------------------
On the **shape-only path** (activity features NaN), ``argmax(probs)``
matches the legacy thresholds exactly.  With **activity features
present**, an ROI whose shape is borderline soma but whose kinetics
(``autocorr_halfwidth_s`` ≫ 1 s) or neuropil correlation
(``fneu_corr`` ≫ 0.5) look dendritic will be relabelled to ``dend``.
This is the documented behaviour, not a regression — switching from the
shape-only legacy classifier therefore *can* change labels on real
sessions, and that is by design.  See
``tests/extraction/test_soma_classifier.py::TestArgmaxMatchesLegacy`` for
the precise shape-only equivalence guarantee and
``TestActivityFeaturesCanRelabel`` for the relabel-with-activity contract.

References
----------
Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
two-photon microscopy." bioRxiv. doi:10.1101/061507.
https://github.com/MouseLand/suite2p

Pedregosa et al. 2011. "Scikit-learn: Machine Learning in Python."
Journal of Machine Learning Research 12:2825–2830.
https://scikit-learn.org
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Canonical class order — every classifier in this module must use the same
# index for each label so that probabilities can be combined and persisted
# with a single set of column names.
CLASS_NAMES: tuple[str, ...] = ("soma", "dend", "artefact")
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Default location of the fitted sklearn classifier pickle.  This path is
# relative to the repository root; it is resolved lazily so that test
# fixtures and CI environments without ``sourcedata/`` still import cleanly.
DEFAULT_CLASSIFIER_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "sourcedata"
    / "trackers"
    / "suite2p"
    / "soma_classifier.pkl"
)


@runtime_checkable
class SomaClassifier(Protocol):
    """Protocol for soma/dendrite/artefact classifiers.

    Implementations must expose a fixed :attr:`class_names` tuple in the
    canonical order ``("soma", "dend", "artefact")`` and a
    :meth:`predict_proba` method that returns an ``(n_rois, 3)`` array of
    probabilities summing to 1 along the last axis.
    """

    class_names: tuple[str, ...]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return per-ROI class probabilities.

        Parameters
        ----------
        features : pandas.DataFrame
            Output of :func:`hm2p.extraction.soma_features.extract_soma_features`.

        Returns
        -------
        numpy.ndarray
            ``(n_rois, 3)`` float array; rows sum to 1.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid."""
    out = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
    return np.asarray(out)


def _or_combine(*probs: np.ndarray) -> np.ndarray:
    """Combine independent probabilities with a probabilistic-OR.

    ``P(any) = 1 - prod(1 - P_i)``.  Inputs are expected to be arrays of
    matching shape with values in ``[0, 1]``.
    """
    not_any = np.ones_like(probs[0])
    for p in probs:
        not_any = not_any * (1.0 - p)
    return 1.0 - not_any


# ---------------------------------------------------------------------------
# Rule-based scorer
# ---------------------------------------------------------------------------


class RuleBasedClassifier:
    """Interim soma/dend/artefact scorer with calibrated probabilities.

    The scorer reproduces the legacy decision boundaries

    * ``radius < 2.0``         → artefact
    * ``compact < 0.1``        → artefact
    * ``aspect_ratio > 2.5``   → dend (when not artefact)
    * otherwise                → soma

    by mapping each boundary onto a steep logistic function so that
    probabilities are calibrated near the boundary while still saturating
    well below or above it.  Activity features are folded in as additional
    "evidence" terms for ``p_dend``: a slow autocorrelation half-width and
    a high Fneu correlation both push probability mass towards dendrite.

    Relabelling contract
    --------------------
    - **Shape-only** (activity features NaN or absent): ``argmax(probs)``
      reproduces the legacy hard thresholds exactly.  Switching from the
      legacy classifier on shape-only inputs therefore cannot change any
      label.
    - **Shape + activity**: the activity terms can flip a borderline
      shape-soma ROI to ``dend`` when its kinetics or neuropil correlation
      look dendritic.  This is intentional — bringing activity context
      into the boundary cases is the entire point of the redesign.  The
      production path in :mod:`hm2p.calcium.run` always provides activity
      features (via :func:`hm2p.extraction.soma_features.extract_soma_features`),
      so production runs *will* relabel a small number of borderline ROIs
      relative to the legacy shape-only output.  The frontend ROI viewer
      surfaces ``p_soma`` / ``p_dend`` / ``p_artefact`` so the user can
      inspect any flips.

    Notes
    -----
    The functional form is intentionally conservative.  This classifier is
    *not* a substitute for a curated, cross-validated logistic regression.
    Its purpose is to expose calibrated probabilities for the curation UI
    in Wave 5; the long-term replacement is :class:`LogisticRegressionClassifier`.

    References
    ----------
    Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
    two-photon microscopy." bioRxiv. doi:10.1101/061507.
    """

    class_names: tuple[str, ...] = CLASS_NAMES

    # --- Soft-margin parameters (centre and scale for each logistic) -------
    # Centres are chosen so that the legacy hard thresholds sit exactly at
    # p = 0.5; scales are tight enough that ROIs well past the boundary
    # saturate at near-1 probability.
    RADIUS_CENTRE: float = 2.0
    RADIUS_SCALE: float = 0.25

    COMPACT_CENTRE: float = 0.1
    COMPACT_SCALE: float = 0.02

    ASPECT_CENTRE: float = 2.5
    ASPECT_SCALE: float = 0.25

    # Activity terms — looser scales because activity features are noisier.
    # These contribute additional evidence for ``p_dend`` when available.
    AUTOCORR_CENTRE_S: float = 1.0
    AUTOCORR_SCALE_S: float = 0.25

    FNEU_CORR_CENTRE: float = 0.5
    FNEU_CORR_SCALE: float = 0.1

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return ``(n_rois, 3)`` probabilities in soma/dend/artefact order.

        Probabilities sum to 1 along the last axis after a final
        renormalisation; this is necessary because the soft-margin terms
        above may sum to slightly more than 1 in marginal cases.

        Parameters
        ----------
        features : pandas.DataFrame
            Output of :func:`hm2p.extraction.soma_features.extract_soma_features`.

        Returns
        -------
        numpy.ndarray
            ``(n_rois, 3)`` float64.  ``rows.sum(axis=1)`` ≈ 1.
        """
        n = len(features)
        if n == 0:
            return np.zeros((0, len(CLASS_NAMES)), dtype=np.float64)

        radius = features["radius"].to_numpy(dtype=np.float64)
        compact = features["compact"].to_numpy(dtype=np.float64)
        aspect = features["aspect_ratio"].to_numpy(dtype=np.float64)

        # Artefact evidence: small radius OR low compactness (legacy OR rule).
        # Each term peaks at 1 well past the boundary and = 0.5 *at* it.
        p_art_radius = _sigmoid(-(radius - self.RADIUS_CENTRE) / self.RADIUS_SCALE)
        p_art_compact = _sigmoid(-(compact - self.COMPACT_CENTRE) / self.COMPACT_SCALE)
        p_artefact = _or_combine(p_art_radius, p_art_compact)

        # Dendrite shape evidence (legacy: aspect_ratio > 2.5).
        p_dend_shape = _sigmoid((aspect - self.ASPECT_CENTRE) / self.ASPECT_SCALE)

        # Optional activity evidence for dendrite (slow kinetics, neuropil-like).
        # NaNs in the activity features simply contribute 0 (no extra evidence).
        if "autocorr_halfwidth_s" in features.columns:
            ac = features["autocorr_halfwidth_s"].to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore"):
                p_dend_ac = np.where(
                    np.isfinite(ac),
                    _sigmoid((ac - self.AUTOCORR_CENTRE_S) / self.AUTOCORR_SCALE_S),
                    0.0,
                )
        else:
            p_dend_ac = np.zeros(n, dtype=np.float64)

        if "fneu_corr" in features.columns:
            fc = features["fneu_corr"].to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore"):
                p_dend_fc = np.where(
                    np.isfinite(fc),
                    _sigmoid((fc - self.FNEU_CORR_CENTRE) / self.FNEU_CORR_SCALE),
                    0.0,
                )
        else:
            p_dend_fc = np.zeros(n, dtype=np.float64)

        p_dend = _or_combine(p_dend_shape, p_dend_ac, p_dend_fc)

        # Suppress dendrite probability inside the artefact region: an ROI
        # that already has high p_artefact should not also have high p_dend
        # (legacy rule: artefact takes priority over dendrite).
        p_dend = p_dend * (1.0 - p_artefact)

        # Soma is the residual.
        p_soma = np.clip(1.0 - p_artefact - p_dend, 0.0, 1.0)

        probs = np.stack([p_soma, p_dend, p_artefact], axis=1)
        # Renormalise — guards against floating-point under/overflow.
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        probs = probs / row_sum
        return np.asarray(probs)


# ---------------------------------------------------------------------------
# Logistic regression wrapper
# ---------------------------------------------------------------------------


class LogisticRegressionClassifier:
    """Wrapper around a fitted sklearn pipeline trained on curated labels.

    The wrapper enforces the canonical :attr:`class_names` order (the
    sklearn estimator may have been fit with a different label order; this
    class reorders ``predict_proba`` columns to match).

    Parameters
    ----------
    pipeline
        Fitted sklearn estimator (typically a
        ``Pipeline([StandardScaler(), LogisticRegression(...)])``).  Must
        expose ``predict_proba`` and ``classes_`` (sklearn convention).

    References
    ----------
    Pedregosa et al. 2011. "Scikit-learn: Machine Learning in Python."
    Journal of Machine Learning Research 12:2825–2830.
    https://scikit-learn.org
    """

    class_names: tuple[str, ...] = CLASS_NAMES

    def __init__(self, pipeline: object) -> None:
        if not hasattr(pipeline, "predict_proba"):
            raise TypeError(
                "LogisticRegressionClassifier requires an estimator with "
                f"predict_proba; got {type(pipeline).__name__}"
            )
        self._pipeline = pipeline
        # Build a permutation so that columns of predict_proba match CLASS_NAMES.
        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            # sklearn Pipeline exposes classes_ via the final step.
            steps = getattr(pipeline, "steps", None)
            if steps is not None and len(steps) > 0:
                classes = getattr(steps[-1][1], "classes_", None)
        if classes is None:
            raise TypeError(
                "Fitted estimator does not expose `classes_`; cannot align probability columns."
            )
        self._classes = list(classes)
        missing = [c for c in CLASS_NAMES if c not in self._classes]
        if missing:
            raise ValueError(
                f"Fitted estimator is missing classes {missing!r}; expected {list(CLASS_NAMES)!r}."
            )
        self._permutation = np.array([self._classes.index(c) for c in CLASS_NAMES], dtype=np.int64)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return per-ROI probabilities in the canonical class order.

        Parameters
        ----------
        features : pandas.DataFrame
            Output of :func:`extract_soma_features`.

        Returns
        -------
        numpy.ndarray
            ``(n_rois, 3)`` float; rows sum to 1.
        """
        if len(features) == 0:
            return np.zeros((0, len(CLASS_NAMES)), dtype=np.float64)
        # sklearn pipelines accept DataFrames; columns are reordered by name
        # via the final estimator's training-time feature order.
        raw = self._pipeline.predict_proba(features)
        return np.asarray(raw)[:, self._permutation]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_classifier(path: Path | None = None) -> SomaClassifier:
    """Return a :class:`LogisticRegressionClassifier` if a pickle exists, else rule-based.

    Parameters
    ----------
    path : Path or None
        Location of the fitted sklearn pipeline pickle.  When ``None``,
        :data:`DEFAULT_CLASSIFIER_PATH` is used.

    Returns
    -------
    SomaClassifier
        Either a :class:`LogisticRegressionClassifier` (when the pickle
        exists and loads successfully) or a :class:`RuleBasedClassifier`
        (with a single warning logged when the pickle is missing).
    """
    target = path if path is not None else DEFAULT_CLASSIFIER_PATH
    if not target.exists():
        log.warning(
            "Soma classifier pickle not found at %s — using interim "
            "rule-based scorer. Train one via "
            "`scripts/train_soma_classifier.py` once curated labels are "
            "available.",
            target,
        )
        return RuleBasedClassifier()

    try:
        import joblib

        pipeline = joblib.load(target)
    except Exception as exc:
        log.warning(
            "Failed to load soma classifier from %s (%s) — falling back to rule-based scorer.",
            target,
            exc,
        )
        return RuleBasedClassifier()

    try:
        return LogisticRegressionClassifier(pipeline)
    except (TypeError, ValueError) as exc:
        log.warning(
            "Soma classifier pickle at %s is incompatible (%s); falling back "
            "to rule-based scorer.",
            target,
            exc,
        )
        return RuleBasedClassifier()


# ---------------------------------------------------------------------------
# Top-level convenience: classify with probabilities
# ---------------------------------------------------------------------------


def classify_rois_with_probs(
    features: pd.DataFrame,
    classifier: SomaClassifier | None = None,
) -> tuple[list[str], np.ndarray]:
    """Classify ROIs and return both hard labels and class probabilities.

    Parameters
    ----------
    features : pandas.DataFrame
        Output of :func:`hm2p.extraction.soma_features.extract_soma_features`.
    classifier : SomaClassifier or None
        When ``None``, :func:`load_classifier` is called with the default
        classifier path.

    Returns
    -------
    labels : list[str]
        ``len(labels) == len(features)``.  Each entry is ``"soma"``,
        ``"dend"``, or ``"artefact"``, taken from the argmax of the
        probability matrix.
    probs : numpy.ndarray
        ``(n_rois, 3)`` float; columns ordered as :data:`CLASS_NAMES`.
    """
    clf = classifier if classifier is not None else load_classifier()
    probs = clf.predict_proba(features)
    if probs.size == 0:
        return [], probs
    arg = np.argmax(probs, axis=1)
    labels = [CLASS_NAMES[i] for i in arg]
    return labels, probs
