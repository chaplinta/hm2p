"""ROI classification inference — apply trained XGBoost model to Suite2p outputs.

Loads the champion XGBoost model from
``sourcedata/trackers/suite2p/roi_classifier_xgb.joblib``, extracts the
26-feature set from a session's Suite2p outputs, and writes classification
results back into the Suite2p directory.

Outputs
-------
For each session's ``suite2p/plane0/`` directory:

- ``iscell.npy`` — **overwritten** to match the classifier output.
  Column 0: 1.0 for soma, 0.0 for dendrite and artefact.
  Column 1: P(soma) from the classifier.
  This ensures the Suite2p GUI shows only soma as "cells", consistent
  with downstream analysis.

- ``roi_class.npy`` — shape ``(n_rois,)``, dtype int8.
  Values: 0 = artefact, 1 = soma, 2 = dendrite.
  This is the primary classification output used by downstream stages.

- ``roi_class_prob.npy`` — shape ``(n_rois, 3)``, dtype float32.
  Columns: [P(artefact), P(soma), P(dendrite)].
  Probabilities from the XGBoost model, for threshold adjustment.

Integration
-----------
Call :func:`classify_session` after Suite2p extraction (end of Stage 1).
The function is self-contained: it loads the model, extracts features,
classifies, and writes outputs.

References
----------
Chen & Guestrin. 2016. "XGBoost: A Scalable Tree Boosting System."
KDD 2016. doi:10.1145/2939672.2939785.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hm2p.extraction.soma_features import FEATURE_COLUMNS, extract_soma_features

log = logging.getLogger(__name__)

# Default paths relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = _REPO_ROOT / "sourcedata" / "trackers" / "suite2p" / "roi_classifier_xgb.joblib"
META_PATH = _REPO_ROOT / "sourcedata" / "trackers" / "suite2p" / "roi_classifier_xgb.json"
MEDIANS_PATH = _REPO_ROOT / "sourcedata" / "trackers" / "suite2p" / "roi_classifier_medians.npy"

# Label encoding — must match training script.
LABEL_NAMES = ("artefact", "soma", "dend")


def load_model(
    model_path: Path | None = None,
    medians_path: Path | None = None,
) -> tuple[Any, np.ndarray, dict]:
    """Load the trained XGBoost classifier and associated metadata.

    Parameters
    ----------
    model_path : Path or None
        Path to the joblib model file. Defaults to the repo-relative path.
    medians_path : Path or None
        Path to the NaN-fill medians. Defaults to the repo-relative path.

    Returns
    -------
    model : XGBClassifier
    medians : (n_features,) float64 array for NaN filling
    meta : dict with model metadata
    """
    mp = model_path or MODEL_PATH
    mdp = medians_path or MEDIANS_PATH
    mtp = META_PATH if model_path is None else mp.with_suffix(".json")

    if not mp.exists():
        raise FileNotFoundError(f"ROI classifier model not found at {mp}")

    model = joblib.load(mp)
    medians = np.load(mdp)
    meta = {}
    if mtp.exists():
        with open(mtp) as f:
            meta = json.load(f)

    log.info(
        "Loaded ROI classifier from %s (v%s, %d features)",
        mp,
        meta.get("version", "?"),
        len(medians),
    )
    return model, medians, meta


def classify_session(
    plane_dir: Path,
    fps: float | None = None,
    model_path: Path | None = None,
    medians_path: Path | None = None,
    neucoeff: float = 0.7,
) -> dict:
    """Classify all ROIs in a Suite2p session directory.

    Reads stat.npy, F.npy, Fneu.npy from ``plane_dir``, extracts features,
    runs the XGBoost classifier, and writes roi_class.npy, roi_class_prob.npy,
    and overwrites iscell.npy.

    Parameters
    ----------
    plane_dir : Path
        Suite2p plane directory (e.g. ``derivatives/.../suite2p/plane0/``).
    fps : float or None
        Imaging frame rate. If None, reads from ops.npy.
    model_path : Path or None
        Override model location.
    medians_path : Path or None
        Override medians location.
    neucoeff : float
        Neuropil subtraction coefficient.

    Returns
    -------
    dict with keys:
        "labels" : (n_rois,) int8 array (0=artefact, 1=soma, 2=dend)
        "probs" : (n_rois, 3) float32 array
        "n_soma" : int
        "n_dend" : int
        "n_artefact" : int
    """
    # Load Suite2p outputs
    stat = list(np.load(plane_dir / "stat.npy", allow_pickle=True))
    F = np.load(plane_dir / "F.npy").astype(np.float32)
    Fneu = np.load(plane_dir / "Fneu.npy").astype(np.float32)

    if fps is None:
        ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
        fps = float(ops.get("fs", 9.6))

    n_rois = len(stat)
    if n_rois == 0:
        labels = np.array([], dtype=np.int8)
        probs = np.zeros((0, 3), dtype=np.float32)
        _write_outputs(plane_dir, labels, probs)
        return {"labels": labels, "probs": probs, "n_soma": 0, "n_dend": 0, "n_artefact": 0}

    # Load model
    model, medians, meta = load_model(model_path, medians_path)

    # Extract features
    features = extract_soma_features(stat, F, Fneu, fps=fps, neucoeff=neucoeff)

    # Fill NaN with training medians
    medians_series = pd.Series(medians, index=list(FEATURE_COLUMNS))
    features = features.fillna(medians_series)

    # Predict
    probs = model.predict_proba(features.values).astype(np.float32)
    labels = np.argmax(probs, axis=1).astype(np.int8)

    # Write outputs
    _write_outputs(plane_dir, labels, probs)

    n_soma = int((labels == 1).sum())
    n_dend = int((labels == 2).sum())
    n_artefact = int((labels == 0).sum())

    log.info(
        "Classified %d ROIs in %s: %d soma, %d dend, %d artefact",
        n_rois,
        plane_dir.name,
        n_soma,
        n_dend,
        n_artefact,
    )

    return {
        "labels": labels,
        "probs": probs,
        "n_soma": n_soma,
        "n_dend": n_dend,
        "n_artefact": n_artefact,
    }


def _write_outputs(
    plane_dir: Path,
    labels: np.ndarray,
    probs: np.ndarray,
) -> None:
    """Write classification outputs to the Suite2p plane directory."""
    n_rois = len(labels)

    # roi_class.npy — primary output
    np.save(plane_dir / "roi_class.npy", labels)

    # roi_class_prob.npy — probabilities for threshold adjustment
    np.save(plane_dir / "roi_class_prob.npy", probs)

    # iscell.npy — overwrite to match classifier (soma = cell)
    iscell = np.zeros((n_rois, 2), dtype=np.float64)
    iscell[:, 0] = (labels == 1).astype(np.float64)  # soma = 1, else 0
    if probs.shape[0] > 0:
        iscell[:, 1] = probs[:, 1]  # P(soma) as probability
    np.save(plane_dir / "iscell.npy", iscell)

    log.info("Wrote roi_class.npy, roi_class_prob.npy, iscell.npy to %s", plane_dir)
