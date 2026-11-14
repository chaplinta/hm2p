"""ROI classifier (soma / dendrite / artefact) quality summary.

Per session: class counts, classifier confidence, agreement between the
stored labels (``roi_class.npy``) and the labels carried into ``ca.h5``
(``roi_types``, which uses a different integer code), reproducibility of the
stored probabilities, the 26 classifier features of every ROI, and ROI
outlines over the mean image.

Reference: leave-one-session-out cross-validated predictions on the manually
labelled legacy sessions, refitted with the champion model's training recipe
(hyper-parameters, objective, balanced class weights). The champion model
itself was scored on one random 80/20
split whose confusion matrix was not saved; the grouped cross-validation is
stricter because no ROI from a test session is seen in training.

XGBoost: Chen T, Guestrin C. 2016. "XGBoost: a scalable tree boosting
system." Proceedings of KDD 2016, 785-794. doi:10.1145/2939672.2939785.
https://github.com/dmlc/xgboost
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from hm2p.qc.common import fnum, hist, rounded

LABELS = ("artefact", "soma", "dend")  # roi_class.npy code: 0, 1, 2
CA_TO_CLASS = {0: 1, 1: 2, 2: 0}  # ca.h5 roi_types (0 soma, 1 dend, 2 artefact) -> roi_class code
AMBIGUOUS_MAX_P = 0.6
MAX_IMG_SIDE = 384


def roi_contour(ypix: npt.ArrayLike, xpix: npt.ArrayLike, tol: float = 0.6) -> list[list[float]]:
    """Outline of an ROI footprint as ``[[x, y], ...]`` (pixel units).

    Traces the longest iso-contour of the binary footprint mask and simplifies
    it with Douglas-Peucker (tolerance *tol* px).
    """
    from skimage.measure import approximate_polygon, find_contours

    yp = np.asarray(ypix, dtype=np.int64)
    xp = np.asarray(xpix, dtype=np.int64)
    if yp.size == 0:
        return []
    y0, x0 = yp.min() - 1, xp.min() - 1
    m = np.zeros((yp.max() - y0 + 2, xp.max() - x0 + 2), dtype=float)
    m[yp - y0, xp - x0] = 1.0
    cs = find_contours(m, 0.5)
    if not cs:
        return []
    c = approximate_polygon(max(cs, key=len), tolerance=tol)
    return [[round(float(px + x0), 1), round(float(py + y0), 1)] for py, px in c]


def image_png_b64(img: npt.ArrayLike, max_side: int = MAX_IMG_SIDE) -> dict:
    """Contrast-stretched (1-99.5 percentile) greyscale PNG, base64, plus scale factor."""
    from PIL import Image

    a = np.asarray(img, dtype=np.float64)
    ok = np.isfinite(a)
    lo, hi = np.percentile(a[ok], [1.0, 99.5]) if ok.any() else (0.0, 1.0)
    u8 = (np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(
        np.uint8
    )
    im = Image.fromarray(u8)
    h, w = u8.shape
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return {
        "png": base64.b64encode(buf.getvalue()).decode("ascii"),
        "width": int(w),
        "height": int(h),
    }


def summarise_rois(
    labels: npt.ArrayLike,
    probs: npt.ArrayLike,
    stat: list[dict[str, Any]] | None = None,
    features: pd.DataFrame | None = None,
    mean_img: npt.ArrayLike | None = None,
    ca_roi_types: npt.ArrayLike | None = None,
    recomputed_probs: npt.ArrayLike | None = None,
    roi_qc: dict[str, npt.ArrayLike] | None = None,
) -> dict:
    """JSON-ready classifier-quality summary of one session.

    Parameters
    ----------
    labels, probs
        ``roi_class.npy`` (0 artefact, 1 soma, 2 dend) and ``roi_class_prob.npy``.
    stat
        Suite2p ``stat.npy`` entries (for outlines and centres).
    features
        Classifier feature table, one row per ROI (``extract_soma_features``).
    mean_img
        Suite2p ``ops["meanImg"]``.
    ca_roi_types
        ``roi_types`` from ``ca.h5`` (0 soma, 1 dend, 2 artefact).
    recomputed_probs
        Probabilities from re-running the current model on *features*.
    roi_qc
        ``ca.h5`` ``roi_qc`` group (per-ROI calcium QC metrics).
    """
    lab = np.asarray(labels).astype(np.int64).ravel()
    p = np.asarray(probs, dtype=np.float64).reshape(-1, 3)
    n = lab.size
    if p.shape[0] != n:
        raise ValueError(f"labels ({n}) and probs ({p.shape[0]}) differ in length")
    pmax = p.max(axis=1) if n else np.zeros(0)
    psort = np.sort(p, axis=1) if n else np.zeros((0, 3))
    margin = psort[:, -1] - psort[:, -2] if n else np.zeros(0)
    out: dict = {
        "n_rois": int(n),
        "counts": {name: int((lab == i).sum()) for i, name in enumerate(LABELS)},
        "argmax_matches_label": bool(np.all(p.argmax(axis=1) == lab)) if n else True,
        "n_ambiguous": int((pmax < AMBIGUOUS_MAX_P).sum()),
        "conf_hist": {name: hist(pmax[lab == i], 1 / 3, 1.0, 20) for i, name in enumerate(LABELS)},
        "margin_hist": hist(margin, 0.0, 1.0, 20),
    }
    if ca_roi_types is not None:
        ca = np.asarray(ca_roi_types).astype(np.int64).ravel()
        if ca.size == n:
            mapped = np.array([CA_TO_CLASS.get(int(v), -1) for v in ca])
            out["ca_label_mismatch"] = int((mapped != lab).sum())
        else:
            out["ca_label_mismatch"] = None
            out["ca_label_len"] = int(ca.size)
    if recomputed_probs is not None:
        rp = np.asarray(recomputed_probs, dtype=np.float64).reshape(-1, 3)
        if rp.shape == p.shape:
            out["recomputed_max_abs_diff"] = fnum(np.abs(rp - p).max() if n else 0.0, 6)
            out["recomputed_label_changes"] = int((rp.argmax(axis=1) != lab).sum())
    rois: list[dict] = []
    for i in range(n):
        r: dict = {"i": i, "c": int(lab[i]), "p": rounded(p[i], 3)}
        if stat is not None and i < len(stat):
            st = stat[i]
            med = st.get("med", [np.nan, np.nan])
            r["med"] = [fnum(med[1], 1), fnum(med[0], 1)]  # x, y
            r["npix"] = int(st.get("npix", len(st.get("ypix", []))))
            r["outline"] = roi_contour(st.get("ypix", []), st.get("xpix", []))
        rois.append(r)
    out["rois"] = rois
    if features is not None:
        out["feature_names"] = list(features.columns)
        out["features"] = [rounded(features[c].to_numpy(dtype=float), 4) for c in features.columns]
    if roi_qc is not None:
        out["roi_qc"] = {
            k: rounded(np.asarray(v, dtype=float), 4)
            for k, v in roi_qc.items()
            if np.asarray(v).dtype.kind in "fiub" and np.asarray(v).size == n
        }
    if mean_img is not None:
        out["image"] = image_png_b64(mean_img)
    return out


def cv_reference(
    features: pd.DataFrame,
    labels: npt.ArrayLike,
    groups: npt.ArrayLike,
    params: dict[str, Any],
    n_jobs: int = 1,
    random_state: int = 42,
) -> dict:
    """Leave-one-session-out predictions on manually labelled ROIs.

    Refits an ``XGBClassifier`` once per held-out session with the same recipe
    as ``scripts/train_roi_classifier.py``: hyper-parameters *params*,
    ``multi:softmax`` objective, balanced class weights
    (``n / (3 * count)``, from the training sessions of the fold) and NaN
    features filled with the training-fold medians (so no statistic of the
    held-out session is used). Returns the confusion matrix,
    per-class precision/recall/F1, a reliability table per class, per-session
    accuracy and feature quantiles per manual label (the reference
    distribution for the per-session feature plots).
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from xgboost import XGBClassifier

    y = np.asarray(labels).astype(np.int64)
    g = np.asarray(groups)
    prob = np.zeros((y.size, 3))
    for sess in np.unique(g):
        te = g == sess
        med = features[~te].median()
        Xtr = features[~te].fillna(med).to_numpy(dtype=np.float64)
        Xte = features[te].fillna(med).to_numpy(dtype=np.float64)
        counts = np.bincount(y[~te], minlength=3).astype(np.float64)
        cw = (~te).sum() / (3.0 * np.maximum(counts, 1.0))
        clf = XGBClassifier(
            **params,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
        )
        clf.fit(Xtr, y[~te], sample_weight=cw[y[~te]])
        prob[te] = clf.predict_proba(Xte)
    pred = prob.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    edges = np.linspace(0, 1, 11)
    reliability = {}
    for k, name in enumerate(LABELS):
        idx = np.clip(np.digitize(prob[:, k], edges) - 1, 0, 9)
        reliability[name] = [
            {
                "p_mean": fnum(prob[idx == b, k].mean(), 3),
                "frac_true": fnum((y[idx == b] == k).mean(), 3),
                "n": int((idx == b).sum()),
            }
            for b in range(10)
            if (idx == b).any()
        ]
    per_session = []
    for sess in np.unique(g):
        m = g == sess
        per_session.append(
            {
                "session": str(sess),
                "n": int(m.sum()),
                "acc": fnum((pred[m] == y[m]).mean(), 3),
                "n_true": [int((y[m] == k).sum()) for k in range(3)],
                "n_pred": [int((pred[m] == k).sum()) for k in range(3)],
            }
        )
    fq: dict[str, dict[str, list[float | None] | None]] = {}
    for c in features.columns:
        fq[c] = {}
        for k, name in enumerate(LABELS):
            v = features[c].to_numpy(dtype=float)[y == k]
            v = v[np.isfinite(v)]
            fq[c][name] = (
                rounded(np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]), 4) if v.size else None
            )
    return {
        "n": int(y.size),
        "n_sessions": int(np.unique(g).size),
        "confusion": cm.astype(int).tolist(),
        "precision": rounded(pr, 3),
        "recall": rounded(rc, 3),
        "f1": rounded(f1, 3),
        "support": sup.astype(int).tolist(),
        "macro_f1": fnum(f1.mean(), 3),
        "reliability": reliability,
        "per_session": per_session,
        "feature_quantiles": fq,
    }


def overview_row(s: dict) -> dict:
    """The few numbers shown per session in the cross-session table."""
    return {
        "n_rois": s["n_rois"],
        "n_soma": s["counts"]["soma"],
        "n_dend": s["counts"]["dend"],
        "n_artefact": s["counts"]["artefact"],
        "n_ambiguous": s["n_ambiguous"],
        "ca_label_mismatch": s.get("ca_label_mismatch"),
        "recomputed_label_changes": s.get("recomputed_label_changes"),
    }
