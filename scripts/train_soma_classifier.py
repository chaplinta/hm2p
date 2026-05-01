"""Train and evaluate a soma/dend/artefact classifier from curated labels.

Reads a CSV with one row per labelled ROI (``session_id, roi_index, label``),
re-extracts features for each ROI from the corresponding ``ca.h5``, fits a
``Pipeline([StandardScaler, LogisticRegression])`` with 5-fold stratified
cross-validation, and saves the fitted pipeline to disk via ``joblib``.

Usage
-----
    python -m scripts.train_soma_classifier \
        --labels labels.csv \
        --output sourcedata/trackers/suite2p/soma_classifier.pkl \
        --report-dir reports/soma_classifier/

Labels CSV
----------
The CSV must contain at least these columns (additional columns are
ignored):

    session_id,roi_index,label
    20220804_13_52_02_1117646,0,soma
    20220804_13_52_02_1117646,1,artefact
    ...

``label`` must be one of ``"soma"``, ``"dend"``, or ``"artefact"``.

ca.h5 location
--------------
Each session's ``ca.h5`` is looked up at
``derivatives/calcium/<sub>/<ses>/ca.h5`` relative to the current working
directory by default; pass ``--ca-root`` to point at a different parent
directory. The script also re-reads the corresponding raw Suite2p
``F.npy``, ``Fneu.npy``, and ``stat.npy`` from
``derivatives/ca_extraction/<sub>/<ses>/suite2p/plane0/``.

Outputs
-------
* ``--output`` — the fitted sklearn pipeline (joblib pickle).
* ``--report-dir/cv_report.csv`` — per-fold metrics (macro F1, per-class P/R/F1).
* ``--report-dir/confusion_matrix.csv`` — aggregate confusion matrix.
* ``--report-dir/feature_coefficients.csv`` — fitted LR coefficients per
  class × feature (for interpretability).

References
----------
Pedregosa et al. 2011. "Scikit-learn: Machine Learning in Python."
Journal of Machine Learning Research 12:2825–2830.
https://scikit-learn.org

Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
two-photon microscopy." bioRxiv. doi:10.1101/061507.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hm2p.extraction.soma_classifier import CLASS_NAMES
from hm2p.extraction.soma_features import FEATURE_COLUMNS, extract_soma_features

log = logging.getLogger("hm2p.train_soma_classifier")


def _session_to_sub_ses(session_id: str) -> tuple[str, str]:
    """Convert ``YYYYMMDD_HH_MM_SS_<animal_id>`` to ``(sub-..., ses-...)``."""
    parts = session_id.split("_")
    if len(parts) < 5:
        raise ValueError(
            f"Unrecognised session_id format: {session_id!r}; expected "
            "YYYYMMDD_HH_MM_SS_<animal_id>."
        )
    animal_id = parts[-1]
    sub = f"sub-{animal_id}"
    ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
    return sub, ses


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Train a soma/dend/artefact classifier from curated Suite2p labels."),
    )
    p.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels CSV (columns: session_id, roi_index, label).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Where to save the fitted sklearn pipeline pickle "
            "(e.g. sourcedata/trackers/suite2p/soma_classifier.pkl)."
        ),
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Directory to write CV reports to (created if absent).",
    )
    p.add_argument(
        "--ca-root",
        type=Path,
        default=Path("derivatives"),
        help=(
            "Root directory containing per-session derivatives. "
            "ca.h5 is read from <ca-root>/calcium/<sub>/<ses>/ca.h5 and "
            "Suite2p output from <ca-root>/ca_extraction/<sub>/<ses>/suite2p/plane0/."
        ),
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the labels CSV and print what would happen, then exit.",
    )
    return p


def _load_labels(path: Path) -> pd.DataFrame:
    """Return a label table with columns ``session_id``, ``roi_index``, ``label``."""
    df = pd.read_csv(path)
    required = {"session_id", "roi_index", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV {path} is missing required columns: {sorted(missing)}")
    valid = set(CLASS_NAMES)
    bad = sorted(set(df["label"].unique()) - valid)
    if bad:
        raise ValueError(
            f"Labels CSV contains unrecognised label values {bad!r}; "
            f"expected one of {sorted(valid)}."
        )
    return df


def _features_for_session(
    session_id: str,
    ca_root: Path,
) -> pd.DataFrame:
    """Re-extract the per-ROI feature table for a single session.

    Reads ``F.npy``, ``Fneu.npy``, and ``stat.npy`` from the Suite2p output
    directory, pulls ``fps_imaging`` from the session's ``ca.h5``, and
    returns the same feature table as used at runtime.
    """
    sub, ses = _session_to_sub_ses(session_id)
    s2p = ca_root / "ca_extraction" / sub / ses / "suite2p" / "plane0"
    ca_h5 = ca_root / "calcium" / sub / ses / "ca.h5"

    F = np.load(s2p / "F.npy").astype(np.float32)
    Fneu = np.load(s2p / "Fneu.npy").astype(np.float32)
    stat = list(np.load(s2p / "stat.npy", allow_pickle=True))

    # fps from ca.h5 attrs (Stage 4 sets it from frame_times).
    import h5py

    with h5py.File(ca_h5, "r") as f:
        fps = float(f.attrs.get("fps_imaging", 9.6))

    return extract_soma_features(stat, F, Fneu, fps=fps)


def _fit_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int,
):
    """Fit a (StandardScaler → LogisticRegression) pipeline with stratified CV.

    Returns
    -------
    final_pipeline
        Pipeline fit on the full training set.
    cv_report : pandas.DataFrame
        Per-fold metrics.
    confusion : numpy.ndarray
        Aggregated confusion matrix in :data:`CLASS_NAMES` order.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    fold_rows: list[dict[str, float | int | str]] = []
    cm_total = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=500,
                    ),
                ),
            ]
        )
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred = pipe.predict(X.iloc[test_idx])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[test_idx],
            y_pred,
            labels=list(CLASS_NAMES),
            zero_division=0,
        )
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "class": cls_name,
                    "precision": float(precision[cls_idx]),
                    "recall": float(recall[cls_idx]),
                    "f1": float(f1[cls_idx]),
                }
            )
        cm_total += confusion_matrix(y[test_idx], y_pred, labels=list(CLASS_NAMES)).astype(
            np.int64
        )

    cv_df = pd.DataFrame(fold_rows)
    macro_f1 = cv_df.groupby("fold")["f1"].mean()
    log.info(
        "Cross-validation macro-F1: mean=%.3f, std=%.3f, per-fold=%s",
        float(macro_f1.mean()),
        float(macro_f1.std()),
        macro_f1.tolist(),
    )

    # Final fit on the full training set.
    final = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=500,
                ),
            ),
        ]
    )
    final.fit(X, y)
    return final, cv_df, cm_total


def _write_reports(
    report_dir: Path,
    cv_df: pd.DataFrame,
    cm_total: np.ndarray,
    final_pipeline,
    feature_columns: tuple[str, ...],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(report_dir / "cv_report.csv", index=False)

    cm_df = pd.DataFrame(
        cm_total,
        index=[f"true_{c}" for c in CLASS_NAMES],
        columns=[f"pred_{c}" for c in CLASS_NAMES],
    )
    cm_df.to_csv(report_dir / "confusion_matrix.csv")

    # Coefficients, in case curators want to look at feature importance.
    clf = final_pipeline.named_steps.get("clf")
    if clf is not None and hasattr(clf, "coef_"):
        # Map sklearn classes_ back to CLASS_NAMES order.
        sk_classes = list(getattr(clf, "classes_", []))
        coef_rows: list[dict[str, float | str]] = []
        for cls_name in CLASS_NAMES:
            if cls_name not in sk_classes:
                continue
            ci = sk_classes.index(cls_name)
            for fi, fname in enumerate(feature_columns):
                coef_rows.append(
                    {"class": cls_name, "feature": fname, "coef": float(clf.coef_[ci, fi])}
                )
        if coef_rows:
            pd.DataFrame(coef_rows).to_csv(report_dir / "feature_coefficients.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_argparser().parse_args(argv)

    labels = _load_labels(args.labels)
    log.info(
        "Loaded %d labelled ROIs across %d sessions from %s",
        len(labels),
        labels["session_id"].nunique(),
        args.labels,
    )

    if args.dry_run:
        for session_id, group in labels.groupby("session_id"):
            counts = group["label"].value_counts().to_dict()
            log.info("  %s: %d ROIs %s", session_id, len(group), counts)
        log.info("Dry run — exiting before feature extraction.")
        return 0

    # Build feature table by re-extracting per session.
    feature_rows: list[pd.DataFrame] = []
    label_arrays: list[np.ndarray] = []
    for session_id, group in labels.groupby("session_id"):
        try:
            features = _features_for_session(str(session_id), args.ca_root)
        except FileNotFoundError as exc:
            log.error("Session %s missing extraction artefact: %s", session_id, exc)
            continue
        roi_indices = group["roi_index"].to_numpy(dtype=np.int64)
        if (roi_indices < 0).any() or (roi_indices >= len(features)).any():
            raise ValueError(
                f"Session {session_id}: roi_index out of range "
                f"(features have {len(features)} rows; received indices "
                f"{roi_indices.min()}..{roi_indices.max()})."
            )
        sub_features = features.iloc[roi_indices].reset_index(drop=True)
        feature_rows.append(sub_features)
        label_arrays.append(group["label"].to_numpy())

    if not feature_rows:
        log.error("No usable sessions — aborting.")
        return 2

    X = pd.concat(feature_rows, ignore_index=True)
    y = np.concatenate(label_arrays)
    log.info(
        "Aggregated feature matrix: %s; class counts: %s",
        X.shape,
        dict(pd.Series(y).value_counts()),
    )

    final_pipeline, cv_df, cm_total = _fit_pipeline(X, y, cv_folds=args.cv_folds)
    _write_reports(args.report_dir, cv_df, cm_total, final_pipeline, FEATURE_COLUMNS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(final_pipeline, args.output)
    log.info("Saved fitted pipeline to %s", args.output)
    log.info("CV report written to %s/", args.report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
