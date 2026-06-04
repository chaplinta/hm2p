"""Train the champion XGBoost model on ALL data and save it.

Uses the best hyperparameters from the Optuna run, but trains on the
full dataset (no train/test split) to maximize the model's exposure to
all labeled ROIs.

Usage
-----
    python scripts/save_champion_model.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.extraction.soma_features import FEATURE_COLUMNS, extract_soma_features

log = logging.getLogger("hm2p.save_champion")

S2P_ROOT = Path("/data/s2p")
OUT_DIR = Path("sourcedata/trackers/suite2p")
LABEL_NAMES = ["artefact", "soma", "dend"]

# Best params from Optuna run (50 trials, 26 features, test F1=0.826)
BEST_PARAMS = {
    "n_estimators": 245,
    "max_depth": 7,
    "learning_rate": 0.176,
    "min_child_weight": 1,
    "subsample": 0.993,
    "colsample_bytree": 0.851,
    "gamma": 0.868,
    "reg_alpha": 5.04e-6,
    "reg_lambda": 7.70e-8,
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    log.info("Loading all sessions from %s", S2P_ROOT)
    t0 = time.time()

    feature_list = []
    label_list = []

    for session_dir in sorted(S2P_ROOT.iterdir()):
        if not session_dir.is_dir():
            continue
        soma_dir = session_dir / "suite2p_soma" / "plane0"
        dend_dir = session_dir / "suite2p_dend" / "plane0"
        if not soma_dir.exists() or not dend_dir.exists():
            continue

        ic_soma = np.load(soma_dir / "iscell.npy")
        ic_dend = np.load(dend_dir / "iscell.npy")
        n_soma = int((ic_soma[:, 0] == 1).sum())
        n_dend = int((ic_dend[:, 0] == 1).sum())
        if n_soma + n_dend == 0:
            continue

        n_rois = len(ic_soma)
        labels = np.zeros(n_rois, dtype=np.int64)
        labels[ic_soma[:, 0] == 1] = 1
        labels[ic_dend[:, 0] == 1] = 2

        stat = list(np.load(soma_dir / "stat.npy", allow_pickle=True))
        F = np.load(soma_dir / "F.npy").astype(np.float32)
        Fneu = np.load(soma_dir / "Fneu.npy").astype(np.float32)
        ops = np.load(soma_dir / "ops.npy", allow_pickle=True).item()
        fps = float(ops.get("fs", 9.6))

        features = extract_soma_features(stat, F, Fneu, fps=fps)
        feature_list.append(features)
        label_list.append(labels)
        log.info("  %s: %d ROIs (%d soma, %d dend)", session_dir.name, n_rois, n_soma, n_dend)

    X = pd.concat(feature_list, ignore_index=True)
    y = np.concatenate(label_list)
    log.info("Total: %d ROIs in %.1fs", len(y), time.time() - t0)

    # NaN fill with median
    train_medians = X.median()
    X = X.fillna(train_medians)

    # Class weights
    class_counts = np.bincount(y, minlength=3)
    n = len(y)
    class_weights = n / (3 * class_counts)
    sample_weights = np.array([class_weights[label] for label in y])

    # Train on ALL data
    log.info("Training XGBoost on full dataset (%d ROIs, %d features)...", len(y), len(FEATURE_COLUMNS))
    clf = XGBClassifier(
        **BEST_PARAMS,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X.values, y, sample_weight=sample_weights)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUT_DIR / "roi_classifier_xgb.joblib"
    joblib.dump(clf, model_path)
    log.info("Saved model to %s", model_path)

    medians_path = OUT_DIR / "roi_classifier_medians.npy"
    np.save(medians_path, train_medians.values)
    log.info("Saved medians to %s", medians_path)

    meta = {
        "version": "1.0",
        "model_type": "XGBClassifier",
        "n_features": len(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "label_names": LABEL_NAMES,
        "label_encoding": {"artefact": 0, "soma": 1, "dend": 2},
        "n_training_rois": len(y),
        "class_counts": {LABEL_NAMES[i]: int(class_counts[i]) for i in range(3)},
        "best_params": BEST_PARAMS,
        "test_f1_macro_from_split": 0.826,
        "test_f1_dend_from_split": 0.691,
    }
    meta_path = OUT_DIR / "roi_classifier_xgb.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata to %s", meta_path)

    log.info("Champion model saved. Ready for inference via hm2p.extraction.roi_classify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
