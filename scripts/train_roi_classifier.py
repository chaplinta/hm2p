"""Train and compare soma/dend/artefact classifiers on legacy Suite2p labels.

Reads the dual-run legacy Suite2p data from /data/s2p/, constructs 3-way
labels, extracts the 26-feature set, and trains two classifiers with Optuna:

1. Logistic Regression (StandardScaler + balanced class weights)
2. XGBoost (softmax multi-class, sample weights for class balancing)

Evaluation:
- 80/20 stratified train/test split (test held out entirely)
- Optuna with 5-fold stratified CV on the train set (50 trials)
- Final evaluation on the held-out test set

References
----------
Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
two-photon microscopy." bioRxiv. doi:10.1101/061507.

Chen & Guestrin. 2016. "XGBoost: A Scalable Tree Boosting System."
KDD 2016. doi:10.1145/2939672.2939785.

Akiba et al. 2019. "Optuna: A Next-generation Hyperparameter Optimization
Framework." KDD 2019. doi:10.1145/3292500.3330701.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.extraction.soma_features import FEATURE_COLUMNS, extract_soma_features

log = logging.getLogger("hm2p.train_roi_classifier")

S2P_ROOT = Path("/data/s2p")

LABEL_NAMES = ["artefact", "soma", "dend"]
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 50
N_CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_session_data(session_dir: Path) -> dict | None:
    """Load features and 3-way labels for one session from /data/s2p/."""
    soma_dir = session_dir / "suite2p_soma" / "plane0"
    dend_dir = session_dir / "suite2p_dend" / "plane0"

    if not soma_dir.exists() or not dend_dir.exists():
        return None

    ic_soma = np.load(soma_dir / "iscell.npy")
    ic_dend = np.load(dend_dir / "iscell.npy")

    n_soma = int((ic_soma[:, 0] == 1).sum())
    n_dend = int((ic_dend[:, 0] == 1).sum())

    if n_soma + n_dend == 0:
        return None

    n_rois = len(ic_soma)
    labels = np.zeros(n_rois, dtype=np.int64)
    labels[ic_soma[:, 0] == 1] = 1
    labels[ic_dend[:, 0] == 1] = 2

    overlap = ((ic_soma[:, 0] == 1) & (ic_dend[:, 0] == 1)).sum()
    if overlap > 0:
        log.warning(
            "%s: %d ROIs labeled as both soma and dend — skipping",
            session_dir.name, overlap,
        )
        return None

    stat = list(np.load(soma_dir / "stat.npy", allow_pickle=True))
    F = np.load(soma_dir / "F.npy").astype(np.float32)
    Fneu = np.load(soma_dir / "Fneu.npy").astype(np.float32)
    ops = np.load(soma_dir / "ops.npy", allow_pickle=True).item()
    fps = float(ops.get("fs", 9.6))

    features = extract_soma_features(stat, F, Fneu, fps=fps)

    return {
        "session_id": session_dir.name,
        "features": features,
        "labels": labels,
        "n_soma": n_soma,
        "n_dend": n_dend,
        "n_artefact": n_rois - n_soma - n_dend,
        "n_rois": n_rois,
    }


def load_all_sessions() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load features, labels, and session IDs for all usable sessions."""
    feature_list = []
    label_list = []
    group_list = []

    for session_dir in sorted(S2P_ROOT.iterdir()):
        if not session_dir.is_dir():
            continue
        data = load_session_data(session_dir)
        if data is None:
            log.info("Skipping %s (no labeled cells)", session_dir.name)
            continue

        feature_list.append(data["features"])
        label_list.append(data["labels"])
        group_list.append(np.full(data["n_rois"], data["session_id"]))

        log.info(
            "  %s: %d ROIs (%d soma, %d dend, %d artefact)",
            data["session_id"], data["n_rois"],
            data["n_soma"], data["n_dend"], data["n_artefact"],
        )

    X = pd.concat(feature_list, ignore_index=True)
    y = np.concatenate(label_list)
    groups = np.concatenate(group_list)

    return X, y, groups


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def get_feature_importance(clf, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "named_steps"):
        lr = clf.named_steps.get("clf")
        if lr is not None and hasattr(lr, "coef_"):
            imp = np.abs(lr.coef_).mean(axis=0)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    return pd.DataFrame({
        "feature": feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------


def logreg_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: StratifiedKFold,
) -> float:
    C = trial.suggest_float("C", 1e-4, 100.0, log=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])

    scores = cross_val_score(
        pipe, X_train, y_train, cv=cv,
        scoring="f1_macro", n_jobs=-1,
    )
    return scores.mean()


def xgboost_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    cv: StratifiedKFold,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        clf = XGBClassifier(
            **params,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=1,  # single-threaded per fold; parallelism via Optuna
        )
        clf.fit(
            X_train[train_idx], y_train[train_idx],
            sample_weight=sample_weights[train_idx],
        )
        y_pred = clf.predict(X_train[val_idx])
        f1 = f1_score(
            y_train[val_idx], y_pred,
            labels=[0, 1, 2], average="macro", zero_division=0,
        )
        scores.append(f1)

    return np.mean(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log.info("Loading sessions from %s", S2P_ROOT)
    t0 = time.time()
    X, y, groups = load_all_sessions()
    t_load = time.time() - t0

    n_sessions = len(np.unique(groups))
    counts = {LABEL_NAMES[i]: int((y == i).sum()) for i in range(3)}
    log.info(
        "Loaded %d ROIs from %d sessions in %.1fs: %s",
        len(y), n_sessions, t_load, counts,
    )

    nan_counts = X.isna().sum()
    if nan_counts.any():
        log.info("NaN counts per feature:\n%s", nan_counts[nan_counts > 0].to_string())

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    log.info(
        "Train: %d ROIs (%s)",
        len(y_train),
        {LABEL_NAMES[i]: int((y_train == i).sum()) for i in range(3)},
    )
    log.info(
        "Test:  %d ROIs (%s)",
        len(y_test),
        {LABEL_NAMES[i]: int((y_test == i).sum()) for i in range(3)},
    )

    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # Class weights for XGBoost
    class_counts_train = np.bincount(y_train, minlength=3)
    n_train = len(y_train)
    class_weights = n_train / (3 * class_counts_train)
    sample_weights_train = np.array([class_weights[label] for label in y_train])

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    # Try W&B
    # ------------------------------------------------------------------
    wandb_available = False
    try:
        import wandb
        wandb.init(
            project="hm2p-roi-classifier",
            config={
                "n_sessions": n_sessions,
                "n_rois_total": len(y),
                "n_rois_train": len(y_train),
                "n_rois_test": len(y_test),
                "n_features": len(FEATURE_COLUMNS),
                "feature_names": list(FEATURE_COLUMNS),
                "class_counts": counts,
                "n_optuna_trials": N_OPTUNA_TRIALS,
            },
            tags=["roi-classifier", "optuna", "26-features"],
        )
        wandb_available = True
        log.info("W&B run: %s", wandb.run.url)
    except Exception as e:
        log.warning("W&B not available (%s) — printing results only.", e)

    # ------------------------------------------------------------------
    # 1. Logistic Regression
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("Logistic Regression — Optuna (%d trials)", N_OPTUNA_TRIALS)
    log.info("=" * 60)

    t0 = time.time()
    lr_study = optuna.create_study(
        direction="maximize", study_name="logreg",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    lr_study.optimize(
        lambda trial: logreg_objective(trial, X_train, y_train, cv),
        n_trials=N_OPTUNA_TRIALS,
    )
    lr_time = time.time() - t0

    lr_best = lr_study.best_params
    log.info("  Best params: %s", lr_best)
    log.info("  Best CV F1 macro: %.4f (%.1fs)", lr_study.best_value, lr_time)

    lr_final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=lr_best["C"],
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])
    lr_final.fit(X_train, y_train)

    lr_pred = lr_final.predict(X_test)
    lr_f1 = f1_score(y_test, lr_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    lr_f1_per = f1_score(y_test, lr_pred, labels=[0, 1, 2], average=None, zero_division=0)
    lr_cm = confusion_matrix(y_test, lr_pred, labels=[0, 1, 2])
    lr_report = classification_report(y_test, lr_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0)

    log.info("  Test F1 macro: %.4f  (art=%.3f, soma=%.3f, dend=%.3f)",
             lr_f1, lr_f1_per[0], lr_f1_per[1], lr_f1_per[2])
    log.info("\n%s", lr_report)
    log.info("  Confusion matrix:\n%s", lr_cm)

    lr_importance = get_feature_importance(lr_final, list(FEATURE_COLUMNS))
    if len(lr_importance) > 0:
        log.info("\n  Feature importance:\n%s", lr_importance.to_string(index=False))

    # ------------------------------------------------------------------
    # 2. XGBoost
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("XGBoost — Optuna (%d trials)", N_OPTUNA_TRIALS)
    log.info("=" * 60)

    # Convert to numpy for XGBoost CV (avoids DataFrame overhead per fold)
    X_train_np = X_train.values
    y_train_np = y_train

    t0 = time.time()
    xgb_study = optuna.create_study(
        direction="maximize", study_name="xgboost",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    xgb_study.optimize(
        lambda trial: xgboost_objective(trial, X_train_np, y_train_np, sample_weights_train, cv),
        n_trials=N_OPTUNA_TRIALS,
        n_jobs=4,  # parallel Optuna trials; each XGB fit is single-threaded
    )
    xgb_time = time.time() - t0

    xgb_best = xgb_study.best_params
    log.info("  Best params: %s", xgb_best)
    log.info("  Best CV F1 macro: %.4f (%.1fs)", xgb_study.best_value, xgb_time)

    xgb_final = XGBClassifier(
        **xgb_best,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_final.fit(X_train_np, y_train_np, sample_weight=sample_weights_train)

    xgb_pred = xgb_final.predict(X_test.values)
    xgb_f1 = f1_score(y_test, xgb_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    xgb_f1_per = f1_score(y_test, xgb_pred, labels=[0, 1, 2], average=None, zero_division=0)
    xgb_cm = confusion_matrix(y_test, xgb_pred, labels=[0, 1, 2])
    xgb_report = classification_report(y_test, xgb_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0)

    log.info("  Test F1 macro: %.4f  (art=%.3f, soma=%.3f, dend=%.3f)",
             xgb_f1, xgb_f1_per[0], xgb_f1_per[1], xgb_f1_per[2])
    log.info("\n%s", xgb_report)
    log.info("  Confusion matrix:\n%s", xgb_cm)

    xgb_importance = get_feature_importance(xgb_final, list(FEATURE_COLUMNS))
    if len(xgb_importance) > 0:
        log.info("\n  Feature importance:\n%s", xgb_importance.to_string(index=False))

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    if wandb_available:
        import wandb
        for name, study, f1, f1_per, cm, importance, t_elapsed in [
            ("logreg", lr_study, lr_f1, lr_f1_per, lr_cm, lr_importance, lr_time),
            ("xgboost", xgb_study, xgb_f1, xgb_f1_per, xgb_cm, xgb_importance, xgb_time),
        ]:
            wandb.log({
                f"{name}/best_cv_f1_macro": study.best_value,
                f"{name}/test_f1_macro": f1,
                f"{name}/test_f1_artefact": f1_per[0],
                f"{name}/test_f1_soma": f1_per[1],
                f"{name}/test_f1_dend": f1_per[2],
                f"{name}/best_params": study.best_params,
                f"{name}/train_time_s": t_elapsed,
            })
            cm_table = wandb.Table(
                columns=[""] + [f"pred_{n}" for n in LABEL_NAMES],
                data=[[f"true_{LABEL_NAMES[i]}"] + cm[i].tolist() for i in range(3)],
            )
            wandb.log({f"{name}/confusion_matrix": cm_table})
            if len(importance) > 0:
                imp_table = wandb.Table(dataframe=importance)
                wandb.log({f"{name}/feature_importance": wandb.plot.bar(
                    imp_table, "feature", "importance", title=f"{name} Feature Importance")})
        wandb.log({"best_model": "xgboost" if xgb_f1 > lr_f1 else "logreg",
                    "best_test_f1_macro": max(lr_f1, xgb_f1)})
        wandb.finish()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("COMPARISON SUMMARY (held-out test set)")
    log.info("=" * 60)
    log.info("  LogReg:  CV=%.4f  Test=%.4f (art=%.3f soma=%.3f dend=%.3f)  params=%s",
             lr_study.best_value, lr_f1, lr_f1_per[0], lr_f1_per[1], lr_f1_per[2], lr_best)
    log.info("  XGBoost: CV=%.4f  Test=%.4f (art=%.3f soma=%.3f dend=%.3f)  params=%s",
             xgb_study.best_value, xgb_f1, xgb_f1_per[0], xgb_f1_per[1], xgb_f1_per[2], xgb_best)
    log.info("  Winner: %s", "XGBoost" if xgb_f1 > lr_f1 else "Logistic Regression")

    # ------------------------------------------------------------------
    # Save champion model
    # ------------------------------------------------------------------
    import json

    import joblib

    model_dir = Path("sourcedata/trackers/suite2p")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "roi_classifier_xgb.joblib"
    meta_path = model_dir / "roi_classifier_xgb.json"
    medians_path = model_dir / "roi_classifier_medians.npy"

    # Save model
    joblib.dump(xgb_final, model_path)
    log.info("Saved XGBoost model to %s", model_path)

    # Save NaN-fill medians (needed at inference time)
    np.save(medians_path, train_medians.values)
    log.info("Saved train medians to %s", medians_path)

    # Save metadata
    meta = {
        "version": "1.0",
        "model_type": "XGBClassifier",
        "n_features": len(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "label_names": LABEL_NAMES,
        "label_encoding": {"artefact": 0, "soma": 1, "dend": 2},
        "n_training_rois": len(y_train),
        "n_test_rois": len(y_test),
        "class_counts_train": {LABEL_NAMES[i]: int((y_train == i).sum()) for i in range(3)},
        "test_f1_macro": float(xgb_f1),
        "test_f1_per_class": {LABEL_NAMES[i]: float(xgb_f1_per[i]) for i in range(3)},
        "best_params": xgb_best,
        "optuna_trials": N_OPTUNA_TRIALS,
        "random_state": RANDOM_STATE,
        "median_feature_names": list(FEATURE_COLUMNS),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata to %s", meta_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
