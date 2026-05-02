"""Tests for scripts/train_soma_classifier.py."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import train_soma_classifier as tsc  # noqa: E402

# ---------------------------------------------------------------------------
# _session_to_sub_ses
# ---------------------------------------------------------------------------


class TestSessionToSubSes:
    def test_basic(self) -> None:
        sub, ses = tsc._session_to_sub_ses("20220804_13_52_02_1117646")
        assert sub == "sub-1117646"
        assert ses == "ses-20220804T135202"

    def test_invalid_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised session_id"):
            tsc._session_to_sub_ses("not_a_session")


# ---------------------------------------------------------------------------
# _load_labels
# ---------------------------------------------------------------------------


class TestLoadLabels:
    def test_round_trip(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "session_id": ["s1", "s1", "s2"],
                "roi_index": [0, 1, 0],
                "label": ["soma", "dend", "artefact"],
            }
        )
        path = tmp_path / "labels.csv"
        df.to_csv(path, index=False)
        out = tsc._load_labels(path)
        assert list(out.columns) >= ["session_id", "roi_index", "label"]
        assert len(out) == 3

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"session_id": ["s1"], "label": ["soma"]})
        path = tmp_path / "labels.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            tsc._load_labels(path)

    def test_unknown_label_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "session_id": ["s1"],
                "roi_index": [0],
                "label": ["soma_or_dend"],
            }
        )
        path = tmp_path / "labels.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="unrecognised label values"):
            tsc._load_labels(path)


# ---------------------------------------------------------------------------
# _fit_pipeline
# ---------------------------------------------------------------------------


def _synthetic_dataset(n_per_class: int = 30) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    rows = []
    labels = []
    # Soma
    for _ in range(n_per_class):
        rows.append(
            {
                "radius": rng.normal(6, 1),
                "compact": rng.normal(0.6, 0.1),
                "aspect_ratio": rng.normal(1.5, 0.3),
                "npix": rng.normal(200, 20),
                "npix_norm": rng.normal(1.5, 0.3),
                "skew": rng.normal(1, 0.5),
                "std": rng.normal(1, 0.3),
                "peak_to_noise_dff": rng.normal(5, 1),
                "autocorr_halfwidth_s": rng.normal(0.5, 0.1),
                "fneu_corr": rng.normal(0.2, 0.1),
            }
        )
        labels.append("soma")
    # Dend
    for _ in range(n_per_class):
        rows.append(
            {
                "radius": rng.normal(6, 1),
                "compact": rng.normal(0.4, 0.1),
                "aspect_ratio": rng.normal(4.0, 0.5),
                "npix": rng.normal(200, 20),
                "npix_norm": rng.normal(1.5, 0.3),
                "skew": rng.normal(1, 0.5),
                "std": rng.normal(1, 0.3),
                "peak_to_noise_dff": rng.normal(5, 1),
                "autocorr_halfwidth_s": rng.normal(2.0, 0.3),
                "fneu_corr": rng.normal(0.7, 0.1),
            }
        )
        labels.append("dend")
    # Artefact
    for _ in range(n_per_class):
        rows.append(
            {
                "radius": rng.normal(1.0, 0.3),
                "compact": rng.normal(0.05, 0.02),
                "aspect_ratio": rng.normal(1.5, 0.3),
                "npix": rng.normal(50, 10),
                "npix_norm": rng.normal(0.5, 0.2),
                "skew": rng.normal(0.5, 0.5),
                "std": rng.normal(0.5, 0.2),
                "peak_to_noise_dff": rng.normal(2, 1),
                "autocorr_halfwidth_s": rng.normal(0.5, 0.1),
                "fneu_corr": rng.normal(0.2, 0.1),
            }
        )
        labels.append("artefact")
    return pd.DataFrame(rows), np.array(labels)


class TestFitPipeline:
    def test_returns_pipeline_and_reports(self) -> None:
        X, y = _synthetic_dataset(n_per_class=15)
        pipe, cv_df, cm = tsc._fit_pipeline(X, y, cv_folds=3)
        assert hasattr(pipe, "predict_proba")
        # 3 folds × 3 classes = 9 rows.
        assert len(cv_df) == 9
        assert cm.shape == (3, 3)

    def test_pipeline_predicts_well_on_clean_data(self) -> None:
        X, y = _synthetic_dataset(n_per_class=30)
        pipe, _, _ = tsc._fit_pipeline(X, y, cv_folds=3)
        preds = pipe.predict(X)
        # Synthetic data is well-separated; expect ≥ 90% training accuracy.
        assert (preds == y).mean() > 0.9


# ---------------------------------------------------------------------------
# _features_for_session — end-to-end with synthetic Suite2p outputs
# ---------------------------------------------------------------------------


def _write_session_artifacts(
    ca_root: Path,
    session_id: str,
    n_rois: int = 5,
    n_frames: int = 200,
) -> None:
    sub, ses = tsc._session_to_sub_ses(session_id)
    s2p = ca_root / "ca_extraction" / sub / ses / "suite2p" / "plane0"
    s2p.mkdir(parents=True)
    rng = np.random.default_rng(0)
    F = rng.uniform(100, 500, (n_rois, n_frames)).astype(np.float32)
    Fneu = rng.uniform(50, 200, (n_rois, n_frames)).astype(np.float32)
    np.save(s2p / "F.npy", F)
    np.save(s2p / "Fneu.npy", Fneu)
    stat = [
        {
            "radius": 5.0,
            "compact": 0.7,
            "aspect_ratio": 1.5,
            "npix": 200,
            "npix_norm": 1.5,
            "skew": 1.0,
            "std": 1.0,
        }
        for _ in range(n_rois)
    ]
    np.save(s2p / "stat.npy", np.array(stat, dtype=object), allow_pickle=True)

    ca_dir = ca_root / "calcium" / sub / ses
    ca_dir.mkdir(parents=True)
    with h5py.File(ca_dir / "ca.h5", "w") as f:
        f.attrs["fps_imaging"] = 9.6


class TestFeaturesForSession:
    def test_features_round_trip(self, tmp_path: Path) -> None:
        session = "20220804_13_52_02_1117646"
        _write_session_artifacts(tmp_path, session, n_rois=5, n_frames=200)
        df = tsc._features_for_session(session, tmp_path)
        from hm2p.extraction.soma_features import FEATURE_COLUMNS

        assert list(df.columns) == list(FEATURE_COLUMNS)
        assert len(df) == 5


class TestCurationCsvFlag:
    """``--curation-csv`` resolves an append-only CSV via load_latest_labels."""

    def test_dry_run_with_curation_csv(self, tmp_path: Path) -> None:
        from hm2p.extraction.curation import append_curation_row

        csv_path = tmp_path / "roi_curation.csv"
        # Two labels for one ROI — latest timestamp should win.
        append_curation_row(
            csv_path,
            "20220804_13_52_02_1117646",
            0,
            "soma",
            "alice",
            "2026-01-01T00:00:00+00:00",
        )
        append_curation_row(
            csv_path,
            "20220804_13_52_02_1117646",
            0,
            "dend",
            "alice",
            "2026-01-02T00:00:00+00:00",
        )

        rc = tsc.main(
            [
                "--curation-csv",
                str(csv_path),
                "--output",
                str(tmp_path / "model.pkl"),
                "--report-dir",
                str(tmp_path / "reports"),
                "--dry-run",
            ]
        )
        assert rc == 0

    def test_mutually_exclusive_with_labels(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            tsc.main(
                [
                    "--labels",
                    "x.csv",
                    "--curation-csv",
                    "y.csv",
                    "--output",
                    str(tmp_path / "model.pkl"),
                    "--report-dir",
                    str(tmp_path / "reports"),
                ]
            )
