"""Tests for hm2p.extraction.curation — curation CSV I/O and runtime resolver."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from hm2p.extraction.curation import (
    CURATION_CSV_COLUMNS,
    append_curation_row,
    apply_curation_to_ca_h5,
    effective_roi_label,
    labels_for_session,
    load_latest_labels,
)

SES_A = "20220804_13_52_02_1117646"
SES_B = "20221015_10_00_00_1116663"


# ---------------------------------------------------------------------------
# append_curation_row
# ---------------------------------------------------------------------------


class TestAppendCurationRow:
    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "roi_curation.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-04-30T12:00:00+00:00")
        text = csv_path.read_text()
        # Header is written exactly once.
        assert text.startswith(",".join(CURATION_CSV_COLUMNS))
        assert text.count("session_id,roi_index") == 1
        assert "alice" in text

    def test_append_does_not_duplicate_header(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "roi_curation.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-04-30T12:00:00+00:00")
        append_curation_row(csv_path, SES_A, 1, "dend", "alice", "2026-04-30T12:00:01+00:00")
        text = csv_path.read_text()
        assert text.count("session_id,roi_index") == 1
        assert text.count("\n") >= 3  # header + 2 rows

    def test_invalid_label_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        with pytest.raises(ValueError, match="Invalid label"):
            append_curation_row(csv_path, SES_A, 0, "neuron", "alice")

    def test_negative_roi_index_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        with pytest.raises(ValueError, match="non-negative"):
            append_curation_row(csv_path, SES_A, -1, "soma", "alice")

    def test_empty_session_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        with pytest.raises(ValueError, match="session_id"):
            append_curation_row(csv_path, "", 0, "soma", "alice")

    def test_empty_curator_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        with pytest.raises(ValueError, match="curator"):
            append_curation_row(csv_path, SES_A, 0, "soma", "")

    def test_default_timestamp_is_iso_utc(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice")
        df = pd.read_csv(csv_path)
        ts = df["timestamp"].iloc[0]
        # ISO-8601 in UTC ends with +00:00 and contains a 'T' separator.
        assert "T" in ts
        assert ts.endswith("+00:00")

    def test_columns_match_canonical_order(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        append_curation_row(csv_path, SES_A, 2, "artefact", "bob", "2026-04-30T00:00:00+00:00")
        df = pd.read_csv(csv_path)
        assert list(df.columns) == list(CURATION_CSV_COLUMNS)


# ---------------------------------------------------------------------------
# load_latest_labels — append-only, latest wins
# ---------------------------------------------------------------------------


class TestLoadLatestLabels:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        df = load_latest_labels(tmp_path / "does_not_exist.csv")
        assert len(df) == 0
        assert list(df.columns) == list(CURATION_CSV_COLUMNS)

    def test_basic_round_trip(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-01-01T00:00:00+00:00")
        append_curation_row(csv_path, SES_A, 1, "dend", "alice", "2026-01-01T00:00:01+00:00")
        df = load_latest_labels(csv_path)
        assert len(df) == 2
        assert set(df["label"]) == {"soma", "dend"}

    def test_latest_wins_on_relabel(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        # First label: soma at 12:00.
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-01-01T12:00:00+00:00")
        # Re-label: dend at 13:00 — should win.
        append_curation_row(csv_path, SES_A, 0, "dend", "alice", "2026-01-01T13:00:00+00:00")
        # Earlier-timestamp re-label appended later — still loses on read.
        append_curation_row(csv_path, SES_A, 0, "artefact", "bob", "2026-01-01T11:00:00+00:00")

        df = load_latest_labels(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["label"] == "dend"
        assert df.iloc[0]["curator"] == "alice"

    def test_independent_sessions(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-01-01T00:00:00+00:00")
        append_curation_row(csv_path, SES_B, 0, "dend", "alice", "2026-01-01T00:00:00+00:00")
        df = load_latest_labels(csv_path)
        assert len(df) == 2

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"session_id": [SES_A], "label": ["soma"]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_latest_labels(csv_path)

    def test_unrecognised_label_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"session_id": [SES_A], "roi_index": [0], "label": ["neuron"]}).to_csv(
            csv_path, index=False
        )
        with pytest.raises(ValueError, match="unrecognised label values"):
            load_latest_labels(csv_path)

    def test_empty_csv_returns_empty(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "empty.csv"
        pd.DataFrame({col: [] for col in CURATION_CSV_COLUMNS}).to_csv(csv_path, index=False)
        df = load_latest_labels(csv_path)
        assert len(df) == 0

    def test_compatible_with_train_soma_classifier_schema(self, tmp_path: Path) -> None:
        """The first three columns must match scripts.train_soma_classifier."""
        csv_path = tmp_path / "x.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-01-01T00:00:00+00:00")
        df = load_latest_labels(csv_path)
        # train_soma_classifier._load_labels checks for these columns.
        assert {"session_id", "roi_index", "label"}.issubset(df.columns)


class TestLabelsForSession:
    def test_returns_per_session_dict(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-01-01T00:00:00+00:00")
        append_curation_row(csv_path, SES_A, 5, "dend", "alice", "2026-01-01T00:00:01+00:00")
        append_curation_row(csv_path, SES_B, 0, "artefact", "alice", "2026-01-01T00:00:02+00:00")
        out = labels_for_session(csv_path, SES_A)
        assert out == {0: "soma", 5: "dend"}

    def test_missing_session_returns_empty(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-01-01T00:00:00+00:00")
        assert labels_for_session(csv_path, SES_B) == {}

    def test_no_csv_returns_empty(self, tmp_path: Path) -> None:
        assert labels_for_session(tmp_path / "nope.csv", SES_A) == {}


# ---------------------------------------------------------------------------
# apply_curation_to_ca_h5
# ---------------------------------------------------------------------------


def _write_minimal_ca_h5(path: Path, n_rois: int = 4, n_frames: int = 50) -> None:
    """Create a tiny ca.h5 with just enough fields for the curation writer."""
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("dff", data=rng.normal(size=(n_rois, n_frames)).astype(np.float32))
        f.attrs["session_id"] = "test"


class TestApplyCurationToCaH5:
    def test_writes_curated_label_array(self, tmp_path: Path) -> None:
        ca_h5 = tmp_path / "ca.h5"
        _write_minimal_ca_h5(ca_h5, n_rois=4)
        csv_path = tmp_path / "labels.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-01-01T00:00:00+00:00")
        append_curation_row(csv_path, SES_A, 2, "dend", "alice", "2026-01-01T00:00:01+00:00")

        n_written = apply_curation_to_ca_h5(csv_path, SES_A, ca_h5)
        assert n_written == 2

        with h5py.File(ca_h5, "r") as f:
            assert "roi_qc/curated_label" in f
            arr = f["roi_qc/curated_label"][:]
        decoded = [v.decode() if isinstance(v, bytes) else v for v in arr]
        assert decoded == ["soma", "", "dend", ""]

    def test_missing_ca_h5_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            apply_curation_to_ca_h5(tmp_path / "labels.csv", SES_A, tmp_path / "nope.h5")

    def test_no_labels_writes_all_empty_strings(self, tmp_path: Path) -> None:
        ca_h5 = tmp_path / "ca.h5"
        _write_minimal_ca_h5(ca_h5, n_rois=3)
        csv_path = tmp_path / "labels.csv"  # never created

        n_written = apply_curation_to_ca_h5(csv_path, SES_A, ca_h5)
        assert n_written == 0
        with h5py.File(ca_h5, "r") as f:
            arr = f["roi_qc/curated_label"][:]
        decoded = [v.decode() if isinstance(v, bytes) else v for v in arr]
        assert decoded == ["", "", ""]

    def test_overwrite_replaces_old_array(self, tmp_path: Path) -> None:
        ca_h5 = tmp_path / "ca.h5"
        _write_minimal_ca_h5(ca_h5, n_rois=3)
        csv_path = tmp_path / "labels.csv"
        append_curation_row(csv_path, SES_A, 0, "soma", "alice", "2026-01-01T00:00:00+00:00")
        apply_curation_to_ca_h5(csv_path, SES_A, ca_h5)
        # Re-label, overwriting.
        append_curation_row(csv_path, SES_A, 0, "dend", "alice", "2026-01-01T00:00:01+00:00")
        apply_curation_to_ca_h5(csv_path, SES_A, ca_h5)
        with h5py.File(ca_h5, "r") as f:
            arr = f["roi_qc/curated_label"][:]
        decoded = [v.decode() if isinstance(v, bytes) else v for v in arr]
        assert decoded[0] == "dend"

    def test_roi_index_out_of_range_raises(self, tmp_path: Path) -> None:
        ca_h5 = tmp_path / "ca.h5"
        _write_minimal_ca_h5(ca_h5, n_rois=2)
        csv_path = tmp_path / "labels.csv"
        append_curation_row(csv_path, SES_A, 5, "soma", "alice", "2026-01-01T00:00:00+00:00")
        with pytest.raises(ValueError, match="roi_index=5"):
            apply_curation_to_ca_h5(csv_path, SES_A, ca_h5)

    def test_missing_dff_raises(self, tmp_path: Path) -> None:
        ca_h5 = tmp_path / "ca.h5"
        ca_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(ca_h5, "w") as f:
            f.attrs["session_id"] = "test"
        csv_path = tmp_path / "labels.csv"
        with pytest.raises(KeyError, match="dff"):
            apply_curation_to_ca_h5(csv_path, SES_A, ca_h5)


# ---------------------------------------------------------------------------
# effective_roi_label
# ---------------------------------------------------------------------------


class TestEffectiveRoiLabel:
    def test_no_roi_qc_returns_soma_default(self) -> None:
        out = effective_roi_label(None, n_rois=3)
        assert list(out) == ["soma", "soma", "soma"]

    def test_falls_back_to_argmax_when_no_curation(self) -> None:
        roi_qc = {
            "p_soma": np.array([0.8, 0.1, 0.2], dtype=np.float32),
            "p_dend": np.array([0.1, 0.8, 0.1], dtype=np.float32),
            "p_artefact": np.array([0.1, 0.1, 0.7], dtype=np.float32),
        }
        out = effective_roi_label(roi_qc, n_rois=3)
        assert list(out) == ["soma", "dend", "artefact"]

    def test_curated_overrides_argmax(self) -> None:
        roi_qc = {
            "p_soma": np.array([0.8, 0.1, 0.2], dtype=np.float32),
            "p_dend": np.array([0.1, 0.8, 0.1], dtype=np.float32),
            "p_artefact": np.array([0.1, 0.1, 0.7], dtype=np.float32),
            "curated_label": np.array(["dend", "", "soma"], dtype=object),
        }
        out = effective_roi_label(roi_qc, n_rois=3)
        # ROI 0: curated wins, ROI 1: empty curated → argmax (dend),
        # ROI 2: curated overrides argmax (artefact → soma).
        assert list(out) == ["dend", "dend", "soma"]

    def test_ignores_invalid_curated_label(self) -> None:
        roi_qc = {
            "p_soma": np.array([0.8], dtype=np.float32),
            "p_dend": np.array([0.1], dtype=np.float32),
            "p_artefact": np.array([0.1], dtype=np.float32),
            "curated_label": np.array(["junk"], dtype=object),
        }
        out = effective_roi_label(roi_qc, n_rois=1)
        assert list(out) == ["soma"]

    def test_handles_h5py_bytes_strings(self) -> None:
        roi_qc = {
            "p_soma": np.array([0.8, 0.1], dtype=np.float32),
            "p_dend": np.array([0.1, 0.8], dtype=np.float32),
            "p_artefact": np.array([0.1, 0.1], dtype=np.float32),
            "curated_label": np.array([b"artefact", b""]),
        }
        out = effective_roi_label(roi_qc, n_rois=2)
        assert list(out) == ["artefact", "dend"]

    def test_nan_probs_fall_back_to_default(self) -> None:
        roi_qc = {
            "p_soma": np.array([np.nan, 0.1], dtype=np.float32),
            "p_dend": np.array([np.nan, 0.8], dtype=np.float32),
            "p_artefact": np.array([np.nan, 0.1], dtype=np.float32),
        }
        out = effective_roi_label(roi_qc, n_rois=2)
        assert out[0] == "soma"  # NaN row → conservative default
        assert out[1] == "dend"

    def test_length_mismatched_probs_use_default(self) -> None:
        # When n_rois disagrees with the probability arrays, we cannot
        # resolve argmax safely.  Fall back to "soma".
        roi_qc = {
            "p_soma": np.array([0.8], dtype=np.float32),
            "p_dend": np.array([0.1], dtype=np.float32),
            "p_artefact": np.array([0.1], dtype=np.float32),
        }
        out = effective_roi_label(roi_qc, n_rois=3)
        assert list(out) == ["soma", "soma", "soma"]

    def test_curated_string_array(self) -> None:
        roi_qc = {"curated_label": np.array(["dend", "soma"], dtype="<U10")}
        out = effective_roi_label(roi_qc, n_rois=2)
        assert list(out) == ["dend", "soma"]

    def test_n_rois_zero(self) -> None:
        out = effective_roi_label({}, n_rois=0)
        assert len(out) == 0

    def test_negative_n_rois_raises(self) -> None:
        with pytest.raises(ValueError):
            effective_roi_label({}, n_rois=-1)

    def test_curated_label_with_none_entries(self) -> None:
        # Object arrays from h5py may surface ``None`` for empty cells.
        roi_qc = {"curated_label": np.array([None, "soma"], dtype=object)}
        out = effective_roi_label(roi_qc, n_rois=2)
        # First entry: None → ignored → falls through to default.
        assert out[0] == "soma"
        assert out[1] == "soma"


class TestLoadLatestLabelsSchemaFallbacks:
    def test_csv_without_curator_column_assigns_unknown(self, tmp_path: Path) -> None:
        # Old-style CSVs that pre-date append_curation_row should still load.
        path = tmp_path / "old.csv"
        pd.DataFrame(
            {
                "session_id": [SES_A],
                "roi_index": [0],
                "label": ["soma"],
            }
        ).to_csv(path, index=False)
        df = load_latest_labels(path)
        assert df.iloc[0]["curator"] == "unknown"
        assert df.iloc[0]["timestamp"] == ""
