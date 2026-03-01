"""Tests for the hm2p CLI (cli.py).

Commands are tested via typer.testing.CliRunner with synthetic temp
directories — no real data is read.  Env vars HM2P_DATA_ROOT and
HM2P_METADATA_DIR redirect all I/O to tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from hm2p.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_ID = "20220804_13_52_02_1117646"
_ANIMAL_ID = "1117646"


def _write_metadata(metadata_dir: Path) -> None:
    """Write minimal synthetic CSV files to metadata_dir.

    Column layout matches the real metadata CSVs:
    - animals.csv: animal_id, celltype, gcamp, virus_id
    - experiments.csv: exp_id (= session ID), orientation, bad_behav_times
      (animal_id is derived from exp_id in load_registry; no explicit column needed)
    """
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "animals.csv").write_text(
        "animal_id,celltype,gcamp,virus_id\n"
        f"{_ANIMAL_ID},penk,7f,ADD3\n"
    )
    (metadata_dir / "experiments.csv").write_text(
        "exp_id,orientation,bad_behav_times\n"
        f"{_SESSION_ID},0.0,\n"
    )


def _invoke(args: list[str], data_root: Path, metadata_dir: Path) -> object:
    """Invoke CLI with HM2P env vars pointing at tmp directories."""
    return runner.invoke(
        app,
        args,
        env={
            "HM2P_DATA_ROOT": str(data_root),
            "HM2P_METADATA_DIR": str(metadata_dir),
        },
    )


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


def test_validate_missing_files_exits_nonzero(tmp_path: Path) -> None:
    """validate exits with code 1 and prints [FAIL] when raw files are absent."""
    metadata_dir = tmp_path / "metadata"
    _write_metadata(metadata_dir)

    result = _invoke(["validate", "--session", _SESSION_ID], tmp_path, metadata_dir)
    assert result.exit_code == 1
    assert "[FAIL]" in result.output


def test_validate_all_files_present_exits_zero(tmp_path: Path) -> None:
    """validate exits 0 and prints [OK] when all required raw files exist."""
    metadata_dir = tmp_path / "metadata"
    _write_metadata(metadata_dir)

    # Build the NeuroBlueprint rawdata tree with dummy files
    ses_dir = (
        tmp_path
        / "rawdata"
        / f"sub-{_ANIMAL_ID}"
        / "ses-20220804T135202"
    )
    funcimg = ses_dir / "funcimg"
    behav = ses_dir / "behav"
    funcimg.mkdir(parents=True)
    behav.mkdir(parents=True)
    (funcimg / "dummy_XYT.tif").write_bytes(b"")
    (funcimg / "dummy-di.tdms").write_bytes(b"")
    (funcimg / "dummy.meta.txt").write_bytes(b"")
    (behav / "dummy_overhead.camera.mp4").write_bytes(b"")

    result = _invoke(["validate", "--session", _SESSION_ID], tmp_path, metadata_dir)
    assert result.exit_code == 0
    assert "[OK]" in result.output


def test_validate_unknown_session_raises(tmp_path: Path) -> None:
    """validate exits non-zero for a session ID not in the registry."""
    metadata_dir = tmp_path / "metadata"
    _write_metadata(metadata_dir)

    result = _invoke(
        ["validate", "--session", "99991231_00_00_00_9999999"],
        tmp_path,
        metadata_dir,
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------


def test_ingest_no_tdms_exits_nonzero(tmp_path: Path) -> None:
    """ingest exits 1 and prints [FAIL] when no TDMS file is present."""
    metadata_dir = tmp_path / "metadata"
    _write_metadata(metadata_dir)

    result = _invoke(["ingest", "--session", _SESSION_ID], tmp_path, metadata_dir)
    assert result.exit_code == 1
    assert "[FAIL]" in result.output


# ---------------------------------------------------------------------------
# calcium command
# ---------------------------------------------------------------------------


def test_calcium_exits_zero_with_info_message(tmp_path: Path) -> None:
    """calcium exits 0 and prints a helpful message (GPU/cloud-only stage)."""
    metadata_dir = tmp_path / "metadata"
    _write_metadata(metadata_dir)

    result = _invoke(["calcium", "--session", _SESSION_ID], tmp_path, metadata_dir)
    assert result.exit_code == 0
    assert "GPU" in result.output or "cloud" in result.output or "Snakemake" in result.output


# ---------------------------------------------------------------------------
# sync command
# ---------------------------------------------------------------------------


def test_sync_missing_kinematics_exits_nonzero(tmp_path: Path) -> None:
    """sync exits 1 when kinematics.h5 is not found."""
    metadata_dir = tmp_path / "metadata"
    _write_metadata(metadata_dir)

    result = _invoke(["sync", "--session", _SESSION_ID], tmp_path, metadata_dir)
    assert result.exit_code == 1
    assert "[FAIL]" in result.output
    assert "kinematics.h5" in result.output


def test_sync_missing_ca_exits_nonzero(tmp_path: Path) -> None:
    """sync exits 1 when ca.h5 is not found (kinematics.h5 present)."""
    metadata_dir = tmp_path / "metadata"
    _write_metadata(metadata_dir)

    from hm2p.io.hdf5 import write_h5

    # Create a kinematics.h5 in the expected derivatives path
    kin_dir = (
        tmp_path
        / "derivatives"
        / "movement"
        / f"sub-{_ANIMAL_ID}"
        / "ses-20220804T135202"
    )
    kin_dir.mkdir(parents=True)
    T = 180
    write_h5(
        kin_dir / "kinematics.h5",
        {
            "frame_times": np.linspace(0.0, 6.0, T, dtype=np.float64),
            "hd_deg": np.zeros(T, dtype=np.float32),
            "x_mm": np.zeros(T, dtype=np.float32),
            "y_mm": np.zeros(T, dtype=np.float32),
            "speed_cm_s": np.ones(T, dtype=np.float32),
            "ahv_deg_s": np.zeros(T, dtype=np.float32),
            "active": np.ones(T, dtype=bool),
            "light_on": np.zeros(T, dtype=bool),
            "bad_behav": np.zeros(T, dtype=bool),
        },
        attrs={"session_id": _SESSION_ID, "fps_imaging": 30.0},
    )

    result = _invoke(["sync", "--session", _SESSION_ID], tmp_path, metadata_dir)
    assert result.exit_code == 1
    assert "[FAIL]" in result.output
    assert "ca.h5" in result.output


def test_sync_success(tmp_path: Path) -> None:
    """sync exits 0 and writes sync.h5 when both input files are present."""
    metadata_dir = tmp_path / "metadata"
    _write_metadata(metadata_dir)

    from hm2p.io.hdf5 import write_h5

    T = 180
    n_rois = 5
    sub = f"sub-{_ANIMAL_ID}"
    ses = "ses-20220804T135202"

    kin_dir = tmp_path / "derivatives" / "movement" / sub / ses
    ca_dir = tmp_path / "derivatives" / "calcium" / sub / ses
    kin_dir.mkdir(parents=True)
    ca_dir.mkdir(parents=True)

    frame_times = np.linspace(0.0, 6.0, T, dtype=np.float64)

    write_h5(
        kin_dir / "kinematics.h5",
        {
            "frame_times": frame_times,
            "hd_deg": np.zeros(T, dtype=np.float32),
            "x_mm": np.zeros(T, dtype=np.float32),
            "y_mm": np.zeros(T, dtype=np.float32),
            "speed_cm_s": np.ones(T, dtype=np.float32),
            "ahv_deg_s": np.zeros(T, dtype=np.float32),
            "active": np.ones(T, dtype=bool),
            "light_on": np.zeros(T, dtype=bool),
            "bad_behav": np.zeros(T, dtype=bool),
        },
        attrs={"session_id": _SESSION_ID, "fps_imaging": 30.0},
    )
    write_h5(
        ca_dir / "ca.h5",
        {
            "frame_times": frame_times,
            "dff": np.zeros((n_rois, T), dtype=np.float32),
        },
        attrs={"session_id": _SESSION_ID, "fps_imaging": 30.0, "extractor": "suite2p"},
    )

    result = _invoke(["sync", "--session", _SESSION_ID], tmp_path, metadata_dir)
    assert result.exit_code == 0, result.output
    assert "[OK]" in result.output

    sync_h5 = tmp_path / "derivatives" / "sync" / sub / ses / "sync.h5"
    assert sync_h5.exists()
