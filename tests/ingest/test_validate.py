"""Tests for ingest/validate.py — raw file completeness checks."""

from __future__ import annotations

from pathlib import Path

from hm2p.ingest.validate import ValidationResult, validate_session
from hm2p.session import Session


def test_validate_session_missing_files(penk_session: Session, tmp_path: Path) -> None:
    """validate_session returns ok=False when raw files are absent."""
    result = validate_session(penk_session, rawdata_root=tmp_path)
    assert isinstance(result, ValidationResult)
    assert result.session_id == penk_session.session_id
    assert not result.ok
    assert len(result.missing) > 0


def test_validate_session_ok(penk_session: Session, tmp_path: Path) -> None:
    """validate_session returns ok=True when all required files exist."""
    ses_root = tmp_path / penk_session.neuroblueprint_sub / penk_session.neuroblueprint_ses
    funcimg = ses_root / "funcimg"
    behav = ses_root / "behav"
    funcimg.mkdir(parents=True)
    behav.mkdir(parents=True)

    # Create the four required files
    (funcimg / "20220804_13_52_02_1117646_XYT.tif").write_bytes(b"\x00")
    (funcimg / "20220804_13_52_02_1117646-di.tdms").write_bytes(b"\x00")
    (funcimg / "20220804_13_52_02_1117646.meta.txt").write_text("dummy")
    (behav / "sub-1117646_ses-20220804T135202_overhead.camera.mp4").write_bytes(b"\x00")

    result = validate_session(penk_session, rawdata_root=tmp_path)
    assert result.ok
    assert result.missing == []
