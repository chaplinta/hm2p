"""Tests for ingest/validate.py — raw file completeness checks."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    # TODO: create synthetic raw file structure and assert ok=True
    pytest.skip("Requires synthetic raw file structure — implement with validate_session()")
