"""Stage 0 — validate raw file completeness for each session.

Checks that all expected raw files are present before any processing begins:
- TIFF imaging stacks in funcimg/
- Behavioural video in behav/
- Camera calibration .npz
- Meta.txt crop / scale metadata
- TDMS DAQ file
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hm2p.session import Session


@dataclass
class ValidationResult:
    """Result of validating a single session's raw file set."""

    session_id: str
    ok: bool
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_session(session: Session, rawdata_root: Path) -> ValidationResult:
    """Check that all required raw files exist for a session.

    Args:
        session: Session metadata.
        rawdata_root: Root of the rawdata/ tree.

    Returns:
        ValidationResult — ok=True if all required files present.
    """
    raise NotImplementedError


def validate_all(sessions: list[Session], rawdata_root: Path) -> list[ValidationResult]:
    """Validate all sessions; return one result per session.

    Args:
        sessions: All sessions loaded from the registry.
        rawdata_root: Root of the rawdata/ tree.

    Returns:
        List of ValidationResult, one per session.
    """
    return [validate_session(s, rawdata_root) for s in sessions]
