"""Tests for scripts/launch_suite2p_ec2.py (no AWS calls)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import launch_suite2p_ec2 as ls  # noqa: E402

SESSIONS = [
    {"exp_id": "20990101_10_00_00_0000001", "sub": "sub-0000001", "ses": "ses-20990101T100000"},
    {"exp_id": "20990102_10_00_00_0000002", "sub": "sub-0000002", "ses": "ses-20990102T100000"},
]


def test_select_sessions(monkeypatch):
    monkeypatch.setattr(ls, "get_sessions", lambda: SESSIONS)
    assert len(ls.select_sessions(None)) == 2
    assert [s["sub"] for s in ls.select_sessions(["20990102_10_00_00_0000002"])] == ["sub-0000002"]
    with pytest.raises(SystemExit):
        ls.select_sessions(["nope"])


def test_user_data_uses_movement_timestamps_and_pinned_suite2p(monkeypatch):
    monkeypatch.setattr(ls, "get_s3_credentials", lambda: ("k", "s", "r"), raising=False)
    ud = ls.build_user_data(SESSIONS, use_instance_profile=True, branch="feat/x", tag="t")
    # timestamps.h5 is written by Stage 0 under movement/; the old timestamps/ path never existed
    assert "/movement/{sub}/{ses_id}/timestamps.h5" in ud
    assert "/timestamps/{sub}/" not in ud
    assert "timestamps.h5 missing" in ud  # a missing file fails the session (no fps fallback)
    assert f"suite2p=={ls.SUITE2P_VERSION}" in ud
    assert "uv venv -q -p 3.12" in ud and "@feat/x" in ud
    assert "_progress_t.json" in ud and "_suite2p_t.log" in ud
