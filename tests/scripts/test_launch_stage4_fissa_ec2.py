"""Tests for scripts/launch_stage4_fissa_ec2.py (no AWS calls)."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import launch_stage4_fissa_ec2 as lf  # noqa: E402

SESSIONS = [
    {"exp_id": "20990101_10_00_00_0000001", "sub": "sub-0000001", "ses": "ses-20990101T100000"},
    {"exp_id": "20990102_10_00_00_0000002", "sub": "sub-0000002", "ses": "ses-20990102T100000"},
]


def _args(**kw) -> Namespace:
    base = {"all_fixed": False, "session": None, "sessions": None}
    base.update(kw)
    return Namespace(**base)


def test_resolve_sessions_list(monkeypatch):
    monkeypatch.setattr(lf, "get_sessions", lambda: SESSIONS)
    sel = lf.resolve_sessions(_args(sessions=["20990102_10_00_00_0000002"]))
    assert [s["sub"] for s in sel] == ["sub-0000002"]
    with pytest.raises(SystemExit):
        lf.resolve_sessions(_args(sessions=["nope"]))
    with pytest.raises(SystemExit):
        lf.resolve_sessions(_args())


def test_keys_per_tag():
    pk, lk, sf = lf._keys("a")
    assert pk == "calcium/_fissa_progress_a.json" and lk == "calcium/_fissa_a.log"
    assert sf.name == ".hm2p-fissa-instance_a.json"
    assert lf._keys("")[0] == "calcium/_fissa_progress.json"


def test_user_data_branch_tag_and_envs(monkeypatch):
    monkeypatch.setattr(lf, "get_s3_credentials", lambda: ("k", "s", "r"))
    ud = lf.build_user_data(SESSIONS, branch="feat/x", tag="b")
    assert "git clone --branch feat/x" in ud
    assert "_fissa_progress_b.json" in ud and "_fissa_b.log" in ud
    assert "uv venv -q -p 3.12 /opt/hm2p" in ud
    assert "--ignore-requires-python" in ud  # FISSA env stays on Python 3.10
    assert "skip_existing = 0" in ud  # explicit session list is never skipped
