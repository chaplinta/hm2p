"""Tests for scripts/declare_dlc_champion.py — champion-manifest writer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import declare_dlc_champion as ddc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeNoSuchKey(Exception):
    """Stands in for botocore's NoSuchKey exception class."""


def _make_fake_s3(existing: dict[str, bytes] | None = None) -> MagicMock:
    """Build a mock S3 client backed by an in-memory dict.

    ``existing`` maps S3 key → raw body bytes for objects that exist before
    the call. ``put_object`` and ``delete_object`` mutate the same dict so
    test bodies can verify final state.
    """
    store: dict[str, bytes] = dict(existing or {})
    s3 = MagicMock()
    s3.exceptions.NoSuchKey = _FakeNoSuchKey

    def _get(Bucket: str, Key: str):
        if Key not in store:
            raise _FakeNoSuchKey(f"no key {Key}")
        body = MagicMock()
        body.read.return_value = store[Key]
        return {"Body": body}

    def _put(Bucket: str, Key: str, Body: bytes, ContentType: str = ""):
        store[Key] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}

    def _delete(Bucket: str, Key: str):
        store.pop(Key, None)
        return {}

    s3.get_object.side_effect = _get
    s3.put_object.side_effect = _put
    s3.delete_object.side_effect = _delete
    s3._store = store  # exposed for assertions
    return s3


@pytest.fixture
def patch_imds_and_git(monkeypatch):
    """Patch IMDS + git lookups so tests don't depend on host environment."""
    monkeypatch.setattr(ddc, "_imds_instance_id", lambda: "i-test-instance")
    monkeypatch.setattr(ddc, "_git_sha", lambda repo: "deadbee")


# ---------------------------------------------------------------------------
# declare_champion — first-time declaration
# ---------------------------------------------------------------------------


def test_declare_champion_writes_manifest_with_required_fields(patch_imds_and_git):
    s3 = _make_fake_s3(existing={})
    out = ddc.declare_champion(
        model_name="hm2p_hrnetw32_shuffle1",
        architecture="HrnetW32",
        snapshot="290",
        training_run_id="retrain-20260423T142500Z",
        notes="auto",
        s3_client=s3,
        bucket="hm2p-derivatives",
    )
    # Required identifier fields
    assert out["model_name"] == "hm2p_hrnetw32_shuffle1"
    assert out["architecture"] == "HrnetW32"
    assert out["snapshot"] == "290"
    assert out["training_run_id"] == "retrain-20260423T142500Z"
    assert out["promoted_by_ec2_instance"] == "i-test-instance"
    assert out["promoted_by_git_sha"] == "deadbee"
    assert out["champion_id"].startswith("dlc-") and "hrnetw32-snap290" in out["champion_id"]
    assert out["note"] == ""
    assert out["notes"] == "auto"
    # Promoted_at is ISO 8601 UTC with trailing Z
    assert out["promoted_at"].endswith("Z") and "T" in out["promoted_at"]


def test_declare_champion_writes_to_s3_at_expected_key(patch_imds_and_git):
    s3 = _make_fake_s3(existing={})
    out = ddc.declare_champion(
        model_name="m", architecture="HrnetW32", snapshot="100",
        training_run_id="rid", s3_client=s3, bucket="b",
    )
    written = json.loads(s3._store[ddc.CHAMPION_MANIFEST_KEY].decode("utf-8"))
    assert written == out


def test_declare_champion_appends_promotions_log(patch_imds_and_git):
    s3 = _make_fake_s3(existing={})
    ddc.declare_champion(
        model_name="m", architecture="HrnetW32", snapshot="100",
        training_run_id="rid", s3_client=s3, bucket="b",
    )
    log_bytes = s3._store[ddc.PROMOTIONS_LOG_KEY]
    line = log_bytes.decode("utf-8").strip()
    parts = line.split("\t")
    assert len(parts) == 4
    # Fields: promoted_at, champion_id, instance_id, git_sha
    assert parts[2] == "i-test-instance"
    assert parts[3] == "deadbee"


def test_declare_champion_clears_pipeline_rerun_marker(patch_imds_and_git):
    s3 = _make_fake_s3(existing={ddc.PIPELINE_RERUN_KEY: b'{"reason":"in flight"}'})
    ddc.declare_champion(
        model_name="m", architecture="HrnetW32", snapshot="100",
        training_run_id="rid", s3_client=s3, bucket="b",
    )
    assert ddc.PIPELINE_RERUN_KEY not in s3._store


# ---------------------------------------------------------------------------
# declare_champion — re-declaration with prior champion present
# ---------------------------------------------------------------------------


def test_declare_champion_archives_previous(patch_imds_and_git):
    prior = {
        "champion_id": "dlc-20260101-hrnetw32-snap100",
        "model_name": "old_model",
        "architecture": "HrnetW32",
        "snapshot": "100",
        "training_date": "2026-01-01",
        "training_run_id": "old-rid",
        "promoted_by_ec2_instance": "i-old",
        "promoted_by_git_sha": "abc1234",
        "promoted_at": "2026-01-01T00:00:00Z",
        "training_s3_prefix": "dlc-retrain/models/",
        "note": "",
        "notes": "old",
    }
    s3 = _make_fake_s3(existing={
        ddc.CHAMPION_MANIFEST_KEY: json.dumps(prior).encode("utf-8"),
    })
    ddc.declare_champion(
        model_name="new_model", architecture="HrnetW32", snapshot="290",
        training_run_id="new-rid", s3_client=s3, bucket="b",
    )
    archive_key = f"{ddc.HISTORY_PREFIX}/{prior['champion_id']}.json"
    assert archive_key in s3._store
    archived = json.loads(s3._store[archive_key].decode("utf-8"))
    assert archived == prior


def test_declare_champion_overwrites_current_manifest(patch_imds_and_git):
    prior = {
        "champion_id": "dlc-20260101-hrnetw32-snap100",
        "model_name": "old", "architecture": "HrnetW32", "snapshot": "100",
        "training_date": "2026-01-01", "training_run_id": "old-rid",
        "promoted_by_ec2_instance": "i-old", "promoted_by_git_sha": "abc1234",
        "promoted_at": "2026-01-01T00:00:00Z",
        "training_s3_prefix": "dlc-retrain/models/", "note": "", "notes": "",
    }
    s3 = _make_fake_s3(existing={
        ddc.CHAMPION_MANIFEST_KEY: json.dumps(prior).encode("utf-8"),
    })
    new = ddc.declare_champion(
        model_name="new", architecture="HrnetW32", snapshot="290",
        training_run_id="new-rid", s3_client=s3, bucket="b",
    )
    current = json.loads(s3._store[ddc.CHAMPION_MANIFEST_KEY].decode("utf-8"))
    assert current["champion_id"] == new["champion_id"]
    assert current["champion_id"] != prior["champion_id"]


def test_declare_champion_appends_to_existing_promotions_log(patch_imds_and_git):
    s3 = _make_fake_s3(existing={
        ddc.PROMOTIONS_LOG_KEY: b"2026-01-01T00:00:00Z\told-id\ti-old\tabc1234\n",
    })
    ddc.declare_champion(
        model_name="m", architecture="HrnetW32", snapshot="290",
        training_run_id="rid", s3_client=s3, bucket="b",
    )
    log_text = s3._store[ddc.PROMOTIONS_LOG_KEY].decode("utf-8")
    assert log_text.count("\n") == 2  # one prior + one new
    assert "old-id" in log_text


# ---------------------------------------------------------------------------
# declare_champion — dry-run path
# ---------------------------------------------------------------------------


def test_declare_champion_dry_run_does_not_touch_s3(patch_imds_and_git):
    s3 = _make_fake_s3(existing={})
    out = ddc.declare_champion(
        model_name="m", architecture="HrnetW32", snapshot="100",
        training_run_id="rid", dry_run=True, s3_client=s3, bucket="b",
    )
    assert out["champion_id"].startswith("dlc-")
    s3.put_object.assert_not_called()
    s3.delete_object.assert_not_called()


# ---------------------------------------------------------------------------
# declare_champion — note vs notes are separate fields
# ---------------------------------------------------------------------------


def test_declare_champion_note_and_notes_kept_separate(patch_imds_and_git):
    s3 = _make_fake_s3(existing={})
    out = ddc.declare_champion(
        model_name="m", architecture="HrnetW32", snapshot="100",
        training_run_id="rid", notes="auto-summary", note="manual hint",
        s3_client=s3, bucket="b",
    )
    assert out["notes"] == "auto-summary"
    assert out["note"] == "manual hint"
