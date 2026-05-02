"""Tests for hm2p.pose.select — DLC model selection logic."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from hm2p.pose.select import (
    CHAMPION_MANIFEST_KEY,
    _snapshot_number,
    compute_champion_id,
    extract_architecture,
    extract_dlc_provenance,
    get_champion_manifest,
    resolve_champion_id,
    select_best_dlc_h5,
    select_best_dlc_h5_s3,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Realistic DLC output keys used across tests.  Short prefix keeps lines under 99 chars.
_PFX = "pose/s/a"
_SA_KEY = f"{_PFX}/video_DLC_SuperAnimalTopViewMouse.h5"  # no snapshot in filename
_FT_290 = f"{_PFX}/videoDLC_HrnetW32_hm2p-retrainMar20_shuffle1_snapshot-best-290.h5"
_FT_110 = f"{_PFX}/videoDLC_HrnetW32_hm2p-retrainMar20_shuffle1_snapshot-best-110.h5"
_FT_50 = f"{_PFX}/videoDLC_Resnet50_hm2p-retrainMar20_shuffle1_snapshot-best-50.h5"
_FILTERED = (
    f"{_PFX}/videoDLC_HrnetW32_hm2p-retrainMar20_shuffle1_snapshot-best-290_filtered.h5"
)
_SINGLE = (
    f"{_PFX}/videoDLC_HrnetW32_hm2p-retrainMar20_shuffle1_snapshot-best-290_single.h5"
)


# ---------------------------------------------------------------------------
# _snapshot_number
# ---------------------------------------------------------------------------


def test_snapshot_number_with_hyphen_separator():
    assert _snapshot_number(_FT_290) == 290


def test_snapshot_number_with_underscore_separator():
    key = "pose/sub-1/ses-A/videoDLC_HrnetW32_proj_shuffle1_snapshot_best_150.h5"
    assert _snapshot_number(key) == 150


def test_snapshot_number_no_match_returns_minus_one():
    assert _snapshot_number(_SA_KEY) == -1


# ---------------------------------------------------------------------------
# select_best_dlc_h5
# ---------------------------------------------------------------------------


def test_select_empty_list_returns_none():
    assert select_best_dlc_h5([]) is None


def test_select_only_filtered_variants_returns_none():
    """Keys that are only _single or _filtered variants → None."""
    assert select_best_dlc_h5([_FILTERED, _SINGLE]) is None


def test_select_skips_filtered_and_single():
    """_single and _filtered files are excluded from consideration."""
    keys = [_FT_110, _FILTERED, _SINGLE]
    result = select_best_dlc_h5(keys)
    assert result == _FT_110


def test_select_picks_highest_snapshot_finetuned():
    """Among multiple finetuned snapshots, highest number wins."""
    keys = [_FT_110, _FT_290, _FT_50]
    result = select_best_dlc_h5(keys)
    assert result == _FT_290


def test_select_finetuned_preferred_over_superanimal():
    """Finetuned model is preferred over superanimal baseline."""
    keys = [_SA_KEY, _FT_110]
    result = select_best_dlc_h5(keys)
    assert result == _FT_110


def test_select_falls_back_to_first_non_finetuned():
    """When no finetuned file exists, return the first valid key."""
    keys = [_SA_KEY]
    assert select_best_dlc_h5(keys) == _SA_KEY


def test_select_hrnet_and_resnet_both_count_as_finetuned():
    """Both Hrnet and Resnet are accepted as finetuned architectures."""
    hrnet = "pose/sub-1/ses-A/videoDLC_HrnetW32_proj_shuffle1_snapshot-best-200.h5"
    resnet = "pose/sub-1/ses-A/videoDLC_Resnet50_proj_shuffle1_snapshot-best-100.h5"
    keys = [hrnet, resnet, _SA_KEY]
    result = select_best_dlc_h5(keys)
    assert result == hrnet  # hrnet has snapshot 200 > resnet 100


def test_select_resnet_wins_when_higher_snapshot():
    resnet_300 = "pose/sub-1/ses-A/videoDLC_Resnet50_proj_shuffle1_snapshot-best-300.h5"
    hrnet_200 = "pose/sub-1/ses-A/videoDLC_HrnetW32_proj_shuffle1_snapshot-best-200.h5"
    result = select_best_dlc_h5([hrnet_200, resnet_300])
    assert result == resnet_300


def test_select_ignores_non_h5_keys():
    """Non-.h5 keys are silently ignored."""
    keys = [
        "pose/sub-1/ses-A/labelled_30fps.mp4",
        "pose/sub-1/ses-A/meta.json",
        _FT_290,
    ]
    assert select_best_dlc_h5(keys) == _FT_290


# ---------------------------------------------------------------------------
# select_best_dlc_h5_s3
# ---------------------------------------------------------------------------


def _make_s3_client(keys: list[str], promoted: dict | None = None) -> MagicMock:
    """Build a mock boto3 S3 client for select_best_dlc_h5_s3 tests."""
    client = MagicMock()
    client.exceptions = MagicMock()

    # list_objects_v2 paginator
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": k} for k in keys]}
    ]
    client.get_paginator.return_value = paginator

    # get_object for promoted.json
    if promoted is not None:
        body = MagicMock()
        body.read.return_value = json.dumps(promoted).encode()
        client.get_object.return_value = {"Body": body}
    else:
        # Simulate NoSuchKey when promoted.json doesn't exist.
        no_such_key = type("NoSuchKey", (Exception,), {})
        client.exceptions.NoSuchKey = no_such_key
        client.get_object.side_effect = no_such_key()

    return client


def test_s3_no_files_returns_none():
    client = _make_s3_client([])
    result = select_best_dlc_h5_s3(client, "hm2p-derivatives", "pose/sub-1/ses-A/")
    assert result is None


def test_s3_heuristic_picks_best_snapshot_when_no_promoted():
    keys = [_FT_110, _FT_290, _SA_KEY]
    client = _make_s3_client(keys, promoted=None)
    result = select_best_dlc_h5_s3(client, "hm2p-derivatives", "pose/sub-1/ses-A/")
    assert result == _FT_290


def test_s3_promoted_json_overrides_heuristic():
    """When promoted.json exists and points to a specific file, use it."""
    # Promoted file has snapshot 110, heuristic would choose 290.
    filename_110 = _FT_110.split("/")[-1]
    promoted = {"h5_filename": filename_110, "snapshot": "110"}
    keys = [_FT_110, _FT_290]
    client = _make_s3_client(keys, promoted=promoted)
    result = select_best_dlc_h5_s3(client, "hm2p-derivatives", "pose/sub-1/ses-A/")
    assert result == _FT_110


def test_s3_promoted_json_missing_file_falls_back():
    """promoted.json names a file that isn't in the listing → use heuristic."""
    promoted = {"h5_filename": "nonexistent_snapshot-best-999.h5"}
    keys = [_FT_110, _FT_290]
    client = _make_s3_client(keys, promoted=promoted)
    result = select_best_dlc_h5_s3(client, "hm2p-derivatives", "pose/sub-1/ses-A/")
    assert result == _FT_290


def test_s3_promoted_json_empty_h5_filename_falls_back():
    """promoted.json with empty h5_filename → heuristic."""
    promoted = {"h5_filename": ""}
    keys = [_FT_110, _FT_290]
    client = _make_s3_client(keys, promoted=promoted)
    result = select_best_dlc_h5_s3(client, "hm2p-derivatives", "pose/sub-1/ses-A/")
    assert result == _FT_290


# ---------------------------------------------------------------------------
# extract_dlc_provenance
# ---------------------------------------------------------------------------


def test_provenance_finetuned_hrnet():
    filename = "videoDLC_HrnetW32_hm2p-retrainMar20_shuffle1_snapshot-best-290.h5"
    model_name, snapshot = extract_dlc_provenance(filename)
    assert model_name == "hm2p-retrainMar20"
    assert snapshot == "290"


def test_provenance_finetuned_resnet():
    filename = "videoDLC_Resnet50_hm2p-retrainMar20_shuffle1_snapshot-best-50.h5"
    model_name, snapshot = extract_dlc_provenance(filename)
    assert model_name == "hm2p-retrainMar20"
    assert snapshot == "50"


def test_provenance_superanimal_baseline():
    """Files without architecture tag → superanimal_topviewmouse."""
    filename = "video_DLC_snapshot-best-0.h5"
    model_name, snapshot = extract_dlc_provenance(filename)
    assert model_name == "superanimal_topviewmouse"
    assert snapshot == "superanimal"


def test_provenance_finetuned_no_snapshot_match():
    """Finetuned filename without recognisable snapshot → snapshot='unknown'."""
    filename = "videoDLC_HrnetW32_myproject_shuffle1_nosnapinfo.h5"
    model_name, snapshot = extract_dlc_provenance(filename)
    assert snapshot == "unknown"


def test_provenance_finetuned_no_project_match():
    """Finetuned filename without DLC_arch_project_shuffle → model='unknown'."""
    filename = "somefileHrnetW32_noprojectname.h5"
    model_name, snapshot = extract_dlc_provenance(filename)
    assert model_name == "unknown"


def test_provenance_underscore_snapshot_separator():
    """Handles both snapshot-best-290 and snapshot_best_290 forms."""
    filename = "videoDLC_HrnetW32_hm2p-retrainMar20_shuffle1_snapshot_best_290.h5"
    _, snapshot = extract_dlc_provenance(filename)
    assert snapshot == "290"


def test_provenance_no_underscore_before_shuffle():
    """Real-world DLC filenames sometimes lack the _ before 'shuffle' (e.g.
    ``...Mar20shuffle1...``). The regex must tolerate that."""
    filename = (
        "20210823_17_00_04_1114353_maze-rose_overhead.camera-cropped_30fpsDLC_"
        "HrnetW32_hm2p-retrainMar20shuffle1_snapshot_best-290.h5"
    )
    model_name, snapshot = extract_dlc_provenance(filename)
    assert model_name == "hm2p-retrainMar20"
    assert snapshot == "290"


# ---------------------------------------------------------------------------
# extract_architecture
# ---------------------------------------------------------------------------


def test_extract_architecture_hrnetw32():
    fn = "videoDLC_HrnetW32_proj_shuffle1_snapshot-best-290.h5"
    assert extract_architecture(fn) == "HrnetW32"


def test_extract_architecture_resnet50():
    fn = "videoDLC_Resnet50_proj_shuffle1_snapshot-best-50.h5"
    assert extract_architecture(fn) == "Resnet50"


def test_extract_architecture_other_hrnet_variant():
    fn = "videoDLC_HrnetW48_proj_shuffle1_snapshot-best-100.h5"
    assert extract_architecture(fn) == "HrnetW48"


def test_extract_architecture_returns_none_for_baseline():
    fn = "video_DLC_SuperAnimalTopViewMouse.h5"
    assert extract_architecture(fn) is None


# ---------------------------------------------------------------------------
# extract_architecture: init source agnostic (SA-finetune design §1.5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename, expected_arch",
    [
        # Existing ImageNet-init shuffle 1, snap 110.
        (
            "video_DLC_HrnetW32_hm2p-retrain_2026-03-20_shuffle1_snapshot-best-110.h5",
            "HrnetW32",
        ),
        # SA-init shuffle 2, snap 60 — same architecture token.
        (
            "video_DLC_HrnetW32_hm2p-retrain_2026-03-20_shuffle2_snapshot-best-60.h5",
            "HrnetW32",
        ),
        # Legacy ResNet50 (TF) shuffle.
        (
            "video_DLC_Resnet50_hm2p-retrain_shuffle1_snapshot-50000.h5",
            "Resnet50",
        ),
    ],
)
def test_extract_architecture_init_source_agnostic(filename, expected_arch):
    """Architecture token is unchanged by init source (ImageNet vs SA).

    Locks the design decision (SA fine-tune §1.5): SA fine-tune is captured
    in the champion manifest's ``notes`` field, not in the architecture
    token. The frontend's staleness check therefore continues to work
    against the existing architecture extraction.
    """
    assert extract_architecture(filename) == expected_arch


# ---------------------------------------------------------------------------
# compute_champion_id
# ---------------------------------------------------------------------------


def test_compute_champion_id_format():
    cid = compute_champion_id(
        model_name="hm2p_hrnetw32_shuffle1",
        architecture="HrnetW32",
        snapshot="50000",
        training_date="2026-04-23",
    )
    assert cid == "dlc-20260423-hrnetw32-snap50000"


def test_compute_champion_id_lowercases_architecture():
    cid = compute_champion_id(
        model_name="x", architecture="Resnet50", snapshot="100",
        training_date="2026-01-01",
    )
    assert "resnet50" in cid and "Resnet" not in cid


def test_compute_champion_id_deterministic():
    """Same inputs always produce the same id (no embedded clock when date given)."""
    a = compute_champion_id("x", "HrnetW32", "10", training_date="2026-04-23")
    b = compute_champion_id("x", "HrnetW32", "10", training_date="2026-04-23")
    assert a == b


def test_compute_champion_id_uses_today_if_no_date():
    """When training_date is None, today's UTC date is used. Just verify form."""
    import datetime
    today = datetime.datetime.now(datetime.UTC).date().isoformat().replace("-", "")
    cid = compute_champion_id("x", "HrnetW32", "10")
    assert cid == f"dlc-{today}-hrnetw32-snap10"


# ---------------------------------------------------------------------------
# get_champion_manifest
# ---------------------------------------------------------------------------


class _FakeS3NoSuchKey(Exception):
    """Simulates the boto3 ClientError-derived NoSuchKey class."""


def _make_s3_with_manifest(manifest_dict: dict | None) -> MagicMock:
    """Build a MagicMock S3 client that returns ``manifest_dict`` (or 404)."""
    s3 = MagicMock()
    s3.exceptions.NoSuchKey = _FakeS3NoSuchKey
    if manifest_dict is None:
        s3.get_object.side_effect = _FakeS3NoSuchKey()
    else:
        body = MagicMock()
        body.read.return_value = json.dumps(manifest_dict).encode("utf-8")
        s3.get_object.return_value = {"Body": body}
    return s3


def test_get_champion_manifest_returns_dict():
    expected = {
        "champion_id": "dlc-20260423-hrnetw32-snap290",
        "model_name": "hm2p_hrnetw32_shuffle1",
        "architecture": "HrnetW32",
        "snapshot": "290",
    }
    s3 = _make_s3_with_manifest(expected)
    got = get_champion_manifest(s3, "hm2p-derivatives")
    assert got == expected
    s3.get_object.assert_called_once_with(
        Bucket="hm2p-derivatives", Key=CHAMPION_MANIFEST_KEY,
    )


def test_get_champion_manifest_returns_none_when_absent():
    s3 = _make_s3_with_manifest(None)
    assert get_champion_manifest(s3, "hm2p-derivatives") is None


def test_get_champion_manifest_returns_none_on_corrupt_json():
    s3 = MagicMock()
    s3.exceptions.NoSuchKey = _FakeS3NoSuchKey
    body = MagicMock()
    body.read.return_value = b"{not json"
    s3.get_object.return_value = {"Body": body}
    assert get_champion_manifest(s3, "hm2p-derivatives") is None


# ---------------------------------------------------------------------------
# resolve_champion_id
# ---------------------------------------------------------------------------


_MANIFEST = {
    "champion_id": "dlc-20260423-hrnetw32-snap290",
    "model_name": "hm2p_hrnetw32_shuffle1",
    "architecture": "HrnetW32",
    "snapshot": "290",
}


def test_resolve_champion_id_full_match():
    assert resolve_champion_id(
        "hm2p_hrnetw32_shuffle1", "HrnetW32", "290", _MANIFEST,
    ) == "dlc-20260423-hrnetw32-snap290"


def test_resolve_champion_id_snapshot_mismatch_returns_unknown():
    assert resolve_champion_id(
        "hm2p_hrnetw32_shuffle1", "HrnetW32", "100", _MANIFEST,
    ) == "unknown"


def test_resolve_champion_id_architecture_mismatch_returns_unknown():
    assert resolve_champion_id(
        "hm2p_hrnetw32_shuffle1", "Resnet50", "290", _MANIFEST,
    ) == "unknown"


def test_resolve_champion_id_model_name_mismatch_returns_unknown():
    assert resolve_champion_id(
        "different_model", "HrnetW32", "290", _MANIFEST,
    ) == "unknown"


def test_resolve_champion_id_no_manifest_returns_unknown():
    assert resolve_champion_id(
        "any_model", "HrnetW32", "290", None,
    ) == "unknown"


def test_resolve_champion_id_no_architecture_returns_unknown():
    """SuperAnimal baseline files have no architecture marker → unknown."""
    assert resolve_champion_id(
        "superanimal_topviewmouse", None, "superanimal", _MANIFEST,
    ) == "unknown"


def test_resolve_champion_id_handles_int_snapshot_in_manifest():
    """Manifest snapshots may sometimes be JSON-int; comparison must coerce."""
    manifest = {**_MANIFEST, "snapshot": 290}
    assert resolve_champion_id(
        "hm2p_hrnetw32_shuffle1", "HrnetW32", "290", manifest,
    ) == "dlc-20260423-hrnetw32-snap290"
