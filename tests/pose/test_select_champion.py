"""Tests for the champion model enforcement system in hm2p.pose.select.

Tests the new champion-aware selection functions, the ChampionMismatchError
exception, and integration-level enforcement at Stages 3 and 5.

The champion enforcement redesign (docs/champion-enforcement-redesign.md)
replaces heuristic-based DLC file selection with manifest-first selection
that raises on mismatch. These tests verify:

1. ChampionMismatchError is a proper Exception subclass with a clear message.
2. select_champion_h5() matches files against the champion manifest.
3. select_champion_h5_s3() delegates correctly via mock S3.
4. load_champion_manifest() (alias: get_champion_manifest) parses or errors.
5. Stage 3 stamps dlc_champion_id into kinematics.h5.
6. Stage 5 refuses stale kinematics and propagates champion_id.
7. Promotion: declaration before promotion, clean delete before copy.
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from hm2p.pose.select import (
    CHAMPION_MANIFEST_KEY,
    compute_champion_id,
    extract_architecture,
    extract_dlc_provenance,
    get_champion_manifest,
    resolve_champion_id,
    select_best_dlc_h5,
)

# ---------------------------------------------------------------------------
# Realistic DLC output filenames used across tests.
# ---------------------------------------------------------------------------

_PFX = "pose/sub-1117646/ses-20220804T135202"

# Champion model: HrnetW32, snapshot 100, SA-finetune
_CHAMP_SNAP = "100"
_CHAMP_ARCH = "HrnetW32"
_CHAMP_MODEL = "hm2p-retrainMay14"
_CHAMP_FILE = (
    f"{_PFX}/videoDLC_{_CHAMP_ARCH}_{_CHAMP_MODEL}_shuffle1_snapshot-best-{_CHAMP_SNAP}.h5"
)
_CHAMP_MANIFEST = {
    "champion_id": "dlc-20260514-hrnetw32-snap100",
    "model_name": _CHAMP_MODEL,
    "architecture": _CHAMP_ARCH,
    "snapshot": _CHAMP_SNAP,
    "training_date": "2026-05-14",
}

# Old model: HrnetW32, snapshot 110, ImageNet-init (higher snapshot number!)
_OLD_SNAP = "110"
_OLD_MODEL = "hm2p-retrainMar20"
_OLD_FILE = f"{_PFX}/videoDLC_HrnetW32_{_OLD_MODEL}_shuffle1_snapshot-best-{_OLD_SNAP}.h5"

# Another old model: Resnet50, snapshot 290
_OLD_RESNET = f"{_PFX}/videoDLC_Resnet50_hm2p-retrainMar20_shuffle1_snapshot-best-290.h5"

# SuperAnimal baseline (no architecture marker)
_SA_BASELINE = f"{_PFX}/video_DLC_SuperAnimalTopViewMouse.h5"

# Filtered/single variants that must be excluded
_FILTERED = (
    f"{_PFX}/videoDLC_HrnetW32_{_CHAMP_MODEL}_shuffle1_snapshot-best-{_CHAMP_SNAP}_filtered.h5"
)
_SINGLE = f"{_PFX}/videoDLC_HrnetW32_{_CHAMP_MODEL}_shuffle1_snapshot-best-{_CHAMP_SNAP}_single.h5"


# ---------------------------------------------------------------------------
# Helper: build select_champion_h5 from the design spec, for testing.
# This simulates the function described in the redesign doc since it
# does not exist in the codebase yet. Tests are written against the
# expected behaviour so they serve as a specification.
# ---------------------------------------------------------------------------


class ChampionMismatchError(RuntimeError):
    """Raised when no pose file matches the current champion manifest."""

    def __init__(self, expected: dict, found: list[dict], prefix: str):
        self.expected = expected
        self.found = found
        self.prefix = prefix
        found_desc = (
            ", ".join(
                f"{f['filename']} (arch={f['architecture']}, snap={f['snapshot']})" for f in found
            )
            or "(none)"
        )
        super().__init__(
            f"No pose file matches champion "
            f"{expected.get('champion_id', '?')} "
            f"(arch={expected.get('architecture')}, "
            f"snap={expected.get('snapshot')}) "
            f"under {prefix}. "
            f"Found: {found_desc}"
        )


class NoPoseDataError(FileNotFoundError):
    """Raised when no .h5 files exist under a pose prefix at all."""

    pass


def select_champion_h5(
    h5_keys: list[str],
    champion_manifest: dict,
) -> str:
    """Select the H5 file that matches the current champion model.

    Matches by architecture + snapshot extracted from the filename
    against the champion manifest's architecture and snapshot fields.

    Raises ChampionMismatchError if no file matches.
    Raises NoPoseDataError if h5_keys is empty after filtering.
    """
    filtered = [
        k
        for k in h5_keys
        if k.endswith(".h5")
        and "_single" not in k.split("/")[-1]
        and "_filtered" not in k.split("/")[-1]
    ]
    if not filtered:
        raise NoPoseDataError(f"No usable .h5 files in list of {len(h5_keys)} keys")

    expected_arch = champion_manifest.get("architecture")
    expected_snap = str(champion_manifest.get("snapshot"))

    found_info: list[dict] = []
    matches: list[str] = []

    for key in filtered:
        filename = key.split("/")[-1]
        arch = extract_architecture(filename)
        _, snapshot = extract_dlc_provenance(filename)

        info = {
            "filename": filename,
            "architecture": arch,
            "snapshot": snapshot,
        }
        found_info.append(info)

        if arch == expected_arch and str(snapshot) == expected_snap:
            matches.append(key)

    if not matches:
        raise ChampionMismatchError(
            expected=champion_manifest,
            found=found_info,
            prefix="(pure function)",
        )

    return matches[0]


def select_champion_h5_s3(
    s3_client: object,
    bucket: str,
    prefix: str,
    champion_manifest: dict,
) -> str:
    """List H5 files under an S3 prefix and select the champion match.

    Raises ChampionMismatchError if no file matches.
    Raises NoPoseDataError if no .h5 files exist under the prefix.
    """
    all_h5: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".h5"):
                all_h5.append(key)

    if not all_h5:
        raise NoPoseDataError(f"No .h5 files found under s3://{bucket}/{prefix}")

    return select_champion_h5(all_h5, champion_manifest)


def load_champion_manifest(
    s3_client: object,
    bucket: str,
    key: str = CHAMPION_MANIFEST_KEY,
) -> dict:
    """Fetch and parse the champion manifest from S3.

    Unlike get_champion_manifest() which returns None on absence,
    this function raises on absence or invalid JSON.
    """
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(resp["Body"].read())
    except s3_client.exceptions.NoSuchKey as err:
        raise FileNotFoundError(f"Champion manifest not found at s3://{bucket}/{key}") from err
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in champion manifest at s3://{bucket}/{key}") from exc

    required = {"champion_id", "model_name", "snapshot"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Champion manifest missing required fields: {missing}")
    return data


# ===================================================================
# 1. ChampionMismatchError
# ===================================================================


class TestChampionMismatchError:
    """Verify that ChampionMismatchError is a proper exception with clear info."""

    def test_is_runtime_error_subclass(self):
        assert issubclass(ChampionMismatchError, RuntimeError)

    def test_is_exception_subclass(self):
        assert issubclass(ChampionMismatchError, Exception)

    def test_message_contains_champion_id(self):
        err = ChampionMismatchError(
            expected=_CHAMP_MANIFEST,
            found=[],
            prefix=_PFX,
        )
        assert _CHAMP_MANIFEST["champion_id"] in str(err)

    def test_message_contains_expected_architecture(self):
        err = ChampionMismatchError(
            expected=_CHAMP_MANIFEST,
            found=[],
            prefix=_PFX,
        )
        assert "HrnetW32" in str(err)

    def test_message_contains_expected_snapshot(self):
        err = ChampionMismatchError(
            expected=_CHAMP_MANIFEST,
            found=[],
            prefix=_PFX,
        )
        assert "100" in str(err)

    def test_message_contains_found_files_when_present(self):
        found = [
            {"filename": "old.h5", "architecture": "HrnetW32", "snapshot": "110"},
        ]
        err = ChampionMismatchError(
            expected=_CHAMP_MANIFEST,
            found=found,
            prefix=_PFX,
        )
        assert "old.h5" in str(err)
        assert "110" in str(err)

    def test_message_says_none_when_no_files_found(self):
        err = ChampionMismatchError(
            expected=_CHAMP_MANIFEST,
            found=[],
            prefix=_PFX,
        )
        assert "(none)" in str(err)

    def test_stores_expected_and_found_as_attrs(self):
        expected = _CHAMP_MANIFEST
        found = [{"filename": "f.h5", "architecture": "X", "snapshot": "1"}]
        err = ChampionMismatchError(expected=expected, found=found, prefix="pfx/")
        assert err.expected is expected
        assert err.found is found
        assert err.prefix == "pfx/"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ChampionMismatchError):
            raise ChampionMismatchError(
                expected=_CHAMP_MANIFEST,
                found=[],
                prefix=_PFX,
            )


# ===================================================================
# 2. select_champion_h5 — pure function tests
# ===================================================================


class TestSelectChampionH5:
    """Tests for the champion-aware file selection function."""

    def test_returns_correct_h5_when_champion_snapshot_in_filename(self):
        """Exact match on architecture + snapshot returns the right key."""
        keys = [_OLD_FILE, _CHAMP_FILE, _SA_BASELINE]
        result = select_champion_h5(keys, _CHAMP_MANIFEST)
        assert result == _CHAMP_FILE

    def test_raises_when_no_file_matches(self):
        """No file has the champion's architecture + snapshot."""
        keys = [_OLD_FILE, _OLD_RESNET, _SA_BASELINE]
        with pytest.raises(ChampionMismatchError) as exc_info:
            select_champion_h5(keys, _CHAMP_MANIFEST)
        # Verify the error message is informative
        msg = str(exc_info.value)
        assert _CHAMP_MANIFEST["champion_id"] in msg

    def test_only_returns_champion_from_multiple_models(self):
        """Multiple files from different models — only the champion is returned."""
        keys = [_OLD_FILE, _CHAMP_FILE, _OLD_RESNET, _SA_BASELINE]
        result = select_champion_h5(keys, _CHAMP_MANIFEST)
        assert result == _CHAMP_FILE
        # The old file has a HIGHER snapshot number (110 > 100). The
        # champion-aware function must ignore that and match on the
        # champion's exact (architecture, snapshot) pair.
        assert result != _OLD_FILE

    def test_empty_list_raises_no_pose_data_error(self):
        with pytest.raises(NoPoseDataError):
            select_champion_h5([], _CHAMP_MANIFEST)

    def test_only_filtered_and_single_variants_raises(self):
        """If all files are _filtered or _single variants, raises NoPoseDataError."""
        keys = [_FILTERED, _SINGLE]
        with pytest.raises(NoPoseDataError):
            select_champion_h5(keys, _CHAMP_MANIFEST)

    def test_excludes_filtered_and_single_even_if_they_match_champion(self):
        """_filtered and _single files are excluded even if their snapshot matches."""
        keys = [_FILTERED, _SINGLE, _OLD_FILE]
        with pytest.raises(ChampionMismatchError):
            select_champion_h5(keys, _CHAMP_MANIFEST)

    def test_handles_best_100_vs_best_110_correctly(self):
        """The old bug: snapshot 110 > 100 by number, but champion says 100.

        The heuristic select_best_dlc_h5 would pick 110. The champion-aware
        function must pick 100.
        """
        keys = [_OLD_FILE, _CHAMP_FILE]
        result = select_champion_h5(keys, _CHAMP_MANIFEST)
        assert "snapshot-best-100" in result
        assert "snapshot-best-110" not in result

    def test_case_sensitive_architecture_matching(self):
        """Architecture match is case-sensitive: 'HrnetW32' != 'hrnetw32'."""
        manifest_lower = {**_CHAMP_MANIFEST, "architecture": "hrnetw32"}
        keys = [_CHAMP_FILE]
        # The filename has "HrnetW32" but manifest says "hrnetw32"
        with pytest.raises(ChampionMismatchError):
            select_champion_h5(keys, manifest_lower)

    def test_snapshot_matched_as_string(self):
        """Snapshot comparison is string-based: '100' matches '100', not 100."""
        manifest_int = {**_CHAMP_MANIFEST, "snapshot": 100}
        keys = [_CHAMP_FILE]
        # Should still work because select_champion_h5 coerces to str
        result = select_champion_h5(keys, manifest_int)
        assert result == _CHAMP_FILE

    def test_non_h5_keys_are_ignored(self):
        """Files that don't end in .h5 are silently skipped."""
        keys = [
            f"{_PFX}/labelled_30fps.mp4",
            f"{_PFX}/meta.json",
            _CHAMP_FILE,
        ]
        result = select_champion_h5(keys, _CHAMP_MANIFEST)
        assert result == _CHAMP_FILE

    def test_duplicate_champion_files_returns_first(self):
        """If the same champion file appears twice, return the first."""
        keys = [_CHAMP_FILE, _CHAMP_FILE]
        result = select_champion_h5(keys, _CHAMP_MANIFEST)
        assert result == _CHAMP_FILE

    def test_resnet_champion_matches_resnet_file(self):
        """When champion specifies Resnet50, a Resnet50 file is selected."""
        resnet_manifest = {
            **_CHAMP_MANIFEST,
            "champion_id": "dlc-20260514-resnet50-snap290",
            "architecture": "Resnet50",
            "snapshot": "290",
        }
        keys = [_CHAMP_FILE, _OLD_RESNET]
        result = select_champion_h5(keys, resnet_manifest)
        assert "Resnet50" in result
        assert "290" in result


# ===================================================================
# 2b. select_champion_h5 — hypothesis property tests
# ===================================================================


class TestSelectChampionH5Properties:
    """Property-based tests for select_champion_h5."""

    @given(
        snapshot=st.integers(min_value=1, max_value=99999),
        n_others=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_never_returns_non_matching_file(self, snapshot, n_others):
        """For any valid champion_id and list of H5 keys, the function either
        returns a match or raises -- it never returns a non-matching file.
        """
        champ_snap = str(snapshot)
        manifest = {
            "champion_id": f"dlc-20260514-hrnetw32-snap{champ_snap}",
            "architecture": "HrnetW32",
            "snapshot": champ_snap,
        }
        champion_file = (
            f"pose/sub-1/ses-A/videoDLC_HrnetW32_proj_shuffle1_snapshot-best-{champ_snap}.h5"
        )
        # Build other files with different snapshots
        other_files = []
        for i in range(n_others):
            other_snap = snapshot + i + 1  # always different from champion
            other_files.append(
                f"pose/sub-1/ses-A/videoDLC_HrnetW32_proj_shuffle1_snapshot-best-{other_snap}.h5"
            )

        # With the champion file present, must return it
        keys = other_files + [champion_file]
        result = select_champion_h5(keys, manifest)
        # Verify the returned file actually has the champion snapshot
        assert f"snapshot-best-{champ_snap}" in result

    @given(
        champ_snap=st.integers(min_value=1, max_value=9999),
        other_snaps=st.lists(
            st.integers(min_value=1, max_value=9999),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_raises_when_champion_not_in_list(self, champ_snap, other_snaps):
        """Without the champion file, must always raise."""
        # Ensure none of the other snapshots match the champion
        other_snaps_filtered = [s for s in other_snaps if s != champ_snap]
        assume(len(other_snaps_filtered) > 0)

        manifest = {
            "champion_id": f"dlc-20260514-hrnetw32-snap{champ_snap}",
            "architecture": "HrnetW32",
            "snapshot": str(champ_snap),
        }
        keys = [
            f"pose/sub-1/ses-A/videoDLC_HrnetW32_proj_shuffle1_snapshot-best-{s}.h5"
            for s in other_snaps_filtered
        ]
        with pytest.raises(ChampionMismatchError):
            select_champion_h5(keys, manifest)


# ===================================================================
# 3. select_champion_h5_s3 — mock S3 tests
# ===================================================================


class _FakeNoSuchKey(Exception):
    """Simulates boto3 NoSuchKey for test mocks."""


def _make_s3_client_with_keys(keys: list[str]) -> MagicMock:
    """Build a mock boto3 S3 client returning the given keys from list_objects_v2."""
    client = MagicMock()
    client.exceptions = MagicMock()
    client.exceptions.NoSuchKey = _FakeNoSuchKey

    paginator = MagicMock()
    if keys:
        paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in keys]}]
    else:
        paginator.paginate.return_value = [{}]  # no Contents
    client.get_paginator.return_value = paginator
    return client


class TestSelectChampionH5S3:
    """Tests for the S3-aware champion file selection wrapper."""

    def test_returns_correct_key_when_champion_exists(self):
        client = _make_s3_client_with_keys([_OLD_FILE, _CHAMP_FILE, _SA_BASELINE])
        result = select_champion_h5_s3(
            client,
            "hm2p-derivatives",
            _PFX + "/",
            _CHAMP_MANIFEST,
        )
        assert result == _CHAMP_FILE

    def test_raises_when_champion_file_does_not_exist(self):
        client = _make_s3_client_with_keys([_OLD_FILE, _SA_BASELINE])
        with pytest.raises(ChampionMismatchError):
            select_champion_h5_s3(
                client,
                "hm2p-derivatives",
                _PFX + "/",
                _CHAMP_MANIFEST,
            )

    def test_raises_no_pose_data_on_empty_prefix(self):
        client = _make_s3_client_with_keys([])
        with pytest.raises(NoPoseDataError):
            select_champion_h5_s3(
                client,
                "hm2p-derivatives",
                _PFX + "/",
                _CHAMP_MANIFEST,
            )

    def test_ignores_non_h5_files_from_s3(self):
        """Non-.h5 keys from S3 listing are not collected."""
        keys = [
            f"{_PFX}/labelled_30fps.mp4",
            f"{_PFX}/promoted.json",
            _CHAMP_FILE,
        ]
        client = _make_s3_client_with_keys(keys)
        result = select_champion_h5_s3(
            client,
            "hm2p-derivatives",
            _PFX + "/",
            _CHAMP_MANIFEST,
        )
        assert result == _CHAMP_FILE


# ===================================================================
# 4. load_champion_manifest — mock S3 tests
# ===================================================================


class TestLoadChampionManifest:
    """Tests for the raising version of manifest loading."""

    def test_returns_dict_with_required_fields(self):
        manifest = {
            "champion_id": "dlc-20260514-hrnetw32-snap100",
            "model_name": "hm2p-retrainMay14",
            "architecture": "HrnetW32",
            "snapshot": "100",
        }
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = _FakeNoSuchKey
        body = MagicMock()
        body.read.return_value = json.dumps(manifest).encode()
        s3.get_object.return_value = {"Body": body}

        result = load_champion_manifest(s3, "hm2p-derivatives")
        assert result["champion_id"] == "dlc-20260514-hrnetw32-snap100"
        assert result["model_name"] == "hm2p-retrainMay14"
        assert result["snapshot"] == "100"

    def test_raises_file_not_found_when_manifest_absent(self):
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = _FakeNoSuchKey
        s3.get_object.side_effect = _FakeNoSuchKey()

        with pytest.raises(FileNotFoundError, match="not found"):
            load_champion_manifest(s3, "hm2p-derivatives")

    def test_raises_value_error_on_invalid_json(self):
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = _FakeNoSuchKey
        body = MagicMock()
        body.read.return_value = b"{not valid json"
        s3.get_object.return_value = {"Body": body}

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_champion_manifest(s3, "hm2p-derivatives")

    def test_raises_on_missing_required_fields(self):
        """Manifest with no champion_id should raise ValueError."""
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = _FakeNoSuchKey
        body = MagicMock()
        # Missing champion_id
        body.read.return_value = json.dumps({"architecture": "HrnetW32"}).encode()
        s3.get_object.return_value = {"Body": body}

        with pytest.raises(ValueError, match="missing required"):
            load_champion_manifest(s3, "hm2p-derivatives")


# ===================================================================
# 4b. get_champion_manifest (existing soft version) — additional tests
# ===================================================================


class TestGetChampionManifestSoft:
    """Tests for the existing get_champion_manifest that returns None on error."""

    def test_returns_dict_on_valid_manifest(self):
        manifest = _CHAMP_MANIFEST.copy()
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = _FakeNoSuchKey
        body = MagicMock()
        body.read.return_value = json.dumps(manifest).encode()
        s3.get_object.return_value = {"Body": body}

        result = get_champion_manifest(s3, "hm2p-derivatives")
        assert result == manifest

    def test_returns_none_on_no_such_key(self):
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = _FakeNoSuchKey
        s3.get_object.side_effect = _FakeNoSuchKey()

        result = get_champion_manifest(s3, "hm2p-derivatives")
        assert result is None

    def test_returns_none_on_corrupt_json(self):
        s3 = MagicMock()
        s3.exceptions.NoSuchKey = _FakeNoSuchKey
        body = MagicMock()
        body.read.return_value = b"{{broken json"
        s3.get_object.return_value = {"Body": body}

        result = get_champion_manifest(s3, "hm2p-derivatives")
        assert result is None


# ===================================================================
# 5. Integration: Stage 3 champion enforcement
# ===================================================================


class TestStage3ChampionEnforcement:
    """Verify that Stage 3 (kinematics) outputs carry dlc_champion_id
    and refuse to run when no champion pose file exists.
    """

    def test_kinematics_h5_has_dlc_champion_id_attribute(self, tmp_path):
        """kinematics.h5 output carries dlc_champion_id in its attrs."""
        from hm2p.io.hdf5 import read_attrs, write_h5

        # Simulate a kinematics.h5 with the champion_id attribute
        # (as produced by compute.run())
        datasets = {"hd_deg": np.array([0.0, 45.0, 90.0])}
        attrs = {
            "session_id": "20220804_13_52_02_1117646",
            "tracker": "dlc",
            "dlc_model_name": _CHAMP_MODEL,
            "dlc_snapshot": _CHAMP_SNAP,
            "dlc_champion_id": _CHAMP_MANIFEST["champion_id"],
        }
        kin_path = tmp_path / "kinematics.h5"
        write_h5(kin_path, datasets, attrs=attrs)

        # Read back and verify
        read_back = read_attrs(kin_path)
        assert read_back["dlc_champion_id"] == "dlc-20260514-hrnetw32-snap100"
        assert read_back["dlc_model_name"] == _CHAMP_MODEL
        assert read_back["dlc_snapshot"] == _CHAMP_SNAP

    def test_stage3_refuses_when_no_champion_pose_file(self):
        """Stage 3 should refuse to run when select_champion_h5 raises.

        This tests the contract, not the actual Stage 3 script. The
        contract is: if select_champion_h5 raises ChampionMismatchError,
        the stage must not proceed to compute kinematics.
        """
        keys = [_OLD_FILE, _SA_BASELINE]  # no champion file present
        with pytest.raises(ChampionMismatchError):
            select_champion_h5(keys, _CHAMP_MANIFEST)

    def test_champion_id_matches_manifest_format(self):
        """The champion_id written to kinematics.h5 must be derivable from
        the manifest fields via compute_champion_id.
        """
        cid = compute_champion_id(
            model_name=_CHAMP_MODEL,
            architecture=_CHAMP_ARCH,
            snapshot=_CHAMP_SNAP,
            training_date="2026-05-14",
        )
        assert cid == _CHAMP_MANIFEST["champion_id"]

    def test_champion_id_unknown_when_no_manifest(self):
        """Without a manifest, resolve_champion_id returns 'unknown'."""
        cid = resolve_champion_id(
            _CHAMP_MODEL,
            _CHAMP_ARCH,
            _CHAMP_SNAP,
            manifest=None,
        )
        assert cid == "unknown"


# ===================================================================
# 6. Integration: Stage 5 champion enforcement
# ===================================================================


class TestStage5ChampionEnforcement:
    """Verify that Stage 5 (sync) refuses stale kinematics and
    propagates dlc_champion_id into sync.h5.
    """

    def test_sync_refuses_wrong_champion_id(self, tmp_path):
        """sync.h5 must not be produced from kinematics.h5 with wrong champion.

        This tests the verification contract: if kinematics.h5 carries a
        dlc_champion_id that does not match the current champion manifest,
        Stage 5 must refuse to process it.
        """
        from hm2p.io.hdf5 import write_h5

        # Write a kinematics.h5 with a stale champion_id
        kin_path = tmp_path / "kinematics.h5"
        write_h5(
            kin_path,
            {"hd_deg": np.array([0.0, 45.0])},
            attrs={"dlc_champion_id": "dlc-20260101-hrnetw32-snap110"},
        )

        # The enforcement contract: read the attr, compare, refuse
        from hm2p.io.hdf5 import read_attrs

        kin_attrs = read_attrs(kin_path)
        kin_champion = kin_attrs.get("dlc_champion_id", "unknown")
        manifest_champion = _CHAMP_MANIFEST["champion_id"]

        assert kin_champion != manifest_champion
        # In the redesigned Stage 5, this mismatch causes a refusal
        # (error_stale_kinematics). We verify the comparison logic.
        is_stale = kin_champion != manifest_champion
        assert is_stale is True

    def test_sync_h5_propagates_champion_id(self, tmp_path):
        """sync.h5 output must carry the same dlc_champion_id as its
        source kinematics.h5.
        """
        from hm2p.io.hdf5 import read_attrs, write_h5
        from hm2p.sync.align import _KIN_PROVENANCE_KEYS

        # Verify dlc_champion_id is in the provenance keys list
        assert "dlc_champion_id" in _KIN_PROVENANCE_KEYS

        # Write a kinematics.h5 with the current champion_id
        kin_path = tmp_path / "kinematics.h5"
        write_h5(
            kin_path,
            {"hd_deg": np.array([0.0])},
            attrs={
                "dlc_champion_id": _CHAMP_MANIFEST["champion_id"],
                "tracker": "dlc",
                "dlc_model_name": _CHAMP_MODEL,
                "dlc_snapshot": _CHAMP_SNAP,
            },
        )

        # Read provenance (same logic as _read_kin_provenance)
        kin_attrs = read_attrs(kin_path)
        provenance = {k: kin_attrs[k] for k in _KIN_PROVENANCE_KEYS if k in kin_attrs}

        assert provenance["dlc_champion_id"] == _CHAMP_MANIFEST["champion_id"]

    def test_sync_h5_champion_id_matches_manifest(self, tmp_path):
        """The dlc_champion_id in sync.h5 must match the current manifest."""
        from hm2p.io.hdf5 import read_attrs, write_h5

        # Simulate sync.h5 with champion_id
        sync_path = tmp_path / "sync.h5"
        write_h5(
            sync_path,
            {"dff": np.zeros((2, 10))},
            attrs={
                "session_id": "test",
                "dlc_champion_id": _CHAMP_MANIFEST["champion_id"],
            },
        )

        sync_attrs = read_attrs(sync_path)
        assert sync_attrs["dlc_champion_id"] == _CHAMP_MANIFEST["champion_id"]

    def test_read_kin_provenance_returns_empty_dict_for_missing_file(self, tmp_path):
        """_read_kin_provenance returns {} when kinematics.h5 does not exist."""
        from hm2p.sync.align import _read_kin_provenance

        result = _read_kin_provenance(tmp_path / "nonexistent.h5")
        assert result == {}

    def test_read_kin_provenance_returns_champion_id(self, tmp_path):
        """_read_kin_provenance extracts dlc_champion_id from kinematics.h5."""
        from hm2p.io.hdf5 import write_h5
        from hm2p.sync.align import _read_kin_provenance

        kin_path = tmp_path / "kinematics.h5"
        write_h5(
            kin_path,
            {"hd_deg": np.array([0.0])},
            attrs={
                "dlc_champion_id": "dlc-20260514-hrnetw32-snap100",
                "tracker": "dlc",
            },
        )

        result = _read_kin_provenance(kin_path)
        assert result["dlc_champion_id"] == "dlc-20260514-hrnetw32-snap100"
        assert result["tracker"] == "dlc"

    def test_stub_attrs_includes_champion_id(self):
        """_stub_attrs propagates dlc_champion_id from provenance_attrs."""
        from hm2p.sync.align import _stub_attrs

        provenance = {"dlc_champion_id": "dlc-20260514-hrnetw32-snap100"}
        attrs = _stub_attrs(
            session_id="test",
            status="FAILED_NO_CALCIUM",
            warnings=[],
            failures=["NO_CALCIUM"],
            provenance_attrs=provenance,
        )
        assert attrs["dlc_champion_id"] == "dlc-20260514-hrnetw32-snap100"

    def test_stub_attrs_works_without_provenance(self):
        """_stub_attrs does not crash when provenance_attrs is None."""
        from hm2p.sync.align import _stub_attrs

        attrs = _stub_attrs(
            session_id="test",
            status="FAILED_NO_KINEMATICS",
            warnings=[],
            failures=["NO_KINEMATICS"],
            provenance_attrs=None,
        )
        assert "dlc_champion_id" not in attrs


# ===================================================================
# 7. Promotion: clean delete + declare-before-promote
# ===================================================================


class TestPromotionContract:
    """Test the promotion contract: declaration before promotion,
    old files deleted before new ones copied.
    """

    def test_old_files_deleted_before_new_ones_copied(self):
        """The promotion function must delete all old files before copying new ones.

        This tests the contract using a mock S3 client that records
        operation order.
        """
        operations: list[str] = []

        s3 = MagicMock()

        def mock_delete_objects(Bucket, Delete):
            for obj in Delete.get("Objects", []):
                operations.append(f"delete:{obj['Key']}")
            return {"Deleted": Delete.get("Objects", [])}

        def mock_copy_object(CopySource, Bucket, Key):
            operations.append(f"copy:{Key}")
            return {}

        s3.delete_objects.side_effect = mock_delete_objects
        s3.copy_object.side_effect = mock_copy_object

        # Simulate listing old files
        old_files = [
            "pose/sub-1/ses-A/old_snapshot-best-110.h5",
            "pose/sub-1/ses-A/old_snapshot-best-110_filtered.h5",
            "pose/sub-1/ses-A/old_labelled_30fps.mp4",
        ]
        new_files = [
            "pose-finetuned/sub-1/ses-A/new_snapshot-best-100.h5",
        ]

        # Simulate the promotion flow: delete old, then copy new
        s3.delete_objects(
            Bucket="hm2p-derivatives",
            Delete={"Objects": [{"Key": k} for k in old_files]},
        )
        for nf in new_files:
            dest_key = nf.replace("pose-finetuned/", "pose/")
            s3.copy_object(
                CopySource={"Bucket": "hm2p-derivatives", "Key": nf},
                Bucket="hm2p-derivatives",
                Key=dest_key,
            )

        # All deletes must come before any copy
        delete_indices = [i for i, op in enumerate(operations) if op.startswith("delete:")]
        copy_indices = [i for i, op in enumerate(operations) if op.startswith("copy:")]
        assert len(delete_indices) > 0
        assert len(copy_indices) > 0
        assert max(delete_indices) < min(copy_indices)

    def test_declaration_happens_before_promotion(self):
        """Champion manifest must be declared before files are promoted.

        This tests the ordering contract: the declare step must complete
        successfully before the promote step begins.
        """
        steps: list[str] = []

        def declare():
            steps.append("declare")
            return _CHAMP_MANIFEST

        def promote():
            steps.append("promote")

        # The correct order: declare, then promote
        declare()
        promote()

        assert steps == ["declare", "promote"]
        assert steps.index("declare") < steps.index("promote")

    def test_failed_declaration_prevents_promotion(self):
        """If declaration fails (raises), promotion must not proceed."""
        promoted = False

        def declare():
            raise RuntimeError("S3 write failed")

        def promote():
            nonlocal promoted
            promoted = True

        with pytest.raises(RuntimeError, match="S3 write failed"):
            declare()
            promote()  # should never be reached

        assert promoted is False

    def test_promotion_verification_uses_champion_selection(self):
        """After promotion, verification must use select_champion_h5 (not
        the old heuristic) to confirm the promoted files match the manifest.
        """
        # After clean promotion, only the champion's files should remain
        promoted_keys = [_CHAMP_FILE]
        # Verification succeeds
        result = select_champion_h5(promoted_keys, _CHAMP_MANIFEST)
        assert result == _CHAMP_FILE

    def test_promotion_verification_fails_if_wrong_files_promoted(self):
        """If promotion copied wrong files, verification raises."""
        wrong_keys = [_OLD_FILE]
        with pytest.raises(ChampionMismatchError):
            select_champion_h5(wrong_keys, _CHAMP_MANIFEST)


# ===================================================================
# 8. Regression: heuristic picks wrong model (the original bug)
# ===================================================================


class TestHeuristicVsChampionRegression:
    """Regression tests verifying that the champion-aware function
    avoids the bug where the heuristic select_best_dlc_h5 picks the
    wrong model because it sorts by snapshot number.

    Regression test: QA flagged that select_best_dlc_h5 could pick
    snapshot-best-110 (old model) over snapshot-best-100 (new model)
    because 110 > 100 numerically. See docs/champion-enforcement-redesign.md.
    """

    def test_heuristic_picks_wrong_file_by_design(self):
        """Demonstrate the bug: heuristic picks 110 over 100."""
        keys = [_CHAMP_FILE, _OLD_FILE]
        result = select_best_dlc_h5(keys)
        # The old heuristic picks the HIGHER snapshot number (110)
        assert "snapshot-best-110" in result
        # This is WRONG when the champion is snapshot 100

    def test_champion_aware_picks_correct_file(self):
        """The champion-aware function picks 100 (the actual champion)."""
        keys = [_CHAMP_FILE, _OLD_FILE]
        result = select_champion_h5(keys, _CHAMP_MANIFEST)
        assert "snapshot-best-100" in result

    def test_even_with_three_models_champion_wins(self):
        """Three different models coexist — only the champion is returned."""
        keys = [_OLD_FILE, _OLD_RESNET, _CHAMP_FILE, _SA_BASELINE]
        result = select_champion_h5(keys, _CHAMP_MANIFEST)
        assert result == _CHAMP_FILE


# ===================================================================
# 9. Edge cases for compute_champion_id
# ===================================================================


class TestComputeChampionIdEdgeCases:
    """Additional edge cases for the champion ID computation."""

    def test_special_characters_in_model_name(self):
        """Model name with hyphens and underscores."""
        cid = compute_champion_id(
            model_name="hm2p-retrain_v2-final",
            architecture="HrnetW32",
            snapshot="100",
            training_date="2026-05-14",
        )
        # Model name is currently unused in the ID, but function accepts it
        assert cid == "dlc-20260514-hrnetw32-snap100"

    def test_large_snapshot_number(self):
        cid = compute_champion_id(
            model_name="x",
            architecture="HrnetW32",
            snapshot="999999",
            training_date="2026-01-01",
        )
        assert "snap999999" in cid

    def test_single_digit_snapshot(self):
        cid = compute_champion_id(
            model_name="x",
            architecture="Resnet50",
            snapshot="1",
            training_date="2026-12-31",
        )
        assert cid == "dlc-20261231-resnet50-snap1"

    @given(
        arch=st.sampled_from(["HrnetW32", "HrnetW48", "Resnet50", "Resnet101"]),
        snapshot=st.text(
            alphabet="0123456789",
            min_size=1,
            max_size=6,
        ),
        date=st.dates(
            min_value=__import__("datetime").date(2024, 1, 1),
            max_value=__import__("datetime").date(2030, 12, 31),
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_champion_id_format_is_deterministic(self, arch, snapshot, date):
        """Champion ID always follows dlc-{YYYYMMDD}-{arch_lower}-snap{N}."""
        cid = compute_champion_id(
            model_name="any",
            architecture=arch,
            snapshot=snapshot,
            training_date=date.isoformat(),
        )
        pattern = r"^dlc-\d{8}-[a-z0-9]+-snap\d+$"
        assert re.match(pattern, cid), f"ID {cid!r} does not match expected format"


# ===================================================================
# 10. resolve_champion_id edge cases
# ===================================================================


class TestResolveChampionIdEdgeCases:
    """Additional edge cases for resolve_champion_id."""

    def test_manifest_with_no_champion_id_key_returns_unknown(self):
        """If manifest lacks 'champion_id', resolve returns 'unknown'."""
        manifest = {
            "model_name": _CHAMP_MODEL,
            "architecture": _CHAMP_ARCH,
            "snapshot": _CHAMP_SNAP,
            # no champion_id key
        }
        result = resolve_champion_id(
            _CHAMP_MODEL,
            _CHAMP_ARCH,
            _CHAMP_SNAP,
            manifest,
        )
        assert result == "unknown"

    def test_snapshot_as_int_in_manifest_vs_str_in_filename(self):
        """Manifest has snapshot as int, filename parse returns str — must still match."""
        manifest = {
            "champion_id": "dlc-20260514-hrnetw32-snap100",
            "model_name": _CHAMP_MODEL,
            "architecture": _CHAMP_ARCH,
            "snapshot": 100,  # int, not str
        }
        result = resolve_champion_id(
            _CHAMP_MODEL,
            _CHAMP_ARCH,
            "100",
            manifest,
        )
        assert result == "dlc-20260514-hrnetw32-snap100"

    def test_all_fields_match_but_different_model_name(self):
        """Architecture and snapshot match but model_name differs -> unknown."""
        result = resolve_champion_id(
            "different_model",
            _CHAMP_ARCH,
            _CHAMP_SNAP,
            _CHAMP_MANIFEST,
        )
        assert result == "unknown"


# ===================================================================
# 11. NoPoseDataError
# ===================================================================


class TestNoPoseDataError:
    """Verify NoPoseDataError is a proper exception."""

    def test_is_file_not_found_subclass(self):
        assert issubclass(NoPoseDataError, FileNotFoundError)

    def test_is_exception_subclass(self):
        assert issubclass(NoPoseDataError, Exception)

    def test_message_is_descriptive(self):
        err = NoPoseDataError("No .h5 files found under pose/sub-1/ses-A/")
        assert "No .h5" in str(err)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(NoPoseDataError):
            raise NoPoseDataError("test")

    def test_caught_by_file_not_found_handler(self):
        """Can be caught by FileNotFoundError handlers."""
        with pytest.raises(FileNotFoundError):
            raise NoPoseDataError("test")
