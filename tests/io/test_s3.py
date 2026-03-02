"""Tests for hm2p.io.s3 — all boto3 calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hm2p.io.s3 import (
    download_file,
    download_session,
    list_sessions,
    upload_file,
    upload_session,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client():
    """Return a fresh MagicMock standing in for the boto3 S3 client."""
    return MagicMock()


def _patch_client(mock_client):
    """Context manager that replaces hm2p.io.s3._client with a callable returning mock_client."""
    return patch("hm2p.io.s3._client", return_value=mock_client)


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------


class TestUploadFile:
    def test_calls_boto3_upload(self, tmp_path):
        f = tmp_path / "file.h5"
        f.write_bytes(b"data")
        client = _mock_client()
        with _patch_client(client):
            upload_file(f, "my-bucket", "some/key.h5", confirm=False)
        client.upload_file.assert_called_once_with(str(f), "my-bucket", "some/key.h5")

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            upload_file(tmp_path / "missing.h5", "bucket", "key", confirm=False)

    def test_profile_passed_to_client(self, tmp_path):
        f = tmp_path / "x.h5"
        f.write_bytes(b"x")
        with patch("hm2p.io.s3._client") as mock_factory:
            mock_factory.return_value = _mock_client()
            upload_file(f, "bucket", "key", profile="my-profile", confirm=False)
        mock_factory.assert_called_once_with("my-profile")

    def test_confirm_calls_confirm_or_abort(self, tmp_path, monkeypatch):
        f = tmp_path / "data.h5"
        f.write_bytes(b"x" * 100)
        client = _mock_client()
        monkeypatch.setattr("builtins.input", lambda _: "y")
        with _patch_client(client):
            upload_file(f, "bucket", "key", confirm=True)
        client.upload_file.assert_called_once()

    def test_confirm_false_skips_prompt(self, tmp_path):
        f = tmp_path / "data.h5"
        f.write_bytes(b"x")
        client = _mock_client()
        with _patch_client(client):
            upload_file(f, "bucket", "key", confirm=False)
        client.upload_file.assert_called_once()


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    def test_calls_boto3_download(self, tmp_path):
        dest = tmp_path / "sub" / "file.h5"
        client = _mock_client()
        with _patch_client(client):
            download_file("my-bucket", "some/key.h5", dest, confirm=False)
        client.download_file.assert_called_once_with("my-bucket", "some/key.h5", str(dest))

    def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "a" / "b" / "c.h5"
        client = _mock_client()
        with _patch_client(client):
            download_file("bucket", "key", dest, confirm=False)
        assert dest.parent.exists()

    def test_confirm_calls_head_object(self, tmp_path, monkeypatch):
        dest = tmp_path / "out.h5"
        client = _mock_client()
        client.head_object.return_value = {"ContentLength": 5000}
        monkeypatch.setattr("builtins.input", lambda _: "y")
        with _patch_client(client):
            download_file("bucket", "key", dest, confirm=True)
        client.head_object.assert_called_once_with(Bucket="bucket", Key="key")
        client.download_file.assert_called_once()


# ---------------------------------------------------------------------------
# upload_session
# ---------------------------------------------------------------------------


class TestUploadSession:
    def test_uploads_all_files(self, tmp_path):
        (tmp_path / "func.tif").write_bytes(b"1")
        (tmp_path / "daq.tdms").write_bytes(b"2")

        client = _mock_client()
        with _patch_client(client):
            keys = upload_session(tmp_path, "hm2p-rawdata", "rawdata/sub-X/ses-Y", confirm=False)

        assert len(keys) == 2
        assert client.upload_file.call_count == 2

    def test_excludes_side_left(self, tmp_path):
        (tmp_path / "video_side_left.mp4").write_bytes(b"side")
        (tmp_path / "video_top.mp4").write_bytes(b"top")

        client = _mock_client()
        with _patch_client(client):
            keys = upload_session(tmp_path, "bucket", "prefix", confirm=False)

        assert len(keys) == 1
        assert keys[0].endswith("video_top.mp4")

    def test_excludes_red_tif(self, tmp_path):
        (tmp_path / "red.tif").write_bytes(b"red")
        (tmp_path / "green.tif").write_bytes(b"green")

        client = _mock_client()
        with _patch_client(client):
            keys = upload_session(tmp_path, "bucket", "prefix", confirm=False)

        assert len(keys) == 1
        assert keys[0].endswith("green.tif")

    def test_key_format(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "data.h5").write_bytes(b"x")

        client = _mock_client()
        with _patch_client(client):
            keys = upload_session(tmp_path, "bucket", "rawdata/sub-X/ses-Y", confirm=False)

        assert keys == ["rawdata/sub-X/ses-Y/subdir/data.h5"]

    def test_raises_if_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            upload_session(tmp_path / "nope", "bucket", "prefix", confirm=False)

    def test_empty_dir_returns_empty_list(self, tmp_path):
        client = _mock_client()
        with _patch_client(client):
            keys = upload_session(tmp_path, "bucket", "prefix", confirm=False)
        assert keys == []

    def test_confirm_prompts_once_for_all_files(self, tmp_path, monkeypatch):
        (tmp_path / "a.tif").write_bytes(b"1")
        (tmp_path / "b.tif").write_bytes(b"2")
        client = _mock_client()
        call_count = 0

        def fake_input(_):
            nonlocal call_count
            call_count += 1
            return "y"

        monkeypatch.setattr("builtins.input", fake_input)
        with _patch_client(client):
            keys = upload_session(tmp_path, "bucket", "prefix", confirm=True)
        assert len(keys) == 2
        assert call_count == 1  # single prompt for all files


# ---------------------------------------------------------------------------
# download_session
# ---------------------------------------------------------------------------


class TestDownloadSession:
    def _make_paginator(
        self, keys_and_sizes: list[tuple[str, int]] | None = None, keys: list[str] | None = None
    ):
        """Build a mock paginator that yields given keys with sizes."""
        if keys is not None:
            # Legacy: keys-only, default Size=1000
            items = [{"Key": k, "Size": 1000} for k in keys]
        else:
            items = [{"Key": k, "Size": s} for k, s in (keys_and_sizes or [])]
        page = {"Contents": items}
        paginator = MagicMock()
        paginator.paginate.return_value = [page]
        return paginator

    def test_downloads_all_objects(self, tmp_path):
        prefix = "rawdata/sub-X/ses-Y/"
        keys = [f"{prefix}funcimg/data.tif", f"{prefix}behav/vid.mp4"]

        client = _mock_client()
        client.get_paginator.return_value = self._make_paginator(keys=keys)

        with _patch_client(client):
            written = download_session("bucket", "rawdata/sub-X/ses-Y", tmp_path, confirm=False)

        assert len(written) == 2
        assert client.download_file.call_count == 2

    def test_creates_nested_dirs(self, tmp_path):
        prefix = "rawdata/sub-X/ses-Y/"
        keys = [f"{prefix}a/b/c.h5"]

        client = _mock_client()
        client.get_paginator.return_value = self._make_paginator(keys=keys)

        with _patch_client(client):
            download_session("bucket", "rawdata/sub-X/ses-Y", tmp_path, confirm=False)

        assert (tmp_path / "a" / "b").exists()

    def test_empty_bucket_returns_empty_list(self, tmp_path):
        client = _mock_client()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        client.get_paginator.return_value = paginator

        with _patch_client(client):
            written = download_session("bucket", "prefix", tmp_path, confirm=False)

        assert written == []

    def test_confirm_uses_size_from_listing(self, tmp_path, monkeypatch):
        prefix = "data/"
        items = [(f"{prefix}a.h5", 5000), (f"{prefix}b.h5", 3000)]
        client = _mock_client()
        client.get_paginator.return_value = self._make_paginator(keys_and_sizes=items)
        monkeypatch.setattr("builtins.input", lambda _: "y")

        with _patch_client(client):
            written = download_session("bucket", "data", tmp_path, confirm=True)

        assert len(written) == 2
        assert client.download_file.call_count == 2


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_returns_sorted_prefixes(self):
        client = _mock_client()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "rawdata/sub-B/"},
                    {"Prefix": "rawdata/sub-A/"},
                ]
            }
        ]
        client.get_paginator.return_value = paginator

        with _patch_client(client):
            result = list_sessions("bucket", prefix="rawdata/")

        assert result == ["rawdata/sub-A/", "rawdata/sub-B/"]

    def test_empty_bucket(self):
        client = _mock_client()
        paginator = MagicMock()
        paginator.paginate.return_value = [{}]
        client.get_paginator.return_value = paginator

        with _patch_client(client):
            result = list_sessions("bucket")

        assert result == []
