"""Tests for ``run_stage5_sync.attach_syllables`` (fake S3, synthetic files)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_stage5_sync as rs5  # noqa: E402


class _FakeS3:
    def __init__(self, payload: bytes | None) -> None:
        self.payload = payload

    def get_object(self, Bucket, Key):  # noqa: N803
        if self.payload is None:
            raise RuntimeError("NoSuchKey")
        return {"Body": io.BytesIO(self.payload)}


def _npz(ids: np.ndarray, name: str = "syllable_id") -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{name: ids})
    return buf.getvalue()


def _kin(path: Path, n: int) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("frame_times", data=np.arange(n) / 30.0)


def test_attach_ok(tmp_path: Path) -> None:
    kin = tmp_path / "kinematics.h5"
    _kin(kin, 100)
    ok = rs5.attach_syllables(_FakeS3(_npz(np.arange(100) % 5)), "sub-1", "ses-1", kin)
    assert ok
    with h5py.File(kin) as f:
        assert f["syllable_id"].shape == (100,) and f["syllable_id"].dtype == np.int16
        assert f["syllable_id"].attrs["backend"] == "keypoint-moseq"


def test_attach_missing_file(tmp_path: Path) -> None:
    kin = tmp_path / "kinematics.h5"
    _kin(kin, 10)
    assert rs5.attach_syllables(_FakeS3(None), "s", "e", kin) is False
    with h5py.File(kin) as f:
        assert "syllable_id" not in f


def test_attach_length_mismatch(tmp_path: Path) -> None:
    kin = tmp_path / "kinematics.h5"
    _kin(kin, 10)
    assert (
        rs5.attach_syllables(_FakeS3(_npz(np.zeros(11, dtype=np.int16))), "s", "e", kin) is False
    )


def test_attach_wrong_key(tmp_path: Path) -> None:
    kin = tmp_path / "kinematics.h5"
    _kin(kin, 10)
    payload = _npz(np.zeros(10, dtype=np.int16), name="other")
    assert rs5.attach_syllables(_FakeS3(payload), "s", "e", kin) is False
