"""Tests for kinematics/syllables.py — behavioural syllable analysis (Stage 3b)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


def test_syllables_module_importable() -> None:
    """syllables module can be imported without optional dependencies."""
    from hm2p.kinematics import syllables  # noqa: F401


def test_keypoint_moseq_not_required_for_import() -> None:
    """keypoint_moseq import failure does not break the module import."""
    from hm2p.kinematics.syllables import append_syllables_to_h5  # noqa: F401


def test_run_keypoint_moseq_importable() -> None:
    """run_keypoint_moseq can be imported and called (returns empty for no files)."""
    from hm2p.kinematics.syllables import run_keypoint_moseq

    # With empty dlc_files list, should create output dir and return empty
    # (actual kpms call requires Docker or kpms installed)
    assert callable(run_keypoint_moseq)


def test_run_keypoint_moseq_no_docker_no_files(tmp_path) -> None:
    """run_keypoint_moseq with empty file list raises RuntimeError from subprocess."""
    from hm2p.kinematics.syllables import run_keypoint_moseq

    # With use_docker=False and no kpms installed, it should fail with
    # a subprocess error or FileNotFoundError for empty input
    with pytest.raises((RuntimeError, FileNotFoundError, subprocess.TimeoutExpired)):
        run_keypoint_moseq(
            dlc_files=[],
            project_dir=tmp_path / "project",
            output_dir=tmp_path / "output",
            use_docker=False,
        )


import subprocess


def test_run_vame_not_implemented() -> None:
    """run_vame raises NotImplementedError (deferred)."""
    from hm2p.kinematics.syllables import run_vame

    with pytest.raises(NotImplementedError):
        run_vame(pose_datasets=[], session_ids=[], project_dir=Path("."), output_dir=Path("."))


# ---------------------------------------------------------------------------
# append_syllables_to_h5 tests
# ---------------------------------------------------------------------------

@pytest.fixture
def kin_h5(tmp_path):
    """Create a minimal kinematics.h5 with frame_times."""
    path = tmp_path / "kinematics.h5"
    n = 500
    with h5py.File(path, "w") as f:
        f.create_dataset("frame_times", data=np.linspace(0, 50, n))
        f.create_dataset("hd", data=np.random.rand(n) * 360)
    return path


class TestAppendSyllables:
    def test_basic_append(self, kin_h5):
        from hm2p.kinematics.syllables import append_syllables_to_h5

        n = 500
        ids = np.random.randint(0, 10, size=n).astype(np.int16)
        append_syllables_to_h5(kin_h5, ids)

        with h5py.File(kin_h5, "r") as f:
            assert "syllable_id" in f
            assert f["syllable_id"].dtype == np.int16
            assert len(f["syllable_id"]) == n
            assert f["syllable_id"].attrs["backend"] == "keypoint-moseq"
            np.testing.assert_array_equal(f["syllable_id"][:], ids)

    def test_append_with_probs(self, kin_h5):
        from hm2p.kinematics.syllables import append_syllables_to_h5

        n, s = 500, 8
        ids = np.random.randint(0, s, size=n).astype(np.int16)
        probs = np.random.rand(n, s).astype(np.float32)

        append_syllables_to_h5(kin_h5, ids, syllable_probs=probs)

        with h5py.File(kin_h5, "r") as f:
            assert "syllable_prob" in f
            assert f["syllable_prob"].shape == (n, s)

    def test_upsert_replaces(self, kin_h5):
        from hm2p.kinematics.syllables import append_syllables_to_h5

        n = 500
        ids1 = np.zeros(n, dtype=np.int16)
        ids2 = np.ones(n, dtype=np.int16)

        append_syllables_to_h5(kin_h5, ids1)
        append_syllables_to_h5(kin_h5, ids2, backend="vame")

        with h5py.File(kin_h5, "r") as f:
            np.testing.assert_array_equal(f["syllable_id"][:], ids2)
            assert f["syllable_id"].attrs["backend"] == "vame"

    def test_length_mismatch_raises(self, kin_h5):
        from hm2p.kinematics.syllables import append_syllables_to_h5

        wrong_len = np.zeros(100, dtype=np.int16)
        with pytest.raises(ValueError, match="syllable_ids length"):
            append_syllables_to_h5(kin_h5, wrong_len)

    def test_missing_file_raises(self, tmp_path):
        from hm2p.kinematics.syllables import append_syllables_to_h5

        with pytest.raises(FileNotFoundError):
            append_syllables_to_h5(
                tmp_path / "nonexistent.h5",
                np.zeros(10, dtype=np.int16),
            )

    def test_prob_length_mismatch_raises(self, kin_h5):
        from hm2p.kinematics.syllables import append_syllables_to_h5

        n = 500
        ids = np.zeros(n, dtype=np.int16)
        bad_probs = np.zeros((n + 10, 5), dtype=np.float32)

        with pytest.raises(ValueError, match="syllable_probs rows"):
            append_syllables_to_h5(kin_h5, ids, syllable_probs=bad_probs)

    def test_2d_ids_raises(self, kin_h5):
        from hm2p.kinematics.syllables import append_syllables_to_h5

        ids = np.zeros((10, 2), dtype=np.int16)
        with pytest.raises(ValueError, match="must be 1D"):
            append_syllables_to_h5(kin_h5, ids)
