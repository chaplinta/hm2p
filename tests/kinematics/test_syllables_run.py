"""Tests for run_keypoint_moseq orchestration (subprocess mocked).

No real keypoint-MoSeq / Docker is invoked: subprocess.run is replaced by
a fake that writes a synthetic ``*_syllables.npz`` so the output-loading
branch is exercised deterministically.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from hm2p.kinematics import syllables
from hm2p.kinematics.syllables import run_keypoint_moseq


def _make_dlc_file(tmp_path: Path, name: str = "sess1.h5") -> Path:
    """A stand-in DLC .h5 file — contents are irrelevant (just copied)."""
    p = tmp_path / name
    p.write_bytes(b"dummy-dlc")
    return p


def _fake_run_factory(output_dir: Path, session_id: str = "sess1"):
    """Return a subprocess.run replacement that writes syllable output."""

    def _fake_run(cmd, capture_output=True, text=True, timeout=None):
        npz = output_dir / f"{session_id}_syllables.npz"
        np.savez(npz, syllable_id=np.array([0, 1, 1, 2, 0], dtype=np.int16))
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    return _fake_run


def test_run_keypoint_moseq_docker_success(monkeypatch, tmp_path: Path) -> None:
    """Docker path: subprocess succeeds and outputs are loaded from npz."""
    dlc = _make_dlc_file(tmp_path)
    project_dir = tmp_path / "project"
    output_dir = tmp_path / "out"
    monkeypatch.setattr(syllables.subprocess, "run", _fake_run_factory(output_dir), raising=True)

    result = run_keypoint_moseq([dlc], project_dir, output_dir, use_docker=True, num_iters=1)
    assert "sess1" in result
    assert result["sess1"].dtype == np.int16
    assert result["sess1"].tolist() == [0, 1, 1, 2, 0]
    # Directories are created as a side-effect.
    assert project_dir.exists()
    assert output_dir.exists()


def test_run_keypoint_moseq_subprocess_path(monkeypatch, tmp_path: Path) -> None:
    """use_docker=False builds the direct-python command and loads output."""
    dlc = _make_dlc_file(tmp_path, name="sessB.h5")
    project_dir = tmp_path / "project"
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        syllables.subprocess,
        "run",
        _fake_run_factory(output_dir, session_id="sessB"),
        raising=True,
    )

    result = run_keypoint_moseq(
        [dlc],
        project_dir,
        output_dir,
        use_docker=False,
        bodyparts=["nose", "left_ear", "right_ear"],
        num_iters=1,
    )
    assert "sessB" in result
    assert len(result["sessB"]) == 5


def test_run_keypoint_moseq_failure_raises(monkeypatch, tmp_path: Path) -> None:
    """Non-zero subprocess exit raises RuntimeError."""
    dlc = _make_dlc_file(tmp_path)
    project_dir = tmp_path / "project"
    output_dir = tmp_path / "out"

    def _fail_run(cmd, capture_output=True, text=True, timeout=None):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(syllables.subprocess, "run", _fail_run, raising=True)

    raised = False
    try:
        run_keypoint_moseq([dlc], project_dir, output_dir, num_iters=1)
    except RuntimeError as exc:
        raised = True
        assert "keypoint-MoSeq failed" in str(exc)
    assert raised
