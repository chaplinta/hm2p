"""Tests for hm2p.anatomy.register."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hm2p.anatomy.register import (
    batch_register,
    find_signal_channel,
    run_brainreg,
)


# ── find_signal_channel ─────────────────────────────────────────────


def test_find_signal_channel(tmp_path: Path) -> None:
    """Finds the single green-channel TIFF in a directory."""
    green_file = tmp_path / "brain_green_channel.tif"
    green_file.write_text("fake")
    # Unrelated file — should be ignored.
    (tmp_path / "brain_red_channel.tif").write_text("fake")

    result = find_signal_channel(tmp_path)
    assert result == green_file


def test_find_signal_channel_case_insensitive(tmp_path: Path) -> None:
    """Name matching is case-insensitive."""
    green_file = tmp_path / "Brain_GREEN_ch.tif"
    green_file.write_text("fake")

    assert find_signal_channel(tmp_path) == green_file


def test_find_signal_channel_not_found(tmp_path: Path) -> None:
    """Raises FileNotFoundError when no green file exists."""
    (tmp_path / "brain_red.tif").write_text("fake")

    with pytest.raises(FileNotFoundError, match="green"):
        find_signal_channel(tmp_path)


def test_find_signal_channel_multiple(tmp_path: Path) -> None:
    """Raises RuntimeError when multiple green files exist."""
    (tmp_path / "a_green.tif").write_text("fake")
    (tmp_path / "b_green.tif").write_text("fake")

    with pytest.raises(RuntimeError, match="Multiple"):
        find_signal_channel(tmp_path)


def test_find_signal_channel_in_subdirectory(tmp_path: Path) -> None:
    """Finds the green file even when nested in a subdirectory."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    green_file = subdir / "green_volume.tif"
    green_file.write_text("fake")

    assert find_signal_channel(tmp_path) == green_file


# ── run_brainreg ────────────────────────────────────────────────────


def test_run_brainreg_command(tmp_path: Path) -> None:
    """Builds the correct subprocess command and calls it."""
    signal = tmp_path / "green.tif"
    signal.write_text("fake")
    out = tmp_path / "output"

    with patch("hm2p.anatomy.register.subprocess.run") as mock_run:
        result = run_brainreg(signal, out, voxel_size=(25.0, 25.0, 25.0))

    assert result == out
    assert out.exists()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]

    assert cmd[0] == "brainreg"
    assert str(signal) in cmd
    assert str(out) in cmd
    # Voxel sizes should appear after -v flag.
    v_idx = cmd.index("-v")
    assert cmd[v_idx + 1 : v_idx + 4] == ["25.0", "25.0", "25.0"]
    assert "--orientation" in cmd
    assert "psl" in cmd
    assert "--atlas" in cmd
    assert "allen_mouse_25um" in cmd
    assert "--save-original-orientation" in cmd


def test_run_brainreg_missing_signal(tmp_path: Path) -> None:
    """Raises FileNotFoundError if signal file does not exist."""
    signal = tmp_path / "nonexistent.tif"
    out = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        run_brainreg(signal, out)


def test_run_brainreg_custom_params(tmp_path: Path) -> None:
    """Custom voxel_size, orientation, and atlas are forwarded."""
    signal = tmp_path / "green.tif"
    signal.write_text("fake")
    out = tmp_path / "out"

    with patch("hm2p.anatomy.register.subprocess.run") as mock_run:
        run_brainreg(
            signal,
            out,
            voxel_size=(10.0, 10.0, 10.0),
            orientation="asr",
            atlas="allen_mouse_50um",
        )

    cmd = mock_run.call_args[0][0]
    v_idx = cmd.index("-v")
    assert cmd[v_idx + 1 : v_idx + 4] == ["10.0", "10.0", "10.0"]
    assert "asr" in cmd
    assert "allen_mouse_50um" in cmd


# ── batch_register ──────────────────────────────────────────────────


def test_batch_register(tmp_path: Path) -> None:
    """Registers all brain subdirectories, collecting results."""
    brains = tmp_path / "brains"
    brains.mkdir()
    out = tmp_path / "output"

    # Two brain directories, each with a green file.
    for name in ("brain_A", "brain_B"):
        d = brains / name
        d.mkdir()
        (d / f"{name}_green.tif").write_text("fake")

    with patch("hm2p.anatomy.register.run_brainreg") as mock_reg:
        mock_reg.return_value = out / "dummy"
        results = batch_register(brains, out)

    assert len(results) == 2
    assert all(r["status"] == "success" for r in results)
    assert mock_reg.call_count == 2


def test_batch_register_error_handling(tmp_path: Path) -> None:
    """Records errors without stopping the batch."""
    brains = tmp_path / "brains"
    brains.mkdir()
    out = tmp_path / "output"

    # One directory has a green file, the other does not.
    good = brains / "good_brain"
    good.mkdir()
    (good / "green.tif").write_text("fake")

    bad = brains / "bad_brain"
    bad.mkdir()
    (bad / "red.tif").write_text("fake")

    with patch("hm2p.anatomy.register.run_brainreg") as mock_reg:
        mock_reg.return_value = out / "dummy"
        results = batch_register(brains, out)

    assert len(results) == 2
    statuses = {r["brain_dir"]: r["status"] for r in results}
    assert statuses[str(bad)] == "error"
    assert statuses[str(good)] == "success"


def test_batch_register_empty_dir(tmp_path: Path) -> None:
    """Returns empty list when brains_dir has no subdirectories."""
    brains = tmp_path / "brains"
    brains.mkdir()
    out = tmp_path / "output"

    results = batch_register(brains, out)
    assert results == []
