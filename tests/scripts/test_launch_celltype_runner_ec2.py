"""Tests for ``scripts/launch_celltype_runner_ec2.py`` (no AWS calls)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import launch_celltype_runner_ec2 as L  # noqa: E402


def test_user_data_contents() -> None:
    ud = L.build_user_data("mazerun", ["spikes", "dff"], extra_args="--n-perms 100")
    assert ud.startswith("#!/bin/bash")
    assert "run_celltype_programme.py mazerun" in ud and "--n-perms 100" in ud
    assert 'printf "%s\\n" spikes dff | xargs -P 2' in ud
    assert "s3://hm2p-derivatives/celltype_programme/mazerun/_done.json" in ud
    assert ud.rstrip().endswith("shutdown -h now")
    assert f"sleep {L.MAX_HOURS * 3600}" in ud


def test_user_data_is_valid_bash(tmp_path: Path) -> None:
    f = tmp_path / "ud.sh"
    f.write_text(L.build_user_data("mazerun", ["spikes"], creds_block="export X=1"))
    assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0


def test_user_data_rejects_bad_signal() -> None:
    with pytest.raises(ValueError):
        L.build_user_data("mazerun", ["nope"])
    with pytest.raises(ValueError):
        L.build_user_data("mazerun", [])


def test_paths_and_parser() -> None:
    assert L.done_key("x") == "celltype_programme/x/_done.json"
    assert L.local_dir("mazerun", "dff").parts[-2:] == ("celltype_programme_mazerun_ec2", "dff")
    args = L._build_arg_parser().parse_args(["mazerun", "--signals", "spikes", "dff"])
    assert args.runner == "mazerun" and args.signals == ["spikes", "dff"]
