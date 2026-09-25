"""Tests for ``scripts/launch_cascade_ec2.py`` (no AWS calls)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import launch_cascade_ec2 as lce  # noqa: E402


def test_user_data_contents() -> None:
    ud = lce.build_user_data(model="M", bucket="bkt", chunk=32, git_branch="dev")
    assert ud.startswith("#!/bin/bash")
    assert "s3://bkt/models/cascade/M.zip" in ud
    assert "--model M" in ud and "--chunk 32" in ud
    assert "--branch dev" in ud
    assert f"s3://bkt/{lce.DONE_KEY}" in ud
    assert "TF_USE_LEGACY_KERAS=1" in ud
    assert "shutdown -h now" in ud
    assert "--overwrite" not in ud
    assert "--overwrite" in lce.build_user_data(overwrite=True)


def test_user_data_creds_block_position() -> None:
    ud = lce.build_user_data(creds_block="echo CREDS\n")
    assert ud.index("echo CREDS") < ud.index("apt-get update")


def test_user_data_is_valid_bash(tmp_path: Path) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    script = tmp_path / "ud.sh"
    script.write_text(lce.build_user_data())
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_parser_defaults() -> None:
    args = lce._build_arg_parser().parse_args([])
    assert args.model == lce.DEFAULT_MODEL and args.chunk == 64
    assert args.instance_type == lce.INSTANCE_TYPE and not args.wait
    args = lce._build_arg_parser().parse_args(["--wait", "--chunk", "16", "--overwrite"])
    assert args.wait and args.chunk == 16 and args.overwrite
