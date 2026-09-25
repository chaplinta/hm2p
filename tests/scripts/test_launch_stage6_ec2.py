"""Tests for ``scripts/launch_stage6_ec2.py`` (no AWS calls)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import launch_stage6_ec2 as ls6  # noqa: E402


def test_session_ids_from_registry(tmp_path: Path) -> None:
    p = tmp_path / "experiments.csv"
    p.write_text(
        "exp_index,exp_id,fibre\n1,20220101_10_00_00_1001,SFB\n2,20220102_10_00_00_1002,TFB\n"
    )
    assert ls6.session_ids(p) == ["20220101_10_00_00_1001", "20220102_10_00_00_1002"]
    assert len(ls6.session_ids()) >= 26


def test_user_data_contents() -> None:
    ud = ls6.build_user_data(
        ["A_1", "B_2"], parallel=3, n_shuffles=250, git_branch="dev", bucket="bkt"
    )
    assert ud.startswith("#!/bin/bash")
    assert "xargs -P 3" in ud and "--n-shuffles 250" in ud and "--branch dev" in ud
    assert 'printf "%s\\n" A_1 B_2' in ud
    assert f"s3://bkt/{ls6.DONE_KEY}" in ud and f"s3://bkt/{ls6.LOG_PREFIX}/" in ud
    assert "run_stage6_analysis.py" in ud and "--force" in ud
    assert "shutdown -h now" in ud


def test_user_data_creds_block_position() -> None:
    ud = ls6.build_user_data(["A_1"], creds_block="echo CREDS\n")
    assert ud.index("echo CREDS") < ud.index("apt-get update")


def test_user_data_is_valid_bash(tmp_path: Path) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    script = tmp_path / "ud.sh"
    script.write_text(ls6.build_user_data(["A_1", "B_2"]))
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_parser_defaults() -> None:
    args = ls6._build_arg_parser().parse_args([])
    assert args.parallel == 4 and args.n_shuffles == 500 and not args.wait
    args = ls6._build_arg_parser().parse_args(["--wait", "--parallel", "6"])
    assert args.wait and args.parallel == 6
