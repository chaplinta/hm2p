"""Tests for ``scripts/launch_allen_penk_ec2.py`` (no AWS calls)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import launch_allen_penk_ec2 as lae


def test_user_data_contents() -> None:
    ud = lae.build_user_data(parts=["A", "C"], bucket="bkt", git_branch="dev")
    assert ud.startswith("#!/bin/bash")
    assert "--branch dev" in ud
    assert "uv venv -q -p 3.11 venv" in ud
    for dep in ("anndata", "h5py", "pandas", "numpy", "scipy", "boto3", "pyarrow"):
        assert dep in ud
    assert "abc_atlas_access @ git+https://github.com/AllenInstitute/abc_atlas_access.git" in ud
    assert "for part in A C; do" in ud
    assert "[A]=10800" in ud and "[C]=2700" in ud and "[B]=" not in ud
    assert '--s3-uri "s3://bkt/allen/penk"' in ud
    assert f"s3://bkt/{lae.DONE_KEY}" in ud
    assert f"s3://bkt/{lae.CODE_KEY}" in ud
    assert "--write-done /tmp/done.json" in ud
    assert f"sleep {lae.MAX_HOURS * 3600}" in ud
    assert "shutdown -h now" in ud
    assert ud.rstrip().endswith("shutdown -h now")


def test_user_data_without_staged_script() -> None:
    ud = lae.build_user_data(staged_script=False, bucket="bkt")
    assert lae.CODE_KEY not in ud
    assert "using repository script" in ud
    assert "for part in A B C; do" in ud


def test_user_data_creds_block_position() -> None:
    ud = lae.build_user_data(creds_block="echo CREDS\n")
    assert ud.index("echo CREDS") < ud.index("apt-get update")


def test_user_data_rejects_unknown_parts() -> None:
    with pytest.raises(ValueError):
        lae.build_user_data(parts=["D"])
    with pytest.raises(ValueError):
        lae.build_user_data(parts=[])


@pytest.mark.parametrize("staged", [True, False])
def test_user_data_is_valid_bash(tmp_path: Path, staged: bool) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    script = tmp_path / "ud.sh"
    script.write_text(lae.build_user_data(staged_script=staged, creds_block="echo x\n"))
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_part_timeouts_fit_hard_limit() -> None:
    assert set(lae.PART_TIMEOUT_MIN) == set(lae.PARTS)
    assert sum(lae.PART_TIMEOUT_MIN.values()) < lae.MAX_HOURS * 60


def test_constants() -> None:
    assert lae.INSTANCE_TYPE == "r5.2xlarge"
    assert lae.VOLUME_GB == 300
    assert lae.MAX_HOURS == 6
    assert lae.PREFIX == "allen/penk"
    assert lae.LOCAL_SCRIPT.name == "allen_penk_query.py" and lae.LOCAL_SCRIPT.exists()


def test_parser() -> None:
    args = lae._build_arg_parser().parse_args([])
    assert args.parts == ["A", "B", "C"] and not args.wait and not args.dry_run
    assert args.instance_type == lae.INSTANCE_TYPE and not args.no_stage_script
    args = lae._build_arg_parser().parse_args(
        ["--parts", "B", "--wait", "--no-stage-script", "--branch", "x"]
    )
    assert args.parts == ["B"] and args.wait and args.no_stage_script and args.branch == "x"
    for flag in ("--status", "--terminate", "--dry-run"):
        assert getattr(lae._build_arg_parser().parse_args([flag]), flag[2:].replace("-", "_"))
    with pytest.raises(SystemExit):
        lae._build_arg_parser().parse_args(["--parts", "Z"])


@pytest.mark.parametrize("staged", [True, False])
def test_user_data_exports_repo_src_on_pythonpath(staged: bool) -> None:
    ud = lae.build_user_data(staged_script=staged)
    assert lae.REPO_SRC == "/opt/hm2p/repo/src"
    export = f"export PYTHONPATH={lae.REPO_SRC}"
    assert export in ud
    # after the clone, before any part runs
    assert ud.index("git clone") < ud.index(export) < ud.index('timeout "${TMO[$part]}"')
    assert 'python -c "import hm2p.analysis.continuum"' in ud
