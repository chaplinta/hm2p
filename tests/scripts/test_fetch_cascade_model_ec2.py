"""Tests for ``scripts/fetch_cascade_model_ec2.py`` (no AWS calls)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import fetch_cascade_model_ec2 as fce  # noqa: E402


@pytest.fixture
def catalogue(tmp_path: Path) -> Path:
    p = tmp_path / "available_models.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "Global_EXC_10Hz_smoothing200ms": {"Link": "https://example.org/a/download"},
                "Other": {"Link": "https://example.org/b/download"},
                "NoLink": {"Info": "x"},
            }
        )
    )
    return p


def test_model_links(catalogue: Path) -> None:
    links = fce.model_links(["Global_EXC_10Hz_smoothing200ms", "Other"], catalogue)
    assert links["Other"] == "https://example.org/b/download"
    with pytest.raises(KeyError):
        fce.model_links(["NoLink"], catalogue)
    with pytest.raises(KeyError):
        fce.model_links(["Missing"], catalogue)


def test_build_user_data_contains_each_model_and_done_marker(catalogue: Path) -> None:
    links = fce.model_links(["Global_EXC_10Hz_smoothing200ms", "Other"], catalogue)
    ud = fce.build_user_data(links, bucket="bkt")
    assert ud.startswith("#!/bin/bash")
    assert 'fetch "Global_EXC_10Hz_smoothing200ms" "https://example.org/a/download"' in ud
    assert 'fetch "Other" "https://example.org/b/download"' in ud
    assert f"s3://bkt/{fce.DONE_KEY}" in ud
    assert "shutdown -h now" in ud
    assert 'STATUS="failed"' in ud
    # every line is dedented (no leading 8-space block indentation left over)
    assert not any(line.startswith("        #!/bin") for line in ud.splitlines())


def test_user_data_is_valid_bash_syntax(catalogue: Path, tmp_path: Path) -> None:
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    links = fce.model_links(["Global_EXC_10Hz_smoothing200ms"], catalogue)
    script = tmp_path / "ud.sh"
    script.write_text(fce.build_user_data(links))
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def test_parser_defaults() -> None:
    args = fce._build_arg_parser().parse_args([])
    assert args.models == fce.DEFAULT_MODELS
    assert args.dry_run is False and args.wait is False
    args = fce._build_arg_parser().parse_args(["--dry-run", "--models", "X", "Y"])
    assert args.models == ["X", "Y"] and args.dry_run is True


def test_constants() -> None:
    assert fce.INSTANCE_TYPE.startswith("t3")
    assert fce.MODELS_PREFIX == "models/cascade"


def test_user_data_creds_block_is_inserted(catalogue: Path) -> None:
    links = fce.model_links(["Other"], catalogue)
    ud = fce.build_user_data(links, creds_block="echo CREDS_HERE\n")
    assert "echo CREDS_HERE" in ud
    assert ud.index("echo CREDS_HERE") < ud.index('fetch "Other"')
