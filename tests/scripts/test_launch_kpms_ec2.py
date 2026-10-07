"""Tests for scripts/launch_kpms_ec2.py user-data (no AWS calls)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import launch_kpms_ec2 as lk  # noqa: E402


def test_default_writes_pipeline_prefix_with_skip_existing():
    ud = lk.build_user_data()
    assert "S3_PREFIX=s3://hm2p-derivatives/kinematics" in ud
    assert "--skip-existing" in ud and "--kappa 1e+06" in ud
    assert "{" + "kappa" not in ud  # all placeholders filled


def test_sweep_prefix_never_touches_pipeline_outputs():
    ud = lk.build_user_data(
        kappa=3e4, num_iters=50, s3_prefix="kpms_sweep/kappa_3e4", branch="feat/x"
    )
    assert "S3_PREFIX=s3://hm2p-derivatives/kpms_sweep/kappa_3e4" in ud
    assert "--s3-prefix kpms_sweep/kappa_3e4" in ud
    assert "--skip-existing" not in ud
    assert "--kappa 30000" in ud and "--num-iters 50" in ud
    assert "git clone --branch feat/x" in ud


def test_state_file_per_tag():
    assert lk._state_file("") == lk.STATE_FILE
    assert lk._state_file("k1").name == ".hm2p-kpms-ec2_k1.json"
