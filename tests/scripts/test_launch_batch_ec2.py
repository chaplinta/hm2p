"""Tests for scripts/launch_batch_ec2.py (no network)."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import launch_batch_ec2 as lb  # noqa: E402


class TestParseJobs:
    def test_lines_chains_and_comments(self) -> None:
        text = """
        # a comment
        a scripts/x.py --n 2
        b scripts/y.py ; scripts/z.py --flag  # trailing comment
        """
        assert lb.parse_jobs(text) == [
            ("a", [["scripts/x.py", "--n", "2"]]),
            ("b", [["scripts/y.py"], ["scripts/z.py", "--flag"]]),
        ]

    def test_missing_command(self) -> None:
        with pytest.raises(ValueError, match="no command"):
            lb.parse_jobs("lonely\n")

    def test_duplicate_name(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            lb.parse_jobs("a x.py\na y.py\n")


class TestPacking:
    def test_round_trip_only_listed_outputs(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "results" / "keep").mkdir(parents=True)
        (src / "results" / "keep" / "a.csv").write_text("1")
        (src / "results" / "other").mkdir()
        (src / "results" / "other" / "b.csv").write_text("2")
        (src / "_logs").mkdir()
        (src / "_logs" / "j.log").write_text("log")
        blob = lb.pack_paths(["results/keep", "results/other", "_logs"], root=src)
        dst = tmp_path / "dst"
        dst.mkdir()
        names = lb.unpack_outputs(blob, ["results/keep"], root=dst)
        assert (dst / "results" / "keep" / "a.csv").read_text() == "1"
        assert (dst / "_logs" / "j.log").exists()
        assert not (dst / "results" / "other").exists()
        assert "results/other/b.csv" not in names

    def test_unpack_skips_traversal(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            data = b"x"
            info = tarfile.TarInfo("results/../../evil.txt")
            info.size = 1
            tar.addfile(info, io.BytesIO(data))
        dst = tmp_path / "d"
        dst.mkdir()
        assert lb.unpack_outputs(buf.getvalue(), ["results"], root=dst) == []
        assert not (tmp_path / "evil.txt").exists()

    def test_pack_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            lb.pack_paths(["nope"], root=tmp_path)


class TestUserData:
    def test_contains_steps(self) -> None:
        ud = lb.build_user_data(
            "qc", ["results/qc", "docs/x"], 3, git_branch="feat/b", max_hours=2
        )
        assert ud.startswith("#!/bin/bash")
        assert "--branch feat/b" in ud
        assert "s3://hm2p-derivatives/batch/qc/inputs.tar.gz" in ud
        assert "xargs -P 3" in ud
        assert "tar -czf /tmp/outputs.tar.gz --ignore-failed-read results/qc docs/x _logs" in ud
        assert "sleep 7200" in ud
        assert "HM2P_S3_CACHE" in ud and "PYTHONPATH=/opt/hm2p/repo/src" in ud
        assert "shutdown -h now" in ud.splitlines()[-1]

    def test_rejects_bad_args(self) -> None:
        with pytest.raises(ValueError):
            lb.build_user_data("x", ["a"], 0)
        with pytest.raises(ValueError):
            lb.build_user_data("x", [], 2)
