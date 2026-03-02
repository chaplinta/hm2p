"""Tests for hm2p.io.aws_cost — cost estimation and confirmation gate."""

from __future__ import annotations

import logging

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hm2p.io.aws_cost import (
    CostEstimate,
    confirm_or_abort,
    estimate_download,
    estimate_upload,
    estimate_upload_from_counts,
)

# ---------------------------------------------------------------------------
# CostEstimate dataclass
# ---------------------------------------------------------------------------


class TestCostEstimate:
    def test_total_cost_is_request_plus_transfer(self):
        est = CostEstimate(
            operation="upload",
            n_files=10,
            total_bytes=1000,
            request_cost_usd=0.01,
            transfer_cost_usd=0.05,
            storage_cost_usd=0.02,
        )
        assert est.total_cost_usd == pytest.approx(0.06)

    def test_summary_contains_operation(self):
        est = CostEstimate(
            operation="download",
            n_files=5,
            total_bytes=2_000_000,
            request_cost_usd=0.001,
            transfer_cost_usd=0.002,
            storage_cost_usd=0.0,
        )
        summary = est.summary()
        assert "download" in summary
        assert "5" in summary

    def test_summary_shows_storage_for_uploads(self):
        est = CostEstimate(
            operation="upload",
            n_files=1,
            total_bytes=1_073_741_824,
            request_cost_usd=0.0,
            transfer_cost_usd=0.0,
            storage_cost_usd=0.025,
        )
        assert "Storage" in est.summary()

    def test_summary_hides_storage_when_zero(self):
        est = CostEstimate(
            operation="download",
            n_files=1,
            total_bytes=100,
            request_cost_usd=0.0,
            transfer_cost_usd=0.0,
            storage_cost_usd=0.0,
        )
        assert "Storage" not in est.summary()


# ---------------------------------------------------------------------------
# estimate_upload_from_counts
# ---------------------------------------------------------------------------


class TestEstimateUploadFromCounts:
    def test_zero_files(self):
        est = estimate_upload_from_counts(0, 0)
        assert est.n_files == 0
        assert est.total_bytes == 0
        assert est.request_cost_usd == 0.0
        assert est.transfer_cost_usd == 0.0
        assert est.storage_cost_usd == 0.0

    def test_request_cost_formula(self):
        # 1000 PUT requests → $0.005
        est = estimate_upload_from_counts(1000, 0)
        assert est.request_cost_usd == pytest.approx(0.005)

    def test_storage_cost_formula(self):
        # 1 GB → $0.025/month
        est = estimate_upload_from_counts(1, 1_073_741_824)
        assert est.storage_cost_usd == pytest.approx(0.025)

    def test_transfer_cost_is_zero(self):
        # Upload ingress is free
        est = estimate_upload_from_counts(100, 10_000_000_000)
        assert est.transfer_cost_usd == 0.0

    def test_operation_is_upload(self):
        est = estimate_upload_from_counts(1, 100)
        assert est.operation == "upload"


# ---------------------------------------------------------------------------
# estimate_upload (with real files)
# ---------------------------------------------------------------------------


class TestEstimateUpload:
    def test_sizes_from_real_files(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"x" * 1000)
        f2.write_bytes(b"y" * 2000)

        est = estimate_upload([f1, f2])
        assert est.n_files == 2
        assert est.total_bytes == 3000
        assert est.operation == "upload"

    def test_empty_list(self):
        est = estimate_upload([])
        assert est.n_files == 0
        assert est.total_bytes == 0


# ---------------------------------------------------------------------------
# estimate_download
# ---------------------------------------------------------------------------


class TestEstimateDownload:
    def test_egress_cost_formula(self):
        # 10 GB → $0.90
        est = estimate_download(n_files=1, total_bytes=10 * 1_073_741_824)
        assert est.transfer_cost_usd == pytest.approx(0.90)

    def test_request_cost_formula(self):
        # 1000 GET requests → $0.0004
        est = estimate_download(n_files=1000, total_bytes=0)
        assert est.request_cost_usd == pytest.approx(0.0004)

    def test_storage_cost_is_zero(self):
        est = estimate_download(n_files=5, total_bytes=5_000_000)
        assert est.storage_cost_usd == 0.0

    def test_operation_is_download(self):
        est = estimate_download(n_files=1, total_bytes=100)
        assert est.operation == "download"


# ---------------------------------------------------------------------------
# confirm_or_abort
# ---------------------------------------------------------------------------


class TestConfirmOrAbort:
    def test_yes_flag_skips_prompt(self, caplog):
        est = estimate_upload_from_counts(10, 1000)
        with caplog.at_level(logging.INFO, logger="hm2p.aws_cost"):
            confirm_or_abort(est, yes=True)
        assert "Proceeding without confirmation" in caplog.text

    def test_y_answer_proceeds(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        est = estimate_upload_from_counts(1, 100)
        confirm_or_abort(est)  # should not raise

    def test_yes_answer_proceeds(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "yes")
        est = estimate_upload_from_counts(1, 100)
        confirm_or_abort(est)  # should not raise

    def test_n_answer_aborts(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        est = estimate_upload_from_counts(1, 100)
        with pytest.raises(SystemExit):
            confirm_or_abort(est)

    def test_empty_answer_aborts(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        est = estimate_upload_from_counts(1, 100)
        with pytest.raises(SystemExit):
            confirm_or_abort(est)

    def test_eof_aborts(self, monkeypatch):
        def raise_eof(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        est = estimate_upload_from_counts(1, 100)
        with pytest.raises(SystemExit):
            confirm_or_abort(est)

    def test_logs_estimate_summary(self, monkeypatch, caplog):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        est = estimate_upload_from_counts(50, 5_000_000)
        with caplog.at_level(logging.INFO, logger="hm2p.aws_cost"):
            confirm_or_abort(est)
        assert "upload" in caplog.text
        assert "50" in caplog.text


# ---------------------------------------------------------------------------
# Hypothesis — cost non-negativity
# ---------------------------------------------------------------------------


@given(
    n_files=st.integers(min_value=0, max_value=10_000),
    total_bytes=st.integers(min_value=0, max_value=10**12),
)
def test_upload_costs_nonnegative(n_files, total_bytes):
    est = estimate_upload_from_counts(n_files, total_bytes)
    assert est.request_cost_usd >= 0
    assert est.transfer_cost_usd >= 0
    assert est.storage_cost_usd >= 0
    assert est.total_cost_usd >= 0


@given(
    n_files=st.integers(min_value=0, max_value=10_000),
    total_bytes=st.integers(min_value=0, max_value=10**12),
)
def test_download_costs_nonnegative(n_files, total_bytes):
    est = estimate_download(n_files, total_bytes)
    assert est.request_cost_usd >= 0
    assert est.transfer_cost_usd >= 0
    assert est.storage_cost_usd >= 0
    assert est.total_cost_usd >= 0
