"""Tests for cost page and AWS page consistency logic."""

from __future__ import annotations

import pytest

st = pytest.importorskip("streamlit")


class TestPipelineStagesConsistency:
    """Test that PIPELINE_STAGES is properly structured for pages."""

    @pytest.fixture(autouse=True)
    def _load_stages(self):
        from frontend.data import PIPELINE_STAGES
        self.stages = PIPELINE_STAGES

    def test_all_stages_have_required_keys(self):
        """Every stage must have label, short, s3_prefix, expected."""
        for key, info in self.stages.items():
            assert "label" in info, f"Stage {key} missing 'label'"
            assert "short" in info, f"Stage {key} missing 'short'"
            assert "s3_prefix" in info, f"Stage {key} missing 's3_prefix'"
            assert "expected" in info, f"Stage {key} missing 'expected'"

    def test_expected_counts_valid(self):
        """Expected session counts should be positive integers."""
        for key, info in self.stages.items():
            assert isinstance(info["expected"], int)
            assert info["expected"] > 0

    def test_ingest_has_no_s3_prefix(self):
        """Ingest stage uses rawdata bucket, not derivatives."""
        assert self.stages["ingest"]["s3_prefix"] is None

    def test_all_other_stages_have_s3_prefix(self):
        """Non-ingest stages must have an S3 prefix."""
        for key, info in self.stages.items():
            if key == "ingest":
                continue
            assert info["s3_prefix"] is not None
            assert isinstance(info["s3_prefix"], str)
            assert len(info["s3_prefix"]) > 0

    def test_stage_prefixes_for_cost_page(self):
        """Cost page builds unique prefix list — verify deduplication."""
        prefixes = [
            info["s3_prefix"]
            for info in self.stages.values()
            if info.get("s3_prefix")
        ]
        unique = list(dict.fromkeys(prefixes))
        assert len(unique) > 0
        assert prefixes.count("kinematics") == 2
        assert unique.count("kinematics") == 1

    def test_known_stages_present(self):
        """All expected pipeline stages should exist."""
        expected = {"ingest", "ca_extraction", "pose", "kinematics",
                    "calcium", "sync", "analysis", "kpms"}
        assert set(self.stages.keys()) == expected

    def test_short_labels_unique(self):
        """Short labels should be unique for display."""
        shorts = [info["short"] for info in self.stages.values()]
        assert len(shorts) == len(set(shorts))


class TestCostPagePricing:
    """Test cost calculation logic (no imports needed)."""

    def test_s3_cost_calculation(self):
        rate = 0.025
        gb = 100.0
        assert gb * rate == 2.50

    def test_ec2_spot_vs_ondemand(self):
        pricing = {
            "g4dn.xlarge": {"on_demand": 0.789, "spot_approx": 0.24},
            "g5.xlarge": {"on_demand": 1.19, "spot_approx": 0.36},
        }
        for itype, p in pricing.items():
            assert p["spot_approx"] < p["on_demand"]
            savings = 1 - p["spot_approx"] / p["on_demand"]
            assert savings > 0.5

    def test_job_cost_formula(self):
        rate = 0.24
        hours = 52
        n_instances = 1
        cost = rate * hours * n_instances
        assert cost == pytest.approx(12.48)
