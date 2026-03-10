"""Tests for patching.config — configuration loading and constants."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hm2p.patching.config import (
    FILTER_CUTOFF,
    FILTER_ORDER,
    SAMPLE_RATE,
    SAMPLE_RATE_KHZ,
    PatchConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify recording constants match the MATLAB pipeline."""

    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 20_000

    def test_sample_rate_khz(self) -> None:
        assert SAMPLE_RATE_KHZ == 20

    def test_filter_cutoff(self) -> None:
        assert FILTER_CUTOFF == 1000

    def test_filter_order(self) -> None:
        assert FILTER_ORDER == 4

    def test_sample_rate_consistency(self) -> None:
        assert SAMPLE_RATE == SAMPLE_RATE_KHZ * 1000


# ---------------------------------------------------------------------------
# PatchConfig
# ---------------------------------------------------------------------------


class TestPatchConfig:
    """Verify PatchConfig validation."""

    def test_from_dict(self, tmp_path: Path) -> None:
        cfg = PatchConfig(
            metadata_dir=tmp_path / "meta",
            morph_dir=tmp_path / "morph",
            ephys_dir=tmp_path / "ephys",
            processed_dir=tmp_path / "proc",
            analysis_dir=tmp_path / "analysis",
        )
        assert cfg.metadata_dir == tmp_path / "meta"
        assert cfg.analysis_dir == tmp_path / "analysis"

    def test_extra_fields_ignored(self, tmp_path: Path) -> None:
        """Extra keys in the YAML should not raise."""
        cfg = PatchConfig(
            metadata_dir=tmp_path,
            morph_dir=tmp_path,
            ephys_dir=tmp_path,
            processed_dir=tmp_path,
            analysis_dir=tmp_path,
            unknown_key="should be ignored",
        )
        assert not hasattr(cfg, "unknown_key")

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            PatchConfig(
                metadata_dir=tmp_path,
                # missing morph_dir, etc.
            )


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Verify YAML loading via load_config."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        yaml_data = {
            "metadata_dir": str(tmp_path / "meta"),
            "morph_dir": str(tmp_path / "morph"),
            "ephys_dir": str(tmp_path / "ephys"),
            "processed_dir": str(tmp_path / "proc"),
            "analysis_dir": str(tmp_path / "analysis"),
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(yaml_data))

        cfg = load_config(cfg_path)
        assert cfg.metadata_dir == Path(tmp_path / "meta")
        assert cfg.ephys_dir == Path(tmp_path / "ephys")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_extra_yaml_keys_ignored(self, tmp_path: Path) -> None:
        yaml_data = {
            "metadata_dir": str(tmp_path),
            "morph_dir": str(tmp_path),
            "ephys_dir": str(tmp_path),
            "processed_dir": str(tmp_path),
            "analysis_dir": str(tmp_path),
            "bonus_field": 42,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(yaml_data))
        cfg = load_config(cfg_path)
        assert isinstance(cfg, PatchConfig)
