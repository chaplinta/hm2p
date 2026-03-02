"""Tests for config.py — PipelineConfig + YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hm2p.config import PipelineConfig, load_config

# ---------------------------------------------------------------------------
# PipelineConfig defaults
# ---------------------------------------------------------------------------


class TestPipelineConfigDefaults:
    def test_default_compute_profile(self) -> None:
        cfg = PipelineConfig()
        assert cfg.compute_profile == "local"

    def test_default_data_root(self) -> None:
        cfg = PipelineConfig()
        assert cfg.data_root == Path("data")

    def test_default_neuropil_coefficient(self) -> None:
        cfg = PipelineConfig()
        assert cfg.neuropil_coefficient == 0.7

    def test_default_cascade_model(self) -> None:
        cfg = PipelineConfig()
        assert "Global_EXC" in cfg.cascade_model

    def test_override_via_kwargs(self) -> None:
        cfg = PipelineConfig(compute_profile="aws-batch", neuropil_coefficient=0.5)
        assert cfg.compute_profile == "aws-batch"
        assert cfg.neuropil_coefficient == 0.5

    def test_invalid_compute_profile_raises(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(compute_profile="nonexistent")

    def test_neuropil_coefficient_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(neuropil_coefficient=1.5)
        with pytest.raises(ValidationError):
            PipelineConfig(neuropil_coefficient=-0.1)

    def test_pose_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(pose_confidence_threshold=1.5)


# ---------------------------------------------------------------------------
# load_config — YAML loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_defaults_when_no_yaml(self, tmp_path: Path) -> None:
        """When YAML doesn't exist, returns defaults."""
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.compute_profile == "local"
        assert cfg.neuropil_coefficient == 0.7

    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        """YAML values override defaults."""
        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text(
            "compute_profile: local-gpu\n"
            "neuropil_coefficient: 0.5\n"
            "pose_confidence_threshold: 0.8\n"
        )
        cfg = load_config(yaml_path)
        assert cfg.compute_profile == "local-gpu"
        assert cfg.neuropil_coefficient == 0.5
        assert cfg.pose_confidence_threshold == 0.8

    def test_partial_yaml_uses_defaults_for_rest(self, tmp_path: Path) -> None:
        """YAML with only some keys still gets defaults for others."""
        yaml_path = tmp_path / "partial.yaml"
        yaml_path.write_text("neuropil_coefficient: 0.6\n")
        cfg = load_config(yaml_path)
        assert cfg.neuropil_coefficient == 0.6
        assert cfg.compute_profile == "local"  # default
        assert cfg.dff_baseline_window_s == 60.0  # default

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        """Empty YAML file returns all defaults."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        cfg = load_config(yaml_path)
        assert cfg.compute_profile == "local"

    def test_real_pipeline_yaml(self) -> None:
        """The actual config/pipeline.yaml loads successfully."""
        yaml_path = Path("/workspace/config/pipeline.yaml")
        if yaml_path.exists():
            cfg = load_config(yaml_path)
            assert cfg.compute_profile == "local"
            assert cfg.neuropil_coefficient == 0.7
            assert cfg.cascade_model == "Global_EXC_7.5Hz_smoothing200ms"

    def test_yaml_extra_keys_ignored(self, tmp_path: Path) -> None:
        """Unknown keys in YAML are ignored (extra='ignore' in model_config)."""
        yaml_path = tmp_path / "extra.yaml"
        yaml_path.write_text(
            "compute_profile: local\nunknown_key: some_value\nanother_unknown: 42\n"
        )
        cfg = load_config(yaml_path)
        assert cfg.compute_profile == "local"

    def test_default_path_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no path given, defaults to config/pipeline.yaml."""
        monkeypatch.chdir("/workspace")
        cfg = load_config()
        # Should load from config/pipeline.yaml (which exists)
        assert isinstance(cfg, PipelineConfig)

    def test_env_vars_override_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars (HM2P_*) take precedence over YAML values."""
        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text("data_root: /from/yaml\nneuropil_coefficient: 0.3\n")
        monkeypatch.setenv("HM2P_DATA_ROOT", "/from/env")
        cfg = load_config(yaml_path)
        assert str(cfg.data_root) == "/from/env"  # env wins
        assert cfg.neuropil_coefficient == 0.3  # yaml wins (no env var set)

    def test_env_var_without_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env vars work when no YAML file exists."""
        monkeypatch.setenv("HM2P_COMPUTE_PROFILE", "aws-batch")
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.compute_profile == "aws-batch"
