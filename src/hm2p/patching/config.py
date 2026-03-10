"""Configuration and constants for the patching electrophysiology pipeline.

Centralises all path configuration and recording parameters. Mirrors the
directory setup from the MATLAB ``loadPCDirs.m`` but uses Pydantic
BaseSettings and YAML instead of INI files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Recording constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 20_000
"""Acquisition sample rate in Hz."""

SAMPLE_RATE_KHZ: int = 20
"""Acquisition sample rate in kHz (convenience)."""

FILTER_CUTOFF: int = 1000
"""Default low-pass filter cutoff frequency in Hz."""

FILTER_ORDER: int = 4
"""Default Butterworth filter order (poles)."""


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class PatchConfig(BaseSettings):
    """Paths and settings for the patching pipeline.

    Fields mirror the MATLAB ``loadPCDirs.m`` output struct.
    """

    metadata_dir: Path = Field(
        ..., description="Directory containing animals.csv and cells.csv"
    )
    morph_dir: Path = Field(
        ..., description="Root directory for confocal morphology data"
    )
    ephys_dir: Path = Field(
        ..., description="Root directory for WaveSurfer H5 ephys files"
    )
    processed_dir: Path = Field(
        ..., description="Directory for intermediate processed outputs"
    )
    analysis_dir: Path = Field(
        ..., description="Directory for final analysis outputs (metrics, plots)"
    )

    model_config = {"extra": "ignore"}


def load_config(config_path: Path) -> PatchConfig:
    """Load pipeline configuration from a YAML file.

    Parameters
    ----------
    config_path : Path
        Path to a YAML file with keys matching :class:`PatchConfig` fields.

    Returns
    -------
    PatchConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    return PatchConfig(**raw)
