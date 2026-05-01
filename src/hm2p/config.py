"""Pipeline configuration — Pydantic settings loaded from config/pipeline.yaml.

Paths, compute profile, per-stage parameters, and tool versions are all
defined here. Settings can be overridden via environment variables prefixed
with HM2P_ (e.g. HM2P_DATA_ROOT=/mnt/data).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_confidence_threshold(v: float | str) -> float | str:
    """Validate a confidence-threshold value.

    Accepts either a float in [0, 1] (fixed scalar threshold) or a string
    of the form ``"quantile:Q"`` with Q in [0, 1] (per-keypoint quantile).
    """
    if isinstance(v, str):
        if not v.startswith("quantile:"):
            raise ValueError(f"string threshold must start with 'quantile:', got {v!r}")
        try:
            q = float(v.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"could not parse quantile from {v!r}: {exc}") from exc
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {q}")
        return v
    if not 0.0 <= float(v) <= 1.0:
        raise ValueError(f"float threshold must be in [0, 1], got {v}")
    return float(v)


class PipelineConfig(BaseSettings):
    """Top-level pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="HM2P_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Compute profile ────────────────────────────────────────────────────
    compute_profile: Literal["local", "local-gpu", "aws-batch"] = Field(
        default="local",
        description="Active Snakemake compute profile.",
    )

    # ── Storage roots ──────────────────────────────────────────────────────
    data_root: Path = Field(
        default=Path("data"),
        description=(
            "Root directory for all data. Subdirectories follow NeuroBlueprint layout "
            "(rawdata/, derivatives/, sourcedata/). "
            "Set to an S3 path (s3://hm2p-rawdata) when running in cloud mode."
        ),
    )

    metadata_dir: Path = Field(
        default=Path("metadata"),
        description="Directory containing animals.csv and experiments.csv.",
    )

    # ── Stage 0 — Ingest ───────────────────────────────────────────────────
    raw_fps_camera: float = Field(default=100.0, description="Nominal camera frame rate (Hz).")
    raw_fps_imaging: float | None = Field(
        default=None,
        description=(
            "Nominal 2P imaging rate (Hz). Set to null (default) to read fps per-session "
            "from timestamps.h5 via fps_from_timestamps(). Falls back to 29.97 Hz if "
            "timestamps.h5 is missing."
        ),
    )

    # ── Stage 3 — Kinematics ───────────────────────────────────────────────
    pose_confidence_threshold: float | str = Field(
        default="quantile:0.25",
        description=(
            "Confidence cutoff for keypoint detections. Either a float in "
            "[0, 1] (fixed scalar threshold) or a string of the form "
            '``"quantile:Q"`` for a per-keypoint quantile threshold. '
            'Default ``"quantile:0.25"`` drops the bottom quartile of each '
            "keypoint's confidence distribution per session — the recommended "
            "filter for DLC 3.x PyTorch outputs whose absolute confidence "
            "values are uncalibrated."
        ),
    )

    @field_validator("pose_confidence_threshold")
    @classmethod
    def _check_pose_confidence_threshold(cls, v: float | str) -> float | str:
        return _validate_confidence_threshold(v)

    pose_gap_fill_frames: int = Field(
        default=5,
        ge=0,
        description="Max consecutive NaN frames to interpolate over.",
    )
    speed_active_threshold_cm_s: float = Field(
        default=2.0,
        description="Speed threshold (cm/s) above which the mouse is considered active.",
    )

    # ── Stage 4 — Calcium ──────────────────────────────────────────────────
    neuropil_coefficient: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fixed neuropil subtraction coefficient: F_corr = F - coeff * Fneu.",
    )
    neuropil_method: Literal["fixed", "fissa"] = Field(
        default="fixed",
        description="Neuropil subtraction method. 'fissa' is more accurate but slower.",
    )
    dff_baseline_window_s: float = Field(
        default=60.0,
        description="Sliding window length (s) for baseline F0 computation.",
    )
    cascade_model: str = Field(
        default="Global_EXC_10Hz_smoothing200ms",
        description=(
            "CASCADE pre-trained model name. Select based on indicator + imaging rate. "
            "hm2p records at ~9.6 Hz; Global_EXC_10Hz_smoothing200ms is the closest "
            "available pre-trained model. "
            "See cascade2p.utils.get_model_folder() for available models. "
            "Rupprecht et al. 2021. doi:10.1038/s41593-021-00895-5"
        ),
    )

    # ── Stage 3b — Behavioural syllables (optional) ────────────────────────
    syllable_backend: Literal["keypoint-moseq", "vame"] = Field(
        default="keypoint-moseq",
        description="Backend for zero-label syllable discovery.",
    )
    kpms_kappa: float = Field(
        default=1e6,
        description="keypoint-MoSeq AR-HMM stickiness parameter (higher = longer syllables).",
    )
    kpms_num_pcs: int = Field(
        default=10,
        ge=3,
        description="Number of PCA components for keypoint-MoSeq.",
    )
    kpms_num_iters: int = Field(
        default=50,
        ge=10,
        description="Number of AR-HMM fitting iterations.",
    )
    kpms_bodyparts: list[str] = Field(
        default=["left_ear", "right_ear", "mid_back", "mouse_center", "tail_base"],
        description="Body parts to use for syllable fitting.",
    )

    # ── Stage 5 — Sync ─────────────────────────────────────────────────────
    sync_interp_method: Literal["linear", "nearest"] = Field(
        default="linear",
        description="Interpolation method for resampling behaviour to imaging rate.",
    )


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load PipelineConfig from a YAML file, with env-var overrides.

    Precedence (highest first):
        1. Environment variables (HM2P_<KEY>)
        2. YAML file values
        3. Pydantic field defaults

    Args:
        path: Path to config YAML. Defaults to config/pipeline.yaml relative
              to the current working directory.

    Returns:
        Validated PipelineConfig instance.
    """
    if path is None:
        path = Path("config/pipeline.yaml")

    if path.exists():
        import os

        import yaml  # defer import — only needed when loading config

        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}

        # Env vars (HM2P_<KEY>) should override YAML values.
        # pydantic-settings gives init kwargs highest priority, so we strip
        # YAML keys that have corresponding env vars set.
        env_prefix = "HM2P_"
        for key in list(yaml_data):
            if f"{env_prefix}{key.upper()}" in os.environ:
                del yaml_data[key]

        return PipelineConfig(**yaml_data)

    return PipelineConfig()
