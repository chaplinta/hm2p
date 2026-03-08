"""Stage 3b (OPTIONAL) — zero-label behavioural syllable discovery.

Two backends supported:
    keypoint-MoSeq (primary) — AR-HMM; reads DLC .h5 directly. Nat Methods 2024.
    VAME (alternative)       — VAE on pose timeseries; reads movement xarray natively.

Neither backend requires any labelled frames.

This module is not run as part of the core Stages 0–5 pipeline. It is a
post-hoc optional step that appends syllable_id / syllable_prob to kinematics.h5.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def run_keypoint_moseq(
    dlc_files: list[Path],
    project_dir: Path,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """Fit a keypoint-MoSeq AR-HMM model on one or more sessions.

    Reads DLC .h5 files directly (no movement conversion needed).
    Returns syllable IDs per frame for each session.

    Args:
        dlc_files: List of DLC .h5 pose files, one per session.
        project_dir: Directory for keypoint-MoSeq config + model checkpoints.
        output_dir: Where to write syllable outputs.

    Returns:
        Dict mapping session_id → (N,) int16 syllable ID array.
    """
    raise NotImplementedError


def run_vame(
    pose_datasets: list[xr.Dataset],
    session_ids: list[str],
    project_dir: Path,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """Fit a VAME VAE model on movement xarray Datasets.

    VAME v0.7+ natively accepts movement xarray (VAME issue #111).

    Args:
        pose_datasets: List of movement xarray Datasets (output of Stage 3 load).
        session_ids: Matching list of session identifiers.
        project_dir: VAME project directory.
        output_dir: Where to write syllable outputs.

    Returns:
        Dict mapping session_id → (N,) int16 syllable ID array.
    """
    raise NotImplementedError


def append_syllables_to_h5(
    kinematics_h5: Path,
    syllable_ids: np.ndarray,
    syllable_probs: np.ndarray | None = None,
    backend: Literal["keypoint-moseq", "vame"] = "keypoint-moseq",
) -> None:
    """Append syllable arrays to an existing kinematics.h5 file.

    Adds:
        /syllable_id    (N,) int16   — syllable index per camera frame
        /syllable_prob  (N, S) float32 — posterior over S syllables (if provided)

    Args:
        kinematics_h5: Path to existing kinematics.h5 to update.
        syllable_ids: (N,) int16 array of syllable indices.
        syllable_probs: Optional (N, S) float32 posterior probabilities.
        backend: Which tool produced the syllables (stored as HDF5 attribute).

    Raises:
        FileNotFoundError: If kinematics_h5 does not exist.
        ValueError: If syllable_ids length doesn't match existing frame count.
    """
    import h5py

    if not kinematics_h5.exists():
        raise FileNotFoundError(f"kinematics.h5 not found: {kinematics_h5}")

    syllable_ids = np.asarray(syllable_ids, dtype=np.int16)
    if syllable_ids.ndim != 1:
        raise ValueError(f"syllable_ids must be 1D, got shape {syllable_ids.shape}")

    with h5py.File(kinematics_h5, "a") as f:
        # Validate length matches existing data
        if "frame_times" in f:
            n_frames = len(f["frame_times"])
            if len(syllable_ids) != n_frames:
                raise ValueError(
                    f"syllable_ids length ({len(syllable_ids)}) != "
                    f"frame_times length ({n_frames})"
                )

        # Write syllable_id (overwrite if exists)
        if "syllable_id" in f:
            del f["syllable_id"]
        ds = f.create_dataset("syllable_id", data=syllable_ids)
        ds.attrs["backend"] = backend

        # Write syllable_prob if provided
        if syllable_probs is not None:
            syllable_probs = np.asarray(syllable_probs, dtype=np.float32)
            if syllable_probs.shape[0] != len(syllable_ids):
                raise ValueError(
                    f"syllable_probs rows ({syllable_probs.shape[0]}) != "
                    f"syllable_ids length ({len(syllable_ids)})"
                )
            if "syllable_prob" in f:
                del f["syllable_prob"]
            f.create_dataset("syllable_prob", data=syllable_probs)
