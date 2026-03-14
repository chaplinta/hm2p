"""Stage 3b (OPTIONAL) — zero-label behavioural syllable discovery.

Two backends supported:
    keypoint-MoSeq (primary) — AR-HMM; reads DLC .h5 directly. Nat Methods 2024.
    VAME (alternative)       — VAE on pose timeseries; reads movement xarray natively.

Neither backend requires any labelled frames.

This module is not run as part of the core Stages 0–5 pipeline. It is a
post-hoc optional step that appends syllable_id / syllable_prob to kinematics.h5.

keypoint-MoSeq runs in an **isolated Docker container** (docker/kpms.Dockerfile)
because it pins numpy<=1.26 and JAX versions that conflict with the main
hm2p environment. The `run_keypoint_moseq()` function orchestrates this
via `docker run` or a subprocess call to scripts/run_kpms.py.

References:
    Weinreb et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point
    tracking to pose dynamics." Nature Methods 21:1329-1339.
    doi:10.1038/s41592-024-02318-2
    https://github.com/dattalab/keypoint-moseq
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

log = logging.getLogger(__name__)

KPMS_DOCKER_IMAGE = "hm2p-kpms"


def run_keypoint_moseq(
    dlc_files: list[Path],
    project_dir: Path,
    output_dir: Path,
    bodyparts: list[str] | None = None,
    kappa: float = 1e6,
    num_pcs: int = 10,
    num_iters: int = 50,
    use_docker: bool = True,
) -> dict[str, np.ndarray]:
    """Fit a keypoint-MoSeq AR-HMM model on one or more sessions.

    Runs in an isolated environment because kpms pins numpy<=1.26.

    When ``use_docker=True`` (default), runs via Docker container
    (hm2p-kpms image). When ``use_docker=False``, calls
    scripts/run_kpms.py directly (requires kpms in current env).

    Args:
        dlc_files: List of DLC .h5 pose files, one per session.
        project_dir: Directory for keypoint-MoSeq config + model checkpoints.
        output_dir: Where to write syllable .npz outputs.
        bodyparts: Body parts to use. Defaults to SuperAnimal TopViewMouse set.
        kappa: AR-HMM stickiness parameter.
        num_pcs: Number of PCA components.
        num_iters: Number of fitting iterations.
        use_docker: Whether to run via Docker (True) or subprocess (False).

    Returns:
        Dict mapping session_id → (N,) int16 syllable ID array.

    Raises:
        RuntimeError: If the kpms process fails.
        FileNotFoundError: If Docker image not found or script missing.
    """
    if bodyparts is None:
        bodyparts = [
            "nose", "left_ear", "right_ear", "neck",
            "mid_back", "mouse_center", "mid_backend", "mid_backend2",
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    # Copy DLC files to a temp directory with clean names
    dlc_dir = Path(tempfile.mkdtemp(prefix="kpms_dlc_"))
    for h5 in dlc_files:
        import shutil
        shutil.copy2(h5, dlc_dir / h5.name)

    if use_docker:
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{dlc_dir}:/data/dlc",
            "-v", f"{project_dir}:/data/project",
            "-v", f"{output_dir}:/data/output",
            KPMS_DOCKER_IMAGE,
            "--dlc-dir", "/data/dlc",
            "--project-dir", "/data/project",
            "--output-dir", "/data/output",
            "--bodyparts", *bodyparts,
            "--kappa", str(kappa),
            "--num-pcs", str(num_pcs),
            "--num-iters", str(num_iters),
        ]
    else:
        # Direct subprocess — requires kpms installed in current env
        script = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "run_kpms.py"
        if not script.exists():
            raise FileNotFoundError(f"run_kpms.py not found at {script}")

        cmd = [
            "python", str(script),
            "--dlc-dir", str(dlc_dir),
            "--project-dir", str(project_dir),
            "--output-dir", str(output_dir),
            "--bodyparts", *bodyparts,
            "--kappa", str(kappa),
            "--num-pcs", str(num_pcs),
            "--num-iters", str(num_iters),
        ]

    log.info("Running kpms: %s", " ".join(str(c) for c in cmd[:8]) + "...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        log.error("kpms failed (exit %d):\n%s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(
            f"keypoint-MoSeq failed with exit code {result.returncode}. "
            f"stderr: {result.stderr[-500:]}"
        )

    if result.stdout:
        log.info("kpms stdout:\n%s", result.stdout[-1000:])

    # Load results from output .npz files
    outputs: dict[str, np.ndarray] = {}
    for npz_path in sorted(output_dir.glob("*_syllables.npz")):
        session_id = npz_path.stem.replace("_syllables", "")
        data = np.load(npz_path)
        outputs[session_id] = data["syllable_id"].astype(np.int16)
        log.info("Loaded syllables for %s: %d frames, %d unique",
                 session_id, len(outputs[session_id]),
                 len(np.unique(outputs[session_id])))

    return outputs


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
