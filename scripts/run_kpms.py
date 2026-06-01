#!/usr/bin/env python3
"""Run keypoint-MoSeq syllable discovery on DLC pose outputs.

Designed to run inside the hm2p-kpms Docker container with an isolated
Python environment (keypoint-MoSeq pins numpy<=1.26).

Can run in two modes:
  1. Local:  --dlc-dir /path/to/pose files
  2. S3:     --s3-bucket hm2p-derivatives --all-sessions

Outputs syllable_id (int16) and syllable_prob (float32) arrays as .npz
files, one per session. These are later appended to kinematics.h5 by
the main pipeline (append_syllables_to_h5).

Reference:
    Weinreb et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point
    tracking to pose dynamics." Nature Methods 21:1329-1339.
    doi:10.1038/s41592-024-02318-2
    https://github.com/dattalab/keypoint-moseq
"""

from __future__ import annotations

# JAX must be configured for 64-bit precision BEFORE any jax/kpms import.
# kpms internally uses float64 but DLC pose data is float32 — JAX raises
# ValueError if x64 mode is not enabled.
import jax

jax.config.update("jax_enable_x64", True)

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# Force JAX to CPU-only (avoids noisy CUDA errors on CPU instances)
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kpms")

# The 8 validated project bodyparts as they appear in DLC SuperAnimal output.
# Note: DLC uses "nose" (mapped to project name "nose_tip" downstream in
# kinematics/compute.py).  "head_midpoint" is a custom-trained keypoint.
DEFAULT_BODYPARTS: list[str] = [
    "nose",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]


# ── S3 helpers ──────────────────────────────────────────────────────────────


def get_s3_client(region: str = "ap-southeast-2"):
    import boto3

    return boto3.client("s3", region_name=region)


def download_s3_file(s3, bucket: str, key: str, local_path: Path) -> bool:
    """Download a file from S3. Returns True on success."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))
        log.info("Downloaded s3://%s/%s → %s", bucket, key, local_path)
        return True
    except Exception:
        log.debug("Not found: s3://%s/%s", bucket, key)
        return False


def convert_madlc_to_single(h5_path: Path, bodyparts: list[str]) -> Path:
    """Convert multi-animal DLC .h5 to single-animal format for kpms.

    SuperAnimal TopViewMouse + FasterRCNN produces maDLC output with
    4-level columns (scorer/individuals/bodyparts/coords) and multiple
    detected "animals". We pick the best individual per frame (highest
    mean likelihood across target bodyparts) and output standard DLC
    3-level columns (scorer/bodyparts/coords).

    Returns path to the converted file (same directory, _single.h5 suffix).
    """
    import pandas as pd

    df = pd.read_hdf(h5_path)

    # Check if already single-animal format (3 levels)
    if df.columns.nlevels == 3:
        log.info("  Already single-animal format: %s", h5_path.name)
        return h5_path

    if df.columns.nlevels != 4:
        raise ValueError(f"Expected 3 or 4 column levels, got {df.columns.nlevels}")

    scorer = df.columns.get_level_values("scorer")[0]
    individuals = df.columns.get_level_values("individuals").unique().tolist()
    available_bps = df.columns.get_level_values("bodyparts").unique().tolist()

    # Filter to requested bodyparts that exist in the data
    use_bps = [bp for bp in bodyparts if bp in available_bps]
    if not use_bps:
        raise ValueError(
            f"None of the requested bodyparts {bodyparts} found in file. "
            f"Available: {available_bps}"
        )
    log.info("  Using %d/%d bodyparts: %s", len(use_bps), len(available_bps), use_bps)

    n_frames = len(df)

    # For each frame, pick the individual with highest mean likelihood
    # across the target bodyparts (vectorized)
    log.info(
        "  Selecting best individual per frame (%d frames, %d individuals)...",
        n_frames,
        len(individuals),
    )

    # Build (n_frames, n_individuals) likelihood matrix
    ind_scores = np.full((n_frames, len(individuals)), -1.0)
    for j, ind in enumerate(individuals):
        lk_cols = []
        for bp in use_bps:
            if (scorer, ind, bp, "likelihood") in df.columns:
                lk_cols.append(df[(scorer, ind, bp, "likelihood")].values)
        if lk_cols:
            # Mean likelihood across bodyparts per frame
            ind_scores[:, j] = np.nanmean(np.column_stack(lk_cols), axis=1)

    best_ind_idx = np.argmax(ind_scores, axis=1)  # (n_frames,)

    # Build single-animal dataframe by gathering from best individual per frame
    new_columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, coord) for bp in use_bps for coord in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    new_data = np.empty((n_frames, len(new_columns)), dtype=np.float64)

    col_idx = 0
    for bp in use_bps:
        for coord in ("x", "y", "likelihood"):
            # Stack all individuals' values for this bp+coord: (n_frames, n_individuals)
            all_vals = np.full((n_frames, len(individuals)), np.nan)
            for j, ind in enumerate(individuals):
                if (scorer, ind, bp, coord) in df.columns:
                    all_vals[:, j] = df[(scorer, ind, bp, coord)].values
            # Gather from best individual per frame
            new_data[:, col_idx] = all_vals[np.arange(n_frames), best_ind_idx]
            col_idx += 1

    new_df = pd.DataFrame(new_data, index=df.index, columns=new_columns)

    out_path = h5_path.with_name(h5_path.stem + "_single.h5")
    new_df.to_hdf(out_path, key="df_with_missing", mode="w")
    log.info("  Converted maDLC → single: %s (%d frames)", out_path.name, n_frames)

    return out_path


def upload_s3_file(s3, local_path: Path, bucket: str, key: str):
    """Upload a file to S3."""
    s3.upload_file(str(local_path), bucket, key)
    log.info("Uploaded %s → s3://%s/%s", local_path, bucket, key)


def parse_session_id(exp_id: str) -> tuple[str, str]:
    """Convert exp_id to (sub, ses) NeuroBlueprint names."""
    parts = exp_id.split("_")
    animal = parts[-1]
    sub = f"sub-{animal}"
    ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
    return sub, ses


def get_dlc_champion_id(s3, bucket: str) -> str | None:
    """Read the current DLC champion model ID from S3.

    Returns None if the champion file doesn't exist.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            s3.download_file(bucket, "dlc-champion.json", tmp.name)
            with open(tmp.name) as f:
                data = json.load(f)
            Path(tmp.name).unlink(missing_ok=True)
            return data.get("champion_id", data.get("model_id"))
    except Exception:
        log.warning("Could not read dlc-champion.json from S3")
        return None


# ── Kappa sweep ─────────────────────────────────────────────────────────────


def run_kappa_sweep(
    data: dict,
    metadata: dict,
    pca: dict,
    project_dir: Path,
    kappa_values: list[float],
    sweep_iters: int = 25,
    imaging_fps: float = 30.0,
) -> dict:
    """Test multiple kappa values with short fits and return sweep results.

    For each kappa, fits the model for ``sweep_iters`` iterations and
    computes summary statistics. The recommended kappa is the one whose
    median syllable duration is closest to 400 ms (kpms recommended
    range is 300-500 ms).

    Parameters
    ----------
    data : dict
        Formatted kpms data dict from ``kpms.format_data``.
    metadata : dict
        Metadata dict from ``kpms.format_data``.
    pca : dict
        Fitted PCA dict.
    project_dir : Path
        kpms project directory.
    kappa_values : list of float
        Kappa values to test.
    sweep_iters : int
        Number of fitting iterations per kappa (default 25).
    imaging_fps : float
        Frame rate for converting bout duration to seconds (default 30).

    Returns
    -------
    dict
        Sweep results with keys: ``"kappa_values"``, ``"results"`` (list of
        per-kappa dicts), ``"best_kappa"``, ``"best_idx"``.
    """
    import keypoint_moseq as kpms

    target_duration_s = 0.400  # 400 ms target
    sweep_results: list[dict] = []

    for kappa in kappa_values:
        log.info("  Sweep: kappa=%.0e (fitting %d iters)...", kappa, sweep_iters)

        kpms.update_config(str(project_dir), kappa=kappa)
        cfg = kpms.load_config(str(project_dir))

        model = kpms.init_model(data, pca=pca, **cfg)
        fit_result = kpms.fit_model(
            model=model,
            data=data,
            metadata=metadata,
            project_dir=str(project_dir),
            model_name=f"sweep_k{kappa:.0e}",
            num_iters=sweep_iters,
        )

        if isinstance(fit_result, tuple):
            model_out, model_name = fit_result
        else:
            model_out = fit_result
            model_name = f"sweep_k{kappa:.0e}"

        results = kpms.extract_results(
            model_out,
            metadata,
            str(project_dir),
            model_name=model_name,
        )

        # Compute summary stats across all sessions
        all_syllables = []
        all_bout_durations = []
        for _session_id, session_res in results.items():
            syl = np.array(session_res["syllable"])
            all_syllables.extend(syl.tolist())

            # Compute bout durations: consecutive runs of same syllable
            if len(syl) > 0:
                changes = np.where(np.diff(syl) != 0)[0] + 1
                boundaries = np.concatenate([[0], changes, [len(syl)]])
                bout_lengths = np.diff(boundaries)
                all_bout_durations.extend(bout_lengths.tolist())

        all_syllables = np.array(all_syllables)
        unique_syllables, counts = np.unique(all_syllables, return_counts=True)
        total_frames = len(all_syllables)

        # Effective syllables: those with >0.5% occupancy
        if total_frames > 0:
            occupancy = counts / total_frames
            n_effective = int(np.sum(occupancy > 0.005))
        else:
            n_effective = 0

        # Entropy of syllable distribution
        if total_frames > 0:
            probs = counts / total_frames
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        else:
            entropy = 0.0

        # Median bout duration
        if all_bout_durations:
            median_bout_frames = float(np.median(all_bout_durations))
            median_bout_s = median_bout_frames / imaging_fps
        else:
            median_bout_frames = 0.0
            median_bout_s = 0.0

        result_entry = {
            "kappa": kappa,
            "n_unique_syllables": len(unique_syllables),
            "n_effective_syllables": n_effective,
            "entropy": entropy,
            "median_bout_frames": median_bout_frames,
            "median_bout_duration_s": median_bout_s,
        }
        sweep_results.append(result_entry)
        log.info(
            "    kappa=%.0e: %d effective syllables, median bout=%.3fs, entropy=%.2f",
            kappa,
            n_effective,
            median_bout_s,
            entropy,
        )

    # Select best kappa: closest median duration to 400ms
    best_idx = 0
    best_distance = float("inf")
    for i, r in enumerate(sweep_results):
        distance = abs(r["median_bout_duration_s"] - target_duration_s)
        if distance < best_distance:
            best_distance = distance
            best_idx = i

    best_kappa = sweep_results[best_idx]["kappa"]
    log.info(
        "Kappa sweep complete. Best kappa=%.0e (median bout=%.3fs, target=%.3fs)",
        best_kappa,
        sweep_results[best_idx]["median_bout_duration_s"],
        target_duration_s,
    )

    return {
        "kappa_values": [float(k) for k in kappa_values],
        "results": sweep_results,
        "best_kappa": best_kappa,
        "best_idx": best_idx,
        "target_duration_s": target_duration_s,
    }


# ── keypoint-MoSeq wrapper ─────────────────────────────────────────────────


def fit_kpms(
    dlc_files: dict[str, Path],
    project_dir: Path,
    bodyparts: list[str],
    kappa: float | None = None,
    num_pcs: int = 10,
    num_iters: int = 100,
    kappa_sweep: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Fit keypoint-MoSeq AR-HMM on DLC .h5 files.

    Parameters
    ----------
    dlc_files : dict[str, Path]
        Dict of session_id to DLC .h5 file path.
    project_dir : Path
        Working directory for kpms config/checkpoints.
    bodyparts : list[str]
        List of body part names to use for fitting.
    kappa : float or None
        AR-HMM stickiness (higher = longer syllables). If None and
        kappa_sweep is False, defaults to 1e6.
    num_pcs : int
        Number of PCA components.
    num_iters : int
        Number of fitting iterations.
    kappa_sweep : bool
        If True, run a kappa sweep first to select the value.

    Returns
    -------
    tuple[dict, dict]
        (results_dict, fit_info) where results_dict maps session_id to
        {"syllable_id": (N,) int16, "syllable_prob": (N, S) float32},
        and fit_info contains fitting metadata (kappa used, PCA variance, etc.).
    """
    import shutil

    import keypoint_moseq as kpms

    # Clean project dir contents to avoid "directory already exists" error
    # from kpms.  We clear contents rather than rmtree because the dir may be
    # a Docker bind-mount (rmtree on a mount point raises EBUSY).
    if project_dir.exists():
        for child in project_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    project_dir.mkdir(parents=True, exist_ok=True)

    # ── Load DLC data first to discover all bodyparts ────────────────────
    # We must load data BEFORE setup_project so we know the full bodypart
    # list.  kpms.format_data asserts len(bodyparts) == coordinates.shape[K],
    # so config `bodyparts` must list ALL keypoints from the file, while
    # `use_bodyparts` selects our desired subset.
    import tempfile as _tempfile

    link_dir = Path(_tempfile.mkdtemp(prefix="kpms_links_"))
    for sid, h5_path in dlc_files.items():
        # kpms uses the filename (minus extension) as session key
        link = link_dir / f"{sid}.h5"
        link.symlink_to(h5_path.resolve())

    log.info("Loading %d DLC files via load_keypoints...", len(dlc_files))
    coordinates, confidences, all_bodyparts = kpms.load_keypoints(
        str(link_dir),
        "deeplabcut",
    )
    log.info("Loaded %d bodyparts from DLC files: %s", len(all_bodyparts), all_bodyparts)
    log.info("Sessions loaded: %s", list(coordinates.keys()))

    # Validate that all requested bodyparts exist in the loaded data
    missing_bps = [bp for bp in bodyparts if bp not in all_bodyparts]
    if missing_bps:
        raise ValueError(
            f"Requested bodyparts not found in DLC files: {missing_bps}. "
            f"Available: {all_bodyparts}"
        )
    log.info(
        "Using %d/%d bodyparts for fitting: %s", len(bodyparts), len(all_bodyparts), bodyparts
    )

    # ── Setup project with the FULL bodypart list ─────────────────────────
    # bodyparts = all keypoints in the data (must match coordinates dim K)
    # use_bodyparts = our selected subset for AR-HMM fitting
    log.info("Setting up kpms project...")
    kpms.setup_project(
        project_dir=str(project_dir),
        deeplabcut_config=None,
        bodyparts=all_bodyparts,
        use_bodyparts=bodyparts,
        overwrite=True,
    )

    # Patch config.yml: setup_project writes placeholder BODYPART1/2/3 in
    # skeleton and anterior/posterior that cause load_config to crash.
    kpms.update_config(
        str(project_dir),
        anterior_bodyparts=[bodyparts[0]],  # e.g. "nose"
        posterior_bodyparts=[bodyparts[-1]],  # e.g. "tail_base"
        bodyparts=all_bodyparts,
        use_bodyparts=bodyparts,
        skeleton=[],
    )

    # Helper to load config as a dict
    def config():
        return kpms.load_config(str(project_dir))

    # ── Format data ──────────────────────────────────────────────────────────
    log.info(
        "Formatting data (all_bodyparts=%d, use_bodyparts=%d)...",
        len(all_bodyparts),
        len(bodyparts),
    )
    cfg = config()
    log.info("Config bodyparts: %s", cfg.get("bodyparts"))
    log.info("Config use_bodyparts: %s", cfg.get("use_bodyparts"))
    log.info("anterior_bodyparts: %s", cfg.get("anterior_bodyparts"))
    log.info("posterior_bodyparts: %s", cfg.get("posterior_bodyparts"))
    data, metadata = kpms.format_data(coordinates, confidences, **cfg)
    data_keys = list(data.keys()) if isinstance(data, dict) else "N/A"
    log.info("data type: %s, keys: %s", type(data).__name__, data_keys)

    # Cast data arrays to float64 — kpms/JAX requires x64 precision but
    # DLC pose data and format_data output are float32.  Use the library's
    # own converter which handles nested dicts, JAX arrays, and numpy arrays.
    from jax_moseq.utils.debugging import convert_data_precision

    data = convert_data_precision(data)
    log.info("Converted data to 64-bit precision via convert_data_precision")

    # noise_calibration is interactive (requires video frames for a Jupyter
    # widget) — skip it on headless EC2.  The default noise prior works fine
    # for DLC data with confidence scores.

    # ── PCA ────────────────────────────────────────────────────────────────
    log.info("Fitting PCA (num_pcs=%d)...", num_pcs)
    kpms.update_config(str(project_dir), num_pcs=num_pcs)
    cfg = config()
    log.info(
        "anterior_idxs: %s, posterior_idxs: %s",
        cfg.get("anterior_idxs"),
        cfg.get("posterior_idxs"),
    )

    pca = kpms.fit_pca(
        data["Y"],
        data["mask"],
        anterior_idxs=cfg.get("anterior_idxs"),
        posterior_idxs=cfg.get("posterior_idxs"),
        conf=data.get("conf"),
        PCA_fitting_num_frames=cfg.get("PCA_fitting_num_frames", 1000000),
    )

    # Convert numeric arrays in pca to float64.  pca is a dict that may
    # contain sklearn PCA objects (non-numeric) — convert only array leaves.
    if isinstance(pca, dict):
        for k, v in pca.items():
            is_float = hasattr(v, "dtype") and np.issubdtype(v.dtype, np.floating)
            if is_float and v.dtype != np.float64:
                pca[k] = np.asarray(v, dtype=np.float64)
                log.info("  Cast pca['%s'] %s → float64", k, v.dtype)
    log.info("PCA precision check complete")

    # Save PCA explained variance
    pca_variance = {}
    if isinstance(pca, dict):
        for k, v in pca.items():
            if isinstance(v, np.ndarray) and "variance" in k.lower():
                pca_variance[k] = v.tolist()
            elif hasattr(v, "explained_variance_ratio_"):
                # sklearn PCA object
                pca_variance["explained_variance_ratio"] = v.explained_variance_ratio_.tolist()
    log.info("PCA variance keys: %s", list(pca_variance.keys()))

    # ── Kappa sweep (optional) ────────────────────────────────────────────
    sweep_results = None
    if kappa_sweep:
        log.info("Running kappa sweep...")
        kappa_values = [1e3, 1e4, 1e5, 1e6]
        sweep_results = run_kappa_sweep(
            data=data,
            metadata=metadata,
            pca=pca,
            project_dir=project_dir,
            kappa_values=kappa_values,
            sweep_iters=25,
        )
        kappa = sweep_results["best_kappa"]
        log.info("Selected kappa from sweep: %.0e", kappa)
    elif kappa is None:
        kappa = 1e6
        log.info("Using default kappa=%.0e", kappa)

    # ── AR-HMM fitting ─────────────────────────────────────────────────────
    log.info("Fitting AR-HMM (kappa=%.0e, n_iters=%d)...", kappa, num_iters)
    kpms.update_config(str(project_dir), kappa=kappa)

    cfg = config()
    model = kpms.init_model(data, pca=pca, **cfg)

    model_name = "hm2p_kpms"
    # fit_model returns (model_dict, model_name_str) tuple
    fit_result = kpms.fit_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=str(project_dir),
        model_name=model_name,
        num_iters=num_iters,
    )
    # Unpack: fit_model returns (model, model_name) tuple
    if isinstance(fit_result, tuple):
        model, model_name = fit_result
    else:
        model = fit_result

    # ── Capture ELBO/convergence trace if available ────────────────────────
    elbo_trace = None
    if isinstance(model, dict):
        # kpms stores log-likelihood history in some versions
        for key in ("ll_history", "elbo_history", "log_likelihood"):
            if key in model:
                elbo_trace = np.array(model[key]).tolist()
                log.info(
                    "Captured convergence trace from model['%s'] (%d values)",
                    key,
                    len(elbo_trace),
                )
                break

    # Check for saved checkpoint files with ELBO traces
    checkpoint_dir = project_dir / model_name
    if elbo_trace is None and checkpoint_dir.exists():
        for fname in ("history.json", "elbo.json", "log_likelihood.json"):
            fpath = checkpoint_dir / fname
            if fpath.exists():
                try:
                    with open(fpath) as f:
                        elbo_trace = json.load(f)
                    log.info("Loaded convergence trace from %s", fpath)
                    break
                except Exception:
                    pass

    # ── Extract results ────────────────────────────────────────────────────
    log.info("Extracting results for model_name=%s...", model_name)
    log.info(
        "model type: %s, keys: %s",
        type(model).__name__,
        list(model.keys()) if isinstance(model, dict) else "N/A",
    )
    results = kpms.extract_results(
        model,
        metadata,
        str(project_dir),
        model_name=model_name,
    )

    # Clean up symlinks
    shutil.rmtree(link_dir, ignore_errors=True)

    # Build output dict
    output = {}
    for session_id in dlc_files:
        if session_id in results:
            syllable_id = np.array(results[session_id]["syllable"], dtype=np.int16)
            # Get posterior probabilities if available
            if "syllable_probability" in results[session_id]:
                syllable_prob = np.array(
                    results[session_id]["syllable_probability"], dtype=np.float32
                )
            else:
                syllable_prob = None

            output[session_id] = {
                "syllable_id": syllable_id,
                "syllable_prob": syllable_prob,
            }
            log.info(
                "  %s: %d frames, %d unique syllables",
                session_id,
                len(syllable_id),
                len(np.unique(syllable_id)),
            )
        else:
            log.warning("  %s: not in results (skipped by kpms?)", session_id)

    # Build fit_info metadata
    fit_info = {
        "kappa": kappa,
        "num_pcs": num_pcs,
        "num_iters": num_iters,
        "bodyparts": bodyparts,
        "all_bodyparts": list(all_bodyparts),
        "pca_variance": pca_variance,
        "sweep_results": sweep_results,
        "elbo_trace": elbo_trace,
        "fit_timestamp": datetime.now(UTC).isoformat(),
    }

    return output, fit_info


# ── Provenance ──────────────────────────────────────────────────────────────


def build_provenance(
    session_id: str,
    syllable_id: np.ndarray,
    fit_info: dict,
    dlc_champion_id: str | None,
) -> dict:
    """Build provenance metadata for a session's syllable output.

    Parameters
    ----------
    session_id : str
        Session exp_id.
    syllable_id : ndarray
        (N,) syllable ID array for this session.
    fit_info : dict
        Fitting metadata from ``fit_kpms``.
    dlc_champion_id : str or None
        DLC champion model ID (from dlc-champion.json).

    Returns
    -------
    dict
        Provenance metadata suitable for JSON serialization.
    """
    try:
        import keypoint_moseq

        kpms_version = keypoint_moseq.__version__
    except Exception:
        kpms_version = "unknown"

    unique_syllables = np.unique(syllable_id)

    # Compute median bout duration
    if len(syllable_id) > 0:
        changes = np.where(np.diff(syllable_id) != 0)[0] + 1
        boundaries = np.concatenate([[0], changes, [len(syllable_id)]])
        bout_lengths = np.diff(boundaries)
        # Assume 30 fps for DLC output (subsampled from ~100 fps)
        median_bout_s = float(np.median(bout_lengths) / 30.0)
    else:
        median_bout_s = 0.0

    return {
        "dlc_champion_id": dlc_champion_id,
        "kpms_version": kpms_version,
        "kappa": fit_info["kappa"],
        "num_pcs": fit_info["num_pcs"],
        "num_iters": fit_info["num_iters"],
        "bodyparts": fit_info["bodyparts"],
        "fit_timestamp": fit_info["fit_timestamp"],
        "n_unique_syllables": len(unique_syllables),
        "median_bout_duration_s": median_bout_s,
    }


def provenance_matches(existing: dict, current: dict) -> bool:
    """Check if existing provenance matches current run parameters.

    Used by --skip-existing to determine whether a session's output
    is still valid.

    Parameters
    ----------
    existing : dict
        Provenance from an existing syllables.provenance.json.
    current : dict
        Provenance for the current run.

    Returns
    -------
    bool
        True if key parameters match (same DLC champion, kappa, bodyparts, num_iters).
    """
    keys_to_compare = ["dlc_champion_id", "kappa", "bodyparts", "num_pcs", "num_iters"]
    return all(existing.get(key) == current.get(key) for key in keys_to_compare)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run keypoint-MoSeq syllable discovery on DLC outputs."
    )
    parser.add_argument(
        "--dlc-dir",
        type=Path,
        default=None,
        help="Local directory containing DLC .h5 files.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("/tmp/kpms_project"),
        help="Working directory for kpms config/checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory to write syllable .npz files.",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="hm2p-derivatives",
        help="S3 bucket for derivatives.",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Process all sessions from metadata/experiments.csv via S3.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Specific session exp_ids to process.",
    )
    parser.add_argument(
        "--bodyparts",
        nargs="*",
        default=DEFAULT_BODYPARTS,
        help="Body parts to use for fitting (kpms recommends 5-10, no tail tip).",
    )
    parser.add_argument(
        "--kappa", type=float, default=None, help="AR-HMM kappa (stickiness). Overrides sweep."
    )
    parser.add_argument(
        "--kappa-sweep",
        action="store_true",
        help="Run kappa sweep (1e3, 1e4, 1e5, 1e6) with 25 iters each, "
        "then full fit with best kappa. Ignored if --kappa is set.",
    )
    parser.add_argument("--num-pcs", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip per-session upload for sessions whose existing output "
        "has matching provenance. All sessions are still included in "
        "the joint model fit.",
    )

    args = parser.parse_args()

    # If --kappa is explicitly set, disable sweep
    use_sweep = args.kappa_sweep and args.kappa is None
    if args.kappa_sweep and args.kappa is not None:
        log.info("--kappa explicitly set to %.0e, ignoring --kappa-sweep", args.kappa)

    # ── Determine which sessions to process ─────────────────────────────────

    s3 = None
    using_s3 = args.all_sessions or args.sessions

    if args.dlc_dir:
        # Local mode: find DLC .h5 files
        dlc_files = {}
        for h5 in sorted(args.dlc_dir.glob("**/*DLC*.h5")):
            # Infer session_id from directory structure
            session_id = h5.stem.split("DLC")[0].rstrip("_")
            dlc_files[session_id] = h5
        log.info("Found %d DLC files in %s", len(dlc_files), args.dlc_dir)

    elif using_s3:
        # S3 mode: download ALL DLC outputs for joint model fitting.
        # --skip-existing only affects per-session upload, NOT download.
        s3 = get_s3_client()

        # Load experiments
        metadata_dir = Path("metadata")
        if not metadata_dir.exists():
            metadata_dir = Path("/app/metadata")
        csv_path = metadata_dir / "experiments.csv"
        with open(csv_path) as f:
            experiments = list(csv.DictReader(f))

        if args.sessions:
            experiments = [e for e in experiments if e["exp_id"] in args.sessions]

        tmpdir = Path(tempfile.mkdtemp(prefix="kpms_dlc_"))
        dlc_files = {}

        for exp in experiments:
            exp_id = exp["exp_id"]
            sub, ses = parse_session_id(exp_id)

            # Download DLC .h5 from pose/ — always download ALL sessions
            # because kpms fits ONE joint model across all sessions
            pose_prefix = f"pose/{sub}/{ses}/"
            try:
                resp = s3.list_objects_v2(
                    Bucket=args.s3_bucket,
                    Prefix=pose_prefix,
                )
                h5_keys = [
                    obj["Key"]
                    for obj in resp.get("Contents", [])
                    if obj["Key"].endswith(".h5") and not obj["Key"].endswith("_single.h5")
                ]
            except Exception:
                log.warning("No pose data for %s", exp_id)
                continue

            if not h5_keys:
                log.warning("No DLC .h5 found for %s at %s", exp_id, pose_prefix)
                continue

            # Download first matching DLC file
            local_h5 = tmpdir / f"{exp_id}.h5"
            if download_s3_file(s3, args.s3_bucket, h5_keys[0], local_h5):
                # Convert multi-animal DLC to single-animal format
                try:
                    converted = convert_madlc_to_single(local_h5, args.bodyparts)
                    dlc_files[exp_id] = converted
                except Exception as e:
                    log.error("Failed to convert %s: %s", exp_id, e)

        log.info("Downloaded and converted %d DLC files from S3", len(dlc_files))

    else:
        parser.error("Provide --dlc-dir, --all-sessions, or --sessions")
        return

    if not dlc_files:
        log.error("No DLC files to process. Exiting.")
        sys.exit(1)

    # ── Run keypoint-MoSeq ──────────────────────────────────────────────────

    log.info("Starting keypoint-MoSeq fitting on %d sessions...", len(dlc_files))

    results, fit_info = fit_kpms(
        dlc_files=dlc_files,
        project_dir=args.project_dir,
        bodyparts=args.bodyparts,
        kappa=args.kappa,
        num_pcs=args.num_pcs,
        num_iters=args.num_iters,
        kappa_sweep=use_sweep,
    )

    log.info("Fitting complete. %d sessions have results.", len(results))

    # ── Read DLC champion ID for provenance ──────────────────────────────
    dlc_champion_id = None
    if using_s3:
        if s3 is None:
            s3 = get_s3_client()
        dlc_champion_id = get_dlc_champion_id(s3, args.s3_bucket)
        log.info("DLC champion ID: %s", dlc_champion_id)

    # ── Determine which sessions to skip upload for ──────────────────────
    skip_upload_sessions: set[str] = set()
    if args.skip_existing and using_s3:
        if s3 is None:
            s3 = get_s3_client()
        for session_id, data in results.items():
            sub, ses = parse_session_id(session_id)
            prov_key = f"kinematics/{sub}/{ses}/syllables.provenance.json"
            try:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    s3.download_file(args.s3_bucket, prov_key, tmp.name)
                    with open(tmp.name) as f:
                        existing_prov = json.load(f)
                    Path(tmp.name).unlink(missing_ok=True)

                # Build current provenance to compare
                current_prov = build_provenance(
                    session_id, data["syllable_id"], fit_info, dlc_champion_id
                )
                if provenance_matches(existing_prov, current_prov):
                    log.info("Skipping upload for %s (provenance matches)", session_id)
                    skip_upload_sessions.add(session_id)
                else:
                    log.info("Will re-upload %s (provenance mismatch)", session_id)
            except Exception:
                # No existing provenance — upload
                pass

    # ── Save outputs ────────────────────────────────────────────────────────

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for session_id, data in results.items():
        npz_data = {"syllable_id": data["syllable_id"]}
        if data.get("syllable_prob") is not None:
            npz_data["syllable_prob"] = data["syllable_prob"]

        # Build provenance
        provenance = build_provenance(session_id, data["syllable_id"], fit_info, dlc_champion_id)

        if args.output_dir:
            # Save locally
            out_path = args.output_dir / f"{session_id}_syllables.npz"
            np.savez_compressed(out_path, **npz_data)
            log.info("Saved %s", out_path)

            # Save provenance locally
            prov_path = args.output_dir / f"{session_id}_syllables.provenance.json"
            with open(prov_path, "w") as f:
                json.dump(provenance, f, indent=2)

        if using_s3:
            if session_id in skip_upload_sessions:
                continue

            if s3 is None:
                s3 = get_s3_client()
            sub, ses = parse_session_id(session_id)

            # Upload syllables.npz
            s3_key = f"kinematics/{sub}/{ses}/syllables.npz"
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                np.savez_compressed(tmp.name, **npz_data)
                upload_s3_file(s3, Path(tmp.name), args.s3_bucket, s3_key)

            # Upload provenance
            prov_key = f"kinematics/{sub}/{ses}/syllables.provenance.json"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(provenance, tmp, indent=2)
                tmp.flush()
                upload_s3_file(s3, Path(tmp.name), args.s3_bucket, prov_key)

    # ── Upload model artifacts to S3 ───────────────────────────────────────

    if using_s3:
        if s3 is None:
            s3 = get_s3_client()
        model_s3_prefix = "kinematics/kpms_model"
        log.info("Uploading model artifacts to s3://%s/%s/...", args.s3_bucket, model_s3_prefix)

        # Upload results.h5 if it exists
        results_h5 = args.project_dir / "hm2p_kpms" / "results.h5"
        if results_h5.exists():
            upload_s3_file(s3, results_h5, args.s3_bucket, f"{model_s3_prefix}/results.h5")
        else:
            # Search for any results file in the project dir
            for rh5 in args.project_dir.rglob("results.h5"):
                upload_s3_file(s3, rh5, args.s3_bucket, f"{model_s3_prefix}/results.h5")
                break

        # Upload config.yml
        config_yml = args.project_dir / "config.yml"
        if config_yml.exists():
            upload_s3_file(s3, config_yml, args.s3_bucket, f"{model_s3_prefix}/config.yml")

        # Upload PCA explained variance
        if fit_info.get("pca_variance"):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(fit_info["pca_variance"], tmp, indent=2)
                tmp.flush()
                upload_s3_file(
                    s3,
                    Path(tmp.name),
                    args.s3_bucket,
                    f"{model_s3_prefix}/pca_variance.json",
                )

        # Upload sweep results if sweep was run
        if fit_info.get("sweep_results"):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(fit_info["sweep_results"], tmp, indent=2)
                tmp.flush()
                upload_s3_file(
                    s3,
                    Path(tmp.name),
                    args.s3_bucket,
                    f"{model_s3_prefix}/kappa_sweep.json",
                )

        # Upload ELBO/convergence trace if captured
        if fit_info.get("elbo_trace"):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump({"elbo_trace": fit_info["elbo_trace"]}, tmp, indent=2)
                tmp.flush()
                upload_s3_file(
                    s3,
                    Path(tmp.name),
                    args.s3_bucket,
                    f"{model_s3_prefix}/convergence.json",
                )

        # Upload fit_info (complete metadata)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            # fit_info may contain non-serializable items; sanitize
            serializable_info = {
                k: v
                for k, v in fit_info.items()
                if k != "pca_variance"  # already uploaded separately
            }
            json.dump(serializable_info, tmp, indent=2, default=str)
            tmp.flush()
            upload_s3_file(
                s3,
                Path(tmp.name),
                args.s3_bucket,
                f"{model_s3_prefix}/fit_info.json",
            )

    # ── Summary ─────────────────────────────────────────────────────────────

    total_syllables = set()
    for data in results.values():
        total_syllables.update(np.unique(data["syllable_id"]).tolist())

    summary = {
        "n_sessions": len(results),
        "n_unique_syllables": len(total_syllables),
        "sessions": {
            sid: {
                "n_frames": len(d["syllable_id"]),
                "n_syllables": len(np.unique(d["syllable_id"])),
            }
            for sid, d in results.items()
        },
        "params": {
            "kappa": fit_info["kappa"],
            "num_pcs": fit_info["num_pcs"],
            "num_iters": fit_info["num_iters"],
            "bodyparts": fit_info["bodyparts"],
            "kappa_sweep": use_sweep,
        },
    }

    log.info(
        "Summary: %d sessions, %d unique syllables across all sessions",
        summary["n_sessions"],
        summary["n_unique_syllables"],
    )

    # Save summary
    if args.output_dir:
        with open(args.output_dir / "kpms_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    if using_s3:
        if s3 is None:
            s3 = get_s3_client()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(summary, tmp, indent=2)
            tmp.flush()
            upload_s3_file(s3, Path(tmp.name), args.s3_bucket, "kinematics/kpms_summary.json")


if __name__ == "__main__":
    main()
