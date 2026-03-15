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

import argparse
import csv
import io
import json
import logging
import sys
import tempfile
from pathlib import Path

import os

# Force JAX to CPU-only (avoids noisy CUDA errors on CPU instances)
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kpms")


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
    log.info("  Selecting best individual per frame (%d frames, %d individuals)...",
             n_frames, len(individuals))

    # Build (n_frames, n_individuals) likelihood matrix
    ind_scores = np.full((n_frames, len(individuals)), -1.0)
    for j, ind in enumerate(individuals):
        lk_cols = []
        for bp in use_bps:
            try:
                lk_cols.append(df[(scorer, ind, bp, "likelihood")].values)
            except KeyError:
                pass
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
                try:
                    all_vals[:, j] = df[(scorer, ind, bp, coord)].values
                except KeyError:
                    pass
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


# ── keypoint-MoSeq wrapper ─────────────────────────────────────────────────

def fit_kpms(
    dlc_files: dict[str, Path],
    project_dir: Path,
    bodyparts: list[str],
    kappa: float = 1e6,
    num_pcs: int = 10,
    num_iters: int = 50,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit keypoint-MoSeq AR-HMM on DLC .h5 files.

    Args:
        dlc_files: Dict of session_id → DLC .h5 file path.
        project_dir: Working directory for kpms config/checkpoints.
        bodyparts: List of body part names to use.
        kappa: AR-HMM stickiness (higher = longer syllables).
        num_pcs: Number of PCA components.
        num_iters: Number of fitting iterations.

    Returns:
        Dict of session_id → {"syllable_id": (N,) int16,
                               "syllable_prob": (N, S) float32}
    """
    import keypoint_moseq as kpms

    # Clean project dir contents to avoid "directory already exists" error
    # from kpms.  We clear contents rather than rmtree because the dir may be
    # a Docker bind-mount (rmtree on a mount point raises EBUSY).
    import shutil
    if project_dir.exists():
        for child in project_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    project_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup project ──────────────────────────────────────────────────────
    log.info("Setting up kpms project (bodyparts=%s)...", bodyparts)
    kpms.setup_project(
        project_dir=str(project_dir),
        deeplabcut_config=None,
        bodyparts=bodyparts,
        use_bodyparts=bodyparts,
        overwrite=True,
    )

    # Patch config.yml: setup_project writes placeholder BODYPART1/2/3 in
    # skeleton and anterior/posterior that cause load_config to crash.
    kpms.update_config(
        str(project_dir),
        anterior_bodyparts=[bodyparts[0]],     # e.g. "nose"
        posterior_bodyparts=[bodyparts[-1]],    # e.g. "mid_backend2"
        use_bodyparts=bodyparts,
        skeleton=[],
    )

    # Helper to load config as a dict
    def config():
        return kpms.load_config(str(project_dir))

    # ── Load DLC data (kpms 0.6+ API) ─────────────────────────────────────
    # load_keypoints expects individual file paths — build a temporary dir
    # with symlinks so we can point it at a directory pattern.
    import tempfile
    link_dir = Path(tempfile.mkdtemp(prefix="kpms_links_"))
    for sid, h5_path in dlc_files.items():
        # kpms uses the filename (minus extension) as session key
        link = link_dir / f"{sid}.h5"
        link.symlink_to(h5_path.resolve())

    log.info("Loading %d DLC files via load_keypoints...", len(dlc_files))
    coordinates, confidences, _bodyparts = kpms.load_keypoints(
        str(link_dir), "deeplabcut",
    )
    log.info("Loaded bodyparts: %s", _bodyparts)
    log.info("Sessions loaded: %s", list(coordinates.keys()))

    # ── Format data ──────────────────────────────────────────────────────────
    log.info("Formatting data...")
    cfg = config()
    log.info("Config keys: %s", list(cfg.keys()))
    log.info("anterior_bodyparts: %s", cfg.get("anterior_bodyparts"))
    log.info("posterior_bodyparts: %s", cfg.get("posterior_bodyparts"))
    log.info("anterior_idxs: %s", cfg.get("anterior_idxs"))
    log.info("posterior_idxs: %s", cfg.get("posterior_idxs"))
    log.info("use_bodyparts: %s", cfg.get("use_bodyparts"))
    log.info("bodyparts: %s", cfg.get("bodyparts"))
    data, metadata = kpms.format_data(coordinates, confidences, **cfg)
    log.info("data type: %s, keys: %s", type(data).__name__, list(data.keys()) if isinstance(data, dict) else "N/A")

    # noise_calibration is interactive (requires video frames for a Jupyter
    # widget) — skip it on headless EC2.  The default noise prior works fine
    # for DLC data with confidence scores.

    # ── PCA ────────────────────────────────────────────────────────────────
    log.info("Fitting PCA (num_pcs=%d)...", num_pcs)
    kpms.update_config(str(project_dir), num_pcs=num_pcs)

    # fit_pca(project_dir, data) internally calls load_config(project_dir)
    # which should set anterior_idxs/posterior_idxs. Debug what it produces.
    _pca_cfg = kpms.load_config(str(project_dir))
    log.info("PCA config anterior_idxs: %s (type: %s)",
             _pca_cfg.get("anterior_idxs"), type(_pca_cfg.get("anterior_idxs")).__name__)
    log.info("PCA config posterior_idxs: %s (type: %s)",
             _pca_cfg.get("posterior_idxs"), type(_pca_cfg.get("posterior_idxs")).__name__)

    pca = kpms.fit_pca(str(project_dir), data)

    # ── AR-HMM fitting ─────────────────────────────────────────────────────
    log.info("Fitting AR-HMM (kappa=%.0e, n_iters=%d)...", kappa, num_iters)
    kpms.update_config(str(project_dir), kappa=kappa)

    cfg = config()
    model = kpms.init_model(data, pca=pca, **cfg)

    model = kpms.fit_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=str(project_dir),
        num_iters=num_iters,
    )

    # ── Extract results ────────────────────────────────────────────────────
    log.info("Extracting syllable assignments...")
    results = kpms.extract_results(model, metadata, str(project_dir))

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
                session_id, len(syllable_id), len(np.unique(syllable_id)),
            )
        else:
            log.warning("  %s: not in results (skipped by kpms?)", session_id)

    return output


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run keypoint-MoSeq syllable discovery on DLC outputs."
    )
    parser.add_argument(
        "--dlc-dir", type=Path, default=None,
        help="Local directory containing DLC .h5 files.",
    )
    parser.add_argument(
        "--project-dir", type=Path, default=Path("/tmp/kpms_project"),
        help="Working directory for kpms config/checkpoints.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Local directory to write syllable .npz files.",
    )
    parser.add_argument(
        "--s3-bucket", type=str, default="hm2p-derivatives",
        help="S3 bucket for derivatives.",
    )
    parser.add_argument(
        "--all-sessions", action="store_true",
        help="Process all sessions from metadata/experiments.csv via S3.",
    )
    parser.add_argument(
        "--sessions", nargs="*", default=None,
        help="Specific session exp_ids to process.",
    )
    parser.add_argument(
        "--bodyparts", nargs="*",
        default=[
            "nose", "left_ear", "right_ear", "neck",
            "mid_back", "mouse_center", "mid_backend", "mid_backend2",
        ],
        help="Body parts to use for fitting (kpms recommends 5-10, no tail).",
    )
    parser.add_argument("--kappa", type=float, default=1e6)
    parser.add_argument("--num-pcs", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip sessions that already have syllable output on S3.",
    )

    args = parser.parse_args()

    # ── Determine which sessions to process ─────────────────────────────────

    if args.dlc_dir:
        # Local mode: find DLC .h5 files
        dlc_files = {}
        for h5 in sorted(args.dlc_dir.glob("**/*DLC*.h5")):
            # Infer session_id from directory structure
            session_id = h5.stem.split("DLC")[0].rstrip("_")
            dlc_files[session_id] = h5
        log.info("Found %d DLC files in %s", len(dlc_files), args.dlc_dir)

    elif args.all_sessions or args.sessions:
        # S3 mode: download DLC outputs
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

            # Check if syllable output already exists
            if args.skip_existing:
                syllable_key = f"kinematics/{sub}/{ses}/syllables.npz"
                try:
                    s3.head_object(Bucket=args.s3_bucket, Key=syllable_key)
                    log.info("Skipping %s (syllables already on S3)", exp_id)
                    continue
                except Exception:
                    pass

            # Download DLC .h5 from pose/
            pose_prefix = f"pose/{sub}/{ses}/"
            try:
                resp = s3.list_objects_v2(
                    Bucket=args.s3_bucket, Prefix=pose_prefix,
                )
                h5_keys = [
                    obj["Key"] for obj in resp.get("Contents", [])
                    if obj["Key"].endswith(".h5")
                    and not obj["Key"].endswith("_single.h5")
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

    results = fit_kpms(
        dlc_files=dlc_files,
        project_dir=args.project_dir,
        bodyparts=args.bodyparts,
        kappa=args.kappa,
        num_pcs=args.num_pcs,
        num_iters=args.num_iters,
    )

    log.info("Fitting complete. %d sessions have results.", len(results))

    # ── Save outputs ────────────────────────────────────────────────────────

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for session_id, data in results.items():
        npz_data = {"syllable_id": data["syllable_id"]}
        if data.get("syllable_prob") is not None:
            npz_data["syllable_prob"] = data["syllable_prob"]

        if args.output_dir:
            # Save locally
            out_path = args.output_dir / f"{session_id}_syllables.npz"
            np.savez_compressed(out_path, **npz_data)
            log.info("Saved %s", out_path)

        if args.all_sessions or args.sessions:
            # Upload to S3
            s3 = get_s3_client()
            sub, ses = parse_session_id(session_id)
            s3_key = f"kinematics/{sub}/{ses}/syllables.npz"

            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                np.savez_compressed(tmp.name, **npz_data)
                upload_s3_file(s3, Path(tmp.name), args.s3_bucket, s3_key)

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
            "kappa": args.kappa,
            "num_pcs": args.num_pcs,
            "num_iters": args.num_iters,
            "bodyparts": args.bodyparts,
        },
    }

    log.info("Summary: %d sessions, %d unique syllables across all sessions",
             summary["n_sessions"], summary["n_unique_syllables"])

    # Save summary
    if args.output_dir:
        with open(args.output_dir / "kpms_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    if args.all_sessions or args.sessions:
        s3 = get_s3_client()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(summary, tmp, indent=2)
            tmp.flush()
            upload_s3_file(s3, Path(tmp.name), args.s3_bucket, "kinematics/kpms_summary.json")


if __name__ == "__main__":
    main()
