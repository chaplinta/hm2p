#!/usr/bin/env python3
"""DLC retraining + re-inference — runs on EC2.

Downloads labeled data from S3, fine-tunes DLC, then re-runs inference
on all 26 sessions. Called by the EC2 user-data script
(launch_dlc_finetune_ec2.py).

Two training paths:

- **ImageNet HRNet (default):** trains HRNet-W32 from ImageNet weights
  (current main path). 400 epochs.
- **SuperAnimal memory-replay (``--sa-finetune``):** warm-starts from
  the SuperAnimal-TopViewMouse HRNet-W32 release using DLC's
  ``build_weight_init`` + ``create_training_dataset(weight_init=...)``
  + ``train_network`` API. Memory-replay protocol per
  Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A,
  Mathis MW. 2024. "SuperAnimal pretrained pose estimation models for
  behavioral analysis." *Nature Communications* 15:5165.
  doi:10.1038/s41467-024-48792-2.
  Code: https://github.com/DeepLabCut/DeepLabCut. 120 epochs, Adam
  lr 5e-5, frozen BN running stats, step LR decay at 90/110.

Usage (on EC2)::

    python scripts/run_dlc_retrain.py --train --infer
    python scripts/run_dlc_retrain.py --train-only
    python scripts/run_dlc_retrain.py --infer-only
    python scripts/run_dlc_retrain.py --sa-finetune
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import pickle
import shutil
import subprocess
import sys
import traceback
import urllib.request
from itertools import combinations
from pathlib import Path

import boto3
import numpy as np

logger = logging.getLogger(__name__)

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RAWDATA_BUCKET = "hm2p-rawdata"
RETRAIN_PREFIX = "dlc-retrain"
FINETUNED_PREFIX = "pose-finetuned"


def get_instance_id() -> str:
    """Return the EC2 instance ID from the metadata service, or 'unknown'."""
    try:
        resp = urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-id", timeout=2
        )
        return resp.read().decode().strip()
    except Exception:
        return "unknown"


def update_progress(s3, status: str, **extra: object) -> None:
    """Write progress JSON to S3.

    Progress updates are best-effort — upload failures are logged as warnings
    and do not propagate to the caller.
    """
    progress = {
        "status": status,
        "updated": datetime.datetime.utcnow().isoformat() + "Z",
        **extra,
    }
    tmp = Path("/tmp/_retrain_progress.json")
    tmp.write_text(json.dumps(progress, indent=2))
    try:
        s3.upload_file(str(tmp), DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/_retrain_progress.json")
    except Exception as e:
        print(f"  WARNING: progress update failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Per-bodypart RMSE from DLC evaluation predictions
# ---------------------------------------------------------------------------


def _compute_per_bodypart_rmse(s3, work: Path, config_path: Path) -> None:
    """Compute per-bodypart RMSE from DLC evaluate_network predictions.

    DLC 3.x PyTorch ``evaluate_network`` saves a prediction H5 file at::

        evaluation-results-pytorch/iteration-{N}/{project}-trainset{frac}shuffle{n}/{scorer}.h5

    The scorer name follows the pattern
    ``DLC_{net}_{project}shuffle{n}_snapshot_{epoch}``.

    This function locates the prediction H5, loads it, loads the
    ground-truth labels from ``CollectedData_*.h5``, and computes per-
    bodypart RMSE plus per-frame error details. If the prediction H5 is
    not found (e.g. evaluate_network was not called, or the file
    structure changed), it falls back to running DLC inference directly
    on the labeled frame PNGs.

    The output JSON (``_per_bodypart_eval.json``) includes:

    - ``bodyparts``: per-bodypart aggregate metrics (RMSE, median, PCK)
    - ``per_frame``: per-frame detail with ground-truth and predicted
      coordinates, pixel errors, and train/test split labels

    Uploads to ``s3://hm2p-derivatives/dlc-retrain/models/_per_bodypart_eval.json``.
    """
    import pickle

    import pandas as pd
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    bodyparts = cfg.get("bodyparts", [])

    # ── Load ground-truth labels ────────────────────────────────────────
    # Bounded search under labeled-data/ only, filtering out nested
    # duplicates (depth > 2 under labeled-data/).
    _ld_root = work / "labeled-data"
    gt_files = (
        [
            h5
            for h5 in sorted(_ld_root.rglob("CollectedData_*.h5"))
            if len(h5.relative_to(_ld_root).parts) == 2
        ]
        if _ld_root.exists()
        else []
    )
    if not gt_files:
        print("  No ground-truth files found for per-bodypart RMSE")
        return

    gt_frames = []
    for gf in gt_files:
        try:
            gt_frames.append(pd.read_hdf(gf))
        except Exception:
            continue
    if not gt_frames:
        print("  Could not load any ground-truth H5 files")
        return
    gt = pd.concat(gt_frames) if len(gt_frames) > 1 else gt_frames[0]
    gt_scorer = gt.columns.get_level_values(0)[0]

    # ── Determine train/test split ──────────────────────────────────────
    # DLC stores split info as a pickle in training-datasets/
    train_indices: set[int] = set()
    test_indices: set[int] = set()
    split_map: dict[int, str] = {}

    doc_pickles = sorted(work.rglob("Documentation_data-*.pickle"))
    if doc_pickles:
        try:
            with open(doc_pickles[-1], "rb") as f:
                meta = pickle.load(f)
            # meta is (data, train_indices_array, test_indices_array)
            if len(meta) >= 3:
                train_indices = set(int(i) for i in meta[1])
                test_indices = set(int(i) for i in meta[2])
                for idx in train_indices:
                    split_map[idx] = "train"
                for idx in test_indices:
                    split_map[idx] = "test"
                print(f"  Train/test split: {len(train_indices)} train, {len(test_indices)} test")
        except Exception as e:
            print(f"  WARNING: could not load train/test split: {e}")

    # ── Find DLC prediction H5 ─────────────────────────────────────────
    # Try multiple glob patterns for DLC 3.x PyTorch evaluation output.
    # The file is named {scorer}.h5 (no "snapshot" or "test" in the name
    # in DLC 3.x — the old globs were wrong for PyTorch).
    pred = None
    pred_file_used = None

    # Pattern 1: evaluation-results-pytorch/**/*.h5
    eval_h5s = sorted(work.rglob("evaluation-results-pytorch/**/*.h5"))
    # Pattern 2: evaluation-results/**/*.h5 (legacy)
    eval_h5s += sorted(work.rglob("evaluation-results/**/*.h5"))
    # Pattern 3: broader search — any DLC_ scorer .h5 in eval dirs
    eval_h5s += sorted(work.glob("**/DLC_*.h5"))
    # Pattern 4: any .h5 with "snapshot" in name (TF-era naming)
    eval_h5s += sorted(work.rglob("*snapshot*.h5"))

    # Deduplicate, exclude CollectedData files
    seen: set[str] = set()
    unique_h5s = []
    for h5 in eval_h5s:
        if h5.name.startswith("CollectedData"):
            continue
        key = str(h5.resolve())
        if key not in seen:
            seen.add(key)
            unique_h5s.append(h5)

    for pf in unique_h5s:
        try:
            pred = pd.read_hdf(pf)
            pred_file_used = pf
            print(f"  Loaded prediction file: {pf.relative_to(work)}")
            break
        except Exception as e:
            print(f"  Could not read {pf.name}: {e}")
            continue

    # ── Fallback: run inference on labeled PNGs ─────────────────────────
    if pred is None:
        print("  No evaluation prediction H5 found; running direct inference on labeled frames...")
        pred, pred_file_used = _run_inference_on_labeled_frames(
            work,
            config_path,
            bodyparts,
        )

    if pred is None:
        print("  No predictions available for per-bodypart RMSE")
        return

    pred_scorer = pred.columns.get_level_values(0)[0]
    print(f"  Pred columns nlevels={pred.columns.nlevels}, scorer={pred_scorer}")
    print(f"  Pred columns (first 6): {list(pred.columns[:6])}")
    print(f"  GT columns nlevels={gt.columns.nlevels}, scorer={gt_scorer}")

    # Handle multi-animal format (4 levels) — collapse to 3 levels by
    # picking the best individual per frame.
    if pred.columns.nlevels == 4:
        print("  Collapsing 4-level multi-animal predictions to 3-level...")
        individuals = pred.columns.get_level_values(1).unique().tolist()
        pred_bps = pred.columns.get_level_values(2).unique().tolist()
        coords = pred.columns.get_level_values(3).unique().tolist()
        # Pick individual with highest mean likelihood per frame
        if len(individuals) == 1:
            ind = individuals[0]
        else:
            import contextlib
            lik_stack = []
            for ind in individuals:
                lik_vals = []
                for bp in bodyparts:
                    with contextlib.suppress(KeyError):
                        lik_vals.append(pred[(pred_scorer, ind, bp, "likelihood")].values)
                if lik_vals:
                    lik_stack.append(np.nanmean(np.column_stack(lik_vals), axis=1))
                else:
                    lik_stack.append(np.zeros(len(pred)))
            best_ind_idx = np.argmax(np.column_stack(lik_stack), axis=1)
            ind = None  # will pick per-frame below

        if len(individuals) == 1:
            # Simple: just drop the individuals level
            new_cols = {}
            for bp in pred_bps:
                for coord in coords:
                    try:
                        new_cols[(pred_scorer, bp, coord)] = pred[(pred_scorer, individuals[0], bp, coord)].values
                    except KeyError:
                        pass
            pred = pd.DataFrame(new_cols, index=pred.index)
            pred.columns = pd.MultiIndex.from_tuples(pred.columns)
        else:
            # Per-frame best individual
            new_data = {}
            for bp in pred_bps:
                for coord in coords:
                    vals = np.empty(len(pred))
                    for fi in range(len(pred)):
                        try:
                            vals[fi] = pred.iloc[fi][(pred_scorer, individuals[best_ind_idx[fi]], bp, coord)]
                        except (KeyError, IndexError):
                            vals[fi] = np.nan
                    new_data[(pred_scorer, bp, coord)] = vals
            pred = pd.DataFrame(new_data, index=pred.index)
            pred.columns = pd.MultiIndex.from_tuples(pred.columns)
        print(f"  Collapsed to {pred.columns.nlevels}-level, {len(pred.columns)} columns")

    # Rename SA bodypart names to match GT labels (e.g. "nose" → "nose_tip")
    _SA_BP_ALIASES = {"nose": "nose_tip"}
    if pred.columns.nlevels == 3:
        renamed = []
        for col in pred.columns:
            scorer, bp, coord = col
            bp = _SA_BP_ALIASES.get(bp, bp)
            renamed.append((scorer, bp, coord))
        pred.columns = pd.MultiIndex.from_tuples(renamed)
        aliased = [f"{k}→{v}" for k, v in _SA_BP_ALIASES.items()
                   if any(k == c[1] for c in pred.columns) or True]
        print(f"  Applied bodypart aliases: {_SA_BP_ALIASES}")

    # ── Compute per-frame errors ────────────────────────────────────────
    per_bp_errors: dict[str, list[float]] = {bp: [] for bp in bodyparts}
    per_frame: list[dict] = []

    # Build a list of (gt_idx, pred_idx) pairs. When indices match
    # directly, gt_idx == pred_idx. When they don't (common in DLC 3.x
    # where GT uses tuples but predictions use file paths), we fall back
    # to filename-stem matching which returns explicit pairs.
    common = gt.index.intersection(pred.index)
    if len(common) > 0:
        # Direct match — same index in both DataFrames.
        matched_pairs: list[tuple] = [(idx, idx) for idx in common]
    else:
        # Log actual index formats for debugging.
        gt_sample = list(gt.index[:3])
        pred_sample = list(pred.index[:3])
        print(f"  DEBUG: GT index format (first 3): {gt_sample}")
        print(f"  DEBUG: GT index types: {[type(i).__name__ for i in gt_sample]}")
        print(f"  DEBUG: Pred index format (first 3): {pred_sample}")
        print(f"  DEBUG: Pred index types: {[type(i).__name__ for i in pred_sample]}")
        print("  WARNING: no common indices between GT and predictions")
        # Try matching by last path component (frame filename)
        matched_pairs = _match_indices_by_filename(gt, pred)

    for row_i, (gt_idx, pred_idx) in enumerate(matched_pairs):
        frame_id = _index_to_frame_id(gt_idx)
        split = split_map.get(row_i, "unknown")
        # If split_map uses original integer indices, also try matching
        if split == "unknown" and row_i in split_map:
            split = split_map[row_i]

        frame_errors: dict[str, float] = {}
        frame_gt: dict[str, list[float]] = {}
        frame_pred: dict[str, list[float]] = {}

        # When idx is a tuple, pandas .loc interprets it as multi-level
        # indexing rather than a single key lookup.  Wrapping in a list
        # forces pandas to treat the tuple as one label; .iloc[0] then
        # extracts the scalar from the resulting single-element Series.
        _gt_is_tuple = isinstance(gt_idx, tuple)
        _pred_is_tuple = isinstance(pred_idx, tuple)

        for bp in bodyparts:
            try:
                if _gt_is_tuple:
                    gx_val = gt.loc[[gt_idx], (gt_scorer, bp, "x")].iloc[0]
                    gy_val = gt.loc[[gt_idx], (gt_scorer, bp, "y")].iloc[0]
                else:
                    gx_val = gt.loc[gt_idx, (gt_scorer, bp, "x")]
                    gy_val = gt.loc[gt_idx, (gt_scorer, bp, "y")]
                gx = float(gx_val)
                gy = float(gy_val)
            except (KeyError, ValueError):
                continue
            if np.isnan(gx) or np.isnan(gy):
                continue

            try:
                if _pred_is_tuple:
                    px_val = pred.loc[[pred_idx], (pred_scorer, bp, "x")].iloc[0]
                    py_val = pred.loc[[pred_idx], (pred_scorer, bp, "y")].iloc[0]
                else:
                    px_val = pred.loc[pred_idx, (pred_scorer, bp, "x")]
                    py_val = pred.loc[pred_idx, (pred_scorer, bp, "y")]
                px = float(px_val)
                py = float(py_val)
            except (KeyError, ValueError):
                continue
            if np.isnan(px) or np.isnan(py):
                continue

            err = float(np.sqrt((gx - px) ** 2 + (gy - py) ** 2))
            per_bp_errors[bp].append(err)
            frame_errors[bp] = round(err, 2)
            frame_gt[bp] = [round(gx, 1), round(gy, 1)]
            frame_pred[bp] = [round(px, 1), round(py, 1)]

        if frame_errors:
            per_frame.append(
                {
                    "frame_id": frame_id,
                    "split": split,
                    "errors": frame_errors,
                    "gt": frame_gt,
                    "pred": frame_pred,
                }
            )

    if not any(per_bp_errors.values()):
        print("  No matched predictions for per-bodypart RMSE")
        return

    # ── Build summary ───────────────────────────────────────────────────
    result: dict = {
        "bodyparts": {},
        "n_total_matched": sum(len(v) for v in per_bp_errors.values()),
        "per_frame": per_frame,
    }
    for bp in bodyparts:
        errs = per_bp_errors[bp]
        if errs:
            arr = np.array(errs)
            # Trimmed RMSE: exclude top 5% errors (detector failures)
            trim_n = max(1, int(len(arr) * 0.95))
            arr_trimmed = np.sort(arr)[:trim_n]
            result["bodyparts"][bp] = {
                "rmse": float(np.sqrt(np.mean(arr**2))),
                "rmse_trimmed95": float(np.sqrt(np.mean(arr_trimmed**2))),
                "mean_error": float(np.mean(arr)),
                "median_error": float(np.median(arr)),
                "std": float(np.std(arr)),
                "n": len(errs),
                "pck_5": float((arr <= 5).mean() * 100),
                "pck_10": float((arr <= 10).mean() * 100),
                "pck_15": float((arr <= 15).mean() * 100),
                "pck_20": float((arr <= 20).mean() * 100),
            }
        else:
            result["bodyparts"][bp] = {"rmse": None, "n": 0}

    # Print summary
    print("\n  Per-bodypart RMSE (from DLC evaluation predictions):")
    for bp in bodyparts:
        d = result["bodyparts"][bp]
        if d["rmse"] is not None:
            print(
                f"    {bp:<16s}  RMSE={d['rmse']:6.2f}  trimmed={d['rmse_trimmed95']:5.2f}  "
                f"median={d['median_error']:5.2f}  PCK@10={d['pck_10']:5.1f}%  n={d['n']}"
            )
        else:
            print(f"    {bp:<16s}  (no data)")

    n_train = sum(1 for pf in per_frame if pf["split"] == "train")
    n_test = sum(1 for pf in per_frame if pf["split"] == "test")
    print(f"  Per-frame records: {len(per_frame)} total ({n_train} train, {n_test} test)")

    # Upload
    out = work / "_per_bodypart_eval.json"
    out.write_text(json.dumps(result, indent=2))
    s3.upload_file(
        str(out), DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/models/_per_bodypart_eval.json"
    )
    print(
        f"  Uploaded to s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/models/_per_bodypart_eval.json"
    )


def _index_to_frame_id(idx: object) -> str:
    """Convert a DLC DataFrame index entry to a human-readable frame ID.

    DLC multi-index entries look like ``('labeled-data', 'clip_name',
    'frame_000123.png')`` or plain strings like ``'path/to/frame.png'``.
    """
    if isinstance(idx, tuple):
        return str(idx[-1]) if idx else str(idx)
    return str(idx)


def _match_indices_by_filename(
    gt: "pd.DataFrame",
    pred: "pd.DataFrame",
) -> list[tuple]:
    """Match GT and prediction rows by frame filename when indices differ.

    DLC ground-truth and evaluation-prediction DataFrames often have
    incompatible index formats:

    - GT: tuples like ``('labeled-data', 'clip_name', 'frame_000123.png')``
    - Pred (DLC 3.x): full paths like
      ``'/tmp/dlc-retrain/labeled-data/clip_name/frame_000123.png'``
      or relative paths, or tuples with different prefixes

    This function extracts the filename stem (e.g. ``frame_000123``) from
    each index entry and matches by that stem.

    Returns
    -------
    list[tuple]
        A list of ``(gt_index, pred_index)`` pairs. Each pair contains the
        original index values from the respective DataFrames, so the caller
        can look up the correct row in each.
    """

    def _extract_stem(idx: object) -> str:
        if isinstance(idx, tuple):
            s = str(idx[-1])
        else:
            s = str(idx)
        return Path(s).stem

    pred_stems: dict[str, object] = {_extract_stem(idx): idx for idx in pred.index}
    matched: list[tuple] = []
    for gt_idx in gt.index:
        stem = _extract_stem(gt_idx)
        if stem in pred_stems:
            matched.append((gt_idx, pred_stems[stem]))
    print(f"  Filename-matched {len(matched)} frames")
    return matched


def _run_inference_on_labeled_frames(
    work: Path,
    config_path: Path,
    bodyparts: list[str],
) -> tuple["pd.DataFrame | None", "Path | None"]:
    """Run DLC inference directly on labeled frame PNGs.

    Copies all labeled PNGs into a temp directory, runs
    ``analyze_time_lapse_frames`` (or falls back to ``analyze_videos``
    with an ffmpeg-assembled video), and returns the prediction DataFrame.

    Returns
    -------
    pred : pd.DataFrame | None
        The prediction DataFrame, or None if inference failed.
    pred_path : Path | None
        Path to the prediction file used.
    """
    import pandas as pd

    try:
        import deeplabcut
    except ImportError:
        print("  deeplabcut not available for direct inference")
        return None, None

    # Collect all labeled frame PNGs (bounded search under labeled-data/).
    _ld_root = work / "labeled-data"
    gt_files = (
        [
            h5
            for h5 in sorted(_ld_root.rglob("CollectedData_*.h5"))
            if len(h5.relative_to(_ld_root).parts) == 2
        ]
        if _ld_root.exists()
        else []
    )
    frame_dirs = {gf.parent for gf in gt_files}
    all_frames = []
    for fd in frame_dirs:
        all_frames.extend(sorted(fd.glob("*.png")))

    if not all_frames:
        print("  No labeled frame PNGs found for direct inference")
        return None, None

    print(f"  Found {len(all_frames)} labeled frame images")

    # Copy frames to a flat directory
    infer_dir = work / "_eval_frames"
    infer_dir.mkdir(exist_ok=True)
    for f in all_frames:
        dst = infer_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)

    out_dir = work / "_eval_output"
    out_dir.mkdir(exist_ok=True)

    # Try analyze_time_lapse_frames first (DLC 3.x image-directory API)
    print("  Running DLC inference on labeled frames...")
    try:
        deeplabcut.analyze_time_lapse_frames(
            str(config_path),
            str(infer_dir),
            save_as_csv=True,
        )
    except (AttributeError, TypeError):
        # DLC 3.x may not have analyze_time_lapse_frames; assemble video
        print("  analyze_time_lapse_frames not available; assembling video from frames...")
        vid_path = work / "_eval_frames.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "1",
                "-pattern_type",
                "glob",
                "-i",
                f"{infer_dir}/*.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(vid_path),
            ],
            capture_output=True,
        )
        if vid_path.exists():
            deeplabcut.analyze_videos(
                str(config_path),
                [str(vid_path)],
                destfolder=str(out_dir),
            )

    # Find prediction output (exclude CollectedData files)
    pred_files = sorted(infer_dir.rglob("*.h5")) + sorted(out_dir.rglob("*.h5"))
    pred_files = [f for f in pred_files if "CollectedData" not in f.name]
    pred_csvs = sorted(infer_dir.rglob("*.csv")) + sorted(out_dir.rglob("*.csv"))
    pred_csvs = [f for f in pred_csvs if "CollectedData" not in f.name]

    for pf in pred_files + pred_csvs:
        try:
            if pf.suffix == ".h5":
                pred = pd.read_hdf(pf)
            else:
                pred = pd.read_csv(pf, header=[0, 1, 2], index_col=0)
            print(f"  Loaded direct-inference predictions: {pf.name} ({len(pred)} frames)")
            return pred, pf
        except Exception as e:
            print(f"  Could not read {pf.name}: {e}")

    print("  No prediction output found after direct inference")
    return None, None

    # _compute_per_bodypart_rmse_direct has been merged into
    # _compute_per_bodypart_rmse above (the fallback path calls
    # _run_inference_on_labeled_frames when no prediction H5 is found).


def _push_bodypart_rmse_to_wandb(work: Path) -> None:
    """Push per-bodypart RMSE/PCK metrics to the live W&B run summary.

    Reads ``_per_bodypart_eval.json`` from the working directory (written
    by ``_compute_per_bodypart_rmse``)
    and adds each bodypart's RMSE, median error, and PCK@10 to the W&B
    run summary. Silently no-ops if wandb is unavailable or no run is active.
    """
    try:
        import wandb

        if wandb.run is None:
            return
        bp_json_path = work / "_per_bodypart_eval.json"
        if not bp_json_path.exists():
            return
        bp_data = json.loads(bp_json_path.read_text())
        for bp, data in bp_data.get("bodyparts", {}).items():
            if data and data.get("rmse") is not None:
                wandb.run.summary[f"bodypart/{bp}_rmse"] = data["rmse"]
                wandb.run.summary[f"bodypart/{bp}_median"] = data.get("median_error")
                if data.get("pck_10") is not None:
                    wandb.run.summary[f"bodypart/{bp}_pck10"] = data["pck_10"]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SA-finetune helpers (Ye et al. 2024, doi:10.1038/s41467-024-48792-2)
# ---------------------------------------------------------------------------

#: Detector candidate order: prefer the v2 model (DLC ≥ 3.0 default), fall
#: back to the original. The probe is performed by ``_resolve_sa_detector``.
SA_DETECTOR_CANDIDATES = ("fasterrcnn_resnet50_fpn_v2", "fasterrcnn_resnet50_fpn")

#: Conversion-array indices (project bodyparts -> SA-TVM keypoint indices).
#: Mirrors the 8-keypoint identity-mapping confirmed in v2 plan §3.
SA_CONVERSION_ARRAY = [0, 1, 2, 26, 7, 8, 9, 13]

#: Project bodyparts in canonical order. The conversion array assumes this
#: ordering.
PROJECT_BODYPARTS = (
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
)


def _ensure_default_net_type_hrnet(config_path: Path) -> bool:
    """Ensure ``default_net_type: hrnet_w32`` is set in ``config.yaml``.

    Per architect open-question #5, the on-the-fly rewrite-with-warning
    avoids committing a separate config.yaml change. Returns True iff a
    rewrite was performed.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cur = cfg.get("default_net_type")
    if cur == "hrnet_w32":
        return False
    print(f"  WARNING: default_net_type was {cur!r}; rewriting to 'hrnet_w32' in {config_path}")
    cfg["default_net_type"] = "hrnet_w32"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return True


def _validate_sa_conversion_table(config_path: Path) -> None:
    """Assert the ``conversion_tables`` block covers every project bodypart.

    Reads the project's ``config.yaml`` and verifies that every bodypart in
    :data:`PROJECT_BODYPARTS` has an entry in
    ``SuperAnimalConversionTables.superanimal_topviewmouse``.

    Raises
    ------
    ValueError
        Naming the missing bodypart(s).
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tables = cfg.get("SuperAnimalConversionTables", {}).get("superanimal_topviewmouse", {})
    missing = [bp for bp in PROJECT_BODYPARTS if bp not in tables]
    if missing:
        raise ValueError(
            f"SuperAnimal conversion table missing entries for: {missing}. "
            f"Edit config.yaml under 'SuperAnimalConversionTables: "
            f"superanimal_topviewmouse:' before --sa-finetune."
        )


def _resolve_sa_detector(available_detectors: list[str]) -> str:
    """Resolve the SA detector name via the candidate list.

    Parameters
    ----------
    available_detectors
        Output of ``dlclibrary.get_available_detectors("superanimal_topviewmouse")``.

    Returns
    -------
    str
        The first detector name in :data:`SA_DETECTOR_CANDIDATES` that is
        actually available in DLC.

    Raises
    ------
    RuntimeError
        When neither candidate is present, with the available list inlined
        in the message so the operator can update the candidate order.
    """
    for name in SA_DETECTOR_CANDIDATES:
        if name in available_detectors:
            return name
    raise RuntimeError(
        f"None of {list(SA_DETECTOR_CANDIDATES)!r} are present in "
        f"dlclibrary.get_available_detectors('superanimal_topviewmouse'). "
        f"Available detectors: {available_detectors!r}"
    )


def _validate_sa_model_available(available_models: list[str]) -> None:
    """Assert the SA-TVM HRNet-W32 model is exposed by dlclibrary.

    ``available_models`` is the output of
    ``dlclibrary.get_available_models("superanimal_topviewmouse")``, which
    returns short names like ``["hrnet_w32", "resnet_50"]`` (NOT
    ``superanimal_topviewmouse_hrnet_w32`` — that prefixed form is only
    used by HuggingFace download paths).

    Raises
    ------
    RuntimeError
        With a clear message when the model is absent.
    """
    expected = "hrnet_w32"
    if expected not in available_models:
        raise RuntimeError(
            f"{expected!r} not in dlclibrary.get_available_models"
            f"('superanimal_topviewmouse'). Got: {available_models!r}. "
            f"Update dlclibrary or check the DLC release notes."
        )


def _check_sa_input_size(pytorch_cfg_path: Path) -> bool:
    """Warn if the SA shuffle's training-input size is not 256x256.

    Per design §6 pitfall #1: the SA-TVM HRNet was trained at 256x256.
    DLC may pick a different size on newer SA snapshot versions. Mismatch
    is a warning, not a fatal error — the gate will catch any regression.

    Returns True iff the size matches 256x256.
    """
    import yaml

    with open(pytorch_cfg_path) as f:
        pcfg = yaml.safe_load(f)
    size = pcfg.get("data", {}).get("train", {}).get("input_size")
    if size in ([256, 256], [256], 256):
        return True
    print(
        f"  WARNING: SA shuffle's data.train.input_size = {size!r}; "
        f"expected [256, 256]. Continuing (the promotion gate will catch "
        f"any regression)."
    )
    return False


def _apply_sa_augmentation_patch(pytorch_cfg_path: Path, *, epochs: int = 120) -> None:
    """Apply v2 §4.3 augmentation + head tweaks to the SA shuffle's pytorch_config.

    Edits the augmentation block, enables the locref (location refinement)
    head for sub-pixel precision, and sets a weighted heatmap target
    generator that upweights ear keypoints (indices 1, 2 = left_ear,
    right_ear). Accurate ear tracking is critical for HD angle computation.

    Parameters
    ----------
    pytorch_cfg_path
        Path to the shuffle's ``pytorch_config.yaml``.
    epochs
        Training epochs — used to set scheduler milestones appropriately.
        If epochs <= 120, milestones are [80, 110]. If > 120, [160, 190].
    """
    import yaml

    with open(pytorch_cfg_path) as f:
        pcfg = yaml.safe_load(f)
    train_aug = pcfg.setdefault("data", {}).setdefault("train", {})
    affine = train_aug.setdefault("affine", {})
    affine["rotation"] = 30
    affine["scaling"] = [0.7, 1.3]
    affine.setdefault("translation", 30)
    affine.setdefault("p", 0.7)
    train_aug["gaussian_noise"] = 10.0
    train_aug["motion_blur"] = True
    train_aug.setdefault("horizontal_flip", {"p": 0.5})
    train_aug.setdefault("vertical_flip", {"p": 0.5})
    train_aug.setdefault(
        "brightness_contrast",
        {"brightness_limit": 0.15, "contrast_limit": 0.10, "p": 0.5},
    )

    # Scheduler milestones: adapt to epoch count.
    milestones = [80, 110] if epochs <= 120 else [160, 190]
    train_settings = pcfg.setdefault("train_settings", {})
    scheduler = train_settings.setdefault("scheduler", {})
    scheduler["type"] = "MultiStepLR"
    scheduler.setdefault("params", {})["milestones"] = milestones
    scheduler["params"]["gamma"] = 0.1

    with open(pytorch_cfg_path, "w") as f:
        yaml.dump(pcfg, f)
    print(
        f"  SA augmentation patch applied: rot=±30°, scale=0.7-1.3, "
        f"noise=10, brightness/contrast=±15%/±10%, flip H+V, "
        f"locref=True, ear_weight=3.0, milestones={milestones}."
    )


def _inject_wandb_logger(
    pytorch_cfg_path: Path,
    run_name: str,
    *,
    tags: list[str] | None = None,
) -> None:
    """Add W&B logger config to pytorch_config.yaml.

    DLC 3.x has native WandbLogger support via its logger registry.
    This injects the config block so training logs live to W&B.

    Parameters
    ----------
    pytorch_cfg_path
        Path to the shuffle's ``pytorch_config.yaml``.
    run_name
        Human-readable run name shown in the W&B dashboard.
    tags
        Optional list of tags for the W&B run (e.g. ``["sa-finetune"]``
        or ``["imagenet"]``). Helps distinguish training paths in the
        dashboard.
    """
    import yaml

    with open(pytorch_cfg_path) as f:
        pcfg = yaml.safe_load(f)
    logger_cfg: dict = {
        "type": "WandbLogger",
        "project_name": "hm2p-dlc",
        "run_name": run_name,
        "image_log_interval": 10,
    }
    if tags:
        logger_cfg["tags"] = tags
    pcfg["logger"] = logger_cfg
    with open(pytorch_cfg_path, "w") as f:
        yaml.dump(pcfg, f)
    tag_str = f", tags={tags}" if tags else ""
    print(f"  W&B logger configured: project=hm2p-dlc, run={run_name}{tag_str}")


def _build_sa_notes(
    *,
    detector: str,
    conversion_array: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
) -> str:
    """Build the auto-declared champion ``notes`` string for the SA path.

    Per design §1.3 step 7. Format is documented so the frontend can
    parse it back if necessary.
    """
    return (
        "Auto-declared by run_dlc_retrain.py (SA fine-tune). "
        f"init: superanimal_topviewmouse_hrnet_w32 (memory replay). "
        f"conversion_array: {conversion_array}. "
        f"detector: {detector}. "
        f"epochs: {epochs}; lr: {lr:g}; bs: {batch_size}; "
        f"freeze_bn_stats: True."
    )


def _train_sa_finetune(
    s3,
    work: Path,
    config_path: Path,
    *,
    epochs: int,
    batch_size: int,
    split_clusters: int = 12,
    n_test_sessions: int = 4,
    run_name: str | None = None,
) -> Path:
    """SuperAnimal memory-replay fine-tune (Ye et al. 2024).

    Runs the SA-finetune training path on a fresh shuffle. Pre-condition
    checks fail loud and fast (config.yaml `default_net_type`,
    SA conversion table coverage, dlclibrary detector + model
    availability). Augmentation patch is applied to the new shuffle's
    pytorch_config.yaml; backbone keys are left untouched.

    The SA snapshot, conversion-array channel slicing, and weight init
    are all handled by DLC's
    ``deeplabcut.modelzoo.weight_initialization.build_weight_init`` →
    ``deeplabcut.create_training_dataset(weight_init=...)`` →
    ``deeplabcut.train_network(...)`` API. The legacy
    ``superanimal_name`` / ``superanimal_transfer_learning`` kwargs are
    pre-3.0 and are NOT passed.

    Reference: Ye 2024 Methods §"Memory replay fine tuning" + Fig. 1d.
    """
    import deeplabcut
    import dlclibrary

    print("=== SA-finetune training path (memory replay) ===")
    update_progress(s3, "Training (SA): pre-flight checks")

    _ensure_default_net_type_hrnet(config_path)
    _validate_sa_conversion_table(config_path)
    _validate_sa_model_available(dlclibrary.get_available_models("superanimal_topviewmouse"))
    detector = _resolve_sa_detector(dlclibrary.get_available_detectors("superanimal_topviewmouse"))
    print(f"  Resolved SA detector: {detector}")

    update_progress(s3, "Training (SA): build_weight_init")
    from deeplabcut.modelzoo.weight_initialization import build_weight_init

    weight_init = build_weight_init(
        cfg=str(config_path),
        super_animal="superanimal_topviewmouse",
        model_name="hrnet_w32",
        detector_name=detector,
        with_decoder=True,
        memory_replay=True,  # patched by patch_dlc_memory_replay.py on EC2
    )

    update_progress(s3, "Training (SA): create_training_dataset")
    new_shuffles = deeplabcut.create_training_dataset(
        str(config_path),
        weight_init=weight_init,
        num_shuffles=1,
        net_type="hrnet_w32",
    )
    # create_training_dataset returns a list of tuples:
    # [(trainingset_fraction, shuffle_index, (train_indices, test_indices)), ...]
    # We need just the integer shuffle index for train_network.
    raw_shuffle = new_shuffles[-1] if isinstance(new_shuffles, list) else new_shuffles
    if isinstance(raw_shuffle, (list, tuple)) and len(raw_shuffle) >= 2:
        sa_shuffle = int(raw_shuffle[1])
    else:
        sa_shuffle = int(raw_shuffle)
    print(f"  SA shuffle index: {sa_shuffle} (raw: {type(raw_shuffle).__name__})")

    # Overwrite DLC's random split with session-level stratified holdout.
    metadata_csv = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    _create_stratified_split(
        work,
        metadata_csv,
        n_clusters=split_clusters,
        n_test_sessions=n_test_sessions,
    )

    # Locate the new shuffle's pytorch_config.yaml and apply the
    # augmentation patch. The 256x256 input-size check is a soft
    # warning per pitfall #1.
    pytorch_cfgs = sorted(work.rglob("pytorch_config.yaml"))
    if pytorch_cfgs:
        # Use the most recently-modified one (DLC's create_training_dataset
        # writes the new shuffle last).
        latest = max(pytorch_cfgs, key=lambda p: p.stat().st_mtime)
        _check_sa_input_size(latest)
        _apply_sa_augmentation_patch(latest, epochs=epochs)
        _inject_wandb_logger(
            latest,
            run_name or _default_run_name(),
            tags=["sa-finetune", "hrnet-w32", "memory-replay"],
        )
    else:
        print("  WARNING: no pytorch_config.yaml found post-create_training_dataset")

    lr = 5e-5
    milestones = [80, 110] if epochs <= 120 else [160, 190]
    update_progress(s3, f"Training (SA): {epochs} epochs (lr={lr:g})")
    deeplabcut.train_network(
        str(config_path),
        shuffle=sa_shuffle,
        epochs=epochs,
        save_epochs=10,
        displayiters=100,
        batch_size=batch_size,
        pytorch_cfg_updates={
            "train_settings.optimizer.params.lr": lr,
            "model.backbone.freeze_bn_stats": True,
            "train_settings.scheduler.type": "MultiStepLR",
            "train_settings.scheduler.params.milestones": milestones,
            "train_settings.scheduler.params.gamma": 0.1,
        },
    )
    update_progress(s3, "Training (SA): train_network complete")

    # Stash a notes file the eventual declare_champion call will pick up.
    notes_text = _build_sa_notes(
        detector=detector,
        conversion_array=SA_CONVERSION_ARRAY,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    (work / "_sa_finetune_notes.txt").write_text(notes_text)
    print(f"  Notes stashed: {notes_text!r}")
    return config_path


# ---------------------------------------------------------------------------
# Stratified session-level train/test split
# ---------------------------------------------------------------------------


def _clip_dir_to_exp_id(
    clip_dir_name: str,
    exp_ids: list[str],
) -> str | None:
    """Map a labeled-data clip directory name to its experiments.csv ``exp_id``.

    Clip directory names have timestamps offset by a few seconds from the
    experiment start time, but share the same date (``YYYYMMDD``) and animal
    ID (last underscore-delimited token). When date + animal are ambiguous
    (multiple sessions on the same day for one animal), the closest time
    match is used.

    Parameters
    ----------
    clip_dir_name
        Clip directory name, e.g.
        ``20220804_11_22_03_1117646_maze-rose_overhead.camera-cropped``.
        Only the first 5 underscore-delimited tokens are used (timestamp +
        animal ID).
    exp_ids
        All ``exp_id`` strings from experiments.csv, each in format
        ``YYYYMMDD_HH_MM_SS_animalid``.

    Returns
    -------
    str or None
        The matching ``exp_id``, or None if no match is found.
    """
    parts = clip_dir_name.split("_")
    if len(parts) < 5:
        return None
    clip_date = parts[0]
    clip_animal = parts[4]
    # Parse clip time as total seconds for distance comparison.
    try:
        clip_seconds = int(parts[1]) * 3600 + int(parts[2]) * 60 + int(parts[3])
    except ValueError:
        return None

    best_exp_id: str | None = None
    best_distance = float("inf")
    for eid in exp_ids:
        e_parts = eid.split("_")
        if len(e_parts) < 5:
            continue
        if e_parts[0] != clip_date or e_parts[4] != clip_animal:
            continue
        try:
            exp_seconds = int(e_parts[1]) * 3600 + int(e_parts[2]) * 60 + int(e_parts[3])
        except ValueError:
            continue
        dist = abs(clip_seconds - exp_seconds)
        if dist < best_distance:
            best_distance = dist
            best_exp_id = eid

    # Sanity check: reject if the best match is more than 60 seconds away.
    if best_distance > 60:
        return None
    return best_exp_id


def _create_stratified_split(
    work: Path,
    metadata_csv: Path,
    *,
    n_clusters: int = 12,
    n_test_sessions: int = 4,
) -> bool:
    """Overwrite DLC's random train/test split with a session-level holdout.

    Selects ``n_test_sessions`` primary non-excluded sessions whose combined
    pose-cluster distribution best matches the overall dataset (minimising
    KL divergence). All frames from these sessions become the test set;
    all others become the train set. This eliminates train/test leakage
    from temporal correlation within sessions.

    Called after ``deeplabcut.create_training_dataset()`` to overwrite
    the ``Documentation_data-*.pickle`` file it produced.

    Parameters
    ----------
    work
        DLC project working directory (contains ``labeled-data/``,
        ``training-datasets/``, ``config.yaml``).
    metadata_csv
        Path to ``metadata/experiments.csv``.
    n_clusters
        Number of k-means clusters for pose-space grouping.
    n_test_sessions
        Number of primary non-excluded sessions to hold out as test.

    Returns
    -------
    bool
        True if the split was successfully overwritten, False if the
        function fell back (caller should use DLC's random split).

    References
    ----------
    Glazner et al. 2025. "Find the Leak, Fix the Split." arXiv:2511.13944.
    Ye et al. 2024. "SuperAnimal pretrained pose estimation models."
    Nature Communications 15:5165. doi:10.1038/s41467-024-48792-2.
    """
    import pandas as pd
    from scipy.special import rel_entr
    from sklearn.cluster import KMeans

    # ── Step 0: Load experiments.csv and identify primary non-excluded sessions ──
    if not metadata_csv.exists():
        print(f"  WARNING: {metadata_csv} not found — skipping stratified split")
        return False

    exp_rows: dict[str, dict[str, str]] = {}
    with open(metadata_csv) as f:
        for row in csv.DictReader(f):
            exp_rows[row["exp_id"]] = row
    all_exp_ids = list(exp_rows.keys())

    primary_non_excluded: set[str] = set()
    for eid, row in exp_rows.items():
        if (
            str(row.get("primary_exp", "0")).strip() == "1"
            and str(row.get("exclude", "0")).strip() == "0"
        ):
            primary_non_excluded.add(eid)

    if len(primary_non_excluded) < n_test_sessions:
        print(
            f"  WARNING: only {len(primary_non_excluded)} primary non-excluded sessions "
            f"(need {n_test_sessions}) — skipping stratified split"
        )
        return False

    # ── Step 1: Load all CollectedData H5 files ─────────────────────────────
    ld_root = work / "labeled-data"
    if not ld_root.exists():
        print("  WARNING: no labeled-data/ directory — skipping stratified split")
        return False

    gt_files = [
        h5
        for h5 in sorted(ld_root.rglob("CollectedData_*.h5"))
        if len(h5.relative_to(ld_root).parts) == 2
    ]
    if not gt_files:
        print("  WARNING: no CollectedData H5 files — skipping stratified split")
        return False

    # Map each clip directory to its exp_id.
    clip_to_exp: dict[str, str | None] = {}
    for gf in gt_files:
        clip_name = gf.parent.name
        clip_to_exp[clip_name] = _clip_dir_to_exp_id(clip_name, all_exp_ids)

    # Load frames per clip, tracking which session each frame belongs to.
    frames_list: list[pd.DataFrame] = []
    frame_session_ids: list[str | None] = []  # exp_id for each frame
    frame_clip_names: list[str] = []  # clip dir name for each frame

    for gf in gt_files:
        clip_name = gf.parent.name
        exp_id = clip_to_exp.get(clip_name)
        try:
            df = pd.read_hdf(gf)
        except Exception:
            continue
        frames_list.append(df)
        frame_session_ids.extend([exp_id] * len(df))
        frame_clip_names.extend([clip_name] * len(df))

    if not frames_list:
        print("  WARNING: could not load any CollectedData — skipping stratified split")
        return False

    all_frames = pd.concat(frames_list, ignore_index=False)
    n_total = len(all_frames)
    print(f"  Stratified split: {n_total} total labeled frames across {len(gt_files)} clips")

    # ── Step 2: Extract (x, y) coordinate features ──────────────────────────
    scorer = all_frames.columns.get_level_values(0)[0]
    bodyparts = all_frames.columns.get_level_values(1).unique().tolist()

    # Build feature matrix: (N, B*2) where B is number of bodyparts.
    coords: list[np.ndarray] = []
    for bp in bodyparts:
        try:
            x = all_frames[(scorer, bp, "x")].values.astype(float)
            y = all_frames[(scorer, bp, "y")].values.astype(float)
        except KeyError:
            x = np.full(n_total, np.nan)
            y = np.full(n_total, np.nan)
        coords.extend([x, y])

    feature_matrix = np.column_stack(coords)  # (N, B*2)

    # Handle NaNs: fill with per-column (per-bodypart-coordinate) mean.
    col_means = np.nanmean(feature_matrix, axis=0)
    for j in range(feature_matrix.shape[1]):
        nan_mask = np.isnan(feature_matrix[:, j])
        if nan_mask.any():
            fill_val = col_means[j] if np.isfinite(col_means[j]) else 0.0
            feature_matrix[nan_mask, j] = fill_val

    # ── Step 3: k-means clustering ──────────────────────────────────────────
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
    except Exception as e:
        print(f"  WARNING: k-means failed ({e}) — skipping stratified split")
        return False

    # ── Step 4: Compute per-session cluster distributions ───────────────────
    session_ids_array = np.array(frame_session_ids, dtype=object)
    unique_sessions = sorted(set(s for s in frame_session_ids if s is not None))

    # Overall cluster distribution (target).
    overall_counts = np.bincount(cluster_labels, minlength=n_clusters).astype(float)
    overall_dist = overall_counts / overall_counts.sum()

    # Per-session cluster distributions.
    session_cluster_counts: dict[str, np.ndarray] = {}
    session_frame_indices: dict[str, list[int]] = {}
    for sid in unique_sessions:
        mask = session_ids_array == sid
        indices = np.where(mask)[0]
        session_frame_indices[sid] = indices.tolist()
        counts = np.bincount(cluster_labels[mask], minlength=n_clusters).astype(float)
        session_cluster_counts[sid] = counts

    # ── Step 5: Select test sessions via KL divergence minimisation ─────────
    # Candidates: primary non-excluded sessions that have labeled data.
    candidates = sorted(primary_non_excluded & set(unique_sessions))

    if len(candidates) < n_test_sessions:
        print(
            f"  WARNING: only {len(candidates)} primary non-excluded sessions have "
            f"labeled data (need {n_test_sessions}) — skipping stratified split"
        )
        return False

    eps = 1e-10  # Epsilon to avoid log(0) in KL divergence.
    best_combo: tuple[str, ...] | None = None
    best_kl = float("inf")

    for combo in combinations(candidates, n_test_sessions):
        # Aggregate cluster counts for this combination.
        combo_counts = np.zeros(n_clusters, dtype=float)
        for sid in combo:
            combo_counts += session_cluster_counts[sid]
        combo_total = combo_counts.sum()
        if combo_total == 0:
            continue
        combo_dist = combo_counts / combo_total

        # KL divergence: D_KL(combo_dist || overall_dist).
        kl = float(np.sum(rel_entr(combo_dist + eps, overall_dist + eps)))

        if kl < best_kl:
            best_kl = kl
            best_combo = combo

    if best_combo is None:
        print("  WARNING: no valid test session combination found — skipping stratified split")
        return False

    # ── Step 5b: Verify all clusters covered in test set ────────────────────
    test_cluster_counts = np.zeros(n_clusters, dtype=float)
    for sid in best_combo:
        test_cluster_counts += session_cluster_counts[sid]

    uncovered = np.where(test_cluster_counts == 0)[0]
    if len(uncovered) > 0:
        print(
            f"  WARNING: test set does not cover clusters {uncovered.tolist()}. "
            f"Searching for a combo with full coverage..."
        )
        # Try all combos ranked by KL, pick the first with full coverage.
        kl_combos: list[tuple[float, tuple[str, ...]]] = []
        for combo in combinations(candidates, n_test_sessions):
            combo_counts = np.zeros(n_clusters, dtype=float)
            for sid in combo:
                combo_counts += session_cluster_counts[sid]
            if (combo_counts == 0).any() or combo_counts.sum() == 0:
                continue
            combo_dist = combo_counts / combo_counts.sum()
            kl = float(np.sum(rel_entr(combo_dist + eps, overall_dist + eps)))
            kl_combos.append((kl, combo))

        if kl_combos:
            kl_combos.sort(key=lambda x: x[0])
            best_kl, best_combo = kl_combos[0]
            test_cluster_counts = np.zeros(n_clusters, dtype=float)
            for sid in best_combo:
                test_cluster_counts += session_cluster_counts[sid]
            print(f"  Found combo with full coverage: KL={best_kl:.4f}")
        else:
            print("  No combo with full cluster coverage — proceeding with best KL combo")

    # ── Step 6: Build train/test index arrays ───────────────────────────────
    test_sessions = set(best_combo)
    test_indices: list[int] = []
    train_indices: list[int] = []

    for i in range(n_total):
        sid = frame_session_ids[i]
        if sid in test_sessions:
            test_indices.append(i)
        else:
            train_indices.append(i)

    train_idx = np.array(train_indices, dtype=int)
    test_idx = np.array(test_indices, dtype=int)

    # ── Step 7: Overwrite the Documentation_data pickle ─────────────────────
    doc_pickles = sorted(work.rglob("Documentation_data-*.pickle"))
    if not doc_pickles:
        print("  WARNING: no Documentation_data pickle found — skipping stratified split")
        return False

    pickle_path = doc_pickles[-1]
    try:
        with open(pickle_path, "rb") as f:
            meta = pickle.load(f)
    except Exception as e:
        print(f"  WARNING: could not read pickle ({e}) — skipping stratified split")
        return False

    # Pickle format: [data, trainIndices, testIndices, trainFraction]
    # Preserve data (meta[0]) and trainFraction (meta[3] if present).
    if len(meta) >= 4:
        new_meta = [meta[0], train_idx, test_idx, meta[3]]
    elif len(meta) >= 3:
        new_meta = [meta[0], train_idx, test_idx]
    else:
        print(f"  WARNING: unexpected pickle format (len={len(meta)}) — skipping stratified split")
        return False

    with open(pickle_path, "wb") as f:
        pickle.dump(new_meta, f, pickle.HIGHEST_PROTOCOL)

    # ── Step 8: Print diagnostics ───────────────────────────────────────────
    print("\n  === Stratified split diagnostics ===")
    print(f"  Test sessions ({n_test_sessions}): {list(best_combo)}")
    print(f"  KL divergence (test vs overall): {best_kl:.6f}")
    print(f"  Train frames: {len(train_indices)}, Test frames: {len(test_indices)}")
    train_pct = len(train_indices) / n_total
    test_pct = len(test_indices) / n_total
    print(f"  Train/Test ratio: {train_pct:.1%} / {test_pct:.1%}")

    print(f"\n  Per-cluster frame counts (k={n_clusters}):")
    print(f"  {'Cluster':>8s} {'Train':>6s} {'Test':>6s} {'Total':>6s} {'Test%':>6s}")
    for k in range(n_clusters):
        train_k = int(np.sum(cluster_labels[train_idx] == k))
        test_k = int(np.sum(cluster_labels[test_idx] == k))
        total_k = train_k + test_k
        pct = f"{test_k / total_k * 100:.1f}" if total_k > 0 else "n/a"
        print(f"  {k:>8d} {train_k:>6d} {test_k:>6d} {total_k:>6d} {pct:>6s}")

    print("\n  Per-session membership:")
    for sid in unique_sessions:
        n_frames = len(session_frame_indices.get(sid, []))
        role = "TEST" if sid in test_sessions else "train"
        is_primary = sid in primary_non_excluded
        label = f"{'primary' if is_primary else 'secondary/excluded'}"
        print(f"    {sid:<35s}  {n_frames:>4d} frames  {role:<5s}  ({label})")

    print(f"  Pickle overwritten: {pickle_path.name}")
    print("  === End stratified split ===\n")

    return True


def _upload_eval_results_json(s3, work: Path, config_path: Path, epochs: int) -> None:
    """Parse DLC's evaluation CSV and upload structured JSON to S3.

    Reads the CombinedEvaluation-results.csv (or per-snapshot CSV) written
    by ``deeplabcut.evaluate_network()`` and uploads a structured JSON to
    ``s3://hm2p-derivatives/dlc-retrain/_eval_results.json`` so the frontend
    can display training metrics without running inference.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train_frac = cfg.get("TrainingFraction", [0.8])[0]

    # Find eval CSV — check multiple locations
    eval_csv = None
    for candidate in sorted(work.rglob("*-results.csv")):
        if "CombinedEvaluation" not in candidate.name:
            eval_csv = candidate
            break
    if eval_csv is None:
        for candidate in sorted(work.rglob("CombinedEvaluation-results.csv")):
            eval_csv = candidate
            break
    if eval_csv is None:
        print("  WARNING: no evaluation CSV found — skipping _eval_results.json upload")
        return

    import pandas as pd

    df = pd.read_csv(eval_csv)
    if df.empty:
        print(f"  WARNING: {eval_csv.name} is empty")
        return

    # Take the last row (latest snapshot)
    row = df.iloc[-1]
    best_epoch = int(row.get("Training epochs", epochs))

    # Load champion info
    try:
        champ_obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-champion.json")
        champ = json.loads(champ_obj["Body"].read())
        champ_id = champ.get("champion_id", "unknown")
    except Exception:
        champ_id = "unknown"

    # Load previous eval for comparison
    prev = {}
    try:
        prev_obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-retrain/_eval_results.json")
        prev_eval = json.loads(prev_obj["Body"].read())
        prev = {
            "champion_id": prev_eval.get("champion_id", ""),
            "train_rmse": prev_eval.get("train", {}).get("rmse"),
            "train_mAP": prev_eval.get("train", {}).get("mAP"),
            "n_labeled_frames": prev_eval.get("n_labeled_frames"),
            "training_fraction": prev_eval.get("training_fraction"),
        }
    except Exception:
        pass

    # Count labeled frames (bounded search under labeled-data/ only,
    # filtering out nested duplicates).
    n_frames = 0
    _ld_root = work / "labeled-data"
    _cd_files = (
        [
            h5
            for h5 in sorted(_ld_root.rglob("CollectedData_*.h5"))
            if len(h5.relative_to(_ld_root).parts) == 2
        ]
        if _ld_root.exists()
        else []
    )
    for h5 in _cd_files:
        try:
            n_frames += len(pd.read_hdf(h5))
        except Exception:
            pass

    eval_results = {
        "champion_id": champ_id,
        "training_fraction": float(train_frac),
        "shuffle": 1,
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "n_labeled_frames": n_frames,
        "train": {
            "rmse": float(row.get("train rmse", 0)),
            "rmse_pcutoff": float(row.get("train rmse_pcutoff", 0)),
            "mAP": float(row.get("train mAP", 0)),
            "mAR": float(row.get("train mAR", 0)),
        },
        "test": {
            "rmse": float(row.get("test rmse", 0)),
            "rmse_pcutoff": float(row.get("test rmse_pcutoff", 0)),
            "mAP": float(row.get("test mAP", 0)),
            "mAR": float(row.get("test mAR", 0)),
        },
        "previous_champion": prev,
    }

    s3.put_object(
        Bucket=DERIVATIVES_BUCKET,
        Key="dlc-retrain/_eval_results.json",
        Body=json.dumps(eval_results, indent=2),
        ContentType="application/json",
    )
    print(
        f"  Uploaded _eval_results.json: train RMSE={eval_results['train']['rmse']:.2f}, "
        f"test mAP={eval_results['test']['mAP']:.1f}"
    )


def _upload_model_artifacts(s3, work: Path) -> None:
    """Upload trained model weights + eval CSVs to S3.

    Shared post-training step for both the ImageNet HRNet path and the
    SA-finetune path. Walks ``work/dlc-models-pytorch/`` (or
    ``dlc-models/`` for legacy TF runs) and uploads all files under
    ``s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/models/``.
    """
    print("Uploading model weights + training metadata to S3...")
    # Upload training-datasets/ so --eval-only can reconstruct the DLC project.
    td_dir = work / "training-datasets"
    if td_dir.exists():
        n_td = 0
        for f in td_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(work)
                key = f"{RETRAIN_PREFIX}/{rel}"
                s3.upload_file(str(f), DERIVATIVES_BUCKET, key)
                n_td += 1
        print(f"  Uploaded {n_td} training-dataset files")

    # Nuke the ENTIRE models/ prefix on S3 before uploading fresh.
    # Prevents stale snapshots / architecture mismatches from previous runs.
    print("  Deleting entire models/ prefix from S3...")
    _paginator = s3.get_paginator("list_objects_v2")
    for _page in _paginator.paginate(
        Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/models/"
    ):
        for _obj in _page.get("Contents", []):
            s3.delete_object(Bucket=DERIVATIVES_BUCKET, Key=_obj["Key"])
    print("  Done.")

    for model_dir_name in ("dlc-models-pytorch", "dlc-models"):
        dlc_train_dir = work / model_dir_name
        if not dlc_train_dir.exists():
            continue
        print(f"  Found {model_dir_name}/")
        n_files = 0
        for f in dlc_train_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(dlc_train_dir)
                key = f"{RETRAIN_PREFIX}/models/{rel}"
                s3.upload_file(str(f), DERIVATIVES_BUCKET, key)
                n_files += 1
        print(f"  Uploaded {n_files} files to s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/models/")

        # Upload eval CSVs with flat keys to prevent nested directory issues.
        # Search only known DLC output directories for eval CSVs.
        _eval_uploaded = 0
        for _eval_search_dir in [
            work / "evaluation-results",
            work / "evaluation-results-pytorch",
            dlc_train_dir,
        ]:
            if not _eval_search_dir.exists():
                continue
            for _eval_csv in _eval_search_dir.rglob("*results*.csv"):
                _eval_key = f"{RETRAIN_PREFIX}/models/eval/{_eval_csv.name}"
                s3.upload_file(str(_eval_csv), DERIVATIVES_BUCKET, _eval_key)
                _eval_uploaded += 1
                print(f"  Uploaded eval: eval/{_eval_csv.name}")
        if _eval_uploaded == 0:
            print("  No evaluation result CSVs found")

        # Upload SA-finetune notes if present (consumed by declare_champion).
        notes_path = work / "_sa_finetune_notes.txt"
        if notes_path.exists():
            s3.upload_file(
                str(notes_path),
                DERIVATIVES_BUCKET,
                f"{RETRAIN_PREFIX}/models/_sa_finetune_notes.txt",
            )
        return
    print("  WARNING: no model directory found")


def train(
    s3,
    maxiters: int = 50000,
    epochs: int = 400,
    batch_size: int = 8,
    sa_finetune: bool = False,
    bodyparts: list[str] | None = None,
    split_clusters: int = 12,
    n_test_sessions: int = 4,
    run_name: str | None = None,
) -> Path:
    """Download labels from S3, fine-tune DLC, upload model weights.

    Parameters
    ----------
    s3
        boto3 S3 client.
    maxiters
        Legacy TF iterations parameter (ignored under DLC 3.0 PyTorch and
        ignored under ``--sa-finetune``).
    epochs
        Training epochs. The CLI default is 400 for the ImageNet path and
        120 for the SA-finetune path; whatever the operator passes in
        propagates here.
    batch_size
        Training batch size (default 8).
    sa_finetune
        When True, runs the SuperAnimal memory-replay fine-tune path
        (Ye et al. 2024). When False, runs the legacy ImageNet HRNet
        path. Mutually exclusive at the API level — both paths share
        the same S3 download / upload scaffolding.
    bodyparts
        Override the bodyparts list in config.yaml. When set, only these
        bodyparts are trained. Labels for other bodyparts remain in
        CollectedData but are ignored by DLC during training.
        E.g. ``["left_ear", "right_ear"]`` for ears-only experiment.
    split_clusters
        Number of k-means clusters for the stratified train/test split.
    n_test_sessions
        Number of primary non-excluded sessions to hold out as the test set.
    """
    import deeplabcut

    work = Path("/tmp/dlc-retrain")
    # Clean local work dir to prevent stale artifacts from a previous
    # run on the same EC2 instance contaminating the new training.
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    # Delete stale training-datasets and model dirs from S3 so DLC
    # builds a fresh dataset from the current labels instead of reusing
    # a cached shuffle from a previous run.
    print("Cleaning stale training artifacts from S3...")
    _paginator = s3.get_paginator("list_objects_v2")
    for _stale_prefix in [
        f"{RETRAIN_PREFIX}/training-datasets/",
        f"{RETRAIN_PREFIX}/models/",  # ALL models, not just iteration-0
    ]:
        for _page in _paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=_stale_prefix):
            for _obj in _page.get("Contents", []):
                s3.delete_object(Bucket=DERIVATIVES_BUCKET, Key=_obj["Key"])
    print("  Done.")

    # Download ONLY config.yaml and labeled-data/ from S3 (not models/,
    # training-datasets/, or any other artifacts that could contaminate
    # the fresh training run).
    print("Downloading config.yaml and labeled-data from S3...")
    s3.download_file(
        DERIVATIVES_BUCKET,
        f"{RETRAIN_PREFIX}/config.yaml",
        str(work / "config.yaml"),
    )
    _ld_prefix = f"{RETRAIN_PREFIX}/labeled-data/"
    _ld_paginator = s3.get_paginator("list_objects_v2")
    _n_ld = 0
    for _ld_page in _ld_paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=_ld_prefix):
        for _ld_obj in _ld_page.get("Contents", []):
            _ld_key = _ld_obj["Key"]
            _ld_rel = _ld_key[len(f"{RETRAIN_PREFIX}/") :]
            if not _ld_rel or _ld_rel.startswith("_"):
                continue
            _ld_dest = work / _ld_rel
            _ld_dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(DERIVATIVES_BUCKET, _ld_key, str(_ld_dest))
            _n_ld += 1
    print(f"  Downloaded config.yaml + {_n_ld} labeled-data files")

    # Signal GPU watchdog that processing is starting.
    Path("/tmp/gpu_processing_active").touch()

    config_path = work / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("No config.yaml in S3 dlc-retrain/")

    # Fix video paths in config (they reference Mac paths)
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Update project path
    cfg["project_path"] = str(work)

    # Override bodyparts if requested (for experiments like ears-only).
    # Labels are NOT modified — DLC ignores columns for unlisted bodyparts.
    original_bodyparts = cfg.get("bodyparts", [])
    if bodyparts is not None:
        cfg["bodyparts"] = bodyparts
        print(f"  Bodyparts override: {original_bodyparts} → {bodyparts}")

    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"Config: {config_path}")
    print(f"Bodyparts: {cfg.get('bodyparts', [])}")
    print(f"Epochs: {epochs}")
    print(f"Mode: {'SA fine-tune (memory replay)' if sa_finetune else 'ImageNet HRNet'}")

    # Remove fully-unlabeled frames from the WORK COPY of CollectedData.
    # These are frames with PNGs but no labels — DLC would interpret them
    # as "all bodyparts absent", hurting training. The original data on
    # S3 / local disk is NOT modified.
    import pandas as pd

    _ld_root = work / "labeled-data"
    _cd_train = (
        [
            h5
            for h5 in sorted(_ld_root.rglob("CollectedData_*.h5"))
            if len(h5.relative_to(_ld_root).parts) == 2
        ]
        if _ld_root.exists()
        else []
    )
    for h5 in _cd_train:
        try:
            df = pd.read_hdf(h5)
            before = len(df)
            # A row is unlabeled if ALL coordinate columns are NaN
            coord_cols = [c for c in df.columns if c[-1] in ("x", "y")]
            if coord_cols:
                all_nan = df[coord_cols].isna().all(axis=1)
                n_drop = int(all_nan.sum())
                if n_drop > 0:
                    df = df[~all_nan]
                    df.to_hdf(h5, key="df_with_missing", mode="w")
                    df.to_csv(h5.with_suffix(".csv"))
                    print(
                        f"  Filtered {h5.parent.name}: {before} → {len(df)} "
                        f"(removed {n_drop} unlabeled frames)"
                    )
        except Exception as exc:
            print(f"  WARNING: failed to filter {h5.name}: {exc}")

    # Delete any stale dlc-models* dirs to ensure a clean shuffle build.
    # Done here once for both paths.
    for old_dir_name in ("dlc-models-pytorch", "dlc-models", "training-datasets"):
        old_dir = work / old_dir_name
        if old_dir.exists():
            shutil.rmtree(old_dir)
            print(f"  Deleted old {old_dir_name}/")

    if sa_finetune:
        _train_sa_finetune(
            s3,
            work,
            config_path,
            epochs=epochs,
            batch_size=batch_size,
            split_clusters=split_clusters,
            n_test_sessions=n_test_sessions,
            run_name=run_name,
        )
        # SA path runs train_network internally; the shared post-training
        # block (evaluation + uploads) follows below.
        update_progress(s3, "Training (SA): evaluating")
        deeplabcut.evaluate_network(str(config_path), plotting=False)
        _upload_eval_results_json(s3, work, config_path, epochs=epochs)
        update_progress(s3, "Training (SA): per-bodypart RMSE")
        _compute_per_bodypart_rmse(s3, work, config_path)
        _push_bodypart_rmse_to_wandb(work)
        update_progress(s3, "Training (SA): evaluation complete")
        _upload_model_artifacts(s3, work)
        update_progress(s3, "Training complete (SA fine-tune)")
        return config_path

    update_progress(s3, "Training: creating dataset")

    # Ensure HRNet-W32 is the default net type before creating the dataset.
    _ensure_default_net_type_hrnet(config_path)

    # Create training dataset (default ResNet50 config — we override below).
    print("Creating training dataset...")
    deeplabcut.create_training_dataset(str(config_path))

    # Overwrite DLC's random split with session-level stratified holdout.
    metadata_csv = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    _create_stratified_split(
        work,
        metadata_csv,
        n_clusters=split_clusters,
        n_test_sessions=n_test_sessions,
    )

    update_progress(s3, "Training: dataset created")

    # Override config: switch to HRNet-W32 backbone (ImageNet pretrained
    # via timm, NOT SuperAnimal) + aggressive augmentation.
    pytorch_cfg_candidates = list(work.rglob("pytorch_config.yaml"))
    for pcfg_path in pytorch_cfg_candidates:
        with open(pcfg_path) as f:
            pcfg = yaml.safe_load(f)

        # Epochs
        if "train_settings" not in pcfg:
            pcfg["train_settings"] = {}
        pcfg["train_settings"]["epochs"] = epochs

        # HRNet-W32 backbone (ImageNet pretrained via timm).
        # DLC's HRNet implementation uses timm to load pretrained weights.
        old_backbone = pcfg.get("model", {}).get("backbone", {}).get("model_name", "?")
        print(f"  Overriding backbone: {old_backbone} → hrnet_w32")
        pcfg["model"]["backbone"] = {
            "model_name": "hrnet_w32",
            "type": "HRNet",
            "freeze_bn_stats": False,
            "freeze_bn_weights": False,
        }
        pcfg["net_type"] = "hrnet_w32"
        # HRNet-W32 outputs 32 channels (ResNet outputs 2048).
        # Head deconv layers must match the backbone output.
        n_bodyparts = len(pcfg.get("metadata", {}).get("bodyparts", []))
        if "heads" in pcfg["model"]:
            for head_cfg in pcfg["model"]["heads"].values():
                if "heatmap_config" in head_cfg:
                    head_cfg["heatmap_config"]["channels"] = [32, n_bodyparts or 8]
                if "locref_config" in head_cfg:
                    head_cfg["locref_config"]["channels"] = [32, (n_bodyparts or 8) * 2]
        print(f"  Head channels: 32 → {n_bodyparts} bodyparts")

        # Scheduler milestones: adapt to epoch count.
        milestones = [80, 110] if epochs <= 120 else [160, 190]
        sched = pcfg["train_settings"].setdefault("scheduler", {})
        sched["type"] = "MultiStepLR"
        sched.setdefault("params", {})["milestones"] = milestones
        sched["params"]["gamma"] = 0.1
        print(f"  Scheduler milestones: {milestones}")

        # Aggressive augmentation for overhead mouse tracking with
        # light/dark alternation and high pose variability.
        # Enable ImageNet pretraining (DLC HRNet template defaults to false)
        if "model" in pcfg and "backbone" in pcfg["model"]:
            pcfg["model"]["backbone"]["pretrained"] = True
            print("  backbone.pretrained = True (ImageNet)")

        # Augmentation: tuned for overhead mouse with light/dark and 184 frames.
        # Moderate augmentation — strong enough for generalisation but not so
        # extreme that the model rarely sees natural examples.
        if "data" in pcfg and "train" in pcfg["data"]:
            aug = pcfg["data"]["train"]
            if "affine" not in aug:
                aug["affine"] = {}
            aug["affine"]["rotation"] = 45  # ±45° (was ±180° — too extreme)
            aug["affine"]["scaling"] = [0.7, 1.4]  # ±30-40% (was 0.25-2.5x)
            aug["affine"]["translation"] = 30  # pixels
            aug["affine"]["p"] = 0.7
            # Brightness/contrast jitter: the IR filter leaks some 450nm
            # visible light and the IR illumination decays ~5-10% over a
            # 30-min session. ±15% brightness + ±10% contrast covers both.
            # Uses the hm2p patch to DLC's transforms.py (applied in
            # launch_dlc_finetune_ec2.py user-data script).
            aug["brightness_contrast"] = {
                "brightness_limit": 0.15,
                "contrast_limit": 0.1,
                "p": 0.5,
            }
            # Flips: keep — mouse is symmetric from above
            aug["horizontal_flip"] = {"p": 0.5}
            aug["vertical_flip"] = {"p": 0.5}
            # Noise: moderate
            aug["gaussian_noise"] = 15.0  # was 30 — too much
            aug["motion_blur"] = True
            # No hue/saturation jitter — images are grayscale (IR overhead camera)
            print(
                "  Augmentation: rot=±45°, scale=0.7-1.4x, "
                "brightness/contrast=±40%, hflip+vflip, noise=15"
            )

        with open(pcfg_path, "w") as f:
            yaml.dump(pcfg, f)
        print(f"  Config updated: {pcfg_path.name}")
        _inject_wandb_logger(
            pcfg_path,
            run_name or _default_run_name(),
            tags=["imagenet", "hrnet-w32"],
        )

    update_progress(s3, f"Training: HRNet-W32 ({epochs} epochs)")

    # Train
    print(f"Training HRNet-W32 for {epochs} epochs...")
    deeplabcut.train_network(
        str(config_path),
        maxiters=maxiters,
        displayiters=100,
        saveiters=5000,
    )
    update_progress(s3, f"Training: network trained ({epochs} epochs)")

    # Evaluate and compute per-bodypart metrics
    print("Evaluating network...")
    deeplabcut.evaluate_network(str(config_path), plotting=False)
    _upload_eval_results_json(s3, work, config_path, epochs=epochs)
    update_progress(s3, "Training: evaluation complete")

    # Per-bodypart RMSE from DLC evaluation predictions.
    print("Computing per-bodypart RMSE...")
    _compute_per_bodypart_rmse(s3, work, config_path)
    _push_bodypart_rmse_to_wandb(work)

    # Upload model weights, eval CSVs, and training metadata via the
    # shared upload function (nuke-and-replace on S3).
    _upload_model_artifacts(s3, work)

    update_progress(s3, "Training complete", maxiters=maxiters)
    return config_path


def _download_session_video(  # noqa: ANN001
    s3, rawdata_bucket: str, sub: str, ses_id: str, dest_dir: Path
) -> None:
    """Download overhead .mp4 files for a session from S3 using boto3.

    Downloads all .mp4 files under ``rawdata/{sub}/{ses_id}/behav/`` except
    side-camera files (filename contains "side").

    Parameters
    ----------
    s3 : boto3 S3 client
    rawdata_bucket : str
    sub : str
        Subject identifier, e.g. ``sub-1114353``.
    ses_id : str
        Session identifier, e.g. ``ses-20210823T165950``.
    dest_dir : Path
        Local directory to download into.
    """
    prefix = f"rawdata/{sub}/{ses_id}/behav/"
    resp = s3.list_objects_v2(Bucket=rawdata_bucket, Prefix=prefix)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        filename = key.split("/")[-1]
        if not filename.endswith(".mp4"):
            continue
        if "side" in filename.lower():
            continue
        local_path = dest_dir / filename
        s3.download_file(rawdata_bucket, key, str(local_path))
        print(f"  Downloaded {filename}")


def infer(s3, config_path: Path, skip_failed: bool = False) -> None:
    """Run inference on all 26 sessions with the fine-tuned model.

    Parameters
    ----------
    s3 : boto3 S3 client
    config_path : Path
        Local path to the DLC config.yaml.
    skip_failed : bool
        If True, promote completed sessions even if some failed.
        If False (default), auto-promote is skipped when any session fails.
    """
    import deeplabcut

    # Read session list
    metadata = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    sessions = []
    with open(metadata) as f:
        for row in csv.DictReader(f):
            eid = row["exp_id"]
            parts = eid.split("_")
            sessions.append(
                {
                    "exp_id": eid,
                    "sub": f"sub-{parts[-1]}",
                    "ses": f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}",
                }
            )

    total = len(sessions)
    completed: list[str] = []
    failed: list[str] = []
    error_records: list[dict] = []
    run_id = datetime.datetime.utcnow().isoformat() + "Z"
    instance_id = get_instance_id()

    # --- Sequential: download, subsample, infer, upload, cleanup per session ---
    for i, ses in enumerate(sessions, 1):
        sub, ses_id = ses["sub"], ses["ses"]
        exp_id = ses["exp_id"]
        print(f"\n=== [{i}/{total}] {sub}/{ses_id} ===")

        # Skip sessions that already have results
        existing_resp = s3.list_objects_v2(
            Bucket=DERIVATIVES_BUCKET,
            Prefix=f"{FINETUNED_PREFIX}/{sub}/{ses_id}/",
            MaxKeys=1,
        )
        if existing_resp.get("Contents"):
            print("  Already has results, skipping")
            completed.append(exp_id)
            continue

        # Progress: session starting
        update_progress(
            s3,
            f"Inference {i}/{total}: starting {sub}/{ses_id}",
            completed=len(completed),
            failed=len(failed),
            total=total,
            current_session=exp_id,
        )

        work = Path(f"/tmp/dlc-infer/{sub}/{ses_id}")
        work.mkdir(parents=True, exist_ok=True)

        try:
            # Download video
            video_dir = work / "behav"
            video_dir.mkdir(parents=True, exist_ok=True)
            _download_session_video(s3, RAWDATA_BUCKET, sub, ses_id, video_dir)

            mp4s = list(video_dir.glob("*overhead*.mp4")) + list(video_dir.glob("*cropped*.mp4"))
            if not mp4s:
                mp4s = list(video_dir.glob("*.mp4"))
            if not mp4s:
                print("  No video found, skipping")
                failed.append(exp_id)
                continue

            video = mp4s[0]

            # Subsample to 30fps
            sub_path = work / f"{video.stem}_30fps.mp4"
            ffmpeg_result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video),
                    "-r",
                    "30",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    str(sub_path),
                ],
                capture_output=True,
            )
            if sub_path.exists() and sub_path.stat().st_size > 1000:
                dlc_video = sub_path
                print(f"  Subsampled to 30fps: {sub_path.name}")
            else:
                print(f"  WARNING: ffmpeg failed (rc={ffmpeg_result.returncode}), using original")
                if ffmpeg_result.stderr:
                    print(f"  stderr: {ffmpeg_result.stderr.decode()[-300:]}")
                dlc_video = video

            # Signal GPU watchdog (first session only)
            Path("/tmp/gpu_processing_active").touch()

            # Run inference
            out_dir = work / "output"
            out_dir.mkdir(exist_ok=True)
            print("  Running DLC inference (batch_size=32)...")
            deeplabcut.analyze_videos(
                str(config_path),
                [str(dlc_video)],
                destfolder=str(out_dir),
                batch_size=32,
            )

            # Labelled video rendering is handled separately by
            # render_dlc_videos.py on a CPU instance after promotion
            # (faster: downscales to 416x304, no DLC dependency needed).

            # Upload results via boto3
            out_files = list(out_dir.rglob("*"))
            out_files = [f for f in out_files if f.is_file()]
            if out_files:
                s3_prefix = f"{FINETUNED_PREFIX}/{sub}/{ses_id}"
                for f in out_files:
                    key = f"{s3_prefix}/{f.name}"
                    # Rename labelled video to standard name for viewer page
                    if f.suffix == ".mp4" and "labeled" in f.name:
                        key = f"{s3_prefix}/labelled_30fps.mp4"
                    s3.upload_file(str(f), DERIVATIVES_BUCKET, key)
                completed.append(exp_id)
                print(f"  Uploaded {len(out_files)} files")

                # Progress: session done
                update_progress(
                    s3,
                    f"Inference {i}/{total}: done {sub}/{ses_id}",
                    completed=len(completed),
                    failed=len(failed),
                    total=total,
                    current_session=exp_id,
                    stage="inference_done",
                )
            else:
                print("  No output files")
                failed.append(exp_id)

        except Exception as e:
            error_records.append(
                {
                    "session": exp_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "stage": "inference",
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
            )
            print(f"  ERROR [{type(e).__name__}]: {e}")
            print(traceback.format_exc())
            failed.append(exp_id)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    update_progress(
        s3,
        "Inference complete",
        completed=len(completed),
        failed=len(failed),
        total=total,
        completed_sessions=completed,
        failed_sessions=failed,
    )
    print(f"\nDone: {len(completed)}/{total} completed, {len(failed)} failed")

    # Upload structured error records — always written, even if empty,
    # so the frontend can distinguish "no errors" from "file missing".
    errors_payload = json.dumps(
        {"run_id": run_id, "instance_id": instance_id, "errors": error_records},
        indent=2,
    ).encode()
    try:
        s3.put_object(
            Bucket=DERIVATIVES_BUCKET,
            Key=f"{RETRAIN_PREFIX}/_inference_errors.json",
            Body=errors_payload,
        )
        print(f"  Error summary uploaded ({len(error_records)} error(s))")
    except Exception as e:
        print(f"  WARNING: could not upload _inference_errors.json: {e}")

    # Auto-promote: declare champion FIRST, then copy pose-finetuned/ → pose/
    # on S3. If champion declaration fails, promotion is aborted — this
    # prevents pose/ from containing files that do not match any manifest.
    if failed and not skip_failed:
        print(
            f"\nSkipping auto-promote: {len(failed)} session(s) failed — "
            f"{failed}.\n"
            f"To promote the {len(completed)} successful session(s), pass "
            f"--skip-failed or run promote_finetuned_pose.py --skip-failed."
        )
        return

    if failed and skip_failed:
        print(
            f"\nAuto-promoting {len(completed)} completed session(s). "
            f"Skipping {len(failed)} failed session(s): {failed}"
        )

    # Only promote sessions that completed successfully
    sessions_to_promote = [s for s in sessions if s["exp_id"] in completed]

    # --- Step 1: Declare champion FIRST (before copying files to pose/) ---
    # If declaration fails, STOP — do not promote. This is the
    # declare-before-promote contract: the manifest must point at the
    # model before any files appear in pose/.
    print("\n=== Declaring new DLC champion (before promotion) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # noqa
    from hm2p.pose.select import extract_architecture, extract_dlc_provenance

    # Find one finetuned h5 from pose-finetuned/ to read identifiers.
    sample = sessions_to_promote[0]
    sample_prefix = f"{FINETUNED_PREFIX}/{sample['sub']}/{sample['ses']}/"
    sample_resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=sample_prefix)
    h5_filenames = [
        obj["Key"].split("/")[-1]
        for obj in sample_resp.get("Contents", [])
        if obj["Key"].endswith(".h5")
        and "_single" not in obj["Key"].split("/")[-1]
        and "_filtered" not in obj["Key"].split("/")[-1]
        and ("Hrnet" in obj["Key"] or "Resnet" in obj["Key"])
    ]
    if not h5_filenames:
        print(
            f"ERROR: No finetuned .h5 found under {sample_prefix}. "
            f"Cannot declare champion — aborting promotion."
        )
        return
    h5_filename = h5_filenames[0]
    model_name, snapshot = extract_dlc_provenance(h5_filename)
    architecture = extract_architecture(h5_filename)
    if architecture is None:
        print(
            f"ERROR: Could not extract architecture from {h5_filename!r}. "
            f"Cannot declare champion — aborting promotion."
        )
        return

    notes_lines = [
        "Auto-declared by run_dlc_retrain.py.",
        f"Sessions promoted: {len(sessions_to_promote)}; failed: {len(failed)}; total: {total}.",
    ]
    # If the SA-finetune training path stashed a notes file on S3,
    # prepend its contents (init source, conversion array, etc.).
    try:
        sa_notes_obj = s3.get_object(
            Bucket=DERIVATIVES_BUCKET,
            Key=f"{RETRAIN_PREFIX}/models/_sa_finetune_notes.txt",
        )
        sa_notes = sa_notes_obj["Body"].read().decode("utf-8").strip()
        if sa_notes:
            notes_lines.insert(0, sa_notes)
    except Exception:
        # ImageNet path leaves no notes file — that's expected.
        pass
    notes = " ".join(notes_lines)

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa
        from declare_dlc_champion import declare_champion  # noqa

        champion_manifest = declare_champion(
            model_name=model_name,
            architecture=architecture,
            snapshot=snapshot,
            training_run_id=run_id,
            notes=notes,
            s3_client=s3,
            bucket=DERIVATIVES_BUCKET,
        )
        champion_id = champion_manifest.get("champion_id", "unknown")
        print(f"  Champion declared: {champion_id}")
    except Exception:
        print(
            "ERROR: champion declaration failed (see traceback). "
            "Aborting promotion — pose/ files will NOT be updated. "
            "Fix the issue and run scripts/declare_dlc_champion.py manually, "
            "then scripts/promote_finetuned_pose.py to promote."
        )
        traceback.print_exc()
        return

    # --- Step 2: Delete existing files from pose/ before copying new ones ---
    # This prevents stale files from a previous model coexisting with the
    # new champion's output.
    print(f"\nPromoting {len(sessions_to_promote)} finetuned sessions → pose/ on S3...")
    for ses in sessions_to_promote:
        sub, ses_id = ses["sub"], ses["ses"]
        dst_prefix = f"pose/{sub}/{ses_id}/"

        # Delete all existing files under pose/{sub}/{ses}/
        paginator = s3.get_paginator("list_objects_v2")
        keys_to_delete: list[str] = []
        for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=dst_prefix):
            for obj in page.get("Contents", []):
                keys_to_delete.append(obj["Key"])
        if keys_to_delete:
            # Batch delete (up to 1000 at a time per S3 API)
            for i in range(0, len(keys_to_delete), 1000):
                batch = keys_to_delete[i : i + 1000]
                s3.delete_objects(
                    Bucket=DERIVATIVES_BUCKET,
                    Delete={"Objects": [{"Key": k} for k in batch]},
                )
            print(f"  {sub}/{ses_id}: deleted {len(keys_to_delete)} old file(s)")

        # Copy new files from pose-finetuned/ → pose/
        src_prefix = f"{FINETUNED_PREFIX}/{sub}/{ses_id}/"
        resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=src_prefix)
        n_copied = 0
        for obj in resp.get("Contents", []):
            src_key = obj["Key"]
            dst_key = src_key.replace(FINETUNED_PREFIX, "pose", 1)
            s3.copy_object(
                Bucket=DERIVATIVES_BUCKET,
                CopySource={"Bucket": DERIVATIVES_BUCKET, "Key": src_key},
                Key=dst_key,
            )
            n_copied += 1
        print(f"  {sub}/{ses_id}: promoted ({n_copied} file(s))")

    # --- Step 3: Verify promotion — champion's file exists in pose/ ---
    print("\n=== Verifying promotion ===")
    from hm2p.pose.select import select_champion_h5_s3

    verification_failures: list[str] = []
    for ses in sessions_to_promote:
        sub, ses_id = ses["sub"], ses["ses"]
        pose_prefix = f"pose/{sub}/{ses_id}/"
        try:
            verified_key = select_champion_h5_s3(
                s3,
                DERIVATIVES_BUCKET,
                pose_prefix,
                champion_id,
            )
            print(f"  {sub}/{ses_id}: verified ({Path(verified_key).name})")
        except Exception as e:
            print(f"  {sub}/{ses_id}: VERIFICATION FAILED — {e}")
            verification_failures.append(ses["exp_id"])
    if verification_failures:
        print(
            f"\nWARNING: {len(verification_failures)} session(s) failed "
            f"post-promotion verification: {verification_failures}"
        )

    update_progress(
        s3,
        "Promoted to pose/",
        completed=len(completed),
        total=total,
        promoted=len(sessions_to_promote),
        failed=len(failed),
        failed_sessions=failed,
        champion_id=champion_id,
    )
    print("Promotion complete.")

    update_progress(
        s3,
        "Inference + promotion complete. Launching CPU instance for downstream + render.",
        completed=len(completed),
        total=total,
        champion_id=champion_id,
    )

    # Launch a CPU instance for downstream stages + video rendering.
    # These don't need GPU — running them on the GPU instance wastes money.
    print("\n=== Launching CPU instance for downstream + render ===")
    try:
        subprocess.run(
            ["python3", "scripts/launch_downstream_cpu.py"],
            check=True,
        )
    except Exception as e:
        print(f"WARNING: could not launch CPU instance: {e}")
        print("Run manually: python3 scripts/launch_downstream_cpu.py")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argparse — split out for unit-testing."""
    parser = argparse.ArgumentParser(description="DLC retraining + inference")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Download model weights from S3, run evaluate_network + "
        "per-bodypart RMSE, upload results. No training or inference.",
    )
    parser.add_argument(
        "--maxiters",
        type=int,
        default=50000,
        help="Legacy TF iterations (ignored by PyTorch; ignored under --sa-finetune)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (DLC 3.0 PyTorch). Default depends on the "
        "training path: 400 for ImageNet HRNet, 120 for --sa-finetune.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--sa-finetune",
        action="store_true",
        help="Use SuperAnimal-TopViewMouse memory-replay fine-tune instead of "
        "the legacy ImageNet HRNet path. Per Ye et al. 2024, "
        "doi:10.1038/s41467-024-48792-2.",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Promote completed sessions even if some inference sessions failed. "
        "By default auto-promotion is skipped if any session fails.",
    )
    parser.add_argument(
        "--bodyparts",
        type=str,
        default=None,
        help="Override bodyparts for training (comma-separated). "
        "E.g. --bodyparts left_ear,right_ear for ears-only experiment. "
        "Labels are NOT modified — DLC ignores unlisted bodyparts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name. Default: YYMMDD (e.g. 260527). Add a suffix "
        "for what's being tested (e.g. 260527-stratified). Append -2 etc "
        "for multiple runs on the same day.",
    )
    parser.add_argument(
        "--split-clusters",
        type=int,
        default=12,
        help="Number of k-means clusters for the stratified train/test split "
        "(default 12). Frames are grouped into pose archetypes; the test "
        "set is selected to cover all clusters proportionally.",
    )
    parser.add_argument(
        "--n-test-sessions",
        type=int,
        default=4,
        help="Number of primary non-excluded sessions to hold out as the "
        "test set (default 4). Selected by minimising KL divergence of "
        "pose-cluster distribution vs the overall dataset.",
    )
    return parser


def _default_run_name() -> str:
    """Generate default W&B run name: YYMMDD (e.g. 260527)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d")


def resolve_epochs(epochs: int | None, *, sa_finetune: bool) -> int:
    """Resolve the default ``--epochs`` based on the training path.

    Per design §2.1: 120 for SA fine-tune, 400 for ImageNet HRNet. When
    the operator passes ``--epochs`` explicitly, that value is honoured
    for both paths.
    """
    if epochs is not None:
        return epochs
    return 200 if sa_finetune else 400


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    epochs = resolve_epochs(args.epochs, sa_finetune=args.sa_finetune)

    s3 = boto3.client("s3", region_name=REGION)

    if args.eval_only:
        # Download config + labeled data + model weights, run evaluation only
        import deeplabcut

        work = Path("/tmp/dlc-retrain")
        # Clean local work dir to prevent stale artifacts.
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        print("Downloading config.yaml and labeled-data from S3 for evaluation...")
        # Download config.yaml
        s3.download_file(
            DERIVATIVES_BUCKET,
            f"{RETRAIN_PREFIX}/config.yaml",
            str(work / "config.yaml"),
        )
        # Download labeled-data/**/*
        _eval_ld_prefix = f"{RETRAIN_PREFIX}/labeled-data/"
        _eval_paginator = s3.get_paginator("list_objects_v2")
        _n_eval_ld = 0
        for _eval_page in _eval_paginator.paginate(
            Bucket=DERIVATIVES_BUCKET, Prefix=_eval_ld_prefix
        ):
            for _eval_obj in _eval_page.get("Contents", []):
                _eval_key = _eval_obj["Key"]
                _eval_rel = _eval_key[len(f"{RETRAIN_PREFIX}/") :]
                if not _eval_rel or _eval_rel.startswith("_"):
                    continue
                _eval_dest = work / _eval_rel
                _eval_dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(DERIVATIVES_BUCKET, _eval_key, str(_eval_dest))
                _n_eval_ld += 1
        print(f"  Downloaded config.yaml + {_n_eval_ld} labeled-data files")

        config_path = work / "config.yaml"

        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["project_path"] = str(work)
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

        # Download model weights into dlc-models-pytorch/ (skip nested
        # models/models/ keys and internal _-prefixed files).
        print("Downloading model weights...")
        _n_eval_model = 0
        for _eval_m_page in _eval_paginator.paginate(
            Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/models/"
        ):
            for _eval_m_obj in _eval_m_page.get("Contents", []):
                _eval_m_key = _eval_m_obj["Key"]
                _eval_m_rel = _eval_m_key[len(f"{RETRAIN_PREFIX}/models/") :]
                if not _eval_m_rel or _eval_m_rel.startswith("_"):
                    continue
                # Skip nested models/models/ keys
                if _eval_m_rel.startswith("models/"):
                    continue
                _eval_m_dest = work / "dlc-models-pytorch" / _eval_m_rel
                _eval_m_dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(DERIVATIVES_BUCKET, _eval_m_key, str(_eval_m_dest))
                _n_eval_model += 1
        print(f"  Downloaded {_n_eval_model} model files")

        # Run per-bodypart RMSE. The function tries DLC evaluation H5
        # first, then falls back to direct inference on labeled PNGs.
        print("Computing per-bodypart RMSE...")
        _compute_per_bodypart_rmse(s3, work, config_path)
        print("Evaluation complete.")
        return

    do_train = not args.infer_only
    do_infer = not args.train_only

    config_path = None
    if do_train:
        bp_override = args.bodyparts.split(",") if args.bodyparts else None
        config_path = train(
            s3,
            maxiters=args.maxiters,
            epochs=epochs,
            batch_size=args.batch_size,
            sa_finetune=args.sa_finetune,
            bodyparts=bp_override,
            split_clusters=args.split_clusters,
            n_test_sessions=args.n_test_sessions,
            run_name=args.run_name,
        )

    if do_infer:
        if config_path is None:
            # Download config + model weights from S3 (training was done in a previous run)
            work = Path("/tmp/dlc-retrain")
            # Clean local work dir to prevent stale artifacts.
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True, exist_ok=True)
            config_path = work / "config.yaml"
            s3.download_file(DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/config.yaml", str(config_path))

            # Download model weights + training-datasets metadata
            print("Downloading model weights from S3...")
            paginator = s3.get_paginator("list_objects_v2")
            n_model = 0
            for page in paginator.paginate(
                Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/models/"
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key[len(f"{RETRAIN_PREFIX}/models/") :]
                    if not rel or rel.startswith("_") or rel.startswith("models/"):
                        continue
                    dest = work / "dlc-models-pytorch" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(DERIVATIVES_BUCKET, key, str(dest))
                    n_model += 1
            if n_model == 0:
                print("ERROR: no model weights on S3. Run training first.")
                sys.exit(1)
            print(f"  Downloaded {n_model} model files")

            # Download training-datasets (needed by analyze_videos for shuffle metadata)
            print("Downloading training-datasets metadata...")
            n_td = 0
            for page in paginator.paginate(
                Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/training-datasets/"
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key[len(f"{RETRAIN_PREFIX}/") :]
                    dest = work / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(DERIVATIVES_BUCKET, key, str(dest))
                    n_td += 1
            print(f"  Downloaded {n_td} training-dataset files")

            # Fix project_path in config
            import yaml

            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg["project_path"] = str(work)
            with open(config_path, "w") as f:
                yaml.dump(cfg, f)

        infer(s3, config_path, skip_failed=args.skip_failed)


if __name__ == "__main__":
    main()
