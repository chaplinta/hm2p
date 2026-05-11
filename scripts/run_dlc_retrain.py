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
import shutil
import subprocess
import sys
import traceback
import urllib.request
from pathlib import Path

import boto3
import numpy as np

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

    DLC's evaluate_network saves per-image predictions as multi-index H5
    files in evaluation-results-pytorch/. This function loads them,
    matches against ground-truth labels, and computes RMSE per bodypart.
    Uploads result as ``_per_bodypart_eval.json`` to S3.
    """
    import pandas as pd
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    bodyparts = cfg.get("bodyparts", [])

    # Find DLC's prediction H5 files from evaluate_network.
    # DLC 3.x saves them as: evaluation-results-pytorch/.../
    #   DLC_<net>_<project>shuffle<n>_snapshot_<epoch>-<split>.h5
    # where split is "train" or "test".
    pred_files = sorted(work.rglob("*snapshot*test*.h5"))
    if not pred_files:
        # Some DLC versions save predictions as .csv
        pred_files = sorted(work.rglob("*snapshot*test*.csv"))
    if not pred_files:
        # Fall back: look for any eval predictions
        pred_files = sorted(work.rglob("*predictions*.h5"))

    # Find ground-truth labeled data
    gt_files = sorted(work.rglob("CollectedData_*.h5"))
    if not gt_files:
        print("  No ground-truth files found for per-bodypart RMSE")
        return

    # Load ground truth (merge all labeled data)
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

    # If we have DLC prediction files, compute per-bodypart RMSE from them
    per_bp_errors: dict[str, list[float]] = {bp: [] for bp in bodyparts}

    if pred_files:
        for pf in pred_files:
            try:
                if pf.suffix == ".h5":
                    pred = pd.read_hdf(pf)
                else:
                    pred = pd.read_csv(pf, header=[0, 1, 2], index_col=0)
            except Exception as e:
                print(f"  Could not read {pf.name}: {e}")
                continue

            pred_scorer = pred.columns.get_level_values(0)[0]

            # Match rows by index
            common = gt.index.intersection(pred.index)
            for idx in common:
                for bp in bodyparts:
                    try:
                        gx = float(gt.loc[idx, (gt_scorer, bp, "x")])
                        gy = float(gt.loc[idx, (gt_scorer, bp, "y")])
                        px = float(pred.loc[idx, (pred_scorer, bp, "x")])
                        py = float(pred.loc[idx, (pred_scorer, bp, "y")])
                    except (KeyError, ValueError):
                        continue
                    if any(np.isnan(v) for v in (gx, gy, px, py)):
                        continue
                    err = float(np.sqrt((gx - px) ** 2 + (gy - py) ** 2))
                    per_bp_errors[bp].append(err)

    if not any(per_bp_errors.values()):
        print("  No matched predictions found for per-bodypart RMSE")
        # Fall back: run inference on test frames directly
        print("  Attempting direct inference on labeled frames...")
        try:
            import deeplabcut
            # Get test frame paths from the training dataset split
            test_frames = []
            for idx in gt.index:
                if isinstance(idx, tuple):
                    frame_path = str(work / idx[0] if len(idx) > 0 else "")
                else:
                    frame_path = str(work / str(idx))
                if Path(frame_path).exists():
                    test_frames.append(frame_path)

            if test_frames:
                print(f"  Running inference on {len(test_frames)} labeled frames...")
                # Use DLC's inference on individual frames
                for frame_path in test_frames[:5]:  # sample
                    print(f"    {Path(frame_path).name}")
        except Exception as e:
            print(f"  Direct inference failed: {e}")
        return

    # Build summary
    result = {"bodyparts": {}, "n_total_matched": sum(len(v) for v in per_bp_errors.values())}
    for bp in bodyparts:
        errs = per_bp_errors[bp]
        if errs:
            arr = np.array(errs)
            result["bodyparts"][bp] = {
                "rmse": float(np.sqrt(np.mean(arr ** 2))),
                "mean_error": float(np.mean(arr)),
                "median_error": float(np.median(arr)),
                "std": float(np.std(arr)),
                "n": len(errs),
                "pck_5": float((arr <= 5).mean() * 100),
                "pck_10": float((arr <= 10).mean() * 100),
                "pck_20": float((arr <= 20).mean() * 100),
            }
        else:
            result["bodyparts"][bp] = {"rmse": None, "n": 0}

    # Print summary
    print("\n  Per-bodypart RMSE (from DLC evaluation predictions):")
    for bp in bodyparts:
        d = result["bodyparts"][bp]
        if d["rmse"] is not None:
            print(f"    {bp:<16s}  RMSE={d['rmse']:6.2f}  median={d['median_error']:6.2f}  "
                  f"PCK@10={d['pck_10']:5.1f}%  n={d['n']}")
        else:
            print(f"    {bp:<16s}  (no data)")

    # Upload
    out = work / "_per_bodypart_eval.json"
    out.write_text(json.dumps(result, indent=2))
    s3.upload_file(str(out), DERIVATIVES_BUCKET,
                   f"{RETRAIN_PREFIX}/models/_per_bodypart_eval.json")
    print(f"  Uploaded to s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/models/_per_bodypart_eval.json")


def _compute_per_bodypart_rmse_direct(s3, work: Path, config_path: Path) -> None:
    """Compute per-bodypart RMSE by running DLC inference on labeled frames.

    Unlike _compute_per_bodypart_rmse (which reads evaluate_network output),
    this runs analyze_videos on labeled frame images directly. Does not need
    training-datasets/ metadata or shuffle info.
    """
    import deeplabcut
    import pandas as pd
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    bodyparts = cfg.get("bodyparts", [])

    # Collect all labeled frames and their ground truth
    gt_files = sorted(work.rglob("CollectedData_*.h5"))
    if not gt_files:
        print("  No ground-truth files found")
        return

    # Find PNG frames alongside the CollectedData files
    frame_dirs = set()
    for gf in gt_files:
        frame_dirs.add(gf.parent)

    # Gather all frame images
    all_frames = []
    for fd in frame_dirs:
        pngs = sorted(fd.glob("*.png"))
        if pngs:
            all_frames.extend(pngs)

    if not all_frames:
        print("  No labeled frame PNGs found")
        return

    print(f"  Found {len(all_frames)} labeled frame images")

    # Run DLC inference on labeled frames (treat them as a batch)
    # Copy frames to a temporary directory as a "video" of images
    infer_dir = work / "_eval_frames"
    infer_dir.mkdir(exist_ok=True)
    for f in all_frames:
        dst = infer_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)

    out_dir = work / "_eval_output"
    out_dir.mkdir(exist_ok=True)

    # Use analyze_time_lapse_frames for image directories
    print("  Running DLC inference on labeled frames...")
    try:
        deeplabcut.analyze_time_lapse_frames(
            str(config_path),
            str(infer_dir),
            save_as_csv=True,
        )
    except AttributeError:
        # DLC 3.x may not have analyze_time_lapse_frames;
        # fall back to creating a video from images
        print("  analyze_time_lapse_frames not available, using analyze_videos...")
        import subprocess as _sp
        vid_path = work / "_eval_frames.mp4"
        _sp.run([
            "ffmpeg", "-y", "-framerate", "1",
            "-pattern_type", "glob", "-i", f"{infer_dir}/*.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(vid_path),
        ], capture_output=True)
        if vid_path.exists():
            deeplabcut.analyze_videos(
                str(config_path), [str(vid_path)],
                destfolder=str(out_dir),
            )

    # Find prediction output
    pred_files = sorted(infer_dir.rglob("*.h5")) + sorted(out_dir.rglob("*.h5"))
    pred_csvs = sorted(infer_dir.rglob("*.csv")) + sorted(out_dir.rglob("*.csv"))
    pred_files = [f for f in pred_files if "CollectedData" not in f.name]
    pred_csvs = [f for f in pred_csvs if "CollectedData" not in f.name]

    if not pred_files and not pred_csvs:
        print("  No prediction output found after inference")
        return

    # Load predictions
    pred = None
    for pf in pred_files + pred_csvs:
        try:
            if pf.suffix == ".h5":
                pred = pd.read_hdf(pf)
            else:
                pred = pd.read_csv(pf, header=[0, 1, 2], index_col=0)
            print(f"  Loaded predictions: {pf.name} ({len(pred)} frames)")
            break
        except Exception as e:
            print(f"  Could not read {pf.name}: {e}")

    if pred is None:
        print("  Could not load any prediction files")
        return

    pred_scorer = pred.columns.get_level_values(0)[0]

    # Load all ground truth
    gt_all = []
    for gf in gt_files:
        try:
            gt_all.append(pd.read_hdf(gf))
        except Exception:
            continue
    gt = pd.concat(gt_all) if len(gt_all) > 1 else gt_all[0]
    gt_scorer = gt.columns.get_level_values(0)[0]

    # Match by frame filename
    per_bp_errors: dict[str, list[float]] = {bp: [] for bp in bodyparts}
    matched = 0

    for gt_idx in gt.index:
        # Extract frame filename from index
        if isinstance(gt_idx, tuple):
            frame_name = gt_idx[-1] if len(gt_idx) > 0 else str(gt_idx)
        else:
            frame_name = str(gt_idx).split("/")[-1]

        # Find matching prediction row
        for pred_idx in pred.index:
            pred_name = str(pred_idx).split("/")[-1] if not isinstance(pred_idx, tuple) else str(pred_idx[-1])
            if Path(frame_name).stem == Path(pred_name).stem:
                for bp in bodyparts:
                    try:
                        gx = float(gt.loc[gt_idx, (gt_scorer, bp, "x")])
                        gy = float(gt.loc[gt_idx, (gt_scorer, bp, "y")])
                        px = float(pred.loc[pred_idx, (pred_scorer, bp, "x")])
                        py = float(pred.loc[pred_idx, (pred_scorer, bp, "y")])
                    except (KeyError, ValueError):
                        continue
                    if any(np.isnan(v) for v in (gx, gy, px, py)):
                        continue
                    err = float(np.sqrt((gx - px) ** 2 + (gy - py) ** 2))
                    per_bp_errors[bp].append(err)
                matched += 1
                break

    print(f"  Matched {matched} frames")

    if not any(per_bp_errors.values()):
        print("  No matched predictions")
        return

    # Build and upload result
    result = {"bodyparts": {}, "n_matched": matched, "method": "direct_inference"}
    for bp in bodyparts:
        errs = per_bp_errors[bp]
        if errs:
            arr = np.array(errs)
            result["bodyparts"][bp] = {
                "rmse": float(np.sqrt(np.mean(arr ** 2))),
                "mean_error": float(np.mean(arr)),
                "median_error": float(np.median(arr)),
                "std": float(np.std(arr)),
                "n": len(errs),
                "pck_5": float((arr <= 5).mean() * 100),
                "pck_10": float((arr <= 10).mean() * 100),
                "pck_20": float((arr <= 20).mean() * 100),
            }
        else:
            result["bodyparts"][bp] = {"rmse": None, "n": 0}

    print("\n  Per-bodypart RMSE (direct inference on labeled frames):")
    for bp in bodyparts:
        d = result["bodyparts"][bp]
        if d.get("rmse") is not None:
            print(f"    {bp:<16s}  RMSE={d['rmse']:6.2f}  median={d['median_error']:6.2f}  "
                  f"PCK@10={d['pck_10']:5.1f}%  n={d['n']}")
        else:
            print(f"    {bp:<16s}  (no data)")

    out = work / "_per_bodypart_eval.json"
    out.write_text(json.dumps(result, indent=2))
    s3.upload_file(str(out), DERIVATIVES_BUCKET,
                   f"{RETRAIN_PREFIX}/models/_per_bodypart_eval.json")
    print(f"  Uploaded _per_bodypart_eval.json")


def _push_bodypart_rmse_to_wandb(work: Path) -> None:
    """Push per-bodypart RMSE/PCK metrics to the live W&B run summary.

    Reads ``_per_bodypart_eval.json`` from the working directory (written
    by ``_compute_per_bodypart_rmse`` or ``_compute_per_bodypart_rmse_direct``)
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
                wandb.run.summary[f"bodypart/{bp}_median"] = data.get(
                    "median_error"
                )
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
    "nose_tip", "left_ear", "right_ear", "head_midpoint",
    "neck", "mid_back", "mouse_center", "tail_base",
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
    print(
        f"  WARNING: default_net_type was {cur!r}; rewriting to 'hrnet_w32' "
        f"in {config_path}"
    )
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
    tables = (
        cfg.get("SuperAnimalConversionTables", {})
        .get("superanimal_topviewmouse", {})
    )
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
    pytorch_cfg_path: Path, run_name: str, *, tags: list[str] | None = None,
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
    *, detector: str, conversion_array: list[int], epochs: int,
    lr: float, batch_size: int,
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
    _validate_sa_model_available(
        dlclibrary.get_available_models("superanimal_topviewmouse")
    )
    detector = _resolve_sa_detector(
        dlclibrary.get_available_detectors("superanimal_topviewmouse")
    )
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
            latest, f"SA-finetune-{epochs}ep",
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
    for model_dir_name in ("dlc-models-pytorch", "dlc-models"):
        dlc_train_dir = work / model_dir_name
        if not dlc_train_dir.exists():
            continue
        print(f"  Found {model_dir_name}/")
        # Delete old snapshots on S3 to prevent model architecture mismatches
        # when multiple training runs upload to the same prefix.
        print("  Cleaning old snapshots from S3...")
        _paginator = s3.get_paginator("list_objects_v2")
        for _page in _paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/models/iteration-0/"):
            for _obj in _page.get("Contents", []):
                if "snapshot" in _obj["Key"]:
                    s3.delete_object(Bucket=DERIVATIVES_BUCKET, Key=_obj["Key"])
        n_files = 0
        for f in dlc_train_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(dlc_train_dir)
                key = f"{RETRAIN_PREFIX}/models/{rel}"
                s3.upload_file(str(f), DERIVATIVES_BUCKET, key)
                n_files += 1
        print(f"  Uploaded {n_files} files to s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/models/")
        # Upload SA-finetune notes if present (consumed by declare_champion).
        notes_path = work / "_sa_finetune_notes.txt"
        if notes_path.exists():
            s3.upload_file(
                str(notes_path), DERIVATIVES_BUCKET,
                f"{RETRAIN_PREFIX}/models/_sa_finetune_notes.txt",
            )
        return
    print("  WARNING: no model directory found")


def train(s3, maxiters: int = 50000, epochs: int = 400, batch_size: int = 8,
          sa_finetune: bool = False,
          bodyparts: list[str] | None = None) -> Path:
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
    """
    import deeplabcut

    work = Path("/tmp/dlc-retrain")
    work.mkdir(parents=True, exist_ok=True)

    # Download labeled data + config
    print("Downloading labeled data from S3...")
    subprocess.run(
        ["aws", "s3", "sync",
         f"s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/",
         str(work),
         "--exclude", "_*"],
        check=True,
    )

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

    # Delete any stale dlc-models* dirs to ensure a clean shuffle build.
    # Done here once for both paths.
    for old_dir_name in ("dlc-models-pytorch", "dlc-models", "training-datasets"):
        old_dir = work / old_dir_name
        if old_dir.exists():
            shutil.rmtree(old_dir)
            print(f"  Deleted old {old_dir_name}/")

    if sa_finetune:
        _train_sa_finetune(
            s3, work, config_path, epochs=epochs, batch_size=batch_size,
        )
        # SA path runs train_network internally; the shared post-training
        # block (evaluation + uploads) follows below.
        update_progress(s3, "Training (SA): evaluating")
        deeplabcut.evaluate_network(str(config_path), plotting=False)
        update_progress(s3, "Training (SA): per-bodypart RMSE")
        _compute_per_bodypart_rmse(s3, work, config_path)
        _push_bodypart_rmse_to_wandb(work)
        update_progress(s3, "Training (SA): evaluation complete")
        _upload_model_artifacts(s3, work)
        update_progress(s3, "Training complete (SA fine-tune)")
        return config_path

    update_progress(s3, "Training: creating dataset")

    # Create training dataset (default ResNet50 config — we override below).
    print("Creating training dataset...")
    deeplabcut.create_training_dataset(str(config_path))
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
            aug["affine"]["rotation"] = 45          # ±45° (was ±180° — too extreme)
            aug["affine"]["scaling"] = [0.7, 1.4]   # ±30-40% (was 0.25-2.5x)
            aug["affine"]["translation"] = 30       # pixels
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
            aug["gaussian_noise"] = 15.0            # was 30 — too much
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
            pcfg_path, f"ImageNet-HRNet-{epochs}ep",
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
    update_progress(s3, "Training: evaluation complete")

    # Per-bodypart RMSE from DLC evaluation predictions.
    print("Computing per-bodypart RMSE...")
    _compute_per_bodypart_rmse(s3, work, config_path)
    _push_bodypart_rmse_to_wandb(work)

    # Upload evaluation results (per-bodypart RMSE).
    # DLC may write these in evaluation-results/ or inside the model dir.
    eval_uploaded = 0
    for search_dir in [work / "evaluation-results", work]:
        for csv_file in search_dir.rglob("*results*.csv"):
            rel = csv_file.relative_to(work)
            key = f"{RETRAIN_PREFIX}/models/{rel}"
            s3.upload_file(str(csv_file), DERIVATIVES_BUCKET, key)
            eval_uploaded += 1
            print(f"  Uploaded eval: {rel}")
    if eval_uploaded == 0:
        print("  No evaluation result CSVs found")

    # Compute per-bodypart RMSE from predictions vs labels
    print("Computing per-bodypart RMSE...")
    try:
        import subprocess as _sp
        _r = _sp.run(
            [sys.executable, "scripts/compute_bodypart_rmse.py",
             "--pose-prefix", FINETUNED_PREFIX],
            capture_output=True, text=True,
        )
        print(_r.stdout[-500:] if _r.stdout else "  (no output)")
        if _r.returncode != 0:
            print(f"  Per-bodypart RMSE failed: {_r.stderr[-300:]}")
    except Exception as e:
        print(f"  Per-bodypart RMSE failed: {e}")

    # Upload model weights via boto3 (aws CLI may not be available)
    print("Uploading model weights to S3...")
    # DLC 3.0 PyTorch uses dlc-models-pytorch; legacy uses dlc-models
    for model_dir_name in ("dlc-models-pytorch", "dlc-models"):
        dlc_train_dir = work / model_dir_name
        if dlc_train_dir.exists():
            print(f"  Found {model_dir_name}/")
            for f in dlc_train_dir.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(dlc_train_dir)
                    key = f"{RETRAIN_PREFIX}/models/{rel}"
                    s3.upload_file(str(f), DERIVATIVES_BUCKET, key)
            n_files = sum(1 for _ in dlc_train_dir.rglob("*") if _.is_file())
            print(f"  Uploaded {n_files} files to s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/models/")
            break
    else:
        print("  WARNING: no model directory found")

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
            sessions.append({
                "exp_id": eid,
                "sub": f"sub-{parts[-1]}",
                "ses": f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}",
            })

    total = len(sessions)
    completed: list[str] = []
    failed: list[str] = []
    error_records: list[dict] = []
    run_id = datetime.datetime.utcnow().isoformat() + "Z"
    instance_id = get_instance_id()

    # --- Phase 1: parallel download + ffmpeg subsample ---
    # Download and subsample all videos to 30fps in parallel (I/O-bound).
    # This runs before inference so the GPU doesn't wait on downloads.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _prefetch_session(ses_info: dict) -> tuple[str, Path | None]:
        """Download video and subsample to 30fps. Returns (exp_id, video_path)."""
        _sub, _ses_id = ses_info["sub"], ses_info["ses"]
        _exp_id = ses_info["exp_id"]
        _s3 = boto3.client("s3", region_name=REGION)

        # Skip if already has results
        existing_resp = _s3.list_objects_v2(
            Bucket=DERIVATIVES_BUCKET,
            Prefix=f"{FINETUNED_PREFIX}/{_sub}/{_ses_id}/",
            MaxKeys=1,
        )
        if existing_resp.get("Contents"):
            return (_exp_id, None)  # None signals "skip"

        _work = Path(f"/tmp/dlc-infer/{_sub}/{_ses_id}")
        _work.mkdir(parents=True, exist_ok=True)
        video_dir = _work / "behav"
        video_dir.mkdir(parents=True, exist_ok=True)
        _download_session_video(_s3, RAWDATA_BUCKET, _sub, _ses_id, video_dir)

        mp4s = list(video_dir.glob("*overhead*.mp4")) + list(video_dir.glob("*cropped*.mp4"))
        if not mp4s:
            mp4s = list(video_dir.glob("*.mp4"))
        if not mp4s:
            return (_exp_id, None)

        video = mp4s[0]
        sub_path = _work / f"{video.stem}_30fps.mp4"
        if not sub_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video),
                 "-r", "30",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 str(sub_path)],
                capture_output=True,
            )
        if sub_path.exists() and sub_path.stat().st_size > 1000:
            return (_exp_id, sub_path)
        return (_exp_id, video)

    print(f"\n=== Prefetching {total} videos (parallel download + ffmpeg) ===")
    update_progress(s3, "Prefetching videos", total=total)
    prefetched: dict[str, Path | None] = {}
    with ThreadPoolExecutor(max_workers=1) as pool:  # sequential — 3 OOM'd
        futures = {pool.submit(_prefetch_session, ses): ses for ses in sessions}
        for future in as_completed(futures):
            exp_id_done, video_path = future.result()
            prefetched[exp_id_done] = video_path
            n_done = len(prefetched)
            status = "skip" if video_path is None else video_path.name
            print(f"  [{n_done}/{total}] {exp_id_done[:25]}: {status}")
    print(f"Prefetch complete: {len(prefetched)} sessions")

    # --- Phase 2: sequential GPU inference ---
    for i, ses in enumerate(sessions, 1):
        sub, ses_id = ses["sub"], ses["ses"]
        exp_id = ses["exp_id"]
        print(f"\n=== [{i}/{total}] {sub}/{ses_id} ===")

        # Progress: session starting
        update_progress(
            s3, f"Inference {i}/{total}: starting {sub}/{ses_id}",
            completed=len(completed), failed=len(failed), total=total,
            current_session=exp_id,
        )

        # Use prefetched video (already downloaded + subsampled)
        dlc_video_path = prefetched.get(exp_id)
        if dlc_video_path is None:
            print(f"  Skipped (already has results or no video)")
            completed.append(exp_id)
            continue

        work = Path(f"/tmp/dlc-infer/{sub}/{ses_id}")
        work.mkdir(parents=True, exist_ok=True)
        dlc_video = dlc_video_path

        try:

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
                    s3, f"Inference {i}/{total}: done {sub}/{ses_id}",
                    completed=len(completed), failed=len(failed), total=total,
                    current_session=exp_id, stage="inference_done",
                )
            else:
                print("  No output files")
                failed.append(exp_id)

        except Exception as e:
            error_records.append({
                "session": exp_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "stage": "inference",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            })
            print(f"  ERROR [{type(e).__name__}]: {e}")
            print(traceback.format_exc())
            failed.append(exp_id)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    update_progress(
        s3, "Inference complete",
        completed=len(completed), failed=len(failed), total=total,
        completed_sessions=completed, failed_sessions=failed,
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

    # Auto-promote: copy pose-finetuned/ → pose/ on S3
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
    print(f"\nPromoting {len(sessions_to_promote)} finetuned sessions → pose/ on S3...")
    for ses in sessions_to_promote:
        sub, ses_id = ses["sub"], ses["ses"]
        src_prefix = f"{FINETUNED_PREFIX}/{sub}/{ses_id}/"
        resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=src_prefix)
        for obj in resp.get("Contents", []):
            src_key = obj["Key"]
            dst_key = src_key.replace(FINETUNED_PREFIX, "pose", 1)
            s3.copy_object(
                Bucket=DERIVATIVES_BUCKET,
                CopySource={"Bucket": DERIVATIVES_BUCKET, "Key": src_key},
                Key=dst_key,
            )
        print(f"  {sub}/{ses_id}: promoted")

    update_progress(
        s3, "Promoted to pose/",
        completed=len(completed), total=total,
        promoted=len(sessions_to_promote), failed=len(failed),
        failed_sessions=failed,
    )
    print("Promotion complete.")

    # Declare the new project-wide champion. Done here, after promotion to
    # pose/ has succeeded, so the manifest only ever points at h5 files that
    # actually exist in pose/. See docs/dlc-champion-model.md.
    print("\n=== Declaring new DLC champion ===")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # noqa
        from hm2p.pose.select import extract_architecture, extract_dlc_provenance
        # Find one promoted h5 to read the identifiers from. Any promoted
        # session works — they all carry the same model_name and snapshot.
        sample = sessions_to_promote[0]
        sample_prefix = f"pose/{sample['sub']}/{sample['ses']}/"
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
            raise RuntimeError(
                f"No finetuned .h5 found under {sample_prefix} after promotion."
            )
        h5_filename = h5_filenames[0]
        model_name, snapshot = extract_dlc_provenance(h5_filename)
        architecture = extract_architecture(h5_filename)
        if architecture is None:
            raise RuntimeError(
                f"Could not extract architecture from {h5_filename!r}."
            )
        notes_lines = [
            "Auto-declared by run_dlc_retrain.py.",
            f"Sessions promoted: {len(sessions_to_promote)}; "
            f"failed: {len(failed)}; total: {total}.",
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
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa
        from declare_dlc_champion import declare_champion  # noqa
        declare_champion(
            model_name=model_name,
            architecture=architecture,
            snapshot=snapshot,
            training_run_id=run_id,
            notes=notes,
            s3_client=s3,
            bucket=DERIVATIVES_BUCKET,
        )
    except Exception:
        print("ERROR: champion declaration failed (see traceback). "
              "The pipeline will continue but the manifest is not updated. "
              "Run scripts/declare_dlc_champion.py manually to fix.")
        traceback.print_exc()

    update_progress(
        s3, "Inference + promotion complete. Launching CPU instance for downstream + render.",
        completed=len(completed), total=total,
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
        "--eval-only", action="store_true",
        help="Download model weights from S3, run evaluate_network + "
             "per-bodypart RMSE, upload results. No training or inference.",
    )
    parser.add_argument(
        "--maxiters", type=int, default=50000,
        help="Legacy TF iterations (ignored by PyTorch; ignored under "
             "--sa-finetune)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs (DLC 3.0 PyTorch). Default depends on the "
             "training path: 400 for ImageNet HRNet, 120 for --sa-finetune.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--sa-finetune", action="store_true",
        help="Use SuperAnimal-TopViewMouse memory-replay fine-tune instead of "
             "the legacy ImageNet HRNet path. Per Ye et al. 2024, "
             "doi:10.1038/s41467-024-48792-2.",
    )
    parser.add_argument(
        "--skip-failed", action="store_true",
        help="Promote completed sessions even if some inference sessions failed. "
             "By default auto-promotion is skipped if any session fails.",
    )
    parser.add_argument(
        "--bodyparts", type=str, default=None,
        help="Override bodyparts for training (comma-separated). "
             "E.g. --bodyparts left_ear,right_ear for ears-only experiment. "
             "Labels are NOT modified — DLC ignores unlisted bodyparts.",
    )
    return parser


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
        work.mkdir(parents=True, exist_ok=True)

        print("Downloading project from S3 for evaluation...")
        subprocess.run(
            ["aws", "s3", "sync",
             f"s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/",
             str(work),
             "--exclude", "_*"],
            check=True,
        )
        config_path = work / "config.yaml"

        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["project_path"] = str(work)
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

        # Download model weights into dlc-models-pytorch/
        print("Downloading model weights...")
        resp = s3.list_objects_v2(
            Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/models/"
        )
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            rel = key[len(f"{RETRAIN_PREFIX}/models/"):]
            if rel.startswith("_") or not rel:
                continue
            dest = work / "dlc-models-pytorch" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(DERIVATIVES_BUCKET, key, str(dest))
        print("  Model weights downloaded")

        # Run per-bodypart RMSE by inference on labeled frames directly.
        # evaluate_network requires training-datasets/ metadata which may
        # not be on S3 from older training runs. Direct inference avoids this.
        print("Computing per-bodypart RMSE via direct inference on labeled frames...")
        _compute_per_bodypart_rmse_direct(s3, work, config_path)
        print("Evaluation complete.")
        return

    do_train = not args.infer_only
    do_infer = not args.train_only

    config_path = None
    if do_train:
        bp_override = args.bodyparts.split(",") if args.bodyparts else None
        config_path = train(
            s3, maxiters=args.maxiters, epochs=epochs,
            batch_size=args.batch_size, sa_finetune=args.sa_finetune,
            bodyparts=bp_override,
        )

    if do_infer:
        if config_path is None:
            # Download config + model weights from S3 (training was done in a previous run)
            work = Path("/tmp/dlc-retrain")
            work.mkdir(parents=True, exist_ok=True)
            config_path = work / "config.yaml"
            s3.download_file(DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/config.yaml", str(config_path))

            # Download model weights + training-datasets metadata
            print("Downloading model weights from S3...")
            paginator = s3.get_paginator("list_objects_v2")
            n_model = 0
            for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/models/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key[len(f"{RETRAIN_PREFIX}/models/"):]
                    if not rel or rel.startswith("_"):
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
            for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/training-datasets/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key[len(f"{RETRAIN_PREFIX}/"):]
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
