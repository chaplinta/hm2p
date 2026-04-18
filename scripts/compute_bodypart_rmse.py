#!/usr/bin/env python3
"""Compute per-bodypart RMSE from DLC predictions vs ground truth labels.

Loads labeled data (local) and model predictions (S3), matches frames,
computes per-bodypart pixel error, and uploads results as JSON to S3.

Usage:
    uv run python scripts/compute_bodypart_rmse.py
    uv run python scripts/compute_bodypart_rmse.py --pose-prefix pose-finetuned
    uv run python scripts/compute_bodypart_rmse.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
METADATA_PATH = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
LABELED_DIR = Path(
    "sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data"
)
RETRAIN_DIR = Path("metadata/retrain_frames")

BODYPARTS = [
    "nose_tip", "left_ear", "right_ear", "head_midpoint",
    "neck", "mid_back", "mouse_center", "tail_base",
]

RAW_FPS = 100.0
DLC_FPS = 30.0


def _clip_to_sub_ses(clip_name: str) -> tuple[str, str] | None:
    """Map clip dir name to (sub, ses) by closest time match."""
    parts = clip_name.split("_")
    if len(parts) < 5:
        return None
    date = parts[0]
    clip_time = int(parts[1] + parts[2] + parts[3])
    animal = parts[4].split("-")[0]

    candidates = []
    for f in RETRAIN_DIR.glob("*.json"):
        fp = f.stem.split("_")
        if len(fp) < 2:
            continue
        f_animal = fp[0].replace("sub-", "")
        f_ses = fp[1].replace("ses-", "")
        f_date = f_ses[:8]
        if f_animal == animal and f_date == date:
            f_time = int(f_ses[9:])
            candidates.append((abs(f_time - clip_time), fp[0], fp[1]))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def main():
    parser = argparse.ArgumentParser(description="Compute per-bodypart RMSE")
    parser.add_argument("--pose-prefix", default="pose",
                        help="S3 prefix for predictions (pose or pose-finetuned)")
    parser.add_argument("--output-key", default="dlc-retrain/models/_bodypart_rmse.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)

    # Collect per-bodypart errors across all sessions
    bp_errors: dict[str, list[float]] = {bp: [] for bp in BODYPARTS}
    bp_errors_by_session: dict[str, dict[str, list[float]]] = {}
    total_frames = 0
    total_matched = 0

    for d in sorted(LABELED_DIR.iterdir()):
        if not d.is_dir():
            continue
        h5 = d / "CollectedData_tristan.h5"
        if not h5.exists():
            continue
        try:
            gt = pd.read_hdf(h5)
        except Exception:
            continue
        if len(gt) == 0 or not gt.notna().any().any():
            continue

        gt_scorer = gt.columns.get_level_values(0)[0]

        result = _clip_to_sub_ses(d.name)
        if result is None:
            continue
        sub, ses = result

        # Find pose .h5 on S3
        prefix = f"{args.pose_prefix}/{sub}/{ses}/"
        resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix, MaxKeys=20)
        h5_keys = [
            o["Key"] for o in resp.get("Contents", [])
            if o["Key"].endswith(".h5") and "filtered" not in o["Key"]
        ]
        if not h5_keys:
            continue

        # Pick the latest finetuned model (highest snapshot number)
        import re as _re
        def _snapshot_num(key: str) -> int:
            m = _re.search(r"snapshot[_-]best[_-](\d+)", key)
            return int(m.group(1)) if m else -1

        finetuned = [k for k in h5_keys if "Hrnet" in k or "Resnet" in k]
        if finetuned:
            pred_key = max(finetuned, key=_snapshot_num)
        else:
            pred_key = h5_keys[0]

        try:
            obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=pred_key)
            with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
                tmp.write(obj["Body"].read())
                tmp.flush()
                pred = pd.read_hdf(tmp.name)
        except Exception:
            continue

        pred_scorer = pred.columns.get_level_values(0)[0]
        pred_bps = pred.columns.get_level_values(1).unique().tolist()

        short = f"{d.name.split('_')[0]}_{d.name.split('_')[4].split('-')[0]}"
        ses_errors: dict[str, list[float]] = {bp: [] for bp in BODYPARTS}

        for i in range(len(gt)):
            idx = gt.index[i]
            frame_file = idx[2] if isinstance(idx, tuple) else str(idx).split("/")[-1]
            m = re.match(r"frame_(\d+)\.png", frame_file)
            if not m:
                continue
            raw_fi = int(m.group(1))
            dlc_fi = round(raw_fi * DLC_FPS / RAW_FPS)
            if dlc_fi >= len(pred):
                continue

            total_frames += 1
            matched = False

            for bp in BODYPARTS:
                try:
                    gx = float(gt.iloc[i][(gt_scorer, bp, "x")])
                    gy = float(gt.iloc[i][(gt_scorer, bp, "y")])
                except (KeyError, ValueError):
                    continue
                if np.isnan(gx) or np.isnan(gy):
                    continue

                pbp = bp
                if bp not in pred_bps:
                    if bp == "head_midpoint" and "implant_base_rear" in pred_bps:
                        pbp = "implant_base_rear"
                    else:
                        continue

                try:
                    px = float(pred.iloc[dlc_fi][(pred_scorer, pbp, "x")])
                    py = float(pred.iloc[dlc_fi][(pred_scorer, pbp, "y")])
                except (KeyError, ValueError):
                    continue
                if np.isnan(px) or np.isnan(py):
                    continue

                err = float(np.sqrt((gx - px) ** 2 + (gy - py) ** 2))
                bp_errors[bp].append(err)
                ses_errors[bp].append(err)
                matched = True

            if matched:
                total_matched += 1

        bp_errors_by_session[short] = {
            bp: {
                "mean": float(np.mean(errs)) if errs else None,
                "std": float(np.std(errs)) if errs else None,
                "n": len(errs),
            }
            for bp, errs in ses_errors.items()
        }

    # Build summary
    summary = {
        "bodyparts": {},
        "total_frames": total_frames,
        "total_matched": total_matched,
        "pose_prefix": args.pose_prefix,
        "per_session": bp_errors_by_session,
    }
    for bp in BODYPARTS:
        errs = bp_errors[bp]
        if errs:
            summary["bodyparts"][bp] = {
                "mean_rmse": float(np.sqrt(np.mean(np.array(errs) ** 2))),
                "mean_error": float(np.mean(errs)),
                "std": float(np.std(errs)),
                "median": float(np.median(errs)),
                "n": len(errs),
                "pck_5": float(np.mean(np.array(errs) <= 5) * 100),
                "pck_10": float(np.mean(np.array(errs) <= 10) * 100),
                "pck_15": float(np.mean(np.array(errs) <= 15) * 100),
                "pck_20": float(np.mean(np.array(errs) <= 20) * 100),
            }
        else:
            summary["bodyparts"][bp] = None

    # Print summary
    print(f"Matched {total_matched}/{total_frames} frames")
    print(f"\nPer-bodypart RMSE:")
    for bp in BODYPARTS:
        info = summary["bodyparts"].get(bp)
        if info:
            print(f"  {bp:20s}  RMSE={info['mean_rmse']:>6.1f}px  "
                  f"PCK@10={info['pck_10']:>5.1f}%  n={info['n']}")
        else:
            print(f"  {bp:20s}  no data")

    if args.dry_run:
        print("\n[DRY RUN] — not uploading.")
        return

    # Upload to S3
    payload = json.dumps(summary, indent=2).encode()
    s3.put_object(Bucket=DERIVATIVES_BUCKET, Key=args.output_key, Body=payload)
    print(f"\nUploaded to s3://{DERIVATIVES_BUCKET}/{args.output_key}")


if __name__ == "__main__":
    main()
