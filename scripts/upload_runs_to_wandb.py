#!/usr/bin/env python3
"""Upload existing DLC training runs from S3 to W&B.

Reads learning_stats.csv and evaluation results from S3 and logs them
as a completed W&B run. Run this on a machine that can reach wandb.ai.

Usage:
    # On your Mac:
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
    uv run python scripts/upload_runs_to_wandb.py --name "SA finetune 120ep"
    uv run python scripts/upload_runs_to_wandb.py --name "ImageNet HRNet 400ep"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from pathlib import Path

import boto3

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RETRAIN_PREFIX = "dlc-retrain"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _get_s3_text(s3, key: str) -> str | None:
    try:
        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=key)
        return obj["Body"].read().decode()
    except Exception:
        return None


def _get_s3_json(s3, key: str) -> dict | None:
    text = _get_s3_text(s3, key)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def upload_run(s3, run_name: str) -> None:
    import wandb

    # Load learning stats
    stats_key = (
        f"{RETRAIN_PREFIX}/models/iteration-0/"
        "hm2p-retrainMar20-trainset80shuffle1/train/learning_stats.csv"
    )
    stats_text = _get_s3_text(s3, stats_key)
    if stats_text is None:
        log.error("No learning_stats.csv on S3")
        return

    rows = list(csv.DictReader(io.StringIO(stats_text)))
    log.info("Loaded %d epochs", len(rows))

    # Load eval results
    eval_key = (
        f"{RETRAIN_PREFIX}/models/evaluation-results-pytorch/"
        "iteration-0/CombinedEvaluation-results.csv"
    )
    eval_text = _get_s3_text(s3, eval_key)
    eval_data = None
    if eval_text:
        eval_rows = list(csv.DictReader(io.StringIO(eval_text)))
        if eval_rows:
            eval_data = eval_rows[-1]

    # Load extras
    notes_text = _get_s3_text(s3, f"{RETRAIN_PREFIX}/models/_sa_finetune_notes.txt")
    cost = _get_s3_json(s3, f"{RETRAIN_PREFIX}/_cost_record_launch.json")
    champion = _get_s3_json(s3, "dlc-champion.json")
    bp_eval = _get_s3_json(s3, f"{RETRAIN_PREFIX}/models/_per_bodypart_eval.json")

    # Start W&B run
    config = {
        "architecture": "HRNet-W32",
        "n_bodyparts": 8,
        "n_epochs": len(rows),
    }
    if eval_data:
        config["training_fraction"] = eval_data.get("%Training dataset", "0.8")
        config["pcutoff"] = eval_data.get("pcutoff", "0.4")
    if cost:
        config["mode"] = cost.get("mode", "unknown")
        config["instance_type"] = cost.get("instance_type", "unknown")
    if notes_text:
        config["notes"] = notes_text[:500]
    if champion:
        config["champion_id"] = champion.get("champion_id", "unknown")

    run = wandb.init(
        project="hm2p-dlc",
        name=run_name,
        config=config,
    )

    # Log per-epoch metrics
    for row in rows:
        step = int(row.get("step", 0))
        metrics = {}

        for csv_col, wb_name in [
            ("losses/train.bodypart_total_loss", "train_loss"),
            ("losses/eval.bodypart_total_loss", "eval_loss"),
            ("metrics/test.rmse", "test_rmse"),
            ("metrics/test.rmse_pcutoff", "test_rmse_pcutoff"),
            ("metrics/test.mAP", "test_mAP"),
            ("metrics/test.mAR", "test_mAR"),
        ]:
            val = row.get(csv_col)
            if val:
                metrics[wb_name] = float(val)

        if metrics:
            run.log(metrics, step=step)

    # Log final eval
    if eval_data:
        summary = {}
        for key, name in [
            ("train rmse", "final/train_rmse"),
            ("test rmse", "final/test_rmse"),
            ("test rmse_pcutoff", "final/test_rmse_pcutoff"),
            ("test mAP", "final/test_mAP"),
            ("test mAR", "final/test_mAR"),
        ]:
            val = eval_data.get(key)
            if val:
                summary[name] = float(val)
        if summary:
            run.summary.update(summary)

    # Log per-bodypart
    if bp_eval and "bodyparts" in bp_eval:
        for bp, data in bp_eval["bodyparts"].items():
            if data and data.get("rmse") is not None:
                run.summary[f"bodypart/{bp}_rmse"] = data["rmse"]

    run.finish()
    log.info("Uploaded run '%s' to W&B project hm2p-dlc", run_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload existing DLC training runs from S3 to W&B."
    )
    parser.add_argument("--name", required=True, help="Run name in W&B.")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    upload_run(s3, args.name)


if __name__ == "__main__":
    main()
