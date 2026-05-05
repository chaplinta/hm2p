#!/usr/bin/env python3
"""Log DLC training runs to MLflow.

Reads learning_stats.csv and evaluation results from S3, logs metrics
and parameters to a local MLflow tracking server. Can import existing
runs or be called after each new training run.

Usage:
    # Import the most recent training run from S3:
    uv run python scripts/log_dlc_to_mlflow.py

    # Import with a custom run name:
    uv run python scripts/log_dlc_to_mlflow.py --name "SA finetune v1"

    # Import a specific snapshot's eval results:
    uv run python scripts/log_dlc_to_mlflow.py --snapshot best-120

    # Start the MLflow UI to view results:
    uv run mlflow ui --port 5001

    # Or from this script:
    uv run python scripts/log_dlc_to_mlflow.py --serve
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import boto3
import mlflow

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RETRAIN_PREFIX = "dlc-retrain"
REPO_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DIR = REPO_ROOT / "mlruns"

EXPERIMENT_NAME = "dlc-training"

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
    return json.loads(text)


def import_run(s3, run_name: str | None = None) -> None:
    """Import a DLC training run from S3 into MLflow."""

    # Set up MLflow
    mlflow.set_tracking_uri(str(MLFLOW_DIR))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load learning stats
    stats_key = f"{RETRAIN_PREFIX}/models/iteration-0/hm2p-retrainMar20-trainset80shuffle1/train/learning_stats.csv"
    stats_text = _get_s3_text(s3, stats_key)
    if stats_text is None:
        log.error("No learning_stats.csv found on S3")
        return

    reader = csv.DictReader(io.StringIO(stats_text))
    rows = list(reader)
    if not rows:
        log.error("learning_stats.csv is empty")
        return

    log.info("Loaded %d epochs from learning_stats.csv", len(rows))

    # Load evaluation results
    eval_key = f"{RETRAIN_PREFIX}/models/evaluation-results-pytorch/iteration-0/CombinedEvaluation-results.csv"
    eval_text = _get_s3_text(s3, eval_key)
    eval_data = None
    if eval_text:
        eval_reader = csv.DictReader(io.StringIO(eval_text))
        eval_rows = list(eval_reader)
        if eval_rows:
            eval_data = eval_rows[-1]  # most recent eval

    # Load SA finetune notes
    notes = _get_s3_json(s3, f"{RETRAIN_PREFIX}/models/_sa_finetune_notes.txt")
    notes_text = _get_s3_text(s3, f"{RETRAIN_PREFIX}/models/_sa_finetune_notes.txt")

    # Load cost record
    cost = _get_s3_json(s3, f"{RETRAIN_PREFIX}/_cost_record_launch.json")

    # Load champion manifest
    champion = _get_s3_json(s3, "dlc-champion.json")

    # Load per-bodypart RMSE if available
    bp_rmse = _get_s3_json(s3, f"{RETRAIN_PREFIX}/models/_per_bodypart_eval.json")

    # Load pytorch config for training params
    pytorch_cfg_text = _get_s3_text(
        s3, f"{RETRAIN_PREFIX}/models/iteration-0/hm2p-retrainMar20-trainset80shuffle1/train/pytorch_config.yaml"
    )

    # Determine run name
    if run_name is None:
        mode = cost.get("mode", "unknown") if cost else "unknown"
        n_epochs = len(rows)
        run_name = f"dlc-{mode}-{n_epochs}ep"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        if eval_data:
            mlflow.log_param("training_fraction", eval_data.get("%Training dataset", "0.8"))
            mlflow.log_param("shuffle", eval_data.get("Shuffle number", "1"))
            mlflow.log_param("total_epochs", eval_data.get("Training epochs", len(rows)))
            mlflow.log_param("pcutoff", eval_data.get("pcutoff", "0.4"))

        mlflow.log_param("architecture", "HRNet-W32")
        mlflow.log_param("thumb_size", "64x64")
        mlflow.log_param("n_bodyparts", 8)

        if cost:
            mlflow.log_param("mode", cost.get("mode", "unknown"))
            mlflow.log_param("instance_type", cost.get("instance_type", "unknown"))

        if notes_text:
            mlflow.log_param("notes", notes_text[:250])

        # Log per-epoch metrics
        for row in rows:
            step = int(row.get("step", 0))

            train_loss = row.get("losses/train.bodypart_total_loss")
            if train_loss:
                mlflow.log_metric("train_loss", float(train_loss), step=step)

            eval_loss = row.get("losses/eval.bodypart_total_loss")
            if eval_loss:
                mlflow.log_metric("eval_loss", float(eval_loss), step=step)

            test_rmse = row.get("metrics/test.rmse")
            if test_rmse:
                mlflow.log_metric("test_rmse", float(test_rmse), step=step)

            test_rmse_p = row.get("metrics/test.rmse_pcutoff")
            if test_rmse_p:
                mlflow.log_metric("test_rmse_pcutoff", float(test_rmse_p), step=step)

            test_map = row.get("metrics/test.mAP")
            if test_map:
                mlflow.log_metric("test_mAP", float(test_map), step=step)

            test_mar = row.get("metrics/test.mAR")
            if test_mar:
                mlflow.log_metric("test_mAR", float(test_mar), step=step)

        # Log final evaluation metrics
        if eval_data:
            for key, metric_name in [
                ("train rmse", "final/train_rmse"),
                ("train rmse_pcutoff", "final/train_rmse_pcutoff"),
                ("train mAP", "final/train_mAP"),
                ("train mAR", "final/train_mAR"),
                ("test rmse", "final/test_rmse"),
                ("test rmse_pcutoff", "final/test_rmse_pcutoff"),
                ("test mAP", "final/test_mAP"),
                ("test mAR", "final/test_mAR"),
            ]:
                val = eval_data.get(key)
                if val:
                    mlflow.log_metric(metric_name, float(val))

        # Log per-bodypart RMSE
        if bp_rmse and "bodyparts" in bp_rmse:
            for bp, data in bp_rmse["bodyparts"].items():
                if data and data.get("rmse") is not None:
                    mlflow.log_metric(f"bodypart/{bp}_rmse", data["rmse"])
                    if "median_error" in data:
                        mlflow.log_metric(f"bodypart/{bp}_median", data["median_error"])
                    if "pck_10" in data:
                        mlflow.log_metric(f"bodypart/{bp}_pck10", data["pck_10"])

        # Log champion info
        if champion:
            mlflow.log_param("champion_id", champion.get("champion_id", "unknown"))
            mlflow.log_param("champion_snapshot", champion.get("snapshot", "unknown"))

        # Log raw files as artifacts
        if stats_text:
            artifact_path = Path("/tmp/mlflow_artifacts")
            artifact_path.mkdir(exist_ok=True)
            (artifact_path / "learning_stats.csv").write_text(stats_text)
            if eval_text:
                (artifact_path / "evaluation_results.csv").write_text(eval_text)
            if pytorch_cfg_text:
                (artifact_path / "pytorch_config.yaml").write_text(pytorch_cfg_text)
            mlflow.log_artifacts(str(artifact_path))

        log.info("Logged run '%s' (id: %s)", run_name, run.info.run_id)
        log.info("  %d epochs, final test RMSE: %s",
                 len(rows),
                 eval_data.get("test rmse", "?") if eval_data else "?")


def serve() -> None:
    """Start the MLflow UI."""
    print(f"Starting MLflow UI at http://localhost:5001")
    print(f"Tracking dir: {MLFLOW_DIR}")
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui",
        "--port", "5001",
        "--backend-store-uri", str(MLFLOW_DIR),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log DLC training runs to MLflow."
    )
    parser.add_argument("--name", type=str, default=None,
                        help="Custom run name (default: auto-generated).")
    parser.add_argument("--serve", action="store_true",
                        help="Start MLflow UI on port 5001.")
    args = parser.parse_args()

    if args.serve:
        serve()
        return

    s3 = boto3.client("s3", region_name=REGION)
    import_run(s3, run_name=args.name)

    print(f"\nView results:")
    print(f"  uv run mlflow ui --port 5001 --backend-store-uri {MLFLOW_DIR}")
    print(f"  Then open http://localhost:5001")


if __name__ == "__main__":
    main()
