#!/usr/bin/env python3
"""DLC retraining + re-inference — runs on EC2.

Downloads labeled data from S3, fine-tunes DLC from SuperAnimal weights,
then re-runs inference on all 26 sessions. Called by the EC2 user-data
script (launch_dlc_finetune_ec2.py).

Usage (on EC2):
    python scripts/run_dlc_retrain.py --train --infer
    python scripts/run_dlc_retrain.py --train-only
    python scripts/run_dlc_retrain.py --infer-only
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import boto3

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RAWDATA_BUCKET = "hm2p-rawdata"
RETRAIN_PREFIX = "dlc-retrain"
FINETUNED_PREFIX = "pose-finetuned"


def update_progress(s3, status: str, **extra: object) -> None:
    """Write progress JSON to S3."""
    import datetime

    progress = {
        "status": status,
        "updated": datetime.datetime.utcnow().isoformat() + "Z",
        **extra,
    }
    tmp = Path("/tmp/_retrain_progress.json")
    tmp.write_text(json.dumps(progress, indent=2))
    s3.upload_file(str(tmp), DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/_retrain_progress.json")


def train(s3, maxiters: int = 50000, batch_size: int = 8) -> Path:
    """Download labels from S3, fine-tune DLC, upload model weights."""
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

    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"Config: {config_path}")
    print(f"Bodyparts: {cfg.get('bodyparts', [])}")
    print(f"Max iterations: {maxiters}")

    update_progress(s3, "Training: creating dataset")

    # Create training dataset with SuperAnimal transfer
    print("Creating training dataset (SuperAnimal transfer)...")
    deeplabcut.create_training_dataset(str(config_path))

    update_progress(s3, "Training: fine-tuning network")

    # Train
    print(f"Training for {maxiters} iterations...")
    deeplabcut.train_network(
        str(config_path),
        maxiters=maxiters,
        displayiters=100,
        saveiters=5000,
    )

    # Evaluate
    print("Evaluating network...")
    deeplabcut.evaluate_network(str(config_path))

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


def infer(s3, config_path: Path) -> None:
    """Run inference on all 26 sessions with the fine-tuned model."""
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
    completed = []
    failed = []

    for i, ses in enumerate(sessions, 1):
        sub, ses_id = ses["sub"], ses["ses"]
        exp_id = ses["exp_id"]
        print(f"\n=== [{i}/{total}] {sub}/{ses_id} ===")
        update_progress(s3, f"Inference {i}/{total}: {sub}/{ses_id}",
                        completed=len(completed), failed=len(failed), total=total)

        work = Path(f"/tmp/dlc-infer/{sub}/{ses_id}")
        work.mkdir(parents=True, exist_ok=True)

        try:
            # Download video
            video_dir = work / "behav"
            video_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["aws", "s3", "sync",
                 f"s3://{RAWDATA_BUCKET}/rawdata/{sub}/{ses_id}/behav/",
                 str(video_dir),
                 "--exclude", "*", "--include", "*.mp4", "--exclude", "*side*"],
                check=True, capture_output=True,
            )

            mp4s = list(video_dir.glob("*overhead*.mp4")) + list(video_dir.glob("*cropped*.mp4"))
            if not mp4s:
                mp4s = list(video_dir.glob("*.mp4"))
            if not mp4s:
                print(f"  No video found, skipping")
                failed.append(exp_id)
                continue

            video = mp4s[0]

            # Subsample to 30fps
            sub_path = work / f"{video.stem}_30fps.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video), "-r", "30",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 str(sub_path)],
                capture_output=True,
            )
            dlc_video = sub_path if sub_path.exists() else video

            # Run inference
            out_dir = work / "output"
            out_dir.mkdir(exist_ok=True)
            print(f"  Running DLC inference (batch_size=64)...")
            deeplabcut.analyze_videos(
                str(config_path),
                [str(dlc_video)],
                destfolder=str(out_dir),
                batch_size=64,
            )

            # Upload results
            out_files = list(out_dir.glob("*.h5")) + list(out_dir.glob("*.csv"))
            if out_files:
                s3_dest = f"s3://{DERIVATIVES_BUCKET}/{FINETUNED_PREFIX}/{sub}/{ses_id}/"
                subprocess.run(
                    ["aws", "s3", "sync", str(out_dir), s3_dest],
                    check=True, capture_output=True,
                )
                completed.append(exp_id)
                print(f"  Uploaded {len(out_files)} files")
            else:
                print(f"  No output files")
                failed.append(exp_id)

        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(exp_id)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    update_progress(
        s3, "Inference complete",
        completed=len(completed), failed=len(failed), total=total,
        completed_sessions=completed, failed_sessions=failed,
    )
    print(f"\nDone: {len(completed)}/{total} completed, {len(failed)} failed")

    # Auto-promote: copy pose-finetuned/ → pose/ on S3
    if failed:
        print(f"\nSkipping auto-promote: {len(failed)} sessions failed.")
        return

    print(f"\nPromoting finetuned pose → pose/ on S3...")
    for ses in sessions:
        sub, ses_id = ses["sub"], ses["ses"]
        src_prefix = f"{FINETUNED_PREFIX}/{sub}/{ses_id}/"
        dst_prefix = f"pose/{sub}/{ses_id}/"
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

    update_progress(s3, "Promoted to pose/", completed=len(completed), total=total)
    print("Promotion complete. Downstream stages (3, 3b, 5, 6) need re-running.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DLC retraining + inference")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--maxiters", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    do_train = not args.infer_only
    do_infer = not args.train_only

    config_path = None
    if do_train:
        config_path = train(s3, maxiters=args.maxiters, batch_size=args.batch_size)

    if do_infer:
        if config_path is None:
            # Download config + model weights from S3 (training was done in a previous run)
            work = Path("/tmp/dlc-retrain")
            work.mkdir(parents=True, exist_ok=True)
            config_path = work / "config.yaml"
            s3.download_file(DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/config.yaml", str(config_path))

            # Download model weights
            print("Downloading model weights from S3...")
            resp = s3.list_objects_v2(
                Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/models/"
            )
            model_files = resp.get("Contents", [])
            if not model_files:
                print("ERROR: no model weights on S3. Run training first.")
                sys.exit(1)
            for obj in model_files:
                key = obj["Key"]
                rel = key[len(f"{RETRAIN_PREFIX}/models/"):]
                dest = work / "dlc-models-pytorch" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(DERIVATIVES_BUCKET, key, str(dest))
            print(f"  Downloaded {len(model_files)} model files")

            # Fix project_path in config
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg["project_path"] = str(work)
            with open(config_path, "w") as f:
                yaml.dump(cfg, f)

        infer(s3, config_path)


if __name__ == "__main__":
    main()
