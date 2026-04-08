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
import sys
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


def train(s3, maxiters: int = 50000, epochs: int = 400, batch_size: int = 8) -> Path:
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
    print(f"Epochs: {epochs}")

    update_progress(s3, "Training: creating dataset")

    # Delete old training data so create_training_dataset() builds fresh
    # with SuperAnimal weights instead of reusing a stale ResNet50 split.
    for old_dir_name in ("dlc-models-pytorch", "dlc-models", "training-datasets"):
        old_dir = work / old_dir_name
        if old_dir.exists():
            shutil.rmtree(old_dir)
            print(f"  Deleted old {old_dir_name}/")

    # Create training dataset with SuperAnimal transfer.
    # The superanimal_name parameter initialises the model from
    # SuperAnimal TopViewMouse weights (HRNet-W32), not default ResNet50.
    print("Creating training dataset (SuperAnimal TopViewMouse transfer)...")
    try:
        deeplabcut.create_training_dataset(
            str(config_path),
            superanimal_name="superanimal_topviewmouse",
        )
    except TypeError:
        # Older DLC versions may not support superanimal_name
        print("  WARNING: superanimal_name not supported, using default backbone")
        deeplabcut.create_training_dataset(str(config_path))

    # Set epochs in the pytorch config (DLC 3.0 ignores maxiters)
    pytorch_cfg_candidates = list(work.rglob("pytorch_config.yaml"))
    for pcfg_path in pytorch_cfg_candidates:
        with open(pcfg_path) as f:
            pcfg = yaml.safe_load(f)

        # Set epochs
        if "train_settings" not in pcfg:
            pcfg["train_settings"] = {}
        pcfg["train_settings"]["epochs"] = epochs

        # Verify backbone after create_training_dataset
        backbone = pcfg.get("model", {}).get("backbone", {}).get("model_name", "?")
        print(f"  Backbone: {backbone}")
        if "resnet" in backbone.lower():
            print("  WARNING: backbone is ResNet, not HRNet. SuperAnimal transfer may not have worked.")

        # Aggressive augmentation for overhead mouse tracking with
        # light/dark alternation and high pose variability.
        if "data" in pcfg and "train" in pcfg["data"]:
            aug = pcfg["data"]["train"]
            # Affine: full rotation, wide scale range, translation
            if "affine" not in aug:
                aug["affine"] = {}
            aug["affine"]["rotation"] = 180       # full 360° (mouse faces any direction)
            aug["affine"]["scaling"] = [0.25, 2.5] # extreme scale range
            aug["affine"]["translation"] = 0.15    # shift up to 15% of image
            aug["affine"]["p"] = 0.8               # apply most of the time
            # Brightness/contrast: critical for light on / light off
            # limit=0.6 means brightness can change by ±60%
            aug["brightness"] = {"p": 0.7, "limit": 0.6}
            aug["contrast"] = {"p": 0.7, "limit": 0.6}
            # Flips: mouse is symmetric from above in both axes
            aug["horizontal_flip"] = {"p": 0.5}
            aug["vertical_flip"] = {"p": 0.5}
            # Noise and blur
            aug["gaussian_noise"] = 30.0
            aug["motion_blur"] = True
            print(
                f"  Augmentation: rot=±180°, scale=0.25-2.5x, "
                f"brightness/contrast=±60%, hflip+vflip, noise=30"
            )

        with open(pcfg_path, "w") as f:
            yaml.dump(pcfg, f)
        print(f"  Set epochs={epochs} in {pcfg_path.name}")

    update_progress(s3, f"Training: fine-tuning network ({epochs} epochs)")

    # Train
    print(f"Training for {epochs} epochs...")
    deeplabcut.train_network(
        str(config_path),
        maxiters=maxiters,
        displayiters=100,
        saveiters=5000,
    )

    # Evaluate
    print("Evaluating network...")
    deeplabcut.evaluate_network(str(config_path), plotting=False)

    # Upload evaluation results (per-bodypart RMSE)
    eval_dir = work / "evaluation-results"
    if eval_dir.exists():
        print("Uploading evaluation results...")
        for f in eval_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(work)
                key = f"{RETRAIN_PREFIX}/models/{rel}"
                s3.upload_file(str(f), DERIVATIVES_BUCKET, key)
        print(f"  Uploaded {sum(1 for _ in eval_dir.rglob('*') if _.is_file())} eval files")

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
            # Download video via boto3 (no awscli dependency)
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
            print("  Running DLC inference (batch_size=64)...")
            deeplabcut.analyze_videos(
                str(config_path),
                [str(dlc_video)],
                destfolder=str(out_dir),
                batch_size=64,
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
            else:
                print("  No output files")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="DLC retraining + inference")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--maxiters", type=int, default=50000, help="Legacy TF iterations (ignored by PyTorch)")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs (DLC 3.0 PyTorch)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Promote completed sessions even if some inference sessions failed. "
             "By default auto-promotion is skipped if any session fails.",
    )
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    do_train = not args.infer_only
    do_infer = not args.train_only

    config_path = None
    if do_train:
        config_path = train(s3, maxiters=args.maxiters, epochs=args.epochs, batch_size=args.batch_size)

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

        infer(s3, config_path, skip_failed=args.skip_failed)


if __name__ == "__main__":
    main()
