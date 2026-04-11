#!/usr/bin/env python3
"""Launch EC2 for DLC fine-tuning and re-inference.

Hard requirements:
- GPU verified before processing (abort if CUDA unavailable)
- GPU utilization monitored every 30s, uploaded to S3 every 5 min
- Watchdog aborts if GPU at 0% for 5 min during processing
- Hard 24h timeout — instance terminates regardless
- Instance self-terminates on completion (InstanceInitiatedShutdownBehavior=terminate)

Usage:
    uv run python scripts/launch_dlc_finetune_ec2.py
    uv run python scripts/launch_dlc_finetune_ec2.py --progress
    uv run python scripts/launch_dlc_finetune_ec2.py --status
    uv run python scripts/launch_dlc_finetune_ec2.py --terminate
    uv run python scripts/launch_dlc_finetune_ec2.py --maxiters 100000
    uv run python scripts/launch_dlc_finetune_ec2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys

import boto3
from ec2_constants import (
    AMI_ID,
    DERIVATIVES_BUCKET,
    IAM_PROFILE,
    KEY_NAME,
    REGION,
    SG_ID,
)
from ec2_utils import (
    APT_INSTALL_SNIPPET,
    DPKG_WAIT_SNIPPET,
    PYTORCH_CUDA_INSTALL_SNIPPET,
    build_creds_block,
    format_cost_record_launch,
    format_cost_record_shutdown,
    format_gpu_guard,
    format_hard_timeout,
    format_heartbeat,
    get_s3_credentials,
)

# Keep a local alias for the AMI variable name used below
AMI = AMI_ID
INSTANCE_TYPE = "g4dn.xlarge"  # fallback from g5.xlarge (capacity issues)
TAG_NAME = "hm2p-dlc-retrain"


def build_user_data(
    maxiters: int = 50000,
    epochs: int = 400,
    infer_only: bool = False,
    train_only: bool = False,
) -> str:
    """Build the EC2 user-data script."""
    key_id, secret, region = get_s3_credentials()
    creds = build_creds_block(key_id, secret, region)
    gpu_guard = format_gpu_guard(DERIVATIVES_BUCKET, "dlc-retrain")
    timeout = format_hard_timeout(24)
    heartbeat = format_heartbeat(
        DERIVATIVES_BUCKET, "dlc-retrain", INSTANCE_TYPE, heartbeat_key="_heartbeat.json"
    )

    if infer_only:
        mode_flag = "--infer-only"
        mode_label = "inference only"
        mode = "infer"
    elif train_only:
        mode_flag = f"--train-only --epochs {epochs}"
        mode_label = f"training only ({epochs} epochs)"
        mode = "train"
    else:
        mode_flag = f"--epochs {epochs}"
        mode_label = f"train + inference ({epochs} epochs)"
        mode = "train+infer"

    cost_launch = format_cost_record_launch(
        DERIVATIVES_BUCKET,
        "dlc-retrain",
        instance_type=INSTANCE_TYPE,
        pipeline_step="dlc-retrain-gpu",
        mode=mode,
        launch_key="_cost_record_launch.json",
    )
    cost_shutdown = format_cost_record_shutdown(
        DERIVATIVES_BUCKET,
        "dlc-retrain",
        shutdown_key="_cost_record_shutdown.json",
    )

    return f"""#!/bin/bash
exec > >(tee /var/log/hm2p-dlc-retrain.log) 2>&1
echo "=== DLC retrain ({mode_label}, GPU enforced, 24h timeout) ==="
echo "Started: $(date -u)"

trap 'aws s3 cp /var/log/hm2p-dlc-retrain.log s3://{DERIVATIVES_BUCKET}/dlc-retrain/_gpu_run_log.txt || true; \\
      aws s3 cp /var/log/gpu_monitor.csv s3://{DERIVATIVES_BUCKET}/dlc-retrain/_gpu_monitor.csv || true; \\
      {cost_shutdown}
      shutdown -h now' EXIT

{creds}
{heartbeat}
{cost_launch}
{DPKG_WAIT_SNIPPET}

set -ex

{APT_INSTALL_SNIPPET}
{PYTORCH_CUDA_INSTALL_SNIPPET}
{timeout}
{gpu_guard}

# Upload log immediately after setup (before Python which may crash)
aws s3 cp /var/log/hm2p-dlc-retrain.log s3://{DERIVATIVES_BUCKET}/dlc-retrain/_gpu_run_log.txt || true

# Clone repo and run — disable set -e so Python errors don't kill
# the script before the EXIT trap can upload logs.
set +e
cd /home/ubuntu
git clone https://github.com/chaplinta/hm2p.git
cd hm2p

# Mark GPU as active during processing
touch /tmp/gpu_processing_active
python3 scripts/run_dlc_retrain.py {mode_flag}
rm -f /tmp/gpu_processing_active

echo "=== DLC retrain complete: $(date -u) ==="
shutdown -h now
"""


def launch(maxiters: int, epochs: int = 400, infer_only: bool = False, train_only: bool = False, dry_run: bool = False) -> None:
    """Launch the retraining instance."""
    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    # Check labeled data exists
    try:
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-retrain/config.yaml")
    except Exception:
        print("ERROR: no labeled data on S3. Run upload_dlc_labels.py first.")
        sys.exit(1)

    # In --infer-only mode, verify model weights exist before launching.
    # Without this check the instance downloads config.yaml, finds no weights,
    # exits with sys.exit(1) inside user-data — the instance self-terminates
    # with no visible error from the local machine.
    if infer_only:
        resp = s3.list_objects_v2(
            Bucket=DERIVATIVES_BUCKET, Prefix="dlc-retrain/models/"
        )
        model_files = [
            obj for obj in resp.get("Contents", [])
            if not obj["Key"].endswith("_retrain_progress.json")
            and not obj["Key"].endswith("/")
        ]
        if not model_files:
            print(
                "ERROR: --infer-only requires model weights on S3 but none were found "
                "at s3://hm2p-derivatives/dlc-retrain/models/.\n"
                "Run training first (omit --infer-only, or use --train-only then "
                "--infer-only after training completes)."
            )
            sys.exit(1)
        print(f"Pre-flight: found {len(model_files)} model file(s) at dlc-retrain/models/")

    user_data = build_user_data(maxiters, epochs=epochs, infer_only=infer_only, train_only=train_only)

    if dry_run:
        print(user_data)
        return

    resp = ec2.run_instances(
        ImageId=AMI,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        IamInstanceProfile={"Name": IAM_PROFILE},
        UserData=user_data,
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"},
        }],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": TAG_NAME},
                {"Key": "Project", "Value": "hm2p"},
            ],
        }],
        InstanceInitiatedShutdownBehavior="terminate",
    )

    inst = resp["Instances"][0]
    iid = inst["InstanceId"]
    print(f"Launched: {iid} ({INSTANCE_TYPE})")

    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    desc = ec2.describe_instances(InstanceIds=[iid])
    ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "N/A")
    print(f"IP: {ip}")
    print(f"SSH: ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{ip}")
    print(f"GPU log: aws s3 cp s3://{DERIVATIVES_BUCKET}/dlc-retrain/_gpu_monitor.csv -")


def status() -> None:
    """Check instance status."""
    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [TAG_NAME]},
        {"Name": "instance-state-name", "Values": ["running", "pending"]},
    ])
    for res in r["Reservations"]:
        for inst in res["Instances"]:
            print(f"{inst['InstanceId']}  {inst['State']['Name']}  {inst.get('PublicIpAddress', 'N/A')}")
    if not r["Reservations"]:
        print("No running retrain instances.")


def progress() -> None:
    """Check training/inference progress from S3."""
    s3 = boto3.client("s3", region_name=REGION)
    try:
        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-retrain/_retrain_progress.json")
        p = json.loads(obj["Body"].read())
        print(json.dumps(p, indent=2))
    except Exception:
        print("No progress data yet.")

    # Show GPU utilization summary
    try:
        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-retrain/_gpu_monitor.csv")
        lines = obj["Body"].read().decode().strip().split("\n")
        if len(lines) > 1:
            import csv as _csv
            reader = _csv.reader(lines[1:])
            gpu_pcts = []
            import contextlib
            for row in reader:
                if len(row) >= 2:
                    with contextlib.suppress(ValueError):
                        gpu_pcts.append(int(row[1].strip().replace(" %", "")))
            if gpu_pcts:
                print(f"\nGPU utilization: mean={sum(gpu_pcts)/len(gpu_pcts):.0f}%, "
                      f"max={max(gpu_pcts)}%, readings={len(gpu_pcts)}")
    except Exception:
        pass


def terminate() -> None:
    """Terminate retrain instances."""
    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [TAG_NAME]},
        {"Name": "instance-state-name", "Values": ["running", "pending"]},
    ])
    for res in r["Reservations"]:
        for inst in res["Instances"]:
            ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
            print(f"Terminated {inst['InstanceId']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DLC training or inference on EC2")
    parser.add_argument("--status", action="store_true", help="Check running instances")
    parser.add_argument("--progress", action="store_true", help="Check S3 progress")
    parser.add_argument("--terminate", action="store_true", help="Terminate running instances")
    parser.add_argument("--dry-run", action="store_true", help="Print user-data without launching")
    parser.add_argument("--infer-only", action="store_true",
                        help="Run inference only (skip training). Uses existing model on S3.")
    parser.add_argument("--train-only", action="store_true",
                        help="Run training only (skip inference).")
    parser.add_argument("--maxiters", type=int, default=50000, help="Legacy TF iterations (ignored)")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs (default 400)")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.progress:
        progress()
    elif args.terminate:
        terminate()
    else:
        if args.infer_only and args.train_only:
            print("ERROR: --infer-only and --train-only are mutually exclusive")
            sys.exit(1)
        launch(
            args.maxiters,
            epochs=args.epochs,
            infer_only=args.infer_only,
            train_only=args.train_only,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
