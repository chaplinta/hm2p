#!/usr/bin/env python3
"""Launch EC2 for DLC fine-tuning and re-inference.

Hard requirements:
- GPU verified before processing (abort if CUDA unavailable)
- GPU utilization monitored every 30s, uploaded to S3 every 5 min
- Watchdog aborts if GPU at 0% for 5 min during processing
- Hard 24h timeout — instance terminates regardless
- Instance self-terminates on completion (InstanceInitiatedShutdownBehavior=terminate)

Usage:
    uv run python scripts/launch_dlc_retrain_ec2.py
    uv run python scripts/launch_dlc_retrain_ec2.py --progress
    uv run python scripts/launch_dlc_retrain_ec2.py --status
    uv run python scripts/launch_dlc_retrain_ec2.py --terminate
    uv run python scripts/launch_dlc_retrain_ec2.py --maxiters 100000
    uv run python scripts/launch_dlc_retrain_ec2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys

import boto3

from ec2_utils import (
    APT_INSTALL_SNIPPET,
    DPKG_WAIT_SNIPPET,
    PYTORCH_CUDA_INSTALL_SNIPPET,
    build_creds_block,
    format_gpu_guard,
    format_hard_timeout,
    get_s3_credentials,
)

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
INSTANCE_TYPE = "g5.xlarge"
AMI = "ami-05186a30469f66913"
KEY_NAME = "hm2p-suite2p"
SG_ID = "sg-020161fb424325e6b"
IAM_PROFILE = "hm2p-ec2-role"
TAG_NAME = "hm2p-dlc-retrain"


def build_user_data(
    maxiters: int = 50000,
    infer_only: bool = False,
    train_only: bool = False,
) -> str:
    """Build the EC2 user-data script."""
    key_id, secret, region = get_s3_credentials()
    creds = build_creds_block(key_id, secret, region)
    gpu_guard = format_gpu_guard(DERIVATIVES_BUCKET, "dlc-retrain")
    timeout = format_hard_timeout(24)

    if infer_only:
        mode_flag = "--infer-only"
        mode_label = "inference only"
    elif train_only:
        mode_flag = f"--train-only --maxiters {maxiters}"
        mode_label = "training only"
    else:
        mode_flag = f"--maxiters {maxiters}"
        mode_label = "train + inference"

    return f"""#!/bin/bash
exec > >(tee /var/log/hm2p-dlc-retrain.log) 2>&1
echo "=== DLC retrain ({mode_label}, GPU enforced, 24h timeout) ==="
echo "Started: $(date -u)"

trap 'aws s3 cp /var/log/hm2p-dlc-retrain.log s3://{DERIVATIVES_BUCKET}/dlc-retrain/_retrain_log.txt || true; \\
      aws s3 cp /var/log/gpu_monitor.csv s3://{DERIVATIVES_BUCKET}/dlc-retrain/_gpu_monitor.csv || true; \\
      shutdown -h now' EXIT

{creds}
{DPKG_WAIT_SNIPPET}

set -ex

{APT_INSTALL_SNIPPET}
{PYTORCH_CUDA_INSTALL_SNIPPET}
{timeout}
{gpu_guard}

# Clone repo and run
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


def launch(maxiters: int, infer_only: bool = False, train_only: bool = False, dry_run: bool = False) -> None:
    """Launch the retraining instance."""
    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    # Check labeled data exists
    try:
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-retrain/config.yaml")
    except Exception:
        print("ERROR: no labeled data on S3. Run upload_dlc_labels.py first.")
        sys.exit(1)

    user_data = build_user_data(maxiters, infer_only=infer_only, train_only=train_only)

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
            for row in reader:
                if len(row) >= 2:
                    try:
                        gpu_pcts.append(int(row[1].strip().replace(" %", "")))
                    except ValueError:
                        pass
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
    parser.add_argument("--maxiters", type=int, default=50000, help="Training iterations")
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
            infer_only=args.infer_only,
            train_only=args.train_only,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
