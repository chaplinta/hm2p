#!/usr/bin/env python3
"""Launch EC2 instance for DLC fine-tuning and re-inference.

Prerequisite: labeled data uploaded to S3 via upload_dlc_labels.py.

Usage:
    uv run python scripts/launch_dlc_retrain_ec2.py              # launch
    uv run python scripts/launch_dlc_retrain_ec2.py --status      # check instance
    uv run python scripts/launch_dlc_retrain_ec2.py --progress    # check training progress
    uv run python scripts/launch_dlc_retrain_ec2.py --terminate   # kill instance
    uv run python scripts/launch_dlc_retrain_ec2.py --maxiters 100000
"""

from __future__ import annotations

import argparse
import json
import sys

import boto3

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
INSTANCE_TYPE = "g5.xlarge"
AMI = "ami-05186a30469f66913"
KEY_NAME = "hm2p-suite2p"
SG_ID = "sg-020161fb424325e6b"
IAM_PROFILE = "hm2p-ec2-role"
TAG_NAME = "hm2p-dlc-retrain"


def build_user_data(maxiters: int = 50000) -> str:
    """Build the EC2 user-data script."""
    return f"""#!/bin/bash
exec > >(tee /var/log/hm2p-dlc-retrain.log) 2>&1
echo "=== DLC retrain ==="

trap 'aws s3 cp /var/log/hm2p-dlc-retrain.log s3://{DERIVATIVES_BUCKET}/dlc-retrain/_retrain_log.txt || true' EXIT

export DEBIAN_FRONTEND=noninteractive
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 10; done

apt-get update -qq
apt-get install -y -qq awscli ffmpeg git

# Install DLC with CUDA PyTorch
pip3 install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install --break-system-packages --quiet --pre deeplabcut
pip3 install --break-system-packages pyyaml

nvidia-smi
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {{torch.cuda.get_device_name(0)}}')"
python3 -c "import deeplabcut; print(f'DLC: {{deeplabcut.__version__}}')"

aws s3 cp /var/log/hm2p-dlc-retrain.log s3://{DERIVATIVES_BUCKET}/dlc-retrain/_retrain_log.txt || true

# GPU monitor — log utilization every 30s to verify GPU is used
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 30 >> /var/log/hm2p-gpu-monitor.log 2>&1 &

# Clone repo and run
cd /home/ubuntu
git clone https://github.com/chaplinta/hm2p.git
cd hm2p

python3 scripts/run_dlc_retrain.py --maxiters {maxiters}

echo "=== DLC retrain complete ==="
aws s3 cp /var/log/hm2p-dlc-retrain.log s3://{DERIVATIVES_BUCKET}/dlc-retrain/_retrain_log.txt || true
shutdown -h now
"""


def launch(maxiters: int) -> None:
    """Launch the retraining instance."""
    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    # Check labeled data exists
    try:
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-retrain/config.yaml")
    except Exception:
        print("ERROR: no labeled data on S3. Run upload_dlc_labels.py first.")
        sys.exit(1)

    resp = ec2.run_instances(
        ImageId=AMI,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        IamInstanceProfile={"Name": IAM_PROFILE},
        UserData=build_user_data(maxiters),
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
    print(f"Logs: ssh ... 'tail -f /var/log/hm2p-dlc-retrain.log'")


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
    parser = argparse.ArgumentParser(description="Launch DLC retraining on EC2")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--terminate", action="store_true")
    parser.add_argument("--maxiters", type=int, default=50000)
    args = parser.parse_args()

    if args.status:
        status()
    elif args.progress:
        progress()
    elif args.terminate:
        terminate()
    else:
        launch(args.maxiters)


if __name__ == "__main__":
    main()
