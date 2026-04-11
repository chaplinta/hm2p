#!/usr/bin/env python3
"""Launch a CPU EC2 instance for downstream pipeline + video rendering.

Runs after DLC inference + promotion. Does NOT need GPU:
  1. Downstream stages (kinematics → sync → analysis) with --force
  2. Render labelled videos (416x304, H.264)

Usage:
    python scripts/launch_downstream_cpu.py
    python scripts/launch_downstream_cpu.py --dry-run
    python scripts/launch_downstream_cpu.py --render-only
"""

from __future__ import annotations

import argparse

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
    DPKG_WAIT_SNIPPET,
    IMDS_HELPER_SNIPPET,
    build_creds_block,
    format_cost_record_launch,
    format_cost_record_shutdown,
    format_cpu_log_upload,
    format_hard_timeout,
    format_heartbeat,
    get_s3_credentials,
)

INSTANCE_TYPE = "c5.xlarge"
TAG_NAME = "hm2p-downstream-cpu"


def build_user_data(render_only: bool = False) -> str:
    key_id, secret, region = get_s3_credentials()
    creds = build_creds_block(key_id, secret, region)
    cpu_upload = format_cpu_log_upload(DERIVATIVES_BUCKET, "dlc-retrain")
    timeout = format_hard_timeout(12)
    heartbeat = format_heartbeat(
        DERIVATIVES_BUCKET,
        "dlc-retrain",
        INSTANCE_TYPE,
        heartbeat_key="_downstream_heartbeat.json",
    )

    mode = "render-only" if render_only else "downstream+render"
    cost_launch = format_cost_record_launch(
        DERIVATIVES_BUCKET,
        "dlc-retrain",
        instance_type=INSTANCE_TYPE,
        pipeline_step="downstream-cpu",
        mode=mode,
        launch_key="_downstream_cost_record_launch.json",
    )
    cost_shutdown = format_cost_record_shutdown(
        DERIVATIVES_BUCKET,
        "dlc-retrain",
        shutdown_key="_downstream_cost_record_shutdown.json",
    )

    downstream_cmd = "" if render_only else """
echo "=== Running downstream pipeline (Stages 3, 5, 6) ==="
python3 scripts/run_downstream_pipeline.py --force
"""

    return f"""#!/bin/bash
exec > >(tee /var/log/hm2p-downstream.log) 2>&1
echo "=== hm2p downstream + render (CPU) ==="
echo "Started: $(date -u)"

{creds}
{IMDS_HELPER_SNIPPET}

# Define shutdown handler as a function to avoid single-quote
# nesting issues that break trap '...' syntax.
_hm2p_shutdown() {{
    aws s3 cp /var/log/hm2p-downstream.log s3://{DERIVATIVES_BUCKET}/dlc-retrain/_cpu_run_log.txt || true
    {cost_shutdown}
    shutdown -h now
}}
trap _hm2p_shutdown EXIT

{heartbeat}
{cost_launch}
{cpu_upload}
{timeout}
{DPKG_WAIT_SNIPPET}

set -ex
apt-get update -qq
apt-get install -y -qq awscli ffmpeg git python3-pip python3-opencv

pip3 install --break-system-packages --quiet boto3 pandas numpy tables opencv-python-headless h5py scipy pyyaml shapely xarray netcdf4 movement pynapple

cd /home/ubuntu
git clone https://github.com/chaplinta/hm2p.git
cd hm2p
# Don't pip install -e . (requires Python >=3.11, AMI has 3.10).
# Add src/ to PYTHONPATH instead.
export PYTHONPATH=/home/ubuntu/hm2p/src:$PYTHONPATH

# Run downstream stages and video rendering IN PARALLEL.
# Video rendering only needs pose/ h5 files (already promoted).
# Downstream needs pose/ for kinematics but doesn't touch videos.
echo "=== Starting video rendering in background ==="
python3 scripts/render_dlc_videos.py --all -v &
RENDER_PID=$!
{downstream_cmd}
echo "=== Waiting for video rendering to finish ==="
wait $RENDER_PID || echo "WARNING: render_dlc_videos.py exited with error"

# Update progress
python3 -c "
import boto3, json, datetime
s3 = boto3.client('s3', region_name='{REGION}')
s3.put_object(
    Bucket='{DERIVATIVES_BUCKET}',
    Key='dlc-retrain/_retrain_progress.json',
    Body=json.dumps({{
        'status': 'Pipeline complete',
        'updated': datetime.datetime.utcnow().isoformat() + 'Z',
    }}, indent=2).encode(),
)
"

echo "=== Downstream + render complete: $(date -u) ==="
"""


def launch(render_only: bool = False, dry_run: bool = False) -> None:
    ec2 = boto3.client("ec2", region_name=REGION)

    user_data = build_user_data(render_only=render_only)

    if dry_run:
        print(user_data)
        return

    resp = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch CPU instance for downstream + render")
    parser.add_argument("--render-only", action="store_true", help="Skip downstream, render videos only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    launch(render_only=args.render_only, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
