#!/usr/bin/env python3
"""Deploy a temporary Streamlit frontend on EC2 for testing.

Launches a t3.small instance, installs the app from the git repo, runs
Streamlit, waits for a configurable duration, then terminates the instance.

Run from **macOS** (not the devcontainer):

    uv run scripts/deploy_frontend_test.py              # 5 min default
    uv run scripts/deploy_frontend_test.py --minutes 10
    uv run scripts/deploy_frontend_test.py --dry-run
    uv run scripts/deploy_frontend_test.py --teardown --instance-id i-xxx

Requires: boto3, AWS credentials configured (hm2p-agent profile or default).
"""

from __future__ import annotations

import argparse
import sys
import time

import boto3

REGION = "ap-southeast-2"
INSTANCE_TYPE = "t3.small"
KEY_NAME = "hm2p-suite2p"
SG_ID = "sg-020161fb424325e6b"
IAM_PROFILE = "hm2p-frontend-role"
REPO_URL = "https://github.com/chaplinta/hm2p.git"
BRANCH = "main"


def _get_ubuntu_ami(region: str = REGION) -> str:
    """Look up the latest Ubuntu 22.04 LTS amd64 AMI via EC2 describe-images."""
    ec2 = boto3.client("ec2", region_name=region)
    resp = ec2.describe_images(
        Owners=["099720109477"],  # Canonical
        Filters=[
            {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("No Ubuntu 22.04 AMI found in " + region)
    return images[0]["ImageId"]


def _build_userdata(repo_url: str, branch: str) -> str:
    """Build cloud-init script that installs and starts Streamlit."""
    return f"""#!/bin/bash
set -euo pipefail
exec > /var/log/hm2p-frontend.log 2>&1

echo "=== Installing system packages ==="
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv curl

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

echo "=== Cloning repo ==="
cd /opt
git clone --depth 1 --branch {branch} {repo_url} hm2p
cd hm2p

echo "=== Setting up venv ==="
uv venv .venv --python python3
uv sync --extra dev
uv pip install plotly streamlit-google-auth tifffile

echo "=== Configuring AWS region ==="
mkdir -p /root/.aws
printf '[default]\\nregion = {REGION}\\noutput = json\\n' > /root/.aws/config

echo "=== Starting Streamlit ==="
# No Google auth env vars = local dev mode (auth skipped)
nohup .venv/bin/python -m streamlit run frontend/app.py \\
    --server.port 8501 \\
    --server.address 0.0.0.0 \\
    --server.headless true \\
    --client.toolbarMode minimal \\
    --client.showErrorDetails false \\
    &

echo "=== Frontend running on port 8501 ==="
"""


def launch(dry_run: bool = False) -> str | None:
    """Launch the test frontend instance. Returns instance ID."""
    userdata = _build_userdata(REPO_URL, BRANCH)

    ami = _get_ubuntu_ami()
    if dry_run:
        print("DRY RUN — would launch:")
        print(f"  AMI:           {ami}")
        print(f"  Instance type: {INSTANCE_TYPE}")
        print(f"  Key:           {KEY_NAME}")
        print(f"  SG:            {SG_ID}")
        print(f"  IAM profile:   {IAM_PROFILE}")
        print(f"  Repo:          {REPO_URL} ({BRANCH})")
        print("\nUser data script:")
        print(userdata)
        return None

    ec2 = boto3.client("ec2", region_name=REGION)

    print(f"Launching {INSTANCE_TYPE} instance...")
    print(f"  Using AMI: {ami}")
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        IamInstanceProfile={"Name": IAM_PROFILE},
        MinCount=1,
        MaxCount=1,
        UserData=userdata,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "hm2p-frontend-test"},
                    {"Key": "Project", "Value": "hm2p"},
                ],
            }
        ],
    )

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"  Instance: {instance_id}")
    print("  Waiting for running state...")

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
    ipv6 = ""
    net_interfaces = desc["Reservations"][0]["Instances"][0].get("NetworkInterfaces", [])
    for ni in net_interfaces:
        for addr in ni.get("Ipv6Addresses", []):
            ipv6 = addr.get("Ipv6Address", "")
            break

    print(f"  Public IPv4: {ip or '(none)'}")
    print(f"  Public IPv6: {ipv6 or '(none)'}")
    print()
    print("  Streamlit will be available in ~2-3 minutes at:")
    if ip:
        print(f"    http://{ip}:8501")
    if ipv6:
        print(f"    http://[{ipv6}]:8501")
    print()
    print(f"  SSH: ssh -i ~/.ssh/hm2p-suite2p.pem ubuntu@{ip or ipv6}")
    print(f"  Logs: ssh ... 'tail -f /var/log/hm2p-frontend.log'")

    return instance_id


def terminate(instance_id: str, dry_run: bool = False) -> None:
    """Terminate the test instance."""
    if dry_run:
        print(f"DRY RUN — would terminate {instance_id}")
        return

    ec2 = boto3.client("ec2", region_name=REGION)
    print(f"Terminating {instance_id}...")
    ec2.terminate_instances(InstanceIds=[instance_id])
    print("  Done — instance will be terminated shortly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy/teardown test frontend on EC2")
    parser.add_argument(
        "--minutes",
        type=int,
        default=5,
        help="Minutes to keep the instance running (default: 5)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument(
        "--teardown", action="store_true", help="Terminate an existing instance"
    )
    parser.add_argument("--instance-id", help="Instance ID (for --teardown)")
    args = parser.parse_args()

    if args.teardown:
        if not args.instance_id:
            print("Error: --teardown requires --instance-id", file=sys.stderr)
            sys.exit(1)
        terminate(args.instance_id, dry_run=args.dry_run)
        return

    instance_id = launch(dry_run=args.dry_run)
    if args.dry_run or instance_id is None:
        return

    print(f"Instance will auto-terminate in {args.minutes} minutes.")
    print("Press Ctrl+C to terminate early.\n")

    try:
        for remaining in range(args.minutes * 60, 0, -30):
            mins, secs = divmod(remaining, 60)
            print(f"  {mins}m {secs}s remaining...", flush=True)
            time.sleep(min(30, remaining))
    except KeyboardInterrupt:
        print("\n  Ctrl+C received — terminating now.")

    terminate(instance_id)


if __name__ == "__main__":
    main()
