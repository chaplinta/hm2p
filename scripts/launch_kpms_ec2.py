#!/usr/bin/env python3
"""Launch an EC2 instance to run keypoint-MoSeq on all 26 sessions.

keypoint-MoSeq pins numpy<=1.26 and has JAX dependencies that conflict
with the main hm2p environment. This script launches a CPU EC2 instance
that builds the hm2p-kpms Docker image and runs kpms on all DLC .h5 files
from S3.

Usage:
    python scripts/launch_kpms_ec2.py              # launch instance
    python scripts/launch_kpms_ec2.py --dry-run     # print user-data only
    python scripts/launch_kpms_ec2.py --status      # check instance
    python scripts/launch_kpms_ec2.py --terminate   # terminate instance

Reference:
    Weinreb et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point
    tracking to pose dynamics." Nature Methods 21:1329-1339.
    doi:10.1038/s41592-024-02318-2
    https://github.com/dattalab/keypoint-moseq
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import boto3

REGION = "ap-southeast-2"
INSTANCE_TYPE = "c5.4xlarge"  # 16 vCPU, 32 GB RAM — CPU-only, kpms doesn't need GPU
AMI_ID = "ami-0818a4d7794d429b1"  # Ubuntu 22.04 LTS (ap-southeast-2)
KEY_NAME = "hm2p-suite2p"
SG_NAME = "hm2p-suite2p-sg"
DERIVATIVES_BUCKET = "hm2p-derivatives"
INSTANCE_PROFILE_NAME = "hm2p-ec2-role"
TAG_PROJECT = {"Key": "Project", "Value": "hm2p-kpms"}
STATE_FILE = Path.home() / ".hm2p-kpms-ec2.json"


def _state_file(tag: str) -> Path:
    """Local state file for a run tag (parallel runs, e.g. a kappa sweep)."""
    return STATE_FILE if not tag else Path.home() / f".hm2p-kpms-ec2_{tag}.json"


# DLC .h5 files are ~300 MB each × 26 sessions ≈ 8 GB download
# Plus Docker image + kpms fitting workspace + model artifacts
ROOT_VOLUME_GB = 100


def get_sg_id() -> str:
    ec2 = boto3.client("ec2", region_name=REGION)
    resp = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG_NAME]}])
    groups = resp["SecurityGroups"]
    if not groups:
        raise RuntimeError(f"Security group '{SG_NAME}' not found")
    return groups[0]["GroupId"]


def get_instance_profile_arn() -> str:
    iam = boto3.client("iam")
    resp = iam.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
    return resp["InstanceProfile"]["Arn"]


def build_user_data(
    kappa: float = 1_000_000,
    num_iters: int = 200,
    ar_only_iters: int = 50,
    num_pcs: int = 4,
    s3_prefix: str = "kinematics",
    branch: str = "main",
) -> str:
    """Cloud-init script. ``s3_prefix`` other than "kinematics" writes a separate
    output set (syllables, model, summary, logs) so a sweep does not overwrite the
    pipeline's syllables; ``--skip-existing`` is only used for the pipeline prefix."""
    skip = "--skip-existing \\" if s3_prefix == "kinematics" else "\\"
    return (
        textwrap.dedent("""\
        #!/bin/bash
        # Do NOT use set -e — we want to capture errors and upload logs.
        set -uxo pipefail
        exec > >(tee /var/log/kpms-setup.log) 2>&1

        # ── Trap: always upload logs before shutdown ──────────────────────
        # Ensures logs reach S3 even if the script crashes unexpectedly
        # (e.g. git clone fails, Docker daemon doesn't start, OOM, etc.).
        S3_PREFIX=s3://hm2p-derivatives/{s3_prefix}
        upload_logs_and_shutdown() {
            echo "=== Uploading logs before shutdown $(date) ==="
            # If no status file was written, write a generic failure
            if [ ! -f /tmp/kpms_status.json ]; then
                echo '{"status":"failed","error":"script exited before completion"}' \
                    > /tmp/kpms_status.json
            fi
            aws s3 cp /tmp/kpms_status.json "$S3_PREFIX/kpms_status.json" || true
            aws s3 cp /var/log/kpms-setup.log "$S3_PREFIX/kpms-setup.log" || true
            echo "=== Shutting down $(date) ==="
            shutdown -h now
        }
        trap upload_logs_and_shutdown EXIT

        echo "=== keypoint-MoSeq setup starting $(date) ==="
        export DEBIAN_FRONTEND=noninteractive

        # ── System packages ──────────────────────────────────────────────
        apt-get update -y
        apt-get install -y docker.io awscli python3-pip git
        systemctl start docker
        systemctl enable docker

        # ── Clone repo for Dockerfile + scripts ──────────────────────────
        cd /home/ubuntu
        git clone --branch {branch} --depth 1 https://github.com/chaplinta/hm2p.git
        cd hm2p

        # ── Build kpms Docker image ──────────────────────────────────────
        echo "=== Building hm2p-kpms Docker image $(date) ==="
        docker build -f docker/kpms.Dockerfile -t hm2p-kpms .
        BUILD_EXIT=$?
        if [ $BUILD_EXIT -ne 0 ]; then
            echo "=== Docker build FAILED (exit $BUILD_EXIT) $(date) ==="
            echo '{"status": "failed", "error": "docker build failed"}' > /tmp/kpms_status.json
            exit 1
        fi

        # ── Create working directories ───────────────────────────────────
        mkdir -p /home/ubuntu/kpms_project /home/ubuntu/kpms_output

        # ── Run kpms on all sessions ─────────────────────────────────────
        echo "=== Running keypoint-MoSeq on all sessions $(date) ==="

        # Safety timeout: terminate instance after 36 hours to prevent
        # runaway costs if the fitting hangs or OOMs silently.
        (sleep 129600 && echo "=== 36h safety timeout reached ===" && shutdown -h now) &

        # IAM instance profile credentials are available via EC2 metadata service
        docker run --rm --network host \\
            -v /home/ubuntu/kpms_project:/data/project \\
            -v /home/ubuntu/kpms_output:/data/output \\
            -v /home/ubuntu/hm2p/metadata:/app/metadata:ro \\
            hm2p-kpms \\
            --all-sessions \\
            --s3-bucket hm2p-derivatives \\
            --project-dir /data/project \\
            --output-dir /data/output \\
            {skip}
            --s3-prefix {s3_prefix} \\
            --bodyparts nose left_ear right_ear head_midpoint \\
                neck mid_back mouse_center tail_base \\
            --kappa {kappa:g} \\
            --num-pcs {num_pcs} \\
            --num-iters {num_iters} \\
            --ar-only-iters {ar_only_iters} \\
            --conf-threshold 0.9
        RUN_EXIT=$?

        if [ $RUN_EXIT -ne 0 ]; then
            echo "=== keypoint-MoSeq FAILED (exit $RUN_EXIT) $(date) ==="
            printf '{"status":"failed","error":"docker run exit %d"}' \
                "$RUN_EXIT" > /tmp/kpms_status.json
        else
            echo "=== keypoint-MoSeq complete $(date) ==="
            echo '{"status": "complete"}' > /tmp/kpms_status.json
        fi

        # The EXIT trap handles log upload and shutdown.
    """)
        .replace("{s3_prefix}", s3_prefix)
        .replace("{branch}", branch)
        .replace("{skip}", skip)
        .replace("{kappa:g}", f"{kappa:g}")
        .replace("{num_iters}", str(num_iters))
        .replace("{ar_only_iters}", str(ar_only_iters))
        .replace("{num_pcs}", str(num_pcs))
    )


def launch_instance(dry_run: bool = False, tag: str = "", **ud_kwargs) -> dict | None:
    STATE_FILE = _state_file(tag)  # noqa: N806
    if dry_run:
        print(build_user_data(**ud_kwargs))
        return None

    ec2 = boto3.client("ec2", region_name=REGION)
    sg_id = get_sg_id()

    # Check for existing instance
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        instance_id = state.get("instance_id")
        if instance_id:
            resp = ec2.describe_instances(InstanceIds=[instance_id])
            reservations = resp["Reservations"]
            if reservations:
                inst_state = reservations[0]["Instances"][0]["State"]["Name"]
                if inst_state in ("running", "pending"):
                    print(
                        f"Instance {instance_id} already {inst_state}."
                        " Use --status or --terminate."
                    )
                    return None

    # IAM instance profile validation skipped — the profile is passed
    # by name to run_instances and EC2 validates it at launch time.
    # The IAM API call fails from devcontainers with restricted networking.

    user_data = build_user_data(**ud_kwargs)

    print(f"Launching {INSTANCE_TYPE} instance for keypoint-MoSeq...")
    resp = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        IamInstanceProfile={"Name": INSTANCE_PROFILE_NAME},
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": ROOT_VOLUME_GB,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        InstanceInitiatedShutdownBehavior="terminate",
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    TAG_PROJECT,
                    {"Key": "Name", "Value": "hm2p-kpms" + (f"-{tag}" if tag else "")},
                ],
            }
        ],
    )

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"Launched: {instance_id}")

    # Wait for public IP
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "N/A")

    state = {
        "instance_id": instance_id,
        "instance_type": INSTANCE_TYPE,
        "public_ip": public_ip,
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"Public IP: {public_ip}")
    print(f"SSH: ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{public_ip}")
    print("Logs: ssh ... 'tail -f /var/log/kpms-setup.log'")
    print(f"State saved to {STATE_FILE}")

    return state


def check_status(tag: str = "", s3_prefix: str = "kinematics") -> None:
    STATE_FILE = _state_file(tag)  # noqa: N806
    if not STATE_FILE.exists():
        print("No kpms instance state file found.")
        return

    state = json.loads(STATE_FILE.read_text())
    instance_id = state.get("instance_id")
    if not instance_id:
        print("No instance ID in state file.")
        return

    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        inst_state = inst["State"]["Name"]
        ip = inst.get("PublicIpAddress", "N/A")
        print(f"Instance: {instance_id}")
        print(f"State: {inst_state}")
        print(f"IP: {ip}")
        print(f"Type: {inst.get('InstanceType', 'N/A')}")
        if inst_state == "running":
            print(f"SSH: ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{ip}")
    except Exception as e:
        print(f"Error checking instance: {e}")

    # Check S3 for completion
    try:
        s3 = boto3.client("s3", region_name=REGION)
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=f"{s3_prefix}/kpms_status.json")
        print("\nkpms_status.json found on S3 — job appears complete!")
    except Exception:
        print("\nkpms_status.json not on S3 — job still running or not started.")

    # Check for syllable outputs
    try:
        s3 = boto3.client("s3", region_name=REGION)
        resp = s3.list_objects_v2(
            Bucket=DERIVATIVES_BUCKET,
            Prefix=f"{s3_prefix}/",
        )
        npz_files = [
            obj["Key"] for obj in resp.get("Contents", []) if obj["Key"].endswith("syllables.npz")
        ]
        print(f"Syllable .npz files on S3: {len(npz_files)}")
    except Exception:
        pass


def terminate_instance(tag: str = "") -> None:
    STATE_FILE = _state_file(tag)  # noqa: N806
    if not STATE_FILE.exists():
        print("No kpms instance state file found.")
        return

    state = json.loads(STATE_FILE.read_text())
    instance_id = state.get("instance_id")
    if not instance_id:
        print("No instance ID in state file.")
        return

    ec2 = boto3.client("ec2", region_name=REGION)
    print(f"Terminating {instance_id}...")
    ec2.terminate_instances(InstanceIds=[instance_id])
    print("Terminated.")
    STATE_FILE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch EC2 for keypoint-MoSeq")
    parser.add_argument("--dry-run", action="store_true", help="Print user-data only")
    parser.add_argument("--status", action="store_true", help="Check instance status")
    parser.add_argument("--terminate", action="store_true", help="Terminate instance")
    parser.add_argument("--kappa", type=float, default=1_000_000)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--ar-only-iters", type=int, default=50)
    parser.add_argument("--num-pcs", type=int, default=4)
    parser.add_argument(
        "--s3-prefix",
        default="kinematics",
        help="Output prefix; use e.g. kpms_sweep/kappa_1e5 for sweeps",
    )
    parser.add_argument("--branch", default="main", help="git branch to build from")
    parser.add_argument(
        "--tag", default="", help="run tag (separate state file for parallel runs)"
    )
    args = parser.parse_args()

    if args.status:
        check_status(args.tag, args.s3_prefix)
    elif args.terminate:
        terminate_instance(args.tag)
    else:
        launch_instance(
            dry_run=args.dry_run,
            tag=args.tag,
            kappa=args.kappa,
            num_iters=args.num_iters,
            ar_only_iters=args.ar_only_iters,
            num_pcs=args.num_pcs,
            s3_prefix=args.s3_prefix,
            branch=args.branch,
        )


if __name__ == "__main__":
    main()
