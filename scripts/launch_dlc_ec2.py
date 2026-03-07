#!/usr/bin/env python3
"""Launch an EC2 g4dn.xlarge instance to run DLC pose estimation on all sessions.

Uses the SuperAnimal-TopViewMouse pretrained model (DLC 3.x) — no custom model
weights needed. Tracks nose, ears, neck, spine, tail from overhead video.

Usage:
    python scripts/launch_dlc_ec2.py                  # launch
    python scripts/launch_dlc_ec2.py --use-profile     # launch with IAM role
    python scripts/launch_dlc_ec2.py --progress        # check progress
    python scripts/launch_dlc_ec2.py --status           # instance info
    python scripts/launch_dlc_ec2.py --terminate        # kill early
    python scripts/launch_dlc_ec2.py --dry-run          # print user-data
"""

from __future__ import annotations

import argparse
import configparser
import json
import textwrap
from pathlib import Path

import boto3

REGION = "ap-southeast-2"
INSTANCE_TYPE = "g4dn.xlarge"
AMI_ID = "ami-05186a30469f66913"  # Deep Learning Base OSS Nvidia (Ubuntu 22.04)
KEY_NAME = "hm2p-suite2p"
SG_NAME = "hm2p-suite2p-sg"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"
INSTANCE_PROFILE_NAME = "hm2p-ec2-role"
TAG = {"Key": "Project", "Value": "hm2p-dlc"}
STATE_FILE = Path.home() / ".hm2p-dlc-instance.json"


def has_instance_profile() -> bool:
    """Check if the hm2p EC2 instance profile exists."""
    try:
        iam = boto3.client("iam")
        resp = iam.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        return len(resp["InstanceProfile"]["Roles"]) > 0
    except Exception:
        return False


def get_s3_credentials() -> tuple[str, str, str]:
    """Read hm2p-agent S3 credentials from ~/.aws/credentials."""
    creds = configparser.ConfigParser()
    creds.read(Path.home() / ".aws" / "credentials")
    for profile in ["hm2p-agent", "default"]:
        if profile in creds:
            return (
                creds[profile]["aws_access_key_id"],
                creds[profile]["aws_secret_access_key"],
                REGION,
            )
    raise SystemExit("No AWS credentials found in ~/.aws/credentials")


def get_sessions() -> list[dict]:
    """Read session list from metadata/experiments.csv."""
    import csv

    csv_path = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    sessions = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = row["exp_id"]
            parts = exp_id.split("_")
            animal = parts[-1]
            sub = f"sub-{animal}"
            ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
            sessions.append({"exp_id": exp_id, "sub": sub, "ses": ses})
    return sessions


def build_user_data(sessions: list[dict], use_instance_profile: bool = False) -> str:
    """Build the cloud-init user-data script for DLC."""
    session_json = json.dumps(sessions)

    if use_instance_profile:
        creds_block = textwrap.dedent(f"""\
            mkdir -p /root/.aws
            cat > /root/.aws/config << 'CONF'
            [default]
            region = {REGION}
            output = json
            CONF
            sed -i 's/^            //' /root/.aws/config
            echo "Using IAM instance profile for AWS access"
        """)
    else:
        key_id, secret, region = get_s3_credentials()
        creds_block = textwrap.dedent(f"""\
            mkdir -p /root/.aws
            cat > /root/.aws/credentials << 'CREDS'
            [default]
            aws_access_key_id = {key_id}
            aws_secret_access_key = {secret}
            CREDS
            cat > /root/.aws/config << 'CONF'
            [default]
            region = {region}
            output = json
            CONF
            sed -i 's/^            //' /root/.aws/credentials /root/.aws/config
            echo "Using embedded AWS credentials"
        """)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        exec > >(tee /var/log/hm2p-dlc.log) 2>&1

        echo "=== hm2p DLC pose estimation ==="
        echo "Started: $(date -u)"

{textwrap.indent(creds_block, "        ")}
        # --- System setup ---
        export DEBIAN_FRONTEND=noninteractive

        # Wait for dpkg lock
        echo "Waiting for dpkg lock..."
        while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
            echo "  dpkg locked, waiting 10s..."
            sleep 10
        done
        echo "dpkg lock free"

        apt-get update -qq
        apt-get install -y -qq python3-pip python3-dev awscli ffmpeg

        # --- Install DeepLabCut ---
        # DLC 2.3 needs TensorFlow for SuperAnimal inference
        # --break-system-packages needed on Ubuntu 22.04+ (PEP 668)
        pip3 install --break-system-packages --quiet "tensorflow[and-cuda]"
        pip3 install --break-system-packages --quiet deeplabcut

        # --- Verify GPU + DLC ---
        nvidia-smi || echo "WARNING: No GPU detected"
        python3 -c "import deeplabcut; print('DLC imported OK')" || echo "WARNING: DLC import failed"
        python3 -c "import tensorflow as tf; print(f'TF {{tf.__version__}}, GPUs: {{len(tf.config.list_physical_devices(\"GPU\"))}}')" || echo "WARNING: TF GPU check failed"

        # --- Process sessions ---
        SESSIONS='{session_json}'
        WORK=/tmp/hm2p-work
        mkdir -p $WORK

        echo "$SESSIONS" | python3 -c "
        import json, sys, subprocess, shutil, os, datetime
        from pathlib import Path

        sessions = json.load(sys.stdin)
        work = Path('/tmp/hm2p-work')
        total = len(sessions)
        completed = []
        failed = []
        skipped = []

        def update_progress(status_msg=''):
            progress = {{
                'total': total,
                'completed': len(completed),
                'failed': len(failed),
                'skipped': len(skipped),
                'completed_sessions': completed,
                'failed_sessions': failed,
                'status': status_msg,
                'updated': datetime.datetime.utcnow().isoformat() + 'Z',
            }}
            progress_file = work / 'progress.json'
            progress_file.write_text(json.dumps(progress, indent=2))
            subprocess.run([
                'aws', 's3', 'cp', str(progress_file),
                's3://{DERIVATIVES_BUCKET}/pose/_progress.json',
            ], capture_output=True)

        for i, ses in enumerate(sessions, 1):
            sub, ses_id = ses['sub'], ses['ses']
            exp_id = ses['exp_id']
            print(f'\\n=== [{{i}}/{{total}}] {{sub}}/{{ses_id}} ({{exp_id}}) ===', flush=True)
            update_progress(f'Processing {{i}}/{{total}}: {{sub}}/{{ses_id}}')

            # Skip if already processed on S3
            check = subprocess.run([
                'aws', 's3', 'ls',
                f's3://{DERIVATIVES_BUCKET}/pose/{{sub}}/{{ses_id}}/',
            ], capture_output=True, text=True)
            if check.returncode == 0 and ('.h5' in check.stdout or '.csv' in check.stdout):
                print(f'  SKIP: already processed on S3', flush=True)
                skipped.append(exp_id)
                continue

            # Create working dirs
            video_dir = work / 'input' / sub / ses_id / 'behav'
            out_dir = work / 'output' / sub / ses_id
            video_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Download video from S3
            s3_prefix = f'rawdata/{{sub}}/{{ses_id}}/behav/'
            print(f'  Downloading video from s3://{RAWDATA_BUCKET}/{{s3_prefix}}...', flush=True)
            ret = subprocess.run([
                'aws', 's3', 'sync',
                f's3://{RAWDATA_BUCKET}/{{s3_prefix}}',
                str(video_dir),
                '--exclude', '*',
                '--include', '*.mp4',
                '--exclude', '*side*',
            ], capture_output=True, text=True)
            if ret.returncode != 0:
                print(f'  ERROR downloading: {{ret.stderr}}', flush=True)
                failed.append(exp_id)
                continue

            mp4s = list(video_dir.glob('*overhead*.mp4')) + list(video_dir.glob('*cropped*.mp4'))
            if not mp4s:
                mp4s = list(video_dir.glob('*.mp4'))
            if not mp4s:
                print(f'  SKIP: no overhead .mp4 found', flush=True)
                skipped.append(exp_id)
                continue

            video_path = mp4s[0]
            print(f'  Video: {{video_path.name}} ({{video_path.stat().st_size/1e6:.1f}} MB)', flush=True)

            # Run DLC with SuperAnimal pretrained model
            print(f'  Running DLC (superanimal_topviewmouse)...', flush=True)
            try:
                import deeplabcut
                deeplabcut.video_inference_superanimal(
                    [str(video_path)],
                    'superanimal_topviewmouse',
                    videotype='mp4',
                    dest_folder=str(out_dir),
                )
                print(f'  DLC DONE', flush=True)
            except Exception as e:
                print(f'  ERROR in DLC: {{e}}', flush=True)
                import traceback
                traceback.print_exc()
                failed.append(exp_id)
                continue

            # Upload results to S3
            # video_inference_superanimal outputs: *superanimal* or *DLC* files
            out_files = list(out_dir.glob('*.h5')) + list(out_dir.glob('*.csv')) + list(out_dir.glob('*.json'))
            if out_files:
                s3_dest = f's3://{DERIVATIVES_BUCKET}/pose/{{sub}}/{{ses_id}}/'
                print(f'  Uploading {{len(out_files)}} files to {{s3_dest}}...', flush=True)
                ret = subprocess.run([
                    'aws', 's3', 'sync',
                    str(out_dir),
                    s3_dest,
                ], capture_output=True, text=True)
                if ret.returncode != 0:
                    print(f'  ERROR uploading: {{ret.stderr}}', flush=True)
                    failed.append(exp_id)
                else:
                    print(f'  Upload DONE', flush=True)
                    completed.append(exp_id)
            else:
                print(f'  WARNING: no DLC output found in {{out_dir}}', flush=True)
                failed.append(exp_id)

            # Cleanup
            shutil.rmtree(work / 'input' / sub, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f'  Cleaned up', flush=True)

        print(f'\\n=== ALL SESSIONS COMPLETE ===', flush=True)
        print(f'Completed: {{len(completed)}}/{{total}}', flush=True)
        print(f'Skipped:   {{len(skipped)}}', flush=True)
        print(f'Failed:    {{len(failed)}}', flush=True)
        if failed:
            print(f'Failed sessions: {{failed}}', flush=True)
        update_progress('ALL DONE')
        "

        echo ""
        echo "=== DLC run complete: $(date -u) ==="
        echo "Shutting down in 60 seconds (cancel with: sudo shutdown -c)"
        sleep 60
        shutdown -h now
    """)
    return script


# ---------------------------------------------------------------------------
# EC2 operations (shared with launch_suite2p_ec2.py)
# ---------------------------------------------------------------------------

def ensure_key_pair(ec2) -> str:
    try:
        ec2.describe_key_pairs(KeyNames=[KEY_NAME])
        print(f"Key pair '{KEY_NAME}' already exists")
    except ec2.exceptions.ClientError:
        pem_path = Path.home() / ".ssh" / f"{KEY_NAME}.pem"
        pem_path.parent.mkdir(exist_ok=True)
        resp = ec2.create_key_pair(KeyName=KEY_NAME)
        pem_path.write_text(resp["KeyMaterial"])
        pem_path.chmod(0o600)
        print(f"Created key pair, saved to {pem_path}")
    return KEY_NAME


def ensure_security_group(ec2) -> str:
    try:
        resp = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
        )
        if resp["SecurityGroups"]:
            sg_id = resp["SecurityGroups"][0]["GroupId"]
            print(f"Security group '{SG_NAME}' already exists: {sg_id}")
            return sg_id
    except ec2.exceptions.ClientError:
        pass

    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    resp = ec2.create_security_group(
        GroupName=SG_NAME,
        Description="hm2p cloud run - SSH access",
        VpcId=vpc_id,
    )
    sg_id = resp["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH from anywhere"}],
        }],
    )
    ec2.create_tags(Resources=[sg_id], Tags=[TAG])
    print(f"Created security group: {sg_id}")
    return sg_id


def launch(args):
    ec2 = boto3.client("ec2", region_name=REGION)
    sessions = get_sessions()
    print(f"Will process {len(sessions)} sessions")

    key_name = ensure_key_pair(ec2)
    sg_id = ensure_security_group(ec2)

    use_profile = args.use_profile or has_instance_profile()
    if use_profile:
        print(f"Using IAM instance profile: {INSTANCE_PROFILE_NAME}")
    else:
        print("No IAM instance profile — embedding S3 credentials")

    user_data = build_user_data(sessions, use_instance_profile=use_profile)

    launch_kwargs = {
        "ImageId": AMI_ID,
        "InstanceType": INSTANCE_TYPE,
        "KeyName": key_name,
        "SecurityGroupIds": [sg_id],
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True},
        }],
        "UserData": user_data,
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [TAG, {"Key": "Name", "Value": "hm2p-dlc-run"}],
        }],
    }
    if use_profile:
        launch_kwargs["IamInstanceProfile"] = {"Name": INSTANCE_PROFILE_NAME}

    resp = ec2.run_instances(**launch_kwargs)
    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"\nInstance launched: {instance_id}")
    print(f"Type: {INSTANCE_TYPE}")

    STATE_FILE.write_text(json.dumps({"instance_id": instance_id, "region": REGION}))

    print("Waiting for instance to start...", end="", flush=True)
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    print(" running!")

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    inst = desc["Reservations"][0]["Instances"][0]
    public_ip = inst.get("PublicIpAddress", "no public IP")
    print(f"Public IP: {public_ip}")
    print(f"\nSSH:  ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{public_ip}")
    print(f"Logs: ssh ... 'tail -f /var/log/hm2p-dlc.log'")
    print(f"\nOr run: python scripts/launch_dlc_ec2.py --progress")


def status(args):
    if not STATE_FILE.exists():
        print("No active instance. Run without --status to launch.")
        return
    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=state["region"])
    desc = ec2.describe_instances(InstanceIds=[state["instance_id"]])
    inst = desc["Reservations"][0]["Instances"][0]
    print(f"Instance: {state['instance_id']}")
    print(f"State:    {inst['State']['Name']}")
    print(f"Type:     {inst['InstanceType']}")
    if "PublicIpAddress" in inst:
        print(f"IP:       {inst['PublicIpAddress']}")
        print(f"SSH:      ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{inst['PublicIpAddress']}")


def progress(args):
    s3 = boto3.client("s3", region_name=REGION)
    try:
        resp = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="pose/_progress.json")
        data = json.loads(resp["Body"].read())
        print(f"Status:    {data['status']}")
        print(f"Progress:  {data['completed']}/{data['total']} completed, "
              f"{data['failed']} failed, {data['skipped']} skipped")
        print(f"Updated:   {data['updated']}")
        if data.get("completed_sessions"):
            print(f"\nCompleted: {', '.join(data['completed_sessions'])}")
        if data.get("failed_sessions"):
            print(f"Failed:    {', '.join(data['failed_sessions'])}")
    except Exception:
        print("No progress file found. Processing may not have started yet.")


def terminate(args):
    if not STATE_FILE.exists():
        print("No active instance.")
        return
    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=state["region"])
    ec2.terminate_instances(InstanceIds=[state["instance_id"]])
    print(f"Terminated: {state['instance_id']}")
    STATE_FILE.unlink()


def main():
    parser = argparse.ArgumentParser(description="Launch DLC on EC2 g4dn")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="Check instance status")
    group.add_argument("--progress", action="store_true", help="Check processing progress")
    group.add_argument("--terminate", action="store_true", help="Terminate instance")
    group.add_argument("--dry-run", action="store_true", help="Print user-data")
    parser.add_argument("--use-profile", action="store_true",
                        help="Force use of IAM instance profile")
    args = parser.parse_args()

    if args.status:
        status(args)
    elif args.progress:
        progress(args)
    elif args.terminate:
        terminate(args)
    elif args.dry_run:
        sessions = get_sessions()
        print(build_user_data(sessions))
    else:
        launch(args)


if __name__ == "__main__":
    main()
