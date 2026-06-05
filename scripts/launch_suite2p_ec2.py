#!/usr/bin/env python3
"""Launch an EC2 c5.2xlarge instance to run Suite2p + ROI classification on all sessions.

Usage (from devcontainer or any machine with boto3 + AWS credentials):
    python scripts/launch_suite2p_ec2.py
    python scripts/launch_suite2p_ec2.py --dry-run    # print user-data without launching
    python scripts/launch_suite2p_ec2.py --status      # instance state + SSH info
    python scripts/launch_suite2p_ec2.py --progress    # S3 progress file
    python scripts/launch_suite2p_ec2.py --terminate   # kill early if needed

The script:
1. Launches a c5.2xlarge CPU Spot instance (Suite2p does not need GPU)
2. Installs hm2p from the git repo (includes suite2p, xgboost, scikit-image)
3. For each session: downloads TIFFs from S3, runs run_suite2p() which
   does Suite2p extraction + XGBoost ROI classification, uploads results
4. Self-terminates when complete
"""

from __future__ import annotations

import argparse
import base64
import configparser
import json
import sys
import textwrap
import time
from pathlib import Path

import boto3

REGION = "ap-southeast-2"
INSTANCE_TYPE = "c5.2xlarge"  # 8 vCPU, 16 GB RAM — CPU-only, no GPU needed
AMI_ID = "ami-0df4b2961410d4cff"  # Ubuntu 22.04 LTS amd64 (ap-southeast-2) — Python 3.10
KEY_NAME = "hm2p-suite2p"
SG_NAME = "hm2p-suite2p-sg"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"
INSTANCE_PROFILE_NAME = "hm2p-ec2-role"
CW_LOG_GROUP = "/hm2p/suite2p"
TAG = {"Key": "Project", "Value": "hm2p-suite2p"}
STATE_FILE = Path.home() / ".hm2p-suite2p-instance.json"
GIT_REPO = "https://github.com/chaplinta/hm2p.git"
GIT_BRANCH = "main"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Build the cloud-init user-data script."""
    session_json = json.dumps(sessions)

    # AWS credentials block
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
        exec > >(tee /var/log/hm2p-suite2p.log) 2>&1

        echo "=== hm2p Suite2p + ROI classification ==="
        echo "Started: $(date -u)"

        upload_log() {{
            aws s3 cp /var/log/hm2p-suite2p.log s3://{DERIVATIVES_BUCKET}/ca_extraction/_suite2p.log 2>/dev/null || true
        }}
        trap upload_log EXIT

{textwrap.indent(creds_block, "        ")}

        # --- System setup ---
        export DEBIAN_FRONTEND=noninteractive

        echo "Waiting for dpkg lock..."
        while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
            sleep 10
        done

        apt-get update -qq
        apt-get install -y -qq python3-pip awscli git libhdf5-dev pkg-config

        # Ubuntu 22.04 has Python 3.10. Use a venv to avoid pip restrictions.
        python3 --version
        python3 -m venv /opt/hm2p

        echo "Installing Stage 1 dependencies..."
        /opt/hm2p/bin/pip install --quiet \
            suite2p \
            xgboost \
            scikit-image \
            scikit-learn \
            joblib \
            numpy \
            scipy \
            pandas \
            h5py \
            tqdm \
            structlog \
            rich \
            typer \
            roiextractors \
            pandera \
            boto3

        echo "Installing hm2p (no-deps)..."
        /opt/hm2p/bin/pip install --quiet --no-deps "git+{GIT_REPO}@{GIT_BRANCH}"

        /opt/hm2p/bin/python -c "import suite2p; print(f'suite2p {{suite2p.__version__}}')"
        /opt/hm2p/bin/python -c "import xgboost; print(f'xgboost {{xgboost.__version__}}')"
        /opt/hm2p/bin/python -c "from hm2p.extraction.roi_classify import classify_session; print('ROI classifier OK')"

        # --- Process sessions ---
        SESSIONS='{session_json}'
        WORK=/tmp/hm2p-work
        mkdir -p $WORK

        echo "$SESSIONS" | /opt/hm2p/bin/python -c "
        import json, sys, subprocess, shutil, datetime, gc
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
                's3://{DERIVATIVES_BUCKET}/ca_extraction/_progress.json',
            ], capture_output=True)

        for i, ses in enumerate(sessions, 1):
            sub, ses_id = ses['sub'], ses['ses']
            exp_id = ses['exp_id']
            print(f'\\n=== [{{i}}/{{total}}] {{sub}}/{{ses_id}} ({{exp_id}}) ===', flush=True)
            update_progress(f'Processing {{i}}/{{total}}: {{sub}}/{{ses_id}}')

            tiff_dir = work / 'input' / sub / ses_id / 'funcimg'
            out_dir = work / 'output' / sub / ses_id
            ts_dir = work / 'timestamps' / sub / ses_id
            tiff_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts_dir.mkdir(parents=True, exist_ok=True)

            # Download TIFFs
            s3_prefix = f'rawdata/{{sub}}/{{ses_id}}/funcimg/'
            print(f'  Downloading TIFFs from s3://{RAWDATA_BUCKET}/{{s3_prefix}}...', flush=True)
            ret = subprocess.run([
                'aws', 's3', 'sync',
                f's3://{RAWDATA_BUCKET}/{{s3_prefix}}',
                str(tiff_dir),
                '--exclude', '*',
                '--include', '*.tif',
                '--include', '*.tiff',
            ], capture_output=True, text=True)
            if ret.returncode != 0:
                print(f'  ERROR downloading TIFFs: {{ret.stderr}}', flush=True)
                failed.append(exp_id)
                continue

            tifs = list(tiff_dir.glob('*.tif')) + list(tiff_dir.glob('*.tiff'))
            if not tifs:
                print(f'  SKIP: no TIFFs found', flush=True)
                skipped.append(exp_id)
                continue
            print(f'  Downloaded {{len(tifs)}} TIFF(s)', flush=True)

            # Download timestamps.h5
            ts_s3 = f's3://{DERIVATIVES_BUCKET}/timestamps/{{sub}}/{{ses_id}}/timestamps.h5'
            ts_local = ts_dir / 'timestamps.h5'
            subprocess.run([
                'aws', 's3', 'cp', ts_s3, str(ts_local),
            ], capture_output=True, text=True)

            # Run Suite2p + ROI classifier via run_suite2p()
            print(f'  Running Suite2p + ROI classifier...', flush=True)
            try:
                from hm2p.extraction.run_suite2p import run_suite2p

                ts_path = ts_local if ts_local.exists() else None
                suite2p_dir = run_suite2p(
                    tiff_dir=tiff_dir,
                    output_dir=out_dir,
                    timestamps_h5=ts_path,
                    indicator='GCaMP6s',
                )
                print(f'  Suite2p + classification DONE', flush=True)
            except Exception as e:
                print(f'  ERROR: {{e}}', flush=True)
                import traceback
                traceback.print_exc()
                failed.append(exp_id)
                continue

            # Upload results to S3
            s2p_out = out_dir / 'suite2p'
            if s2p_out.exists():
                s3_dest = f's3://{DERIVATIVES_BUCKET}/ca_extraction/{{sub}}/{{ses_id}}/suite2p/'
                print(f'  Uploading to {{s3_dest}}...', flush=True)
                subprocess.run(['aws', 's3', 'rm', '--recursive', s3_dest], capture_output=True)
                ret = subprocess.run([
                    'aws', 's3', 'sync', str(s2p_out), s3_dest,
                ], capture_output=True, text=True)
                if ret.returncode != 0:
                    print(f'  ERROR uploading: {{ret.stderr}}', flush=True)
                    failed.append(exp_id)
                else:
                    print(f'  Upload DONE', flush=True)
                    completed.append(exp_id)
            else:
                print(f'  WARNING: suite2p output dir not found', flush=True)
                failed.append(exp_id)

            # Cleanup
            shutil.rmtree(work / 'input' / sub, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            gc.collect()

        print(f'\\n=== ALL SESSIONS COMPLETE ===', flush=True)
        print(f'Completed: {{len(completed)}}/{{total}}', flush=True)
        print(f'Skipped:   {{len(skipped)}}', flush=True)
        print(f'Failed:    {{len(failed)}}', flush=True)
        if failed:
            print(f'Failed sessions: {{failed}}', flush=True)
        update_progress('ALL DONE')
        "

        echo ""
        echo "=== Suite2p run complete: $(date -u) ==="
        echo "Shutting down in 60 seconds (cancel with: sudo shutdown -c)"
        sleep 60
        shutdown -h now
    """)
    return script


# ---------------------------------------------------------------------------
# EC2 operations
# ---------------------------------------------------------------------------


def ensure_key_pair(ec2) -> str:
    """Create key pair if it doesn't exist."""
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
    """Create security group with SSH access if it doesn't exist."""
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
        Description="hm2p Suite2p cloud run - SSH access",
        VpcId=vpc_id,
    )
    sg_id = resp["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH from anywhere"}],
        }],
    )
    ec2.create_tags(Resources=[sg_id], Tags=[TAG])
    print(f"Created security group: {sg_id}")
    return sg_id


def launch(args):
    """Launch the Spot instance."""
    ec2 = boto3.client("ec2", region_name=REGION)

    sessions = get_sessions()
    print(f"Will process {len(sessions)} sessions")

    key_name = ensure_key_pair(ec2)
    sg_id = ensure_security_group(ec2)

    use_profile = args.use_profile or has_instance_profile()
    if use_profile:
        print(f"Using IAM instance profile: {INSTANCE_PROFILE_NAME}")
    else:
        print("No IAM instance profile — embedding S3 credentials in user-data")

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
            "Ebs": {
                "VolumeSize": 200,
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }],
        "UserData": user_data,
        "InstanceInitiatedShutdownBehavior": "stop",  # stop (not terminate) so we can debug failures
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                TAG,
                {"Key": "Name", "Value": "hm2p-suite2p-run"},
            ],
        }],
    }

    if use_profile:
        launch_kwargs["IamInstanceProfile"] = {"Name": INSTANCE_PROFILE_NAME}

    resp = ec2.run_instances(**launch_kwargs)

    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"\nInstance launched: {instance_id}")
    print(f"Type: {INSTANCE_TYPE} (~$0.14 USD/hr on-demand)")

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
    print(f"Logs: ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{public_ip} 'tail -f /var/log/hm2p-suite2p.log'")
    print(f"\nOr run: python scripts/launch_suite2p_ec2.py --status")


def status(args):
    """Check instance status."""
    if not STATE_FILE.exists():
        print("No active instance. Run without --status to launch.")
        return

    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=state["region"])

    try:
        desc = ec2.describe_instances(InstanceIds=[state["instance_id"]])
        inst = desc["Reservations"][0]["Instances"][0]
        inst_state = inst["State"]["Name"]
        public_ip = inst.get("PublicIpAddress", "N/A")
        launch_time = inst.get("LaunchTime", "")
        print(f"Instance: {state['instance_id']}")
        print(f"State: {inst_state}")
        print(f"IP: {public_ip}")
        print(f"Launched: {launch_time}")
        if inst_state == "running":
            print(f"\nSSH: ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{public_ip}")
    except Exception as e:
        print(f"Error checking status: {e}")


def progress(args):
    """Download and show the S3 progress file."""
    import tempfile
    s3 = boto3.client("s3", region_name=REGION)
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        try:
            s3.download_file(DERIVATIVES_BUCKET, "ca_extraction/_progress.json", f.name)
            data = json.loads(Path(f.name).read_text())
            print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"No progress file found: {e}")


def terminate(args):
    """Terminate the instance."""
    if not STATE_FILE.exists():
        print("No active instance.")
        return

    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=state["region"])

    print(f"Terminating {state['instance_id']}...")
    ec2.terminate_instances(InstanceIds=[state["instance_id"]])
    print("Terminated.")
    STATE_FILE.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Launch Suite2p + ROI classifier on EC2")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="Check instance status")
    group.add_argument("--progress", action="store_true", help="Show S3 progress")
    group.add_argument("--terminate", action="store_true", help="Terminate instance")
    group.add_argument("--dry-run", action="store_true", help="Print user-data without launching")
    parser.add_argument("--use-profile", action="store_true", help="Force IAM instance profile")
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
