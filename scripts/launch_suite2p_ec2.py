#!/usr/bin/env python3
"""Launch an EC2 g4dn.xlarge instance to run Suite2p on all sessions.

Usage (from devcontainer or any machine with boto3 + AWS credentials):
    python scripts/launch_suite2p_ec2.py

The script:
1. Creates an EC2 key pair (for SSH monitoring) — saved to ~/.ssh/hm2p-suite2p.pem
2. Creates a security group allowing SSH
3. Detects IAM instance profile (if setup_ec2_iam.py was run)
4. Launches a g4dn.xlarge instance with the Deep Learning AMI
5. User-data bootstraps: install suite2p, process all sessions, self-terminate

Monitor:
    python scripts/launch_suite2p_ec2.py --status      # instance state + SSH info
    python scripts/launch_suite2p_ec2.py --progress    # S3 progress file
    python scripts/launch_suite2p_ec2.py --logs        # CloudWatch logs (needs IAM profile)
    python scripts/launch_suite2p_ec2.py --terminate   # kill early if needed

Authentication modes:
    - With IAM instance profile (recommended): run setup_ec2_iam.py first.
      Instance gets S3 + CloudWatch access via IAM role. No embedded credentials.
    - Without profile: S3 credentials from ~/.aws/credentials [hm2p-agent] are
      embedded in user-data. CloudWatch logging not available.
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
INSTANCE_TYPE = "g4dn.xlarge"
AMI_ID = "ami-05186a30469f66913"  # Deep Learning Base OSS Nvidia (Ubuntu 22.04) 20260220
KEY_NAME = "hm2p-suite2p"
SG_NAME = "hm2p-suite2p-sg"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"
INSTANCE_PROFILE_NAME = "hm2p-ec2-role"
CW_LOG_GROUP = "/hm2p/suite2p"
TAG = {"Key": "Project", "Value": "hm2p-suite2p"}
STATE_FILE = Path.home() / ".hm2p-suite2p-instance.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_instance_profile() -> bool:
    """Check if the hm2p EC2 instance profile exists."""
    try:
        iam = boto3.client("iam")
        resp = iam.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        roles = resp["InstanceProfile"]["Roles"]
        return len(roles) > 0
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
    """Build the cloud-init user-data script.

    Args:
        sessions: List of session dicts with exp_id, sub, ses keys.
        use_instance_profile: If True, skip embedding AWS credentials (IAM role
            on the instance provides S3 + CloudWatch access). If False, embed
            credentials from ~/.aws/credentials.
    """
    session_json = json.dumps(sessions)

    # Build credentials block
    if use_instance_profile:
        creds_block = textwrap.dedent(f"""\
            # --- AWS credentials via IAM instance profile (no embedded keys) ---
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
            # --- AWS credentials (embedded — use setup_ec2_iam.py to switch to IAM role) ---
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

    # Build CloudWatch log streaming block
    cw_block = textwrap.dedent(f"""\
        # --- CloudWatch Logs ---
        pip3 install --quiet watchtower 2>/dev/null || true
        python3 -c "
        import logging, watchtower
        handler = watchtower.CloudWatchLogHandler(
            log_group_name='{CW_LOG_GROUP}',
            log_stream_name='$(hostname)-$(date +%Y%m%dT%H%M%S)',
        )
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        logging.info('CloudWatch logging initialized')
        " 2>/dev/null && echo "CloudWatch logging enabled" || echo "CloudWatch logging not available (missing IAM permissions or watchtower)"
    """)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        exec > >(tee /var/log/hm2p-suite2p.log) 2>&1

        echo "=== hm2p Suite2p cloud run ==="
        echo "Started: $(date -u)"

{textwrap.indent(creds_block, "        ")}
        # --- System setup ---
        export DEBIAN_FRONTEND=noninteractive

        # Wait for unattended-upgrades / dpkg lock to release (common on Ubuntu AMIs)
        echo "Waiting for dpkg lock..."
        while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
            echo "  dpkg locked, waiting 10s..."
            sleep 10
        done
        echo "dpkg lock free"

        apt-get update -qq
        apt-get install -y -qq python3-pip awscli

        # --- Install suite2p ---
        # Deep Learning AMI has PyTorch+CUDA pre-installed; install suite2p on top
        pip3 install --quiet suite2p

        # --- Verify GPU ---
        nvidia-smi || echo "WARNING: No GPU detected"
        python3 -c "import suite2p; print('suite2p imported OK')" || echo "WARNING: suite2p import failed"

{textwrap.indent(cw_block, "        ")}
        # --- Download custom classifier from S3 ---
        mkdir -p /tmp/hm2p-config
        aws s3 cp s3://{DERIVATIVES_BUCKET}/config/suite2p/classifier_soma.npy /tmp/hm2p-config/classifier_soma.npy
        echo "Custom classifier downloaded"

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
                's3://{DERIVATIVES_BUCKET}/ca_extraction/_progress.json',
            ], capture_output=True)

        for i, ses in enumerate(sessions, 1):
            sub, ses_id = ses['sub'], ses['ses']
            exp_id = ses['exp_id']
            print(f'\\n=== [{{i}}/{{total}}] {{sub}}/{{ses_id}} ({{exp_id}}) ===', flush=True)
            update_progress(f'Processing {{i}}/{{total}}: {{sub}}/{{ses_id}}')

            # Skip if already processed on S3
            check = subprocess.run([
                'aws', 's3', 'ls',
                f's3://{DERIVATIVES_BUCKET}/ca_extraction/{{sub}}/{{ses_id}}/suite2p/plane0/F.npy',
            ], capture_output=True, text=True)
            if check.returncode == 0:
                print(f'  SKIP: already processed on S3', flush=True)
                skipped.append(exp_id)
                continue

            # Create working dirs
            tiff_dir = work / 'input' / sub / ses_id / 'funcimg'
            out_dir = work / 'output' / sub / ses_id
            tiff_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Download TIFFs from S3
            s3_prefix = f'rawdata/{{sub}}/{{ses_id}}/funcimg/'
            print(f'  Downloading from s3://{RAWDATA_BUCKET}/{{s3_prefix}}...', flush=True)
            ret = subprocess.run([
                'aws', 's3', 'sync',
                f's3://{RAWDATA_BUCKET}/{{s3_prefix}}',
                str(tiff_dir),
                '--exclude', '*',
                '--include', '*.tif',
                '--include', '*.tiff',
            ], capture_output=True, text=True)
            if ret.returncode != 0:
                print(f'  ERROR downloading: {{ret.stderr}}', flush=True)
                failed.append(exp_id)
                continue

            tifs = list(tiff_dir.glob('*.tif')) + list(tiff_dir.glob('*.tiff'))
            if not tifs:
                print(f'  SKIP: no TIFFs found', flush=True)
                skipped.append(exp_id)
                continue
            print(f'  Downloaded {{len(tifs)}} TIFF(s), total {{sum(f.stat().st_size for f in tifs)/1e9:.1f}} GB', flush=True)

            # Run Suite2p (1.0 API: db + settings)
            print(f'  Running Suite2p...', flush=True)
            try:
                import numpy as np
                import suite2p
                import suite2p.detection.sparsedetect as sd

                # Patch mode() bug in suite2p 1.0
                if not getattr(sd.find_best_scale, '_patched', False):
                    _orig_fbs = sd.find_best_scale
                    def _patched_fbs(I, spatial_scale):
                        scale, mode = _orig_fbs(I, spatial_scale)
                        if isinstance(scale, np.ndarray):
                            scale = int(scale.item())
                        return scale, mode
                    _patched_fbs._patched = True
                    sd.find_best_scale = _patched_fbs

                settings = suite2p.default_settings()

                # Core imaging parameters (matching legacy pipeline)
                settings['fs'] = 29.97
                settings['tau'] = 0.7  # GCaMP7f decay time
                settings['diameter'] = [12.0, 12.0]

                # Pipeline control
                settings['run']['do_deconvolution'] = False

                # IO — keep registered binary for frontend viewing
                settings['io']['delete_bin'] = False

                # Registration (matching legacy)
                settings['registration']['nonrigid'] = True
                settings['registration']['block_size'] = (32, 32)  # legacy used 32x32
                settings['registration']['batch_size'] = 10000
                settings['registration']['maxregshift'] = 0.3  # legacy: allow 30% shift
                settings['registration']['smooth_sigma'] = 1.15
                settings['registration']['th_badframes'] = 1.0
                settings['registration']['subpixel'] = 10
                settings['registration']['two_step_registration'] = True
                settings['registration']['keep_movie_raw'] = True

                # Detection (matching legacy)
                settings['detection']['sparse_mode'] = True
                settings['detection']['spatial_scale'] = 0  # auto-detect
                settings['detection']['denoise'] = True
                settings['detection']['connected'] = True  # soma masks connected
                settings['detection']['smooth_masks'] = True
                settings['detection']['threshold_scaling'] = 1.0
                settings['detection']['max_overlap'] = 0.75
                settings['detection']['max_iterations'] = 100
                settings['detection']['nbinned'] = 20000

                # Extraction (matching legacy)
                settings['extraction']['batch_size'] = 500
                settings['extraction']['neuropil_extract'] = True
                settings['extraction']['neuropil_coefficient'] = 0.7
                settings['extraction']['inner_neuropil_radius'] = 2
                settings['extraction']['min_neuropil_pixels'] = 100  # legacy: 100 (fewer pixels)
                settings['extraction']['allow_overlap'] = False

                # Classification - custom soma classifier
                classifier = Path('/tmp/hm2p-config/classifier_soma.npy')
                if classifier.exists():
                    settings['classification']['classifier_path'] = str(classifier)
                    settings['classification']['use_builtin_classifier'] = False
                    print(f'  Using custom classifier: {{classifier}}', flush=True)
                else:
                    settings['classification']['use_builtin_classifier'] = True
                    print(f'  WARNING: custom classifier not found, using builtin', flush=True)

                db = {{
                    'data_path': [str(tiff_dir)],
                    'save_path0': str(out_dir),
                    'nplanes': 1,
                    'nchannels': 1,
                }}
                suite2p.run_s2p(db=db, settings=settings)
                print(f'  Suite2p DONE', flush=True)
            except Exception as e:
                print(f'  ERROR in Suite2p: {{e}}', flush=True)
                import traceback
                traceback.print_exc()
                failed.append(exp_id)
                continue

            # Upload results to S3
            s2p_out = out_dir / 'suite2p'
            if s2p_out.exists():
                s3_dest = f's3://{DERIVATIVES_BUCKET}/ca_extraction/{{sub}}/{{ses_id}}/suite2p/'
                print(f'  Uploading to {{s3_dest}}...', flush=True)
                ret = subprocess.run([
                    'aws', 's3', 'sync',
                    str(s2p_out),
                    s3_dest,
                ], capture_output=True, text=True)
                if ret.returncode != 0:
                    print(f'  ERROR uploading: {{ret.stderr}}', flush=True)
                    failed.append(exp_id)
                else:
                    print(f'  Upload DONE', flush=True)
                    completed.append(exp_id)
            else:
                print(f'  WARNING: suite2p output dir not found at {{s2p_out}}', flush=True)
                failed.append(exp_id)

            # Cleanup to save disk space
            shutil.rmtree(work / 'input' / sub, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f'  Cleaned up local files', flush=True)

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
    """Create key pair if it doesn't exist. Returns key name."""
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

    # Get default VPC
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

    # Check for IAM instance profile (provides S3 + CloudWatch without embedded keys)
    use_profile = args.use_profile or has_instance_profile()
    if use_profile:
        print(f"Using IAM instance profile: {INSTANCE_PROFILE_NAME}")
        print("  (no embedded credentials — S3 + CloudWatch via IAM role)")
    else:
        print("No IAM instance profile found — embedding S3 credentials in user-data")
        print("  Run 'python scripts/setup_ec2_iam.py' to create the profile")

    user_data = build_user_data(sessions, use_instance_profile=use_profile)

    # Build run_instances kwargs
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
                "VolumeSize": 100,
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }],
        "UserData": user_data,
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
    print(f"Type: {INSTANCE_TYPE} Spot (~$0.30/hr)")

    # Save state for monitoring
    STATE_FILE.write_text(json.dumps({"instance_id": instance_id, "region": REGION}))

    # Wait for running state
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
    """Check instance status and show recent log output."""
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
        print(f"Logs:     ssh ... 'tail -f /var/log/hm2p-suite2p.log'")

    # Try to get console output
    try:
        console = ec2.get_console_output(InstanceId=state["instance_id"])
        if "Output" in console and console["Output"]:
            lines = console["Output"].strip().split("\n")
            print(f"\n--- Console output (last 20 lines) ---")
            for line in lines[-20:]:
                print(f"  {line}")
    except Exception:
        pass


def terminate(args):
    """Terminate the instance."""
    if not STATE_FILE.exists():
        print("No active instance.")
        return

    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=state["region"])
    ec2.terminate_instances(InstanceIds=[state["instance_id"]])
    print(f"Terminated: {state['instance_id']}")
    STATE_FILE.unlink()


def logs(args):
    """Stream CloudWatch logs from the running instance."""
    logs_client = boto3.client("logs", region_name=REGION)

    # List log streams in the group
    try:
        resp = logs_client.describe_log_streams(
            logGroupName=CW_LOG_GROUP,
            orderBy="LastEventTime",
            descending=True,
            limit=5,
        )
    except logs_client.exceptions.ResourceNotFoundException:
        print(f"Log group '{CW_LOG_GROUP}' not found.")
        print("Run 'python scripts/setup_ec2_iam.py' to create it.")
        return
    except Exception as e:
        print(f"Error accessing CloudWatch Logs: {e}")
        print("The h2mp-agent user may not have CloudWatch permissions.")
        print("Check IAM policies or use --status for SSH-based log access.")
        return

    streams = resp.get("logStreams", [])
    if not streams:
        print(f"No log streams in {CW_LOG_GROUP}")
        print("The instance may not have started logging yet.")
        return

    # Show the most recent stream
    stream_name = streams[0]["logStreamName"]
    print(f"Log stream: {stream_name}")
    print(f"---")

    resp = logs_client.get_log_events(
        logGroupName=CW_LOG_GROUP,
        logStreamName=stream_name,
        startFromHead=False,
        limit=100,
    )
    for event in resp.get("events", []):
        print(event["message"])


def progress(args):
    """Check processing progress from S3 progress file."""
    s3 = boto3.client("s3", region_name=REGION)
    try:
        resp = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="ca_extraction/_progress.json")
        data = json.loads(resp["Body"].read())
        print(f"Status:    {data['status']}")
        print(f"Progress:  {data['completed']}/{data['total']} completed, "
              f"{data['failed']} failed, {data['skipped']} skipped")
        print(f"Updated:   {data['updated']}")
        if data.get("completed_sessions"):
            print(f"\nCompleted: {', '.join(data['completed_sessions'])}")
        if data.get("failed_sessions"):
            print(f"Failed:    {', '.join(data['failed_sessions'])}")
    except s3.exceptions.NoSuchKey:
        print("No progress file found. Processing may not have started yet.")
    except Exception as e:
        print(f"Error reading progress: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Launch Suite2p on EC2 g4dn Spot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="Check instance status")
    group.add_argument("--progress", action="store_true", help="Check processing progress from S3")
    group.add_argument("--logs", action="store_true", help="Stream CloudWatch logs")
    group.add_argument("--terminate", action="store_true", help="Terminate instance")
    group.add_argument("--dry-run", action="store_true", help="Print user-data without launching")
    parser.add_argument("--use-profile", action="store_true",
                        help="Force use of IAM instance profile (skip embedded credentials)")
    args = parser.parse_args()

    if args.status:
        status(args)
    elif args.progress:
        progress(args)
    elif args.logs:
        logs(args)
    elif args.terminate:
        terminate(args)
    elif args.dry_run:
        sessions = get_sessions()
        print(build_user_data(sessions))
    else:
        launch(args)


if __name__ == "__main__":
    main()
