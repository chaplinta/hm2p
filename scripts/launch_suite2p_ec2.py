#!/usr/bin/env python3
"""Launch an EC2 g4dn.xlarge Spot instance to run Suite2p on all sessions.

Usage (from devcontainer or any machine with boto3 + AWS credentials):
    python scripts/launch_suite2p_ec2.py

The script:
1. Creates an EC2 key pair (for SSH monitoring) — saved to ~/.ssh/hm2p-suite2p.pem
2. Creates a security group allowing SSH
3. Launches a g4dn.xlarge Spot instance with the Deep Learning AMI
4. User-data bootstraps: install suite2p, process all sessions, self-terminate

Monitor:
    python scripts/launch_suite2p_ec2.py --status
    python scripts/launch_suite2p_ec2.py --logs       # streams CloudWatch or console output
    python scripts/launch_suite2p_ec2.py --terminate   # kill early if needed

S3 credentials for the instance are read from ~/.aws/credentials [hm2p-agent] profile
(S3-only access). EC2 launch uses the [default] profile (needs EC2 permissions).
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
TAG = {"Key": "Project", "Value": "hm2p-suite2p"}
STATE_FILE = Path.home() / ".hm2p-suite2p-instance.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def build_user_data(sessions: list[dict]) -> str:
    """Build the cloud-init user-data script."""
    key_id, secret, region = get_s3_credentials()

    session_json = json.dumps(sessions)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        exec > >(tee /var/log/hm2p-suite2p.log) 2>&1

        echo "=== hm2p Suite2p cloud run ==="
        echo "Started: $(date -u)"

        # --- AWS credentials (S3-only) ---
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
        # Fix indentation in heredoc
        sed -i 's/^        //' /root/.aws/credentials /root/.aws/config

        # --- System setup ---
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq python3-pip awscli

        # --- Install suite2p ---
        pip3 install --quiet suite2p

        # --- Verify GPU ---
        nvidia-smi || echo "WARNING: No GPU detected"
        python3 -c "import suite2p; print(f'suite2p {{suite2p.__version__}}')"

        # --- Download custom classifier from S3 ---
        mkdir -p /tmp/hm2p-config
        aws s3 cp s3://{DERIVATIVES_BUCKET}/config/suite2p/classifier_soma.npy /tmp/hm2p-config/classifier_soma.npy
        echo "Custom classifier downloaded"

        # --- Process sessions ---
        SESSIONS='{session_json}'
        WORK=/tmp/hm2p-work
        mkdir -p $WORK

        echo "$SESSIONS" | python3 -c "
        import json, sys, subprocess, shutil, os
        from pathlib import Path

        sessions = json.load(sys.stdin)
        work = Path('/tmp/hm2p-work')
        total = len(sessions)
        completed = []
        failed = []
        skipped = []

        for i, ses in enumerate(sessions, 1):
            sub, ses_id = ses['sub'], ses['ses']
            exp_id = ses['exp_id']
            print(f'\\n=== [{{i}}/{{total}}] {{sub}}/{{ses_id}} ({{exp_id}}) ===', flush=True)

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

                # Core imaging parameters (matching legacy ops_default.npy)
                settings['fs'] = 29.97
                settings['tau'] = 1.0
                settings['diameter'] = [12.0, 12.0]

                # Pipeline control
                settings['run']['do_deconvolution'] = False

                # IO
                settings['io']['delete_bin'] = True

                # Registration (matching legacy)
                settings['registration']['nonrigid'] = True
                settings['registration']['block_size'] = (128, 128)
                settings['registration']['batch_size'] = 100
                settings['registration']['maxregshift'] = 0.1
                settings['registration']['smooth_sigma'] = 1.15
                settings['registration']['th_badframes'] = 1.0
                settings['registration']['subpixel'] = 10

                # Detection (matching legacy)
                settings['detection']['threshold_scaling'] = 1.0
                settings['detection']['max_overlap'] = 0.75
                settings['detection']['sparsery_settings']['highpass_neuropil'] = 25

                # Extraction (matching legacy)
                settings['extraction']['batch_size'] = 500
                settings['extraction']['neuropil_extract'] = True
                settings['extraction']['neuropil_coefficient'] = 0.7
                settings['extraction']['inner_neuropil_radius'] = 2
                settings['extraction']['min_neuropil_pixels'] = 350
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
    user_data = build_user_data(sessions)

    # Request On-Demand instance (Spot quota is 0 on new accounts)
    resp = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=key_name,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": 100,  # 100 GB root volume
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }],
        UserData=user_data,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                TAG,
                {"Key": "Name", "Value": "hm2p-suite2p-run"},
            ],
        }],
    )

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Launch Suite2p on EC2 g4dn Spot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="Check instance status")
    group.add_argument("--terminate", action="store_true", help="Terminate instance")
    group.add_argument("--dry-run", action="store_true", help="Print user-data without launching")
    args = parser.parse_args()

    if args.status:
        status(args)
    elif args.terminate:
        terminate(args)
    elif args.dry_run:
        sessions = get_sessions()
        print(build_user_data(sessions))
    else:
        launch(args)


if __name__ == "__main__":
    main()
