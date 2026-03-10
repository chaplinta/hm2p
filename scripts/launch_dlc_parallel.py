#!/usr/bin/env python3
"""Launch parallel EC2 Spot instances to run DLC pose estimation.

Splits 26 sessions across N parallel g5.xlarge (A10G GPU) Spot instances.
Each instance processes its shard independently and uploads results to S3.

Usage:
    python scripts/launch_dlc_parallel.py                    # launch 4 instances
    python scripts/launch_dlc_parallel.py -n 2               # launch 2 instances
    python scripts/launch_dlc_parallel.py --instance-type g4dn.xlarge  # use T4 instead
    python scripts/launch_dlc_parallel.py --on-demand        # skip Spot, use On-Demand
    python scripts/launch_dlc_parallel.py --progress         # check all shards
    python scripts/launch_dlc_parallel.py --status           # instance info
    python scripts/launch_dlc_parallel.py --terminate        # kill all instances
    python scripts/launch_dlc_parallel.py --dry-run          # print user-data shard 0

Quota requirements (request in AWS Console → Service Quotas → EC2):
    - "All G and VT Spot Instance Requests": need N × 4 vCPUs (e.g. 16 for 4 instances)
    - "Running On-Demand G and VT instances": same, if using --on-demand
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import textwrap
from pathlib import Path

import boto3

REGION = "ap-southeast-2"
DEFAULT_INSTANCE_TYPE = "g5.xlarge"  # A10G 24GB — ~2x faster than T4
FALLBACK_INSTANCE_TYPE = "g4dn.xlarge"  # T4 16GB — fallback
AMI_ID = "ami-05186a30469f66913"  # Deep Learning Base OSS Nvidia (Ubuntu 22.04)
KEY_NAME = "hm2p-suite2p"
SG_NAME = "hm2p-suite2p-sg"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"
INSTANCE_PROFILE_NAME = "hm2p-ec2-role"
TAG_PROJECT = {"Key": "Project", "Value": "hm2p-dlc"}
STATE_FILE = Path.home() / ".hm2p-dlc-parallel.json"

# A10G can handle larger batches (24GB vs 16GB VRAM)
BATCH_SIZES = {
    "g5.xlarge": {"batch_size": 96, "detector_batch_size": 24},
    "g5.2xlarge": {"batch_size": 96, "detector_batch_size": 24},
    "g4dn.xlarge": {"batch_size": 64, "detector_batch_size": 16},
    "g4dn.2xlarge": {"batch_size": 64, "detector_batch_size": 16},
    "p3.2xlarge": {"batch_size": 128, "detector_batch_size": 32},
}


def has_instance_profile() -> bool:
    try:
        iam = boto3.client("iam")
        resp = iam.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        return len(resp["InstanceProfile"]["Roles"]) > 0
    except Exception:
        return False


def get_s3_credentials() -> tuple[str, str, str]:
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


def split_sessions(sessions: list[dict], n_shards: int) -> list[list[dict]]:
    """Split sessions into N roughly equal shards."""
    shard_size = math.ceil(len(sessions) / n_shards)
    return [sessions[i:i + shard_size] for i in range(0, len(sessions), shard_size)]


def build_user_data(
    shard_sessions: list[dict],
    shard_id: int,
    n_shards: int,
    instance_type: str,
    use_instance_profile: bool = False,
) -> str:
    """Build cloud-init script for one shard."""
    session_json = json.dumps(shard_sessions)
    batch_cfg = BATCH_SIZES.get(instance_type, BATCH_SIZES["g4dn.xlarge"])
    batch_size = batch_cfg["batch_size"]
    detector_batch_size = batch_cfg["detector_batch_size"]

    if use_instance_profile:
        creds_block = textwrap.dedent(f"""\
            mkdir -p /root/.aws
            printf '[default]\\nregion = {REGION}\\noutput = json\\n' > /root/.aws/config
        """)
    else:
        key_id, secret, region = get_s3_credentials()
        creds_block = textwrap.dedent(f"""\
            mkdir -p /root/.aws
            printf '[default]\\naws_access_key_id = {key_id}\\naws_secret_access_key = {secret}\\n' > /root/.aws/credentials
            printf '[default]\\nregion = {region}\\noutput = json\\n' > /root/.aws/config
        """)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/hm2p-dlc.log) 2>&1

        echo "=== hm2p DLC shard {shard_id}/{n_shards} ({len(shard_sessions)} sessions) ==="
        echo "Instance type: {instance_type}"
        echo "Batch sizes: pose={batch_size}, detector={detector_batch_size}"
        echo "Started: $(date -u)"

        upload_log() {{
            aws s3 cp /var/log/hm2p-dlc.log \\
                s3://{DERIVATIVES_BUCKET}/pose/_dlc_log_shard{shard_id}.txt 2>/dev/null || true
        }}
        trap upload_log EXIT

{textwrap.indent(creds_block, "        ")}
        export DEBIAN_FRONTEND=noninteractive

        # Wait for dpkg lock
        for i in $(seq 1 30); do
            fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
            echo "  dpkg locked, waiting 10s... ($i/30)"
            sleep 10
        done

        apt-get update -qq
        apt-get install -y -qq python3-pip python3-dev awscli ffmpeg || true

        pip3 install --break-system-packages --quiet --pre deeplabcut
        echo "DLC install exit code: $?"

        nvidia-smi || echo "WARNING: No GPU detected"
        python3 -c "import deeplabcut; print(f'DLC version: {{deeplabcut.__version__}}')" || true
        python3 -c "import torch; print(f'PyTorch {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}, Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}')" || true

        upload_log

        SESSIONS='{session_json}'
        WORK=/tmp/hm2p-work
        mkdir -p $WORK

        echo "$SESSIONS" | python3 -c "
        import json, sys, subprocess, shutil, os, datetime
        from pathlib import Path

        sessions = json.load(sys.stdin)
        work = Path('/tmp/hm2p-work')
        total = len(sessions)
        shard_id = {shard_id}
        n_shards = {n_shards}
        completed = []
        failed = []
        failed_errors = {{}}
        skipped = []

        def update_progress(status_msg=''):
            progress = {{
                'shard_id': shard_id,
                'n_shards': n_shards,
                'total': total,
                'completed': len(completed),
                'failed': len(failed),
                'skipped': len(skipped),
                'completed_sessions': completed,
                'failed_sessions': failed,
                'failed_errors': failed_errors,
                'status': status_msg,
                'updated': datetime.datetime.utcnow().isoformat() + 'Z',
            }}
            progress_file = work / 'progress.json'
            progress_file.write_text(json.dumps(progress, indent=2))
            subprocess.run([
                'aws', 's3', 'cp', str(progress_file),
                f's3://{DERIVATIVES_BUCKET}/pose/_progress_shard{{shard_id}}.json',
            ], capture_output=True)

        for i, ses in enumerate(sessions, 1):
            sub, ses_id = ses['sub'], ses['ses']
            exp_id = ses['exp_id']
            print(f'\\n=== SHARD {{shard_id}} [{{i}}/{{total}}] {{sub}}/{{ses_id}} ===', flush=True)
            update_progress(f'Shard {{shard_id}}: {{i}}/{{total}} {{sub}}/{{ses_id}}')

            # Skip if already on S3
            check = subprocess.run([
                'aws', 's3', 'ls',
                f's3://{DERIVATIVES_BUCKET}/pose/{{sub}}/{{ses_id}}/',
            ], capture_output=True, text=True)
            if check.returncode == 0 and ('.h5' in check.stdout or '.csv' in check.stdout):
                print(f'  SKIP: already processed', flush=True)
                skipped.append(exp_id)
                continue

            video_dir = work / 'input' / sub / ses_id / 'behav'
            out_dir = work / 'output' / sub / ses_id
            video_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            s3_prefix = f'rawdata/{{sub}}/{{ses_id}}/behav/'
            print(f'  Downloading video...', flush=True)
            ret = subprocess.run([
                'aws', 's3', 'sync',
                f's3://{RAWDATA_BUCKET}/{{s3_prefix}}',
                str(video_dir),
                '--exclude', '*', '--include', '*.mp4', '--exclude', '*side*',
            ], capture_output=True, text=True)
            if ret.returncode != 0:
                err_msg = ret.stderr[:500] if ret.stderr else 'download failed'
                print(f'  ERROR downloading: {{err_msg}}', flush=True)
                failed.append(exp_id)
                failed_errors[exp_id] = f'download: {{err_msg}}'
                continue

            mp4s = list(video_dir.glob('*overhead*.mp4')) + list(video_dir.glob('*cropped*.mp4'))
            if not mp4s:
                mp4s = list(video_dir.glob('*.mp4'))
            if not mp4s:
                print(f'  SKIP: no .mp4 found', flush=True)
                skipped.append(exp_id)
                continue

            video_path = mp4s[0]
            print(f'  Video: {{video_path.name}} ({{video_path.stat().st_size/1e6:.1f}} MB)', flush=True)

            # Subsample to 30fps
            subsampled_path = video_dir / f'{{video_path.stem}}_30fps.mp4'
            sub_ret = subprocess.run([
                'ffmpeg', '-y', '-i', str(video_path),
                '-r', '30', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '18', str(subsampled_path),
            ], capture_output=True, text=True)
            if sub_ret.returncode == 0 and subsampled_path.exists():
                dlc_video = subsampled_path
                print(f'  Subsampled to 30fps', flush=True)
            else:
                dlc_video = video_path

            print(f'  Running DLC (batch={batch_size}, det_batch={detector_batch_size})...', flush=True)
            try:
                import deeplabcut
                deeplabcut.video_inference_superanimal(
                    [str(dlc_video)],
                    superanimal_name='superanimal_topviewmouse',
                    model_name='hrnet_w32',
                    detector_name='fasterrcnn_resnet50_fpn_v2',
                    videotype='.mp4',
                    dest_folder=str(out_dir),
                    batch_size={batch_size},
                    detector_batch_size={detector_batch_size},
                    plot_trajectories=False,
                    create_labeled_video=False,
                )
                import json as _json
                meta = {{'tracking_fps': 30 if dlc_video == subsampled_path else 100,
                         'original_fps': 100,
                         'model': 'superanimal_topviewmouse_hrnet_w32',
                         'detector': 'fasterrcnn_resnet50_fpn_v2',
                         'instance_type': '{instance_type}',
                         'shard_id': {shard_id}}}
                (out_dir / 'dlc_meta.json').write_text(_json.dumps(meta, indent=2))
                print(f'  DLC DONE', flush=True)
            except Exception as e:
                err_msg = str(e)
                print(f'  ERROR: {{err_msg}}', flush=True)
                import traceback
                traceback.print_exc()
                failed.append(exp_id)
                failed_errors[exp_id] = err_msg[:500]
                continue

            out_files = list(out_dir.glob('*.h5')) + list(out_dir.glob('*.csv')) + list(out_dir.glob('*.json'))
            if out_files:
                s3_dest = f's3://{DERIVATIVES_BUCKET}/pose/{{sub}}/{{ses_id}}/'
                ret = subprocess.run([
                    'aws', 's3', 'sync', str(out_dir), s3_dest,
                ], capture_output=True, text=True)
                if ret.returncode != 0:
                    failed.append(exp_id)
                    failed_errors[exp_id] = f'upload: {{ret.stderr[:500]}}'
                else:
                    completed.append(exp_id)
                    print(f'  Uploaded to S3', flush=True)
            else:
                all_files = list(out_dir.rglob('*'))
                print(f'  WARNING: no output. Files: {{[str(f) for f in all_files[:20]]}}', flush=True)
                failed.append(exp_id)

            shutil.rmtree(work / 'input' / sub, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

        print(f'\\n=== SHARD {{shard_id}} COMPLETE: {{len(completed)}}/{{total}} done, {{len(failed)}} failed ===', flush=True)
        update_progress(f'SHARD {{shard_id}} DONE')
        "

        echo ""
        echo "=== Shard {shard_id} complete: $(date -u) ==="
        echo "Shutting down in 60 seconds"
        sleep 60
        shutdown -h now
    """)
    return script


# ---------------------------------------------------------------------------
# EC2 operations
# ---------------------------------------------------------------------------

def ensure_key_pair(ec2) -> str:
    try:
        ec2.describe_key_pairs(KeyNames=[KEY_NAME])
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
            return resp["SecurityGroups"][0]["GroupId"]
    except ec2.exceptions.ClientError:
        pass

    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    resp = ec2.create_security_group(
        GroupName=SG_NAME, Description="hm2p cloud run - SSH access", VpcId=vpc_id,
    )
    sg_id = resp["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH from anywhere"}],
        }],
    )
    ec2.create_tags(Resources=[sg_id], Tags=[TAG_PROJECT])
    print(f"Created security group: {sg_id}")
    return sg_id


def launch(args):
    ec2 = boto3.client("ec2", region_name=REGION)
    sessions = get_sessions()
    n_shards = args.n_instances
    instance_type = args.instance_type

    shards = split_sessions(sessions, n_shards)
    # Filter out empty shards
    shards = [s for s in shards if s]
    n_shards = len(shards)

    print(f"Sessions: {len(sessions)} total → {n_shards} shards")
    for i, shard in enumerate(shards):
        print(f"  Shard {i}: {len(shard)} sessions")
    print(f"Instance type: {instance_type}")
    print(f"Spot: {not args.on_demand}")

    key_name = ensure_key_pair(ec2)
    sg_id = ensure_security_group(ec2)
    use_profile = args.use_profile or has_instance_profile()

    batch_cfg = BATCH_SIZES.get(instance_type, BATCH_SIZES["g4dn.xlarge"])
    print(f"Batch config: pose={batch_cfg['batch_size']}, detector={batch_cfg['detector_batch_size']}")

    instances = []
    for shard_id, shard_sessions in enumerate(shards):
        user_data = build_user_data(
            shard_sessions, shard_id, n_shards, instance_type,
            use_instance_profile=use_profile,
        )

        launch_kwargs = {
            "ImageId": AMI_ID,
            "InstanceType": instance_type,
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
                "Tags": [
                    TAG_PROJECT,
                    {"Key": "Name", "Value": f"hm2p-dlc-shard{shard_id}"},
                    {"Key": "Shard", "Value": str(shard_id)},
                ],
            }],
        }
        if use_profile:
            launch_kwargs["IamInstanceProfile"] = {"Name": INSTANCE_PROFILE_NAME}

        # Spot instance request
        if not args.on_demand:
            launch_kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }

        try:
            resp = ec2.run_instances(**launch_kwargs)
            iid = resp["Instances"][0]["InstanceId"]
            instances.append({"instance_id": iid, "shard_id": shard_id})
            print(f"  Shard {shard_id}: launched {iid}")
        except Exception as e:
            print(f"  Shard {shard_id}: FAILED to launch — {e}")
            # If Spot fails, suggest On-Demand
            if "InsufficientInstanceCapacity" in str(e) or "SpotMaxPriceTooLow" in str(e):
                print(f"    → Try --on-demand or a different instance type")

    if not instances:
        print("\nNo instances launched. Check your quota and try again.")
        return

    # Save state
    state = {
        "instances": instances,
        "instance_type": instance_type,
        "n_shards": n_shards,
        "region": REGION,
        "spot": not args.on_demand,
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))

    # Wait for all to start
    instance_ids = [i["instance_id"] for i in instances]
    print(f"\nWaiting for {len(instance_ids)} instances to start...", end="", flush=True)
    waiter = ec2.get_waiter("instance_running")
    try:
        waiter.wait(InstanceIds=instance_ids)
        print(" all running!")
    except Exception as e:
        print(f" error: {e}")

    # Print IPs
    desc = ec2.describe_instances(InstanceIds=instance_ids)
    for res in desc["Reservations"]:
        for inst in res["Instances"]:
            iid = inst["InstanceId"]
            ip = inst.get("PublicIpAddress", "no-ip")
            shard = next(
                (i["shard_id"] for i in instances if i["instance_id"] == iid), "?"
            )
            print(f"  Shard {shard}: {iid} @ {ip}")

    vcpus = len(instances) * 4
    print(f"\nTotal: {len(instances)} instances, {vcpus} vCPUs")
    print(f"Estimated time: ~{math.ceil(max(len(s) for s in shards) * (1.5 if 'g5' in instance_type else 3))}h")
    print(f"\nMonitor: python scripts/launch_dlc_parallel.py --progress")


def status(args):
    if not STATE_FILE.exists():
        print("No active instances. Run without --status to launch.")
        return
    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=state["region"])
    ids = [i["instance_id"] for i in state["instances"]]
    desc = ec2.describe_instances(InstanceIds=ids)

    print(f"Cluster: {state['n_shards']} shards, {state['instance_type']}, "
          f"{'Spot' if state.get('spot') else 'On-Demand'}")
    print()
    for res in desc["Reservations"]:
        for inst in res["Instances"]:
            iid = inst["InstanceId"]
            ip = inst.get("PublicIpAddress", "-")
            st = inst["State"]["Name"]
            shard = next(
                (i["shard_id"] for i in state["instances"] if i["instance_id"] == iid), "?"
            )
            color = "running" if st == "running" else st
            print(f"  Shard {shard}: {iid} | {color} | {ip}")
            if st == "running":
                print(f"    SSH: ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{ip}")


def progress(args):
    s3 = boto3.client("s3", region_name=REGION)

    # Try parallel progress files first
    shard_files = []
    for i in range(20):  # Check up to 20 shards
        key = f"pose/_progress_shard{i}.json"
        try:
            resp = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=key)
            data = json.loads(resp["Body"].read())
            shard_files.append(data)
        except Exception:
            break

    if shard_files:
        total_completed = sum(d["completed"] for d in shard_files)
        total_failed = sum(d["failed"] for d in shard_files)
        total_skipped = sum(d["skipped"] for d in shard_files)
        total_sessions = sum(d["total"] for d in shard_files)

        print(f"=== Parallel DLC Progress ({len(shard_files)} shards) ===")
        print(f"Overall: {total_completed}/{total_sessions} completed, "
              f"{total_failed} failed, {total_skipped} skipped")
        print()
        for d in shard_files:
            sid = d["shard_id"]
            print(f"  Shard {sid}: {d['completed']}/{d['total']} done | {d['status']}")
            if d.get("failed_sessions"):
                for fs in d["failed_sessions"]:
                    print(f"    FAILED: {fs}")
        return

    # Fall back to single-instance progress
    try:
        resp = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="pose/_progress.json")
        data = json.loads(resp["Body"].read())
        print(f"Status:    {data['status']}")
        print(f"Progress:  {data['completed']}/{data['total']} completed, "
              f"{data['failed']} failed, {data['skipped']} skipped")
        print(f"Updated:   {data['updated']}")
    except Exception:
        print("No progress files found. Processing may not have started yet.")


def terminate(args):
    if not STATE_FILE.exists():
        print("No active instances.")
        return
    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=state["region"])
    ids = [i["instance_id"] for i in state["instances"]]
    ec2.terminate_instances(InstanceIds=ids)
    print(f"Terminated {len(ids)} instances: {ids}")
    STATE_FILE.unlink()


def main():
    parser = argparse.ArgumentParser(description="Launch parallel DLC on EC2 Spot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true")
    group.add_argument("--progress", action="store_true")
    group.add_argument("--terminate", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    parser.add_argument("-n", "--n-instances", type=int, default=4,
                        help="Number of parallel instances (default: 4)")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                        help=f"EC2 instance type (default: {DEFAULT_INSTANCE_TYPE})")
    parser.add_argument("--on-demand", action="store_true",
                        help="Use On-Demand instead of Spot")
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
        shards = split_sessions(sessions, args.n_instances)
        print(f"# Shard 0 of {len(shards)} ({len(shards[0])} sessions)")
        print(build_user_data(shards[0], 0, len(shards), args.instance_type))
    else:
        launch(args)


if __name__ == "__main__":
    main()
