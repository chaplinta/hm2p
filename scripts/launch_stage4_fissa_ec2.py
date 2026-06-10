#!/usr/bin/env python3
"""Launch an EC2 instance to reprocess Stage 4 ca.h5 with FISSA neuropil subtraction.

Background
----------
The 26 ca.h5 currently on S3 are not uniform: some sessions used true FISSA
(Keemink et al. 2018) neuropil decontamination, the rest used the fixed-0.7
coefficient. That processing-version difference is confounded with cell type, so
this script reprocesses the fixed-0.7 sessions with FISSA to make the dataset
uniform (see docs / project notes).

FISSA pins ``scikit-learn < 1.2`` whereas the ROI classifier needs
``scikit-learn >= 1.4``; the two cannot coexist in one environment. The instance
therefore builds two virtual environments:

* ``/opt/hm2p``  — main pipeline: suite2p, xgboost, scikit-learn>=1.4, hm2p.
                   Re-registers the movie, builds masks, runs dF/F0 + ca.h5.
* ``/opt/fissa`` — isolated FISSA env: fissa (+ scikit-learn<1.2), hm2p --no-deps.
                   Runs only ``scripts/fissa_bridge.py`` to produce F_corr.npy.

The per-session orchestration lives in ``scripts/run_stage4_fissa.py``
(``run_session_fissa``); this launcher only provisions the instance and drives the
session loop.

Recommended workflow
--------------------
1. ``--session SUB SES --validate-only`` — single-session alignment validation:
   re-registers one session, checks that the regenerated movie reproduces the
   stored F.npy (Spearman), writes NO ca.h5. Confirms registration determinism
   before committing to the batch.
2. ``--session SUB SES`` — full single-session reprocess (writes one ca.h5).
3. ``--all-fixed`` — batch: every session whose ca.h5 is not already FISSA.

Usage:
    python scripts/launch_stage4_fissa_ec2.py --session sub-XXXX ses-YYYY --validate-only
    python scripts/launch_stage4_fissa_ec2.py --session sub-XXXX ses-YYYY
    python scripts/launch_stage4_fissa_ec2.py --all-fixed
    python scripts/launch_stage4_fissa_ec2.py --dry-run --session sub-XXXX ses-YYYY
    python scripts/launch_stage4_fissa_ec2.py --status
    python scripts/launch_stage4_fissa_ec2.py --progress
    python scripts/launch_stage4_fissa_ec2.py --terminate
"""

from __future__ import annotations

import argparse
import configparser
import json
import textwrap
from pathlib import Path

import boto3

REGION = "ap-southeast-2"
INSTANCE_TYPE = "c5.2xlarge"  # 8 vCPU, 16 GB RAM — Suite2p re-registration + FISSA, CPU only
AMI_ID = "ami-0df4b2961410d4cff"  # Ubuntu 22.04 LTS amd64 (ap-southeast-2) — Python 3.10
KEY_NAME = "hm2p-suite2p"
SG_NAME = "hm2p-suite2p-sg"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"
INSTANCE_PROFILE_NAME = "hm2p-ec2-role"
TAG = {"Key": "Project", "Value": "hm2p-fissa"}
STATE_FILE = Path.home() / ".hm2p-fissa-instance.json"
GIT_REPO = "https://github.com/chaplinta/hm2p.git"
GIT_BRANCH = "feat/fissa-reprocessing"
PROGRESS_KEY = "calcium/_fissa_progress.json"

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
    """Read the full session list from metadata/experiments.csv."""
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


def build_creds_block(use_instance_profile: bool) -> str:
    """Build the AWS-credentials portion of the user-data script."""
    if use_instance_profile:
        return textwrap.dedent(f"""\
            mkdir -p /root/.aws
            cat > /root/.aws/config << 'CONF'
            [default]
            region = {REGION}
            output = json
            CONF
            sed -i 's/^            //' /root/.aws/config
            echo "Using IAM instance profile for AWS access"
        """)
    key_id, secret, region = get_s3_credentials()
    return textwrap.dedent(f"""\
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


def build_user_data(
    sessions: list[dict],
    *,
    use_instance_profile: bool = False,
    validate_only: bool = False,
    all_fixed: bool = False,
    alignment_threshold: float = 0.9,
) -> str:
    """Build the cloud-init user-data script.

    Parameters
    ----------
    sessions : list of dict
        Sessions to process (``sub``/``ses``/``exp_id``). For ``--all-fixed`` this
        is the full list; the on-instance loop skips any already on FISSA.
    use_instance_profile : bool
        Use the IAM instance profile instead of embedded credentials.
    validate_only : bool
        Pass ``--validate-only`` to the driver (alignment check, no ca.h5 write).
    all_fixed : bool
        If True, the loop reprocesses every session not already on FISSA. If
        False, it processes exactly the sessions passed.
    alignment_threshold : float
        Minimum median Spearman accepted by the driver's alignment gate.
    """
    creds_block = build_creds_block(use_instance_profile)
    session_json = json.dumps(sessions)
    skip_existing = "1" if all_fixed else "0"

    script = textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/hm2p-fissa.log) 2>&1

        echo "=== hm2p Stage 4 FISSA reprocessing ==="
        echo "Started: $(date -u)"

        upload_log() {{
            aws s3 cp /var/log/hm2p-fissa.log s3://{DERIVATIVES_BUCKET}/calcium/_fissa.log 2>/dev/null || true
        }}
        trap upload_log EXIT

{textwrap.indent(creds_block, "        ")}

        export DEBIAN_FRONTEND=noninteractive
        echo "Waiting for dpkg lock..."
        while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 10; done
        apt-get update -qq
        apt-get install -y -qq python3-pip python3-venv awscli git libhdf5-dev pkg-config

        # --- Clone repo (need scripts/, not just the installed package) ---
        git clone --branch {GIT_BRANCH} --depth 1 {GIT_REPO} /opt/hm2p-repo

        # --- Main env: suite2p re-registration + ROI classifier + dF/F0 + ca.h5 ---
        python3 -m venv /opt/hm2p
        /opt/hm2p/bin/pip install --quiet \\
            suite2p xgboost scikit-image "scikit-learn>=1.4" joblib \\
            numpy scipy pandas h5py tqdm structlog rich typer \\
            roiextractors pandera boto3
        /opt/hm2p/bin/pip install --quiet --no-deps "git+{GIT_REPO}@{GIT_BRANCH}"

        # --- Isolated FISSA env: scikit-learn<1.2 with an ABI-matched numpy.
        # Pin an explicit, mutually-compatible numpy/scipy/scikit-learn trio
        # (scikit-learn 1.1.x era) so the resolver cannot mix a sklearn binary
        # with an incompatible numpy ABI ('numpy.dtype size changed').
        python3 -m venv /opt/fissa
        /opt/fissa/bin/pip install --quiet \\
            "numpy==1.23.5" "scipy==1.9.3" "scikit-learn==1.1.3" fissa h5py tifffile
        /opt/fissa/bin/pip install --quiet --no-deps "git+{GIT_REPO}@{GIT_BRANCH}"

        /opt/hm2p/bin/python -c "import suite2p, sklearn; print('main env OK; sklearn', sklearn.__version__)"
        /opt/fissa/bin/python -c "import fissa, sklearn; print('fissa env OK; sklearn', sklearn.__version__)" || echo "WARN: fissa env import failed (not needed for --validate-only)"

        # --- Process sessions ---
        SESSIONS='{session_json}'
        WORK=/tmp/hm2p-fissa
        mkdir -p $WORK

        echo "$SESSIONS" | /opt/hm2p/bin/python -c "
        import json, sys, subprocess, datetime, tempfile
        from pathlib import Path

        sessions = json.load(sys.stdin)
        work = Path('/tmp/hm2p-fissa')
        skip_existing = {skip_existing}
        total = len(sessions)
        done, rejected, skipped, failed = [], [], [], []

        def already_fissa(sub, ses):
            # Inspect the current ca.h5 neuropil_method attribute on S3.
            import h5py
            url = f's3://{DERIVATIVES_BUCKET}/calcium/{{sub}}/{{ses}}/ca.h5'
            with tempfile.TemporaryDirectory() as td:
                local = Path(td) / 'ca.h5'
                r = subprocess.run(['aws','s3','cp',url,str(local)], capture_output=True)
                if r.returncode != 0 or not local.exists():
                    return False
                try:
                    with h5py.File(local,'r') as h:
                        return str(h.attrs.get('neuropil_method','')) == 'fissa'
                except Exception:
                    return False

        def update_progress(msg=''):
            prog = {{
                'total': total, 'done': len(done), 'rejected': len(rejected),
                'skipped': len(skipped), 'failed': len(failed),
                'done_sessions': done, 'rejected_sessions': rejected,
                'failed_sessions': failed, 'status': msg,
                'updated': datetime.datetime.utcnow().isoformat() + 'Z',
            }}
            pf = work / 'progress.json'
            pf.write_text(json.dumps(prog, indent=2))
            subprocess.run(['aws','s3','cp',str(pf),
                            's3://{DERIVATIVES_BUCKET}/{PROGRESS_KEY}'], capture_output=True)

        sys.path.insert(0, '/opt/hm2p-repo/src')
        sys.path.insert(0, '/opt/hm2p-repo/scripts')
        from run_stage4_fissa import run_session_fissa

        for i, s in enumerate(sessions, 1):
            sub, ses, exp = s['sub'], s['ses'], s['exp_id']
            print(f'\\n=== [{{i}}/{{total}}] {{sub}}/{{ses}} ({{exp}}) ===', flush=True)
            update_progress(f'Processing {{i}}/{{total}}: {{sub}}/{{ses}}')
            if skip_existing and already_fissa(sub, ses):
                print('  SKIP: ca.h5 already on FISSA', flush=True)
                skipped.append(exp); continue
            try:
                res = run_session_fissa(
                    sub, ses, work_dir=work,
                    fissa_python='/opt/fissa/bin/python',
                    validate_only={'True' if validate_only else 'False'},
                    alignment_threshold={alignment_threshold},
                )
                al = res.get('alignment', {{}})
                st = res['status']
                ms = al.get('median_spearman')
                print(f'  status={{st}} median_spearman={{ms}}', flush=True)
                if res['status'] in ('done','validated'):
                    done.append(exp)
                elif res['status'] == 'rejected':
                    rejected.append(exp)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f'  ERROR: {{e}}', flush=True)
                failed.append(exp)

        print(f'\\n=== COMPLETE: done={{len(done)}} rejected={{len(rejected)}} '
              f'skipped={{len(skipped)}} failed={{len(failed)}} ===', flush=True)
        update_progress('ALL DONE')
        "

        echo "=== FISSA reprocessing complete: $(date -u) ==="
        echo "Shutting down in 60 seconds (cancel with: sudo shutdown -c)"
        sleep 60
        shutdown -h now
    """)
    return script


# ---------------------------------------------------------------------------
# EC2 operations (key pair + security group reused from the Suite2p launcher)
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
    """Reuse (or create) the hm2p SSH security group."""
    resp = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
    )
    if resp["SecurityGroups"]:
        sg_id = resp["SecurityGroups"][0]["GroupId"]
        print(f"Security group '{SG_NAME}' already exists: {sg_id}")
        return sg_id

    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]
    resp = ec2.create_security_group(
        GroupName=SG_NAME, Description="hm2p cloud run - SSH access", VpcId=vpc_id
    )
    sg_id = resp["GroupId"]
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
        }],
    )
    ec2.create_tags(Resources=[sg_id], Tags=[TAG])
    print(f"Created security group: {sg_id}")
    return sg_id


def resolve_sessions(args) -> list[dict]:
    """Select the sessions to send to the instance from CLI args."""
    if args.all_fixed:
        return get_sessions()
    if args.session:
        sub, ses = args.session
        exp = f"{sub}/{ses}"
        return [{"sub": sub, "ses": ses, "exp_id": exp}]
    raise SystemExit("Provide --session SUB SES or --all-fixed")


def launch(args):
    """Launch the Spot/on-demand instance."""
    ec2 = boto3.client("ec2", region_name=REGION)
    sessions = resolve_sessions(args)
    print(f"Will process {len(sessions)} session(s)"
          + (" (validate-only)" if args.validate_only else ""))

    key_name = ensure_key_pair(ec2)
    sg_id = ensure_security_group(ec2)

    use_profile = args.use_profile or has_instance_profile()
    print("Using IAM instance profile" if use_profile
          else "Embedding S3 credentials in user-data")

    user_data = build_user_data(
        sessions,
        use_instance_profile=use_profile,
        validate_only=args.validate_only,
        all_fixed=args.all_fixed,
        alignment_threshold=args.alignment_threshold,
    )

    launch_kwargs = {
        "ImageId": AMI_ID,
        "InstanceType": INSTANCE_TYPE,
        "KeyName": key_name,
        "SecurityGroupIds": [sg_id],
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True},
        }],
        "UserData": user_data,
        "InstanceInitiatedShutdownBehavior": "stop",  # stop so failures can be debugged
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [TAG, {"Key": "Name", "Value": "hm2p-fissa-run"}],
        }],
    }
    if use_profile:
        launch_kwargs["IamInstanceProfile"] = {"Name": INSTANCE_PROFILE_NAME}

    resp = ec2.run_instances(**launch_kwargs)
    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"\nInstance launched: {instance_id}")
    print(f"Type: {INSTANCE_TYPE} (~$0.34 USD/hr on-demand [~$0.52 AUD])")

    STATE_FILE.write_text(json.dumps({"instance_id": instance_id, "region": REGION}))

    print("Waiting for instance to start...", end="", flush=True)
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    print(" running!")

    desc = ec2.describe_instances(InstanceIds=[instance_id])
    inst = desc["Reservations"][0]["Instances"][0]
    public_ip = inst.get("PublicIpAddress", "no public IP")
    print(f"Public IP: {public_ip}")
    print(f"\nSSH:  ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{public_ip}")
    print(f"Logs: ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{public_ip} 'tail -f /var/log/hm2p-fissa.log'")
    print("Or:   python scripts/launch_stage4_fissa_ec2.py --progress")


def status(args):
    """Check instance status."""
    if not STATE_FILE.exists():
        print("No active instance.")
        return
    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=state["region"])
    try:
        desc = ec2.describe_instances(InstanceIds=[state["instance_id"]])
        inst = desc["Reservations"][0]["Instances"][0]
        print(f"Instance: {state['instance_id']}")
        print(f"State: {inst['State']['Name']}")
        print(f"IP: {inst.get('PublicIpAddress', 'N/A')}")
        print(f"Launched: {inst.get('LaunchTime', '')}")
    except Exception as e:
        print(f"Error checking status: {e}")


def progress(args):
    """Download and show the S3 progress file."""
    import tempfile

    s3 = boto3.client("s3", region_name=REGION)
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        try:
            s3.download_file(DERIVATIVES_BUCKET, PROGRESS_KEY, f.name)
            print(json.dumps(json.loads(Path(f.name).read_text()), indent=2))
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
    parser = argparse.ArgumentParser(description="Launch Stage 4 FISSA reprocessing on EC2")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true")
    group.add_argument("--progress", action="store_true")
    group.add_argument("--terminate", action="store_true")
    group.add_argument("--dry-run", action="store_true",
                       help="Print user-data without launching")
    parser.add_argument("--session", nargs=2, metavar=("SUB", "SES"),
                        help="Process one session: sub-XXXX ses-YYYYMMDDTHHMMSS")
    parser.add_argument("--all-fixed", action="store_true",
                        help="Process all sessions not already on FISSA")
    parser.add_argument("--validate-only", action="store_true",
                        help="Alignment check only; write no ca.h5")
    parser.add_argument("--alignment-threshold", type=float, default=0.9)
    parser.add_argument("--use-profile", action="store_true",
                        help="Force IAM instance profile")
    args = parser.parse_args()

    if args.status:
        status(args)
    elif args.progress:
        progress(args)
    elif args.terminate:
        terminate(args)
    elif args.dry_run:
        print(build_user_data(
            resolve_sessions(args),
            validate_only=args.validate_only,
            all_fixed=args.all_fixed,
            alignment_threshold=args.alignment_threshold,
        ))
    else:
        launch(args)


if __name__ == "__main__":
    main()
