#!/usr/bin/env python3
"""Run CASCADE spike inference for all sessions on a short-lived EC2 instance.

Builds the same environment as docs/manual-installs.md (uv venv, Python 3.11,
TensorFlow 2.x with legacy Keras, Cascade from source), pulls the pretrained
models from ``s3://hm2p-derivatives/models/cascade/``, clones this repository
for ``scripts/run_cascade.py`` and the session registry, runs every session
(skipping those that already have ``spikes``), uploads the log and a
completion marker, and terminates itself.

Usage
-----
    python scripts/launch_cascade_ec2.py --dry-run   # print user-data
    python scripts/launch_cascade_ec2.py --wait      # launch and poll
    python scripts/launch_cascade_ec2.py --status
    python scripts/launch_cascade_ec2.py --terminate

Reference: Rupprecht et al. 2021. "A database and deep learning toolbox for
noise-optimized, generalized spike inference from calcium imaging."
Nature Neuroscience 24:1324-1337. doi:10.1038/s41593-021-00895-5
https://github.com/HelmchenLabSoftware/Cascade
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec2_constants import (  # noqa: E402
    DERIVATIVES_BUCKET,
    IAM_PROFILE,
    KEY_NAME,
    REGION,
    SG_ID,
)

INSTANCE_TYPE = "c5.4xlarge"  # 16 vCPU, 32 GB; ~0.85 USD/h on-demand
AMI_ID = "ami-0df4b2961410d4cff"  # Ubuntu 22.04 LTS amd64 (ap-southeast-2)
MODELS_PREFIX = "models/cascade"
DONE_KEY = "calcium/_cascade_done.json"
LOG_KEY = "calcium/_cascade.log"
DEFAULT_MODEL = "Global_EXC_10Hz_smoothing200ms"
GIT_REPO = "https://github.com/chaplinta/hm2p.git"
CASCADE_REPO = "https://github.com/HelmchenLabSoftware/Cascade.git"
STATE_FILE = Path.home() / ".hm2p-cascade-run-instance.json"
TAG = {"Key": "Project", "Value": "hm2p-cascade"}
MAX_HOURS = 3


def build_user_data(
    model: str = DEFAULT_MODEL,
    bucket: str = DERIVATIVES_BUCKET,
    chunk: int = 64,
    git_branch: str = "main",
    creds_block: str = "",
    overwrite: bool = False,
) -> str:
    """Cloud-init script: build env, fetch models + repo, run all sessions, shut down."""
    overwrite_flag = " --overwrite" if overwrite else ""
    return textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/hm2p-cascade.log) 2>&1
        set -u
        echo "=== hm2p CASCADE run === $(date -u)"
        export AWS_DEFAULT_REGION={REGION}
        export HOME=/root
{textwrap.indent(creds_block, "        ")}
        # hard timeout: never run longer than {MAX_HOURS} h
        ( sleep {MAX_HOURS * 3600}; echo "hard timeout"; shutdown -h now ) &

        apt-get update -qq && apt-get install -y -qq git curl unzip awscli > /dev/null
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="/root/.local/bin:$PATH"
        mkdir -p /opt/hm2p && cd /opt/hm2p
        git clone -q --depth 1 --branch {git_branch} {GIT_REPO} repo
        git clone -q --depth 1 {CASCADE_REPO} cascade-src
        uv venv -q -p 3.11 venv
        uv pip install -q --python venv/bin/python "tensorflow>=2.15" tf-keras pip \\
            h5py numpy scipy boto3 ruamel.yaml pyyaml tqdm matplotlib seaborn

        mkdir -p cascade-src/Pretrained_models/{model}
        aws s3 cp "s3://{bucket}/{MODELS_PREFIX}/{model}.zip" model.zip
        unzip -q -o model.zip -d cascade-src/Pretrained_models/{model}
        test -f cascade-src/Pretrained_models/{model}/config.yaml || {{ echo "model missing"; exit 1; }}

        STATUS="ok"
        cd repo
        TF_USE_LEGACY_KERAS=1 PYTHONPATH=/opt/hm2p/cascade-src ../venv/bin/python -u \\
            scripts/run_cascade.py --cascade-src /opt/hm2p/cascade-src --model {model} \\
            --chunk {chunk}{overwrite_flag} || STATUS="failed"

        FIN="$(date -u +%FT%TZ)"
        echo "{{\\"status\\": \\"$STATUS\\", \\"finished\\": \\"$FIN\\", \\"model\\": \\"{model}\\"}}" \\
            > /tmp/done.json
        aws s3 cp /tmp/done.json "s3://{bucket}/{DONE_KEY}"
        aws s3 cp /var/log/hm2p-cascade.log "s3://{bucket}/{LOG_KEY}" || true
        echo "done: $STATUS"
        shutdown -h now
    """)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing spikes")
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--terminate", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--timeout-min", type=int, default=120)
    return ap


def launch(args: argparse.Namespace) -> str:  # pragma: no cover - network
    import boto3
    from botocore.exceptions import ClientError

    ec2 = boto3.client("ec2", region_name=REGION)
    kwargs = dict(
        model=args.model, chunk=args.chunk, git_branch=args.branch, overwrite=args.overwrite
    )
    common = dict(
        ImageId=AMI_ID,
        InstanceType=args.instance_type,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        MinCount=1,
        MaxCount=1,
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[
            {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 40, "VolumeType": "gp3"}}
        ],
        TagSpecifications=[
            {"ResourceType": "instance", "Tags": [TAG, {"Key": "Name", "Value": "hm2p-cascade"}]}
        ],
    )
    try:
        resp = ec2.run_instances(
            UserData=build_user_data(**kwargs), IamInstanceProfile={"Name": IAM_PROFILE}, **common
        )
    except ClientError as exc:
        print(f"instance profile attach failed ({exc}); embedding S3 credentials instead")
        from ec2_utils import build_creds_block, get_s3_credentials

        key_id, secret, region = get_s3_credentials()
        resp = ec2.run_instances(
            UserData=build_user_data(
                creds_block=build_creds_block(key_id, secret, region), **kwargs
            ),
            **common,
        )
    instance_id = resp["Instances"][0]["InstanceId"]
    STATE_FILE.write_text(json.dumps({"instance_id": instance_id, "region": REGION}))
    print(f"launched {instance_id} ({args.instance_type}, self-terminating)")
    return instance_id


def instance_state() -> str:  # pragma: no cover - network
    import boto3

    if not STATE_FILE.exists():
        return "no state file"
    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=REGION)
    inst = ec2.describe_instances(InstanceIds=[state["instance_id"]])["Reservations"][0]
    return f"{state['instance_id']}: {inst['Instances'][0]['State']['Name']}"


def terminate() -> None:  # pragma: no cover - network
    import boto3

    if not STATE_FILE.exists():
        print("no state file")
        return
    state = json.loads(STATE_FILE.read_text())
    boto3.client("ec2", region_name=REGION).terminate_instances(InstanceIds=[state["instance_id"]])
    print(f"terminate requested for {state['instance_id']}")


def wait_for_done(timeout_min: int) -> dict | None:  # pragma: no cover - network
    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    deadline = time.time() + 60 * timeout_min
    while time.time() < deadline:
        try:
            obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=DONE_KEY)
            return json.loads(obj["Body"].read())
        except Exception as exc:  # noqa: BLE001
            if "NoSuchKey" not in str(exc) and "404" not in str(exc):
                print(f"poll error: {exc}")
        print(f"  waiting... ({instance_state()})")
        time.sleep(60)
    return None


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - network entry point
    args = _build_arg_parser().parse_args(argv)
    if args.status:
        print(instance_state())
        return 0
    if args.terminate:
        terminate()
        return 0
    if args.dry_run:
        print(build_user_data(args.model, chunk=args.chunk, git_branch=args.branch))
        return 0
    launch(args)
    if args.wait:
        done = wait_for_done(args.timeout_min)
        if done is None:
            print("timed out waiting for completion marker")
            return 1
        print(f"run finished: {done}")
        return 0 if done.get("status") == "ok" else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
