#!/usr/bin/env python3
"""Run one cell-type programme runner on a short-lived EC2 instance.

Runs ``scripts/run_celltype_programme.py <runner> --signal <signal>`` once per
requested signal (in parallel, one process each) on an instance that clones
this repository, installs the runtime dependencies in a uv venv, uploads each
output folder and log to ``s3://hm2p-derivatives/celltype_programme/<runner>/
<signal>/``, writes a completion marker and terminates itself.

Usage
-----
    python scripts/launch_celltype_runner_ec2.py mazerun --dry-run
    python scripts/launch_celltype_runner_ec2.py mazerun --signals spikes dff --wait
    python scripts/launch_celltype_runner_ec2.py mazerun --download
    python scripts/launch_celltype_runner_ec2.py --status | --terminate
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

INSTANCE_TYPE = "c5.2xlarge"  # 8 vCPU, 16 GB
AMI_ID = "ami-0df4b2961410d4cff"  # Ubuntu 22.04 LTS amd64 (ap-southeast-2)
S3_PREFIX = "celltype_programme"
GIT_REPO = "https://github.com/chaplinta/hm2p.git"
STATE_FILE = Path.home() / ".hm2p-celltype-runner-instance.json"
TAG = {"Key": "Project", "Value": "hm2p-celltype"}
MAX_HOURS = 3
RUNTIME_DEPS = "boto3 numpy scipy pandas h5py scikit-learn pyyaml xarray pydantic structlog"
SIGNALS = ("dff", "event_masks", "spikes")
REPO = Path(__file__).resolve().parent.parent


def done_key(runner: str) -> str:
    """S3 key of the completion marker for *runner*."""
    return f"{S3_PREFIX}/{runner}/_done.json"


def build_user_data(
    runner: str,
    signals: list[str],
    extra_args: str = "",
    git_branch: str = "main",
    bucket: str = DERIVATIVES_BUCKET,
    creds_block: str = "",
) -> str:
    """Cloud-init script: env, repo, one runner process per signal, uploads, marker, shutdown."""
    bad = [s for s in signals if s not in SIGNALS]
    if bad or not signals:
        raise ValueError(f"unknown or empty signals: {bad or signals}")
    sigs = " ".join(signals)
    dest = f"s3://{bucket}/{S3_PREFIX}/{runner}"
    return textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/hm2p-celltype.log) 2>&1
        set -u
        echo "=== hm2p celltype runner {runner} ({sigs}) === $(date -u)"
        export AWS_DEFAULT_REGION={REGION}
        export HOME=/root
{textwrap.indent(creds_block, "        ")}
        ( sleep {MAX_HOURS * 3600}; echo "hard timeout"; shutdown -h now ) &

        apt-get update -qq && apt-get install -y -qq git curl awscli > /dev/null
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="/root/.local/bin:$PATH"
        mkdir -p /opt/hm2p && cd /opt/hm2p
        git clone -q --depth 1 --branch {git_branch} {GIT_REPO} repo
        uv venv -q -p 3.12 venv
        uv pip install -q --python venv/bin/python {RUNTIME_DEPS}
        cd repo
        export PYTHONPATH=/opt/hm2p/repo/src
        export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
        echo "commit $(git rev-parse --short HEAD)"

        run_one() {{
            sig="$1"
            out="/tmp/out_$sig"
            mkdir -p "$out"
            /opt/hm2p/venv/bin/python -u scripts/run_celltype_programme.py {runner} \\
                --signal "$sig" --out "$out" {extra_args} > "$out/run.log" 2>&1
            rc=$?
            echo "$sig rc=$rc"
            aws s3 cp --recursive "$out" "{dest}/$sig/" > /dev/null 2>&1 || true
            return $rc
        }}
        export -f run_one
        printf "%s\\n" {sigs} | xargs -P {len(signals)} -I{{}} bash -c 'run_one {{}}' \\
            | tee /tmp/summary.txt
        STATUS="ok"
        grep -q "rc=[1-9]" /tmp/summary.txt && STATUS="failed"

        FIN="$(date -u +%FT%TZ)"
        COMMIT="$(git rev-parse --short HEAD)"
        echo "{{\\"status\\": \\"$STATUS\\", \\"finished\\": \\"$FIN\\", \\"commit\\": \\"$COMMIT\\"}}" \\
            > /tmp/done.json
        aws s3 cp /var/log/hm2p-celltype.log "{dest}/_instance.log" || true
        aws s3 cp /tmp/done.json "{dest}/_done.json"
        echo "done: $STATUS"
        shutdown -h now
    """)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runner", nargs="?", default=None, help="programme runner key, e.g. mazerun")
    ap.add_argument("--signals", nargs="+", default=["spikes"], choices=SIGNALS)
    ap.add_argument("--extra-args", default="", help="passed through to the runner")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--terminate", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--download", action="store_true", help="fetch outputs to results/")
    ap.add_argument("--timeout-min", type=int, default=120)
    return ap


def local_dir(runner: str, signal: str) -> Path:
    """Local results folder for one runner/signal output."""
    return REPO / "results" / f"celltype_programme_{runner}_ec2" / signal


def launch(args: argparse.Namespace) -> str:  # pragma: no cover - network
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=REGION)
    s3.delete_object(Bucket=DERIVATIVES_BUCKET, Key=done_key(args.runner))
    ec2 = boto3.client("ec2", region_name=REGION)
    kwargs = dict(extra_args=args.extra_args, git_branch=args.branch)
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
            {
                "ResourceType": "instance",
                "Tags": [TAG, {"Key": "Name", "Value": f"hm2p-celltype-{args.runner}"}],
            }
        ],
    )
    try:
        resp = ec2.run_instances(
            UserData=build_user_data(args.runner, args.signals, **kwargs),
            IamInstanceProfile={"Name": IAM_PROFILE},
            **common,
        )
    except ClientError as exc:
        print(f"instance profile attach failed ({exc}); embedding S3 credentials instead")
        from ec2_utils import build_creds_block, get_s3_credentials

        key_id, secret, region = get_s3_credentials()
        resp = ec2.run_instances(
            UserData=build_user_data(
                args.runner,
                args.signals,
                creds_block=build_creds_block(key_id, secret, region),
                **kwargs,
            ),
            **common,
        )
    instance_id = resp["Instances"][0]["InstanceId"]
    STATE_FILE.write_text(json.dumps({"instance_id": instance_id, "runner": args.runner}))
    print(f"launched {instance_id} ({args.instance_type}, {args.runner} {args.signals})")
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


def wait_for_done(runner: str, timeout_min: int) -> dict | None:  # pragma: no cover - network
    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    deadline = time.time() + 60 * timeout_min
    while time.time() < deadline:
        try:
            obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=done_key(runner))
            return json.loads(obj["Body"].read())
        except Exception as exc:  # noqa: BLE001
            if "NoSuchKey" not in str(exc) and "404" not in str(exc):
                print(f"poll error: {exc}")
        print(f"  waiting... ({instance_state()})")
        time.sleep(60)
    return None


def download(runner: str, signals: list[str]) -> None:  # pragma: no cover - network
    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    for sig in signals:
        prefix = f"{S3_PREFIX}/{runner}/{sig}/"
        pages = s3.get_paginator("list_objects_v2").paginate(
            Bucket=DERIVATIVES_BUCKET, Prefix=prefix
        )
        for page in pages:
            for obj in page.get("Contents", []):
                dest = local_dir(runner, sig) / obj["Key"][len(prefix) :]
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(DERIVATIVES_BUCKET, obj["Key"], str(dest))
        print(f"downloaded {prefix} -> {local_dir(runner, sig)}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - network entry point
    args = _build_arg_parser().parse_args(argv)
    if args.status:
        print(instance_state())
        return 0
    if args.terminate:
        terminate()
        return 0
    if args.runner is None:
        print("runner is required")
        return 2
    if args.dry_run:
        print(build_user_data(args.runner, args.signals, args.extra_args, args.branch))
        return 0
    if args.download:
        download(args.runner, args.signals)
        return 0
    launch(args)
    if args.wait:
        done = wait_for_done(args.runner, args.timeout_min)
        if done is None:
            print("timed out waiting for completion marker")
            return 1
        print(f"run finished: {done}")
        download(args.runner, args.signals)
        return 0 if done.get("status") == "ok" else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
