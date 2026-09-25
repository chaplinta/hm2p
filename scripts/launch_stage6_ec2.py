#!/usr/bin/env python3
"""Run Stage 6 (analysis.h5) for all sessions in parallel on a short-lived EC2 instance.

Stage 6 reads ``sync.h5`` per session and writes ``analysis.h5`` for every
signal type present (dff, deconv, deconv_norm, events, events_sd, spikes).
It takes 8-20 min per session on four cores, so all 26 sessions run here in
parallel (``--parallel`` sessions at once) on a 16-vCPU instance instead of
serially in the devcontainer. The instance clones this repository, installs
the runtime dependencies in a uv venv, runs ``scripts/run_stage6_analysis.py
--session <exp_id> --force`` for each session, uploads per-session logs and a
completion marker, and terminates itself.

Usage
-----
    python scripts/launch_stage6_ec2.py --dry-run
    python scripts/launch_stage6_ec2.py --wait
    python scripts/launch_stage6_ec2.py --status | --terminate
"""

from __future__ import annotations

import argparse
import csv
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

INSTANCE_TYPE = "c5.4xlarge"  # 16 vCPU, 32 GB
AMI_ID = "ami-0df4b2961410d4cff"  # Ubuntu 22.04 LTS amd64 (ap-southeast-2)
DONE_KEY = "analysis/_stage6_done.json"
LOG_PREFIX = "analysis/_stage6_logs"
GIT_REPO = "https://github.com/chaplinta/hm2p.git"
STATE_FILE = Path.home() / ".hm2p-stage6-instance.json"
TAG = {"Key": "Project", "Value": "hm2p-stage6"}
MAX_HOURS = 4
RUNTIME_DEPS = "boto3 numpy scipy pandas h5py scikit-learn pyyaml xarray pydantic structlog"


def session_ids(csv_path: Path | None = None) -> list[str]:
    """All exp_ids from metadata/experiments.csv (pipeline stages process every session)."""
    csv_path = csv_path or Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    with open(csv_path) as fh:
        return [row["exp_id"] for row in csv.DictReader(fh)]


def build_user_data(
    exp_ids: list[str],
    parallel: int = 4,
    n_shuffles: int = 500,
    git_branch: str = "main",
    bucket: str = DERIVATIVES_BUCKET,
    creds_block: str = "",
) -> str:
    """Cloud-init script: env, repo, parallel Stage 6, logs, marker, shutdown."""
    ids = " ".join(exp_ids)
    return textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/hm2p-stage6.log) 2>&1
        set -u
        echo "=== hm2p Stage 6 (parallel={parallel}) === $(date -u)"
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
        export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
        mkdir -p /tmp/stage6_logs

        run_one() {{
            exp="$1"
            /opt/hm2p/venv/bin/python -u scripts/run_stage6_analysis.py \\
                --session "$exp" --force --n-shuffles {n_shuffles} \\
                > "/tmp/stage6_logs/$exp.log" 2>&1
            rc=$?
            echo "$exp rc=$rc"
            return $rc
        }}
        export -f run_one
        STATUS="ok"
        printf "%s\\n" {ids} | xargs -P {parallel} -I{{}} bash -c 'run_one {{}}' \\
            | tee /tmp/stage6_logs/_summary.txt
        grep -q "rc=[1-9]" /tmp/stage6_logs/_summary.txt && STATUS="failed"

        FIN="$(date -u +%FT%TZ)"
        N_OK=$(grep -c "rc=0" /tmp/stage6_logs/_summary.txt || true)
        echo "{{\\"status\\": \\"$STATUS\\", \\"finished\\": \\"$FIN\\", \\"n_ok\\": $N_OK}}" \\
            > /tmp/done.json
        aws s3 cp --recursive /tmp/stage6_logs "s3://{bucket}/{LOG_PREFIX}/" || true
        aws s3 cp /var/log/hm2p-stage6.log "s3://{bucket}/{LOG_PREFIX}/_instance.log" || true
        aws s3 cp /tmp/done.json "s3://{bucket}/{DONE_KEY}"
        echo "done: $STATUS ($N_OK ok)"
        shutdown -h now
    """)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--n-shuffles", type=int, default=500)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--terminate", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--timeout-min", type=int, default=180)
    return ap


def launch(args: argparse.Namespace) -> str:  # pragma: no cover - network
    import boto3
    from botocore.exceptions import ClientError

    ec2 = boto3.client("ec2", region_name=REGION)
    ids = session_ids()
    kwargs = dict(parallel=args.parallel, n_shuffles=args.n_shuffles, git_branch=args.branch)
    common = dict(
        ImageId=AMI_ID,
        InstanceType=args.instance_type,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        MinCount=1,
        MaxCount=1,
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[
            {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 60, "VolumeType": "gp3"}}
        ],
        TagSpecifications=[
            {"ResourceType": "instance", "Tags": [TAG, {"Key": "Name", "Value": "hm2p-stage6"}]}
        ],
    )
    try:
        resp = ec2.run_instances(
            UserData=build_user_data(ids, **kwargs),
            IamInstanceProfile={"Name": IAM_PROFILE},
            **common,
        )
    except ClientError as exc:
        print(f"instance profile attach failed ({exc}); embedding S3 credentials instead")
        from ec2_utils import build_creds_block, get_s3_credentials

        key_id, secret, region = get_s3_credentials()
        resp = ec2.run_instances(
            UserData=build_user_data(
                ids, creds_block=build_creds_block(key_id, secret, region), **kwargs
            ),
            **common,
        )
    instance_id = resp["Instances"][0]["InstanceId"]
    STATE_FILE.write_text(json.dumps({"instance_id": instance_id, "region": REGION}))
    print(f"launched {instance_id} ({args.instance_type}, {len(ids)} sessions, self-terminating)")
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
        print(build_user_data(session_ids(), args.parallel, args.n_shuffles, args.branch))
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
