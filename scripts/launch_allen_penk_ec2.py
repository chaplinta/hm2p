#!/usr/bin/env python3
"""Query Allen Institute datasets for Penk+ RSP neurons on a short-lived EC2 instance.

The devcontainer cannot reach the Allen hosts or the us-west-2 ABC Atlas bucket,
so ``scripts/allen_penk_query.py`` runs on an r5.2xlarge (64 GB RAM, for
backed AnnData slicing). The instance installs a uv venv (Python 3.11) with
anndata, h5py, pandas, numpy, scipy, boto3, pyarrow and ``abc_atlas_access``
(git), clones this repository, runs each requested part as a separate process
under its own time limit, uploads outputs, ``manifest.json`` and logs to
``s3://hm2p-derivatives/allen/penk/`` after every part, writes ``_done.json``
and terminates itself (hard limit 6 h).

Because the query script may not yet be committed, the launcher by default
stages the local copy at ``allen/penk/_code/allen_penk_query.py`` before
launching; the instance prefers that copy and falls back to the repository
clone (``--no-stage-script`` uses the clone only). Either way the clone's
``src/`` directory is exported on ``PYTHONPATH`` so the script can import
``hm2p.analysis.continuum`` (numpy only) for the continuum vs discrete
analyses; that module must exist on the cloned branch.

Usage
-----
    python scripts/launch_allen_penk_ec2.py --dry-run          # print user-data
    python scripts/launch_allen_penk_ec2.py --wait             # launch and poll
    python scripts/launch_allen_penk_ec2.py --parts A B --wait
    python scripts/launch_allen_penk_ec2.py --status | --terminate

Datasets: Yao et al. 2023 Nature doi:10.1038/s41586-023-06812-z; Zhang et al.
2023 Nature doi:10.1038/s41586-023-06808-9; Scala et al. 2021 Nature
doi:10.1038/s41586-020-2907-3; Gouwens et al. 2020 Cell
doi:10.1016/j.cell.2020.09.057 (see allen_penk_query.py for full citations).
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

INSTANCE_TYPE = "r5.2xlarge"  # 8 vCPU, 64 GB; ~0.65 USD/h on-demand (ap-southeast-2)
AMI_ID = "ami-0df4b2961410d4cff"  # Ubuntu 22.04 LTS amd64 (ap-southeast-2)
VOLUME_GB = 300
PREFIX = "allen/penk"
DONE_KEY = f"{PREFIX}/_done.json"
LOG_KEY = f"{PREFIX}/_instance.log"
CODE_KEY = f"{PREFIX}/_code/allen_penk_query.py"
GIT_REPO = "https://github.com/chaplinta/hm2p.git"
REPO_SRC = "/opt/hm2p/repo/src"  # on PYTHONPATH so the query script imports hm2p.analysis
ABC_PKG = "abc_atlas_access @ git+https://github.com/AllenInstitute/abc_atlas_access.git"
RUNTIME_DEPS = "anndata h5py pandas numpy scipy boto3 pyarrow"
STATE_FILE = Path.home() / ".hm2p-allen-penk-instance.json"
TAG = {"Key": "Project", "Value": "hm2p-allen-penk"}
MAX_HOURS = 6
PARTS = ("A", "B", "C")
# Per-part limits (minutes); sum stays below the hard timeout.
PART_TIMEOUT_MIN = {"A": 180, "B": 120, "C": 45}
LOCAL_SCRIPT = Path(__file__).resolve().parent / "allen_penk_query.py"


def build_user_data(
    parts: list[str] | tuple[str, ...] = PARTS,
    bucket: str = DERIVATIVES_BUCKET,
    git_branch: str = "main",
    creds_block: str = "",
    staged_script: bool = True,
) -> str:
    """Cloud-init script: env, code, one process per part, uploads, marker, shutdown."""
    bad = [p for p in parts if p not in PART_TIMEOUT_MIN]
    if bad or not parts:
        raise ValueError(f"unknown parts {bad}; choose from {PARTS}")
    s3_uri = f"s3://{bucket}/{PREFIX}"
    part_list = " ".join(parts)
    timeouts = " ".join(f"[{p}]={PART_TIMEOUT_MIN[p] * 60}" for p in parts)
    stage = (
        f'aws s3 cp "s3://{bucket}/{CODE_KEY}" /opt/hm2p/allen_penk_query.py '
        f'&& SCRIPT=/opt/hm2p/allen_penk_query.py && echo "using staged script"'
        if staged_script
        else 'echo "using repository script"'
    )
    return textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/hm2p-allen-penk.log) 2>&1
        set -u
        echo "=== hm2p Allen Penk query (parts: {part_list}) === $(date -u)"
        export AWS_DEFAULT_REGION={REGION}
        export HOME=/root
{textwrap.indent(creds_block, "        ")}
        # hard timeout: never run longer than {MAX_HOURS} h
        ( sleep {MAX_HOURS * 3600}; echo "hard timeout"; \\
          aws s3 cp /var/log/hm2p-allen-penk.log "s3://{bucket}/{LOG_KEY}" || true; \\
          shutdown -h now ) &

        apt-get update -qq && apt-get install -y -qq git curl awscli > /dev/null
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="/root/.local/bin:$PATH"
        mkdir -p /opt/hm2p && cd /opt/hm2p
        git clone -q --depth 1 --branch {git_branch} {GIT_REPO} repo
        # shared numpy-only analysis modules (hm2p.analysis.continuum) from the clone
        export PYTHONPATH={REPO_SRC}
        SCRIPT=/opt/hm2p/repo/scripts/allen_penk_query.py
        {stage}
        test -f "$SCRIPT" || {{ echo "query script missing"; shutdown -h now; exit 1; }}
        uv venv -q -p 3.11 venv
        uv pip install -q --python venv/bin/python {RUNTIME_DEPS} "{ABC_PKG}"
        venv/bin/python -c "import anndata, abc_atlas_access; print('anndata', anndata.__version__)"
        venv/bin/python -c "import hm2p.analysis.continuum" \\
            || echo "WARNING: hm2p.analysis.continuum not importable; continuum outputs will fail"

        # periodic instance-log upload (every 5 min)
        ( while true; do sleep 300; \\
            aws s3 cp /var/log/hm2p-allen-penk.log "s3://{bucket}/{LOG_KEY}" > /dev/null 2>&1 || true; \\
          done ) &

        OUT=/opt/hm2p/allen_out
        mkdir -p "$OUT"
        declare -A TMO=({timeouts})
        for part in {part_list}; do
            echo "--- part $part (limit ${{TMO[$part]}} s) $(date -u)"
            timeout "${{TMO[$part]}}" venv/bin/python -u "$SCRIPT" --parts "$part" \\
                --outdir "$OUT" --cache-dir /opt/hm2p/abc_cache --s3-uri "{s3_uri}"
            echo "part $part rc=$?"
            aws s3 sync "$OUT" "{s3_uri}/" --exclude "*.h5ad" > /dev/null || true
            aws s3 cp /var/log/hm2p-allen-penk.log "s3://{bucket}/{LOG_KEY}" > /dev/null || true
        done

        venv/bin/python "$SCRIPT" --outdir "$OUT" --parts {part_list} --write-done /tmp/done.json \\
            || echo '{{"status": "failed", "detail": "manifest summary failed"}}' > /tmp/done.json
        aws s3 cp "$OUT/manifest.json" "{s3_uri}/manifest.json" || true
        aws s3 cp /var/log/hm2p-allen-penk.log "s3://{bucket}/{LOG_KEY}" || true
        aws s3 cp /tmp/done.json "s3://{bucket}/{DONE_KEY}"
        echo "done: $(cat /tmp/done.json)"
        shutdown -h now
    """)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    ap.add_argument(
        "--no-stage-script",
        action="store_true",
        help="do not upload the local query script; use the repository clone",
    )
    ap.add_argument("--dry-run", action="store_true", help="print user-data, launch nothing")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--terminate", action="store_true")
    ap.add_argument("--wait", action="store_true", help="poll S3 for the completion marker")
    ap.add_argument("--timeout-min", type=int, default=MAX_HOURS * 60 + 30)
    return ap


def stage_script(bucket: str = DERIVATIVES_BUCKET) -> str:  # pragma: no cover - network
    """Upload the local query script so the instance runs this exact version."""
    import boto3

    boto3.client("s3", region_name=REGION).upload_file(str(LOCAL_SCRIPT), bucket, CODE_KEY)
    uri = f"s3://{bucket}/{CODE_KEY}"
    print(f"staged {LOCAL_SCRIPT.name} -> {uri}")
    return uri


def launch(args: argparse.Namespace) -> str:  # pragma: no cover - network
    import boto3
    from botocore.exceptions import ClientError

    staged = not args.no_stage_script
    if staged:
        stage_script()
    ec2 = boto3.client("ec2", region_name=REGION)
    kwargs = dict(parts=args.parts, git_branch=args.branch, staged_script=staged)
    common = dict(
        ImageId=AMI_ID,
        InstanceType=args.instance_type,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        MinCount=1,
        MaxCount=1,
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[
            {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": VOLUME_GB, "VolumeType": "gp3"}}
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [TAG, {"Key": "Name", "Value": "hm2p-allen-penk"}],
            }
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
    print(f"launched {instance_id} ({args.instance_type}, self-terminating, max {MAX_HOURS} h)")
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


def wait_for_done(timeout_min: int, launched_after: float) -> dict | None:  # pragma: no cover
    """Poll for a completion marker written after ``launched_after`` (epoch s)."""
    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    deadline = time.time() + 60 * timeout_min
    while time.time() < deadline:
        try:
            obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=DONE_KEY)
            if obj["LastModified"].timestamp() >= launched_after:
                return json.loads(obj["Body"].read())
        except Exception as exc:  # noqa: BLE001
            if "NoSuchKey" not in str(exc) and "404" not in str(exc):
                print(f"poll error: {exc}")
        print(f"  waiting... ({instance_state()})")
        time.sleep(120)
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
        print(
            build_user_data(
                args.parts, git_branch=args.branch, staged_script=not args.no_stage_script
            )
        )
        return 0
    t_launch = time.time()
    launch(args)
    if args.wait:
        done = wait_for_done(args.timeout_min, t_launch)
        if done is None:
            print("timed out waiting for completion marker")
            return 1
        print(f"run finished: {done}")
        return 0 if done.get("status") == "ok" else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
