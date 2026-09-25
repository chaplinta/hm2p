#!/usr/bin/env python3
"""Fetch CASCADE pretrained models to S3 via a short-lived EC2 instance.

The devcontainer cannot reach ``drive.switch.ch`` where the CASCADE
pretrained models are hosted, but an EC2 instance can. This launches a small
on-demand instance whose only job is to download the requested model zips,
copy them to ``s3://hm2p-derivatives/models/cascade/``, write a completion
marker and terminate itself. Inference then runs locally
(``scripts/run_cascade.py``; see docs/manual-installs.md).

Usage
-----
    python scripts/fetch_cascade_model_ec2.py --dry-run     # print user-data
    python scripts/fetch_cascade_model_ec2.py               # launch
    python scripts/fetch_cascade_model_ec2.py --wait        # launch and poll S3
    python scripts/fetch_cascade_model_ec2.py --status
    python scripts/fetch_cascade_model_ec2.py --terminate

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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec2_constants import (  # noqa: E402
    DERIVATIVES_BUCKET,
    IAM_PROFILE,
    KEY_NAME,
    REGION,
    SG_ID,
)

INSTANCE_TYPE = "t3.small"
AMI_ID = (
    "ami-0df4b2961410d4cff"  # Ubuntu 22.04 LTS amd64 (ap-southeast-2), as in launch_suite2p_ec2
)
MODELS_PREFIX = "models/cascade"
DONE_KEY = f"{MODELS_PREFIX}/_fetch_done.json"
DEFAULT_MODELS = ["Global_EXC_10Hz_smoothing200ms", "Global_EXC_10Hz_smoothing200ms_causalkernel"]
STATE_FILE = Path.home() / ".hm2p-cascade-fetch-instance.json"
TAG = {"Key": "Project", "Value": "hm2p-cascade-fetch"}
DEFAULT_YAML = Path(__file__).resolve().parent.parent / ".cascade-src" / "Pretrained_models"
DEFAULT_YAML = DEFAULT_YAML / "available_models.yaml"


def model_links(model_names: list[str], yaml_path: Path) -> dict[str, str]:
    """Resolve download links for *model_names* from Cascade's available_models.yaml."""
    with open(yaml_path) as fh:
        catalogue = yaml.safe_load(fh)
    links: dict[str, str] = {}
    for name in model_names:
        entry = catalogue.get(name)
        if not entry or "Link" not in entry:
            raise KeyError(f"model {name!r} not in {yaml_path}")
        links[name] = str(entry["Link"])
    return links


def build_user_data(
    links: dict[str, str], bucket: str = DERIVATIVES_BUCKET, creds_block: str = ""
) -> str:
    """Cloud-init script: download each model zip, copy to S3, mark done, shut down.

    *creds_block* is an optional shell snippet that writes ``/root/.aws``
    credentials, used when the IAM instance profile cannot be attached.
    """
    fetch_lines = "\n".join(f'fetch "{name}" "{link}"' for name, link in links.items())
    return textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/hm2p-cascade-fetch.log) 2>&1
        set -u
        echo "=== hm2p CASCADE model fetch === $(date -u)"
        export AWS_DEFAULT_REGION={REGION}
{textwrap.indent(creds_block, "        ")}
        apt-get update -qq && apt-get install -y -qq curl unzip awscli > /dev/null
        mkdir -p /tmp/models && cd /tmp/models
        STATUS="ok"
        fetch() {{
            name="$1"; link="$2"
            echo "--- $name"
            if curl -fL --retry 5 --retry-delay 10 -o "$name.zip" "$link"; then
                unzip -q -o "$name.zip" -d "$name.d" || true
                aws s3 cp "$name.zip" "s3://{bucket}/{MODELS_PREFIX}/$name.zip" \\
                    && echo "uploaded $name"
            else
                echo "FAILED $name"; STATUS="failed"
            fi
        }}
{textwrap.indent(fetch_lines, "        ")}
        FIN="$(date -u +%FT%TZ)"
        echo "{{\\"status\\": \\"$STATUS\\", \\"finished\\": \\"$FIN\\"}}" > done.json
        aws s3 cp done.json "s3://{bucket}/{DONE_KEY}"
        aws s3 cp /var/log/hm2p-cascade-fetch.log \\
            "s3://{bucket}/{MODELS_PREFIX}/_fetch.log" || true
        echo "done: $STATUS"
        shutdown -h now
    """)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="CASCADE model names")
    ap.add_argument("--yaml", type=Path, default=DEFAULT_YAML, help="available_models.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print user-data, launch nothing")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--terminate", action="store_true")
    ap.add_argument("--wait", action="store_true", help="poll S3 for the completion marker")
    ap.add_argument("--timeout-min", type=int, default=20)
    return ap


def launch(links: dict[str, str]) -> str:  # pragma: no cover - network
    """Launch with the IAM instance profile; fall back to embedded credentials."""
    import boto3
    from botocore.exceptions import ClientError

    ec2 = boto3.client("ec2", region_name=REGION)
    common = dict(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        MinCount=1,
        MaxCount=1,
        InstanceInitiatedShutdownBehavior="terminate",
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [TAG, {"Key": "Name", "Value": "hm2p-cascade-fetch"}],
            }
        ],
    )
    try:
        resp = ec2.run_instances(
            UserData=build_user_data(links),
            IamInstanceProfile={"Name": IAM_PROFILE},
            **common,
        )
    except ClientError as exc:
        print(f"instance profile attach failed ({exc}); embedding S3 credentials instead")
        from ec2_utils import build_creds_block, get_s3_credentials

        key_id, secret, region = get_s3_credentials()
        resp = ec2.run_instances(
            UserData=build_user_data(links, creds_block=build_creds_block(key_id, secret, region)),
            **common,
        )
    instance_id = resp["Instances"][0]["InstanceId"]
    STATE_FILE.write_text(json.dumps({"instance_id": instance_id, "region": REGION}))
    print(f"launched {instance_id} ({INSTANCE_TYPE}, terminates itself when done)")
    return instance_id


def instance_state() -> str:  # pragma: no cover - network
    import boto3

    if not STATE_FILE.exists():
        return "no state file"
    state = json.loads(STATE_FILE.read_text())
    ec2 = boto3.client("ec2", region_name=REGION)
    desc = ec2.describe_instances(InstanceIds=[state["instance_id"]])
    inst = desc["Reservations"][0]["Instances"][0]
    return f"{state['instance_id']}: {inst['State']['Name']}"


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
        except s3.exceptions.NoSuchKey:
            pass
        except Exception as exc:  # noqa: BLE001
            if "NoSuchKey" not in str(exc) and "404" not in str(exc):
                print(f"poll error: {exc}")
        print(f"  waiting... ({instance_state()})")
        time.sleep(30)
    return None


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - network entry point
    args = _build_arg_parser().parse_args(argv)
    if args.status:
        print(instance_state())
        return 0
    if args.terminate:
        terminate()
        return 0
    links = model_links(args.models, args.yaml)
    if args.dry_run:
        print(build_user_data(links))
        return 0
    launch(links)
    if args.wait:
        done = wait_for_done(args.timeout_min)
        if done is None:
            print("timed out waiting for completion marker")
            return 1
        print(f"fetch finished: {done}")
        return 0 if done.get("status") == "ok" else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
