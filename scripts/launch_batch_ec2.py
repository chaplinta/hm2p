#!/usr/bin/env python3
"""Run a list of repository scripts on a short-lived CPU EC2 instance.

Generic launcher for analysis and QC jobs that read from S3. The instance
clones this repository at ``--branch``, installs a CPU-only runtime (no
PyTorch / Cellpose / Suite2p) in a uv venv, unpacks the staged ``--inputs``,
runs the jobs ``--parallel`` at a time, packs ``--outputs`` into
``s3://hm2p-derivatives/batch/<name>/outputs.tar.gz`` together with one log
per job, writes ``_done.json`` (per-job exit codes) and terminates itself.

Job file format (one job per line, ``#`` comments allowed)::

    <job-name> <script.py> [args ...] [; <script.py> [args ...]]

Scripts in a line separated by ``;`` run in order (a chain stops at the first
failure); separate lines run in parallel. Paths are relative to the repo root
and every script runs with ``PYTHONPATH=src`` and ``HM2P_S3_CACHE`` set.

Usage
-----
    python scripts/launch_batch_ec2.py NAME --jobs jobs.txt --outputs results/x --dry-run
    python scripts/launch_batch_ec2.py NAME --jobs jobs.txt --inputs results/a \\
        --outputs results/x results/y --branch feat/qc-reports --wait
    python scripts/launch_batch_ec2.py NAME --download     # fetch outputs again
"""

from __future__ import annotations

import argparse
import io
import json
import shlex
import sys
import tarfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ec2_constants import DERIVATIVES_BUCKET, IAM_PROFILE, KEY_NAME, REGION, SG_ID  # noqa: E402

AMI_ID = "ami-0df4b2961410d4cff"  # Ubuntu 22.04 LTS amd64 (ap-southeast-2)
INSTANCE_TYPE = "c5.2xlarge"  # 8 vCPU, 16 GB
GIT_REPO = "https://github.com/chaplinta/hm2p.git"
S3_PREFIX = "batch"
TAG = {"Key": "Project", "Value": "hm2p-batch"}
REPO = Path(__file__).resolve().parent.parent
RUNTIME_DEPS = (
    "boto3 numpy scipy pandas h5py scikit-learn joblib xgboost scikit-image shapely "
    "'movement>=0.17,<0.18' xarray bottleneck matplotlib opencv-python-headless "
    "imageio-ffmpeg statsmodels pyyaml pydantic pydantic-settings structlog typer "
    "pandera pingouin plotly tqdm rich ripser pycatch22"
)


def parse_jobs(text: str) -> list[tuple[str, list[list[str]]]]:
    """Parse a job file into ``[(name, [argv, ...]), ...]``.

    Raises ValueError on a line without a script or a duplicate job name.
    """
    jobs: list[tuple[str, list[list[str]]]] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, rest = line.partition(" ")
        chain = [shlex.split(part) for part in rest.split(";") if part.strip()]
        if not chain:
            raise ValueError(f"job {name!r} has no command")
        if name in seen:
            raise ValueError(f"duplicate job name {name!r}")
        seen.add(name)
        jobs.append((name, chain))
    return jobs


def pack_paths(paths: list[str], root: Path = REPO) -> bytes:
    """gzip tarball of *paths* (relative to *root*), keeping their relative names.

    Raises FileNotFoundError if a path does not exist.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in paths:
            src = root / p
            if not src.exists():
                raise FileNotFoundError(src)
            tar.add(src, arcname=p)
    return buf.getvalue()


def unpack_outputs(blob: bytes, outputs: list[str], root: Path = REPO) -> list[str]:
    """Extract members under the listed *outputs* (and ``_logs/``) into *root*.

    Members outside those prefixes, absolute paths and ``..`` components are
    skipped. Returns the extracted member names.
    """
    allowed = tuple(o.rstrip("/") for o in outputs) + ("_logs",)
    done = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            name = m.name
            if name.startswith("/") or ".." in Path(name).parts:
                continue
            if not any(name == a or name.startswith(a + "/") for a in allowed):
                continue
            if not (m.isfile() or m.isdir()):
                continue
            tar.extract(m, root, filter="data")
            done.append(name)
    return done


def build_user_data(
    name: str,
    outputs: list[str],
    parallel: int,
    git_branch: str = "main",
    max_hours: int = 4,
    bucket: str = DERIVATIVES_BUCKET,
    creds_block: str = "",
) -> str:
    """Cloud-init script: env, repo, staged inputs, jobs, packed outputs, marker, shutdown."""
    if parallel < 1:
        raise ValueError("parallel must be >= 1")
    if not outputs:
        raise ValueError("at least one output path is required")
    dest = f"s3://{bucket}/{S3_PREFIX}/{name}"
    outs = " ".join(shlex.quote(o) for o in outputs)
    return textwrap.dedent(f"""\
        #!/bin/bash
        exec > >(tee /var/log/hm2p-batch.log) 2>&1
        set -u
        echo "=== hm2p batch {name} === $(date -u)"
        export AWS_DEFAULT_REGION={REGION}
        export HOME=/root
{textwrap.indent(creds_block, "        ")}
        ( sleep {max_hours * 3600}; echo "hard timeout"; shutdown -h now ) &
        # Scripts open boto3 with profile "hm2p-agent": map it to the embedded
        # credentials if present, else to the instance role.
        mkdir -p /root/.aws
        if [ -f /root/.aws/credentials ]; then
            sed -n '/^\\[default\\]/,/^$/p' /root/.aws/credentials \\
                | sed 's/\\[default\\]/[hm2p-agent]/' >> /root/.aws/credentials
            printf '[profile hm2p-agent]\\nregion = {REGION}\\n' >> /root/.aws/config
        else
            printf '[profile hm2p-agent]\\ncredential_source = Ec2InstanceMetadata\\nregion = {REGION}\\n' \\
                >> /root/.aws/config
        fi

        apt-get update -qq && apt-get install -y -qq git curl awscli > /dev/null
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="/root/.local/bin:$PATH"
        mkdir -p /opt/hm2p && cd /opt/hm2p
        git clone -q --depth 1 --branch {git_branch} {GIT_REPO} repo
        uv venv -q -p 3.12 venv
        uv pip install -q --python venv/bin/python {RUNTIME_DEPS}
        cd repo
        aws s3 cp "{dest}/inputs.tar.gz" /tmp/inputs.tar.gz && tar -xzf /tmp/inputs.tar.gz
        export PYTHONPATH=/opt/hm2p/repo/src HM2P_S3_CACHE=/opt/s3cache
        export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
        COMMIT="$(git rev-parse --short HEAD)"; echo "commit $COMMIT"
        mkdir -p _logs

        run_job() {{
            name="$1"; shift
            log="_logs/$name.log"; rc=0
            IFS=';' read -ra parts <<< "$*"
            for cmd in "${{parts[@]}}"; do
                read -ra argv <<< "$cmd"
                [ ${{#argv[@]}} -eq 0 ] && continue
                echo "### ${{argv[*]}}" >> "$log"
                /opt/hm2p/venv/bin/python -u "${{argv[@]}}" >> "$log" 2>&1 || {{ rc=$?; break; }}
            done
            echo "$name $rc"
        }}
        export -f run_job
        grep -v '^\\s*#' jobs.txt | grep -v '^\\s*$' \\
            | xargs -P {parallel} -L 1 bash -c 'run_job "$@"' _ | tee /tmp/rc.txt

        tar -czf /tmp/outputs.tar.gz --ignore-failed-read {outs} _logs
        aws s3 cp /tmp/outputs.tar.gz "{dest}/outputs.tar.gz"
        STATUS=ok; awk '$2 != 0' /tmp/rc.txt | grep -q . && STATUS=failed
        RCS=$(awk '{{printf "%s\\"%s\\": %s", (NR>1?", ":""), $1, $2}}' /tmp/rc.txt)
        FIN="$(date -u +%FT%TZ)"
        echo "{{\\"status\\": \\"$STATUS\\", \\"finished\\": \\"$FIN\\", \\"commit\\": \\"$COMMIT\\", \\"jobs\\": {{$RCS}}}}" > /tmp/done.json
        aws s3 cp /var/log/hm2p-batch.log "{dest}/_instance.log" || true
        aws s3 cp /tmp/done.json "{dest}/_done.json"
        echo "done: $STATUS"
        shutdown -h now
    """)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="batch name (S3 prefix batch/<name>/)")
    ap.add_argument("--jobs", type=Path, help="job file (see module docstring)")
    ap.add_argument("--inputs", nargs="*", default=[], help="local paths to stage (repo-relative)")
    ap.add_argument("--outputs", nargs="*", default=[], help="paths to bring back (repo-relative)")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--instance-type", default=INSTANCE_TYPE)
    ap.add_argument("--max-hours", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait", action="store_true", help="poll for completion, then download")
    ap.add_argument("--download", action="store_true", help="only fetch and unpack outputs")
    ap.add_argument("--timeout-min", type=int, default=240)
    return ap


def _s3():  # pragma: no cover - network
    import boto3

    return boto3.Session(profile_name="hm2p-agent", region_name=REGION).client("s3")


def stage(name: str, jobs_text: str, inputs: list[str]) -> None:  # pragma: no cover - network
    """Upload jobs.txt plus the staged inputs as one tarball; clear an old marker."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = jobs_text.encode()
        info = tarfile.TarInfo("jobs.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for p in inputs:
            tar.add(REPO / p, arcname=p)
    s3 = _s3()
    s3.put_object(
        Bucket=DERIVATIVES_BUCKET, Key=f"{S3_PREFIX}/{name}/inputs.tar.gz", Body=buf.getvalue()
    )
    for k in ("_done.json", "outputs.tar.gz"):
        s3.delete_object(Bucket=DERIVATIVES_BUCKET, Key=f"{S3_PREFIX}/{name}/{k}")
    print(
        f"staged {len(buf.getvalue()) / 1e6:.1f} MB to s3://{DERIVATIVES_BUCKET}/{S3_PREFIX}/{name}/"
    )


def launch(args: argparse.Namespace) -> str:  # pragma: no cover - network
    import boto3
    from botocore.exceptions import ClientError

    ec2 = boto3.Session(profile_name="hm2p-agent", region_name=REGION).client("ec2")
    kw = dict(git_branch=args.branch, max_hours=args.max_hours)
    common = dict(
        ImageId=AMI_ID,
        InstanceType=args.instance_type,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        MinCount=1,
        MaxCount=1,
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[
            {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}}
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [TAG, {"Key": "Name", "Value": f"hm2p-batch-{args.name}"}],
            }
        ],
    )
    try:
        resp = ec2.run_instances(
            UserData=build_user_data(args.name, args.outputs, args.parallel, **kw),
            IamInstanceProfile={"Name": IAM_PROFILE},
            **common,
        )
    except ClientError as exc:
        print(f"instance profile attach failed ({exc}); embedding S3 credentials instead")
        from ec2_utils import build_creds_block, get_s3_credentials

        key_id, secret, region = get_s3_credentials()
        resp = ec2.run_instances(
            UserData=build_user_data(
                args.name,
                args.outputs,
                args.parallel,
                creds_block=build_creds_block(key_id, secret, region),
                **kw,
            ),
            **common,
        )
    iid = resp["Instances"][0]["InstanceId"]
    print(f"launched {iid} ({args.instance_type}) for batch {args.name}")
    return iid


def wait_for_done(name: str, timeout_min: int) -> dict | None:  # pragma: no cover - network
    s3 = _s3()
    deadline = time.time() + 60 * timeout_min
    while time.time() < deadline:
        try:
            obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=f"{S3_PREFIX}/{name}/_done.json")
            return json.loads(obj["Body"].read())
        except Exception as exc:  # noqa: BLE001
            if "NoSuchKey" not in str(exc) and "404" not in str(exc):
                print(f"poll error: {exc}")
        time.sleep(60)
    return None


def download(name: str, outputs: list[str]) -> None:  # pragma: no cover - network
    import tempfile

    s3 = _s3()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        s3.download_file(DERIVATIVES_BUCKET, f"{S3_PREFIX}/{name}/outputs.tar.gz", tmp.name)
        names = unpack_outputs(Path(tmp.name).read_bytes(), outputs)
    print(f"unpacked {len(names)} entries into {REPO}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - network entry point
    args = _build_arg_parser().parse_args(argv)
    if args.download:
        download(args.name, args.outputs)
        return 0
    if args.jobs is None:
        print("--jobs is required")
        return 2
    jobs_text = args.jobs.read_text()
    jobs = parse_jobs(jobs_text)
    if args.dry_run:
        print(f"{len(jobs)} jobs: {', '.join(n for n, _ in jobs)}")
        print(f"inputs: {args.inputs}\noutputs: {args.outputs}")
        print(build_user_data(args.name, args.outputs, args.parallel, args.branch, args.max_hours))
        return 0
    stage(args.name, jobs_text, args.inputs)
    launch(args)
    if args.wait:
        done = wait_for_done(args.name, args.timeout_min)
        if done is None:
            print("timed out waiting for completion marker")
            return 1
        print(f"batch finished: {done}")
        download(args.name, args.outputs)
        return 0 if done.get("status") == "ok" else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
