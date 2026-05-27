#!/usr/bin/env python3
"""Launch EC2 for DLC fine-tuning and re-inference.

Hard requirements:
- GPU verified before processing (abort if CUDA unavailable)
- GPU utilization monitored every 30s, uploaded to S3 every 5 min
- Watchdog aborts if GPU at 0% for 5 min during processing
- Hard 24h timeout — instance terminates regardless
- Instance self-terminates on completion (InstanceInitiatedShutdownBehavior=terminate)

Usage:
    uv run python scripts/launch_dlc_finetune_ec2.py
    uv run python scripts/launch_dlc_finetune_ec2.py --progress
    uv run python scripts/launch_dlc_finetune_ec2.py --status
    uv run python scripts/launch_dlc_finetune_ec2.py --terminate
    uv run python scripts/launch_dlc_finetune_ec2.py --maxiters 100000
    uv run python scripts/launch_dlc_finetune_ec2.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from ec2_constants import (
    AMI_ID,
    DERIVATIVES_BUCKET,
    IAM_PROFILE,
    KEY_NAME,
    REGION,
    SG_ID,
)
from ec2_utils import (
    APT_INSTALL_SNIPPET,
    DPKG_WAIT_SNIPPET,
    IMDS_HELPER_SNIPPET,
    PYTORCH_CUDA_INSTALL_SNIPPET,
    build_creds_block,
    format_cost_record_launch,
    format_cost_record_shutdown,
    format_gpu_guard,
    format_hard_timeout,
    format_heartbeat,
    get_s3_credentials,
)

# Keep a local alias for the AMI variable name used below
AMI = AMI_ID
INSTANCE_TYPE = "g4dn.2xlarge"  # 32 GB RAM (xlarge OOM'd on parallel prefetch)
TAG_NAME = "hm2p-dlc-retrain"


def build_user_data(
    maxiters: int = 50000,
    epochs: int = 400,
    infer_only: bool = False,
    train_only: bool = False,
    sa_finetune: bool = False,
    eval_only: bool = False,
    bodyparts: str | None = None,
    run_name: str | None = None,
) -> str:
    """Build the EC2 user-data script.

    Parameters
    ----------
    maxiters
        Legacy TF iterations parameter; ignored under DLC 3.0 PyTorch and
        ignored under ``sa_finetune=True``.
    epochs
        Training epochs to pass to ``run_dlc_retrain.py``. The CLI default
        depends on the path (400 ImageNet / 120 SA).
    infer_only, train_only
        Mutually exclusive subset of pipeline stages to run.
    sa_finetune
        When True, append ``--sa-finetune`` to the run_dlc_retrain.py
        invocation and tag the cost record's ``mode`` with ``+sa``. Per
        Ye et al. 2024 (doi:10.1038/s41467-024-48792-2). Compatible with
        ``--infer-only`` (re-running inference of an SA-finetuned model
        is a valid combination).
    """
    key_id, secret, region = get_s3_credentials()
    creds = build_creds_block(key_id, secret, region)
    gpu_guard = format_gpu_guard(DERIVATIVES_BUCKET, "dlc-retrain")
    timeout = format_hard_timeout(24)
    heartbeat = format_heartbeat(
        DERIVATIVES_BUCKET, "dlc-retrain", INSTANCE_TYPE, heartbeat_key="_heartbeat.json"
    )

    if eval_only:
        mode_flag = "--eval-only"
        mode_label = "evaluation only (per-bodypart RMSE)"
        mode = "eval"
    elif infer_only:
        mode_flag = "--infer-only"
        mode_label = "inference only"
        mode = "infer"
    elif train_only:
        mode_flag = f"--train-only --epochs {epochs}"
        mode_label = f"training only ({epochs} epochs)"
        mode = "train"
    else:
        mode_flag = f"--epochs {epochs}"
        mode_label = f"train + inference ({epochs} epochs)"
        mode = "train+infer"

    if sa_finetune:
        mode_flag = f"{mode_flag} --sa-finetune"
        mode_label = f"{mode_label} [SA fine-tune]"
        mode = f"{mode}+sa"

    if bodyparts:
        mode_flag = f"{mode_flag} --bodyparts {bodyparts}"
        mode_label = f"{mode_label} [bodyparts: {bodyparts}]"
        mode = f"{mode}+bp-{bodyparts.replace(',', '-')}"

    if run_name:
        mode_flag = f"{mode_flag} --run-name {run_name}"

    cost_launch = format_cost_record_launch(
        DERIVATIVES_BUCKET,
        "dlc-retrain",
        instance_type=INSTANCE_TYPE,
        pipeline_step="dlc-retrain-gpu",
        mode=mode,
        launch_key="_cost_record_launch.json",
    )
    cost_shutdown = format_cost_record_shutdown(
        DERIVATIVES_BUCKET,
        "dlc-retrain",
        shutdown_key="_cost_record_shutdown.json",
    )

    # W&B API key — read from env or ~/.wandb_api_key
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        wandb_key_file = Path.home() / ".wandb_api_key"
        if wandb_key_file.exists():
            wandb_key = wandb_key_file.read_text().strip()
    wandb_block = ""
    if wandb_key:
        wandb_block = f'export WANDB_API_KEY="{wandb_key}"\nexport WANDB_MODE=online'
    else:
        wandb_block = 'export WANDB_MODE=offline\necho "WARNING: No WANDB_API_KEY — logging offline only"'

    return f"""#!/bin/bash
exec > >(tee /var/log/hm2p-dlc-retrain.log) 2>&1
echo "=== DLC retrain ({mode_label}, GPU enforced, 24h timeout) ==="
echo "Started: $(date -u)"

{creds}
{wandb_block}
{IMDS_HELPER_SNIPPET}

# Define shutdown handler as a function to avoid single-quote
# nesting issues that break trap '...' syntax.
_hm2p_shutdown() {{
    aws s3 cp /var/log/hm2p-dlc-retrain.log s3://{DERIVATIVES_BUCKET}/dlc-retrain/_gpu_run_log.txt || true
    aws s3 cp /var/log/gpu_monitor.csv s3://{DERIVATIVES_BUCKET}/dlc-retrain/_gpu_monitor.csv || true
    {cost_shutdown}
    shutdown -h now
}}
trap _hm2p_shutdown EXIT

{heartbeat}
{cost_launch}
{DPKG_WAIT_SNIPPET}

set -ex

{APT_INSTALL_SNIPPET}
{PYTORCH_CUDA_INSTALL_SNIPPET}
{timeout}
{gpu_guard}

# Upload log immediately after setup (before Python which may crash)
aws s3 cp /var/log/hm2p-dlc-retrain.log s3://{DERIVATIVES_BUCKET}/dlc-retrain/_gpu_run_log.txt || true

# Patch DLC to add brightness/contrast augmentation.
# The IR filter leaks some 450nm light and IR illumination decays
# through sessions, causing ~5-10% brightness variation. DLC 3.0
# PyTorch has no built-in brightness jitter, so we patch transforms.py.
python3 -c "
import deeplabcut, pathlib, inspect
tf_path = pathlib.Path(inspect.getfile(deeplabcut)).parent / 'pose_estimation_pytorch' / 'data' / 'transforms.py'
code = tf_path.read_text()
if 'RandomBrightnessContrast' not in code:
    # Insert brightness/contrast after the hist_eq block
    marker = 'if augmentations.get(\"hist_eq\"'
    patch = '''
    # hm2p patch: brightness/contrast jitter for IR illumination variation
    import albumentations as _A
    _bc = augmentations.get(\"brightness_contrast\", {{}})
    if _bc:
        transforms.append(_A.RandomBrightnessContrast(
            brightness_limit=_bc.get(\"brightness_limit\", 0.15),
            contrast_limit=_bc.get(\"contrast_limit\", 0.1),
            p=_bc.get(\"p\", 0.5),
        ))
'''
    if marker in code:
        idx = code.index(marker)
        # Find the end of the hist_eq block (next 'if augmentations' or end of function)
        rest = code[idx:]
        lines = rest.split('\\n')
        insert_after = 0
        in_block = True
        for k, line in enumerate(lines[1:], 1):
            stripped = line.strip()
            if stripped.startswith('if augmentations') or stripped.startswith('if crop_sampling'):
                insert_after = k
                break
        if insert_after > 0:
            before = '\\n'.join(lines[:insert_after])
            after = '\\n'.join(lines[insert_after:])
            code = code[:idx] + before + '\\n' + patch + '\\n    ' + after
    tf_path.write_text(code)
    print('Patched transforms.py with RandomBrightnessContrast')
else:
    print('transforms.py already patched')
" || echo "WARNING: DLC brightness patch failed (non-fatal)"

# Clone repo and run — disable set -e so Python errors don't kill
# the script before the EXIT trap can upload logs.
set +e
cd /home/ubuntu
git clone https://github.com/chaplinta/hm2p.git
cd hm2p

# Patch DLC memory_replay.py bug (KeyError on missing bboxes/bodyparts).
# Standalone Python script avoids all bash quoting issues.
python3 scripts/patch_dlc_memory_replay.py || echo "WARNING: memory_replay patch failed (non-fatal)"

# Register the custom weighted heatmap target generator so DLC can
# import WeightedHeatmapGaussianGenerator at training time. The module
# lives in scripts/ — adding it to PYTHONPATH is sufficient.
export PYTHONPATH="/home/ubuntu/hm2p/scripts:$PYTHONPATH"

# GPU watchdog flag is created by the Python script AFTER prefetch
# completes (not here), so the watchdog doesn't kill during download.
python3 scripts/run_dlc_retrain.py {mode_flag}
rm -f /tmp/gpu_processing_active

echo "=== DLC retrain complete: $(date -u) ==="
shutdown -h now
"""


def launch(
    maxiters: int,
    epochs: int = 400,
    infer_only: bool = False,
    train_only: bool = False,
    dry_run: bool = False,
    sa_finetune: bool = False,
    eval_only: bool = False,
    bodyparts: str | None = None,
    run_name: str | None = None,
) -> None:
    """Launch the retraining instance.

    Parameters
    ----------
    sa_finetune
        Pass-through to ``run_dlc_retrain.py --sa-finetune``. Bumps the
        EBS root volume from 100 to 120 GB to absorb the SA snapshot
        download (~600 MB) plus the memory-replay pseudo-label cache.
        Instance type stays ``g4dn.xlarge`` (architect open-question #1
        defers any change until first-run feedback).
    """
    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    # Check labeled data exists
    try:
        s3.head_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-retrain/config.yaml")
    except Exception:
        print("ERROR: no labeled data on S3. Run upload_dlc_labels.py first.")
        sys.exit(1)

    # In --infer-only mode, verify model weights exist before launching.
    # Without this check the instance downloads config.yaml, finds no weights,
    # exits with sys.exit(1) inside user-data — the instance self-terminates
    # with no visible error from the local machine.
    if infer_only:
        resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix="dlc-retrain/models/")
        model_files = [
            obj
            for obj in resp.get("Contents", [])
            if not obj["Key"].endswith("_retrain_progress.json") and not obj["Key"].endswith("/")
        ]
        if not model_files:
            print(
                "ERROR: --infer-only requires model weights on S3 but none were found "
                "at s3://hm2p-derivatives/dlc-retrain/models/.\n"
                "Run training first (omit --infer-only, or use --train-only then "
                "--infer-only after training completes)."
            )
            sys.exit(1)
        print(f"Pre-flight: found {len(model_files)} model file(s) at dlc-retrain/models/")

    user_data = build_user_data(
        maxiters,
        epochs=epochs,
        infer_only=infer_only,
        train_only=train_only,
        eval_only=eval_only,
        sa_finetune=sa_finetune,
        bodyparts=bodyparts,
        run_name=run_name,
    )

    if dry_run:
        print(user_data)
        return

    # SA fine-tune downloads the SA-TVM HRNet checkpoint (~600 MB) and
    # caches memory-replay pseudo-labels — bump the root EBS volume to
    # 120 GB to leave headroom (per architect §6 pitfall #5).
    volume_size = 120 if sa_finetune else 100

    resp = ec2.run_instances(
        ImageId=AMI,
        InstanceType=INSTANCE_TYPE,
        MinCount=1,
        MaxCount=1,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        IamInstanceProfile={"Name": IAM_PROFILE},
        UserData=user_data,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": volume_size, "VolumeType": "gp3"},
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": TAG_NAME},
                    {"Key": "Project", "Value": "hm2p"},
                ],
            }
        ],
        InstanceInitiatedShutdownBehavior="terminate",
    )

    inst = resp["Instances"][0]
    iid = inst["InstanceId"]
    print(f"Launched: {iid} ({INSTANCE_TYPE})")

    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    desc = ec2.describe_instances(InstanceIds=[iid])
    ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "N/A")
    print(f"IP: {ip}")
    print(f"SSH: ssh -i ~/.ssh/{KEY_NAME}.pem ubuntu@{ip}")
    print(f"GPU log: aws s3 cp s3://{DERIVATIVES_BUCKET}/dlc-retrain/_gpu_monitor.csv -")


def status() -> None:
    """Check instance status."""
    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [TAG_NAME]},
            {"Name": "instance-state-name", "Values": ["running", "pending"]},
        ]
    )
    for res in r["Reservations"]:
        for inst in res["Instances"]:
            print(
                f"{inst['InstanceId']}  {inst['State']['Name']}  {inst.get('PublicIpAddress', 'N/A')}"
            )
    if not r["Reservations"]:
        print("No running retrain instances.")


def progress() -> None:
    """Check training/inference progress from S3."""
    s3 = boto3.client("s3", region_name=REGION)
    try:
        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-retrain/_retrain_progress.json")
        p = json.loads(obj["Body"].read())
        print(json.dumps(p, indent=2))
    except Exception:
        print("No progress data yet.")

    # Show GPU utilization summary
    try:
        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key="dlc-retrain/_gpu_monitor.csv")
        lines = obj["Body"].read().decode().strip().split("\n")
        if len(lines) > 1:
            import csv as _csv

            reader = _csv.reader(lines[1:])
            gpu_pcts = []
            import contextlib

            for row in reader:
                if len(row) >= 2:
                    with contextlib.suppress(ValueError):
                        gpu_pcts.append(int(row[1].strip().replace(" %", "")))
            if gpu_pcts:
                print(
                    f"\nGPU utilization: mean={sum(gpu_pcts) / len(gpu_pcts):.0f}%, "
                    f"max={max(gpu_pcts)}%, readings={len(gpu_pcts)}"
                )
    except Exception:
        pass


def terminate() -> None:
    """Terminate retrain instances."""
    ec2 = boto3.client("ec2", region_name=REGION)
    r = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [TAG_NAME]},
            {"Name": "instance-state-name", "Values": ["running", "pending"]},
        ]
    )
    for res in r["Reservations"]:
        for inst in res["Instances"]:
            ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
            print(f"Terminated {inst['InstanceId']}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Argparse for the launcher — split out for unit-testing."""
    parser = argparse.ArgumentParser(description="Launch DLC training or inference on EC2")
    parser.add_argument("--status", action="store_true", help="Check running instances")
    parser.add_argument("--progress", action="store_true", help="Check S3 progress")
    parser.add_argument("--terminate", action="store_true", help="Terminate running instances")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print user-data without launching",
    )
    parser.add_argument(
        "--infer-only",
        action="store_true",
        help="Run inference only (skip training). Uses existing model on S3.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run training only (skip inference).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only — per-bodypart RMSE from existing model on S3.",
    )
    parser.add_argument(
        "--maxiters",
        type=int,
        default=50000,
        help="Legacy TF iterations (ignored under DLC 3.0 PyTorch and "
        "ignored under --sa-finetune)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. Default depends on the path: 400 for "
        "ImageNet HRNet, 120 for --sa-finetune.",
    )
    parser.add_argument(
        "--sa-finetune",
        action="store_true",
        help="Use SuperAnimal-TopViewMouse memory-replay fine-tune instead "
        "of the legacy ImageNet HRNet path. Bumps EBS root from 100 to "
        "120 GB. Compatible with --infer-only. "
        "Cite: Ye et al. 2024, doi:10.1038/s41467-024-48792-2.",
    )
    parser.add_argument(
        "--bodyparts", type=str, default=None,
        help="Override bodyparts (comma-separated). "
        "E.g. --bodyparts left_ear,right_ear for ears-only.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="W&B run name. Default: YYMMDD. Add suffix for what's being "
        "tested (e.g. 260527-stratified). Append -2 for same-day reruns.",
    )
    return parser


def _resolve_epochs(epochs: int | None, *, sa_finetune: bool) -> int:
    """Resolve the default ``--epochs`` to 120 (SA) or 400 (ImageNet)."""
    if epochs is not None:
        return epochs
    return 200 if sa_finetune else 400


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    epochs = _resolve_epochs(args.epochs, sa_finetune=args.sa_finetune)

    if args.status:
        status()
    elif args.progress:
        progress()
    elif args.terminate:
        terminate()
    else:
        if sum([args.infer_only, args.train_only, args.eval_only]) > 1:
            print("ERROR: --infer-only, --train-only, and --eval-only are mutually exclusive")
            sys.exit(1)
        launch(
            args.maxiters,
            epochs=epochs,
            infer_only=args.infer_only,
            train_only=args.train_only,
            dry_run=args.dry_run,
            sa_finetune=args.sa_finetune,
            eval_only=args.eval_only,
            bodyparts=args.bodyparts,
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
