"""Data access layer — loads metadata CSVs and S3 pipeline status."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import boto3
import streamlit as st

REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"
METADATA_DIR = Path(__file__).resolve().parent.parent / "metadata"

STAGE_PREFIXES = {
    "ca_extraction": "Stage 1 — Suite2p",
    "pose": "Stage 2 — DLC",
    "kinematics": "Stage 3 — Kinematics",
    "calcium": "Stage 4 — Calcium",
    "sync": "Stage 5 — Sync",
}


@st.cache_data(ttl=300)
def load_experiments() -> list[dict[str, str]]:
    """Load experiments.csv into a list of dicts."""
    csv_path = METADATA_DIR / "experiments.csv"
    with open(csv_path) as f:
        return list(csv.DictReader(f))


@st.cache_data(ttl=300)
def load_animals() -> list[dict[str, str]]:
    """Load animals.csv into a list of dicts."""
    csv_path = METADATA_DIR / "animals.csv"
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def parse_session_id(exp_id: str) -> tuple[str, str]:
    """Convert exp_id to (sub, ses) NeuroBlueprint names."""
    parts = exp_id.split("_")
    animal = parts[-1]
    sub = f"sub-{animal}"
    ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
    return sub, ses


def get_s3_client():
    """Get boto3 S3 client."""
    return boto3.client("s3", region_name=REGION)


@st.cache_data(ttl=60)
def get_pipeline_status() -> dict[str, dict[str, bool]]:
    """Check which pipeline stages have outputs for each session.

    Returns dict[exp_id -> dict[stage_prefix -> bool]].
    """
    s3 = get_s3_client()
    experiments = load_experiments()
    status: dict[str, dict[str, bool]] = {}

    for exp in experiments:
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        status[exp_id] = {}
        for prefix in STAGE_PREFIXES:
            s3_prefix = f"{prefix}/{sub}/{ses}/"
            try:
                resp = s3.list_objects_v2(
                    Bucket=DERIVATIVES_BUCKET, Prefix=s3_prefix, MaxKeys=1
                )
                status[exp_id][prefix] = resp.get("KeyCount", 0) > 0
            except Exception:
                status[exp_id][prefix] = False

    return status


@st.cache_data(ttl=30)
def get_progress(stage: str) -> dict[str, Any] | None:
    """Get _progress.json for a pipeline stage."""
    s3 = get_s3_client()
    try:
        obj = s3.get_object(
            Bucket=DERIVATIVES_BUCKET, Key=f"{stage}/_progress.json"
        )
        return json.loads(obj["Body"].read())
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_ec2_instances() -> list[dict]:
    """Get running/pending hm2p EC2 instances."""
    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "instance-state-name", "Values": ["running", "pending"]},
                {"Name": "tag:Project", "Values": ["hm2p-suite2p", "hm2p-dlc"]},
            ]
        )
        instances = []
        for res in resp["Reservations"]:
            for inst in res["Instances"]:
                tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                instances.append(
                    {
                        "id": inst["InstanceId"],
                        "type": inst["InstanceType"],
                        "state": inst["State"]["Name"],
                        "ip": inst.get("PublicIpAddress", ""),
                        "launch_time": str(inst.get("LaunchTime", "")),
                        "project": tags.get("Project", ""),
                    }
                )
        return instances
    except Exception:
        return []


@st.cache_data(ttl=60)
def list_s3_session_files(bucket: str, prefix: str) -> list[dict]:
    """List files in an S3 prefix."""
    s3 = get_s3_client()
    files = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                files.append(
                    {
                        "key": obj["Key"],
                        "size_mb": obj["Size"] / 1e6,
                        "modified": str(obj["LastModified"]),
                    }
                )
    except Exception:
        pass
    return files


def download_s3_bytes(bucket: str, key: str) -> bytes | None:
    """Download an S3 object as bytes."""
    s3 = get_s3_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except Exception:
        return None


def download_s3_numpy(bucket: str, key: str):
    """Download and load a .npy file from S3."""
    import numpy as np

    data = download_s3_bytes(bucket, key)
    if data is None:
        return None
    return np.load(io.BytesIO(data), allow_pickle=True)
