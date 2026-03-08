"""Data access layer — loads metadata CSVs and S3 pipeline status."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any

import boto3
import streamlit as st

log = logging.getLogger("hm2p.frontend")

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
    "analysis": "Stage 6 — Analysis",
}


@st.cache_data(ttl=300)
def load_experiments() -> list[dict[str, str]]:
    """Load experiments.csv into a list of dicts."""
    csv_path = METADATA_DIR / "experiments.csv"
    log.info("Loading experiments from %s", csv_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %d experiments", len(rows))
    return rows


@st.cache_data(ttl=300)
def load_animals() -> list[dict[str, str]]:
    """Load animals.csv into a list of dicts."""
    csv_path = METADATA_DIR / "animals.csv"
    log.info("Loading animals from %s", csv_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %d animals", len(rows))
    return rows


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
    log.info("Checking pipeline status on S3 (26 sessions x %d stages)", len(STAGE_PREFIXES))
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
                log.exception("Error checking S3 %s/%s", DERIVATIVES_BUCKET, s3_prefix)
                status[exp_id][prefix] = False

    done_counts = {
        prefix: sum(1 for s in status.values() if s.get(prefix))
        for prefix in STAGE_PREFIXES
    }
    log.info("Pipeline status: %s", done_counts)
    return status


@st.cache_data(ttl=30)
def get_progress(stage: str) -> dict[str, Any] | None:
    """Get _progress.json for a pipeline stage."""
    s3 = get_s3_client()
    try:
        obj = s3.get_object(
            Bucket=DERIVATIVES_BUCKET, Key=f"{stage}/_progress.json"
        )
        data = json.loads(obj["Body"].read())
        log.info("Progress for %s: %s", stage, data.get("status", "?"))
        return data
    except s3.exceptions.NoSuchKey:
        return None
    except Exception:
        log.exception("Error fetching progress for %s", stage)
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
        log.info("Found %d running EC2 instances", len(instances))
        return instances
    except Exception:
        log.exception("Error listing EC2 instances")
        return []


@st.cache_data(ttl=60)
def list_s3_session_files(bucket: str, prefix: str) -> list[dict]:
    """List files in an S3 prefix."""
    log.info("Listing S3 files: s3://%s/%s", bucket, prefix)
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
        log.info("Found %d files in s3://%s/%s", len(files), bucket, prefix)
    except Exception:
        log.exception("Error listing S3 files: s3://%s/%s", bucket, prefix)
    return files


def download_s3_bytes(bucket: str, key: str) -> bytes | None:
    """Download an S3 object as bytes."""
    log.debug("Downloading s3://%s/%s", bucket, key)
    s3 = get_s3_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        log.info("Downloaded s3://%s/%s (%.1f KB)", bucket, key, len(data) / 1024)
        return data
    except Exception:
        log.exception("Error downloading s3://%s/%s", bucket, key)
        return None


@st.cache_data(ttl=300)
def load_all_sync_data() -> dict:
    """Load sync.h5 data for ALL sessions that have it.

    Returns dict with:
        ``"sessions"`` — list of dicts, each with keys:
            exp_id, sub, ses, animal_id, celltype, dff, hd_deg, speed_cm_s,
            light_on, active, bad_behav, n_rois, n_frames, frame_times
        ``"n_sessions"`` — number of sessions loaded
        ``"n_total_rois"`` — total ROIs across all sessions
    """
    import h5py
    import numpy as np

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}
    s3 = get_s3_client()
    sessions = []

    for exp in experiments:
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        animal_id = exp_id.split("_")[-1]
        animal_info = animal_map.get(animal_id, {})

        key = f"sync/{sub}/{ses}/sync.h5"
        try:
            resp = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=key)
            data = resp["Body"].read()
        except Exception:
            continue

        try:
            buf = io.BytesIO(data)
            with h5py.File(buf, "r") as f:
                dff = f["dff"][:]  # (n_rois, n_frames)
                hd_deg = f["hd_deg"][:]
                speed = f["speed_cm_s"][:] if "speed_cm_s" in f else np.zeros(len(hd_deg))
                light_on = f["light_on"][:] if "light_on" in f else np.ones(len(hd_deg), dtype=bool)
                active = f["active"][:] if "active" in f else np.ones(len(hd_deg), dtype=bool)
                bad_behav = f["bad_behav"][:] if "bad_behav" in f else np.zeros(len(hd_deg), dtype=bool)
                frame_times = f["frame_times"][:] if "frame_times" in f else np.arange(len(hd_deg), dtype=float)
                # roi_types: 0=soma, 1=dend, 2=artefact; default all soma
                roi_types = f["roi_types"][:] if "roi_types" in f else np.zeros(dff.shape[0], dtype=np.uint8)

            sessions.append({
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "animal_id": animal_id,
                "celltype": animal_info.get("celltype", "unknown"),
                "dff": dff,
                "hd_deg": hd_deg,
                "speed_cm_s": speed,
                "light_on": light_on,
                "active": active,
                "bad_behav": bad_behav,
                "roi_types": roi_types,
                "n_rois": dff.shape[0],
                "n_frames": dff.shape[1],
                "frame_times": frame_times,
            })
        except Exception:
            log.exception("Error reading sync.h5 for %s", exp_id)
            continue

    return {
        "sessions": sessions,
        "n_sessions": len(sessions),
        "n_total_rois": sum(s["n_rois"] for s in sessions),
    }


def session_filter_sidebar(sessions: list[dict], show_roi_filter: bool = True) -> list[dict]:
    """Add optional sidebar filters for celltype, animal, and ROI type.

    When ``show_roi_filter`` is True, adds a soma/dendrite selector. If the user
    selects "Soma only" (default), each session's ``dff`` and ``roi_types`` are
    filtered to keep only soma ROIs, and ``n_rois`` is updated.

    Returns filtered (and optionally ROI-subsetted) list.
    """
    if not sessions:
        return sessions

    celltypes = sorted(set(s["celltype"] for s in sessions))
    animals = sorted(set(s["animal_id"] for s in sessions))

    ROI_TYPE_MAP = {0: "soma", 1: "dend", 2: "artefact"}

    with st.sidebar:
        st.header("Filters (optional)")
        sel_celltypes = st.multiselect(
            "Cell type", celltypes, default=celltypes, key="filter_celltype",
        )
        sel_animals = st.multiselect(
            "Animal", animals, default=animals, key="filter_animal",
        )
        if show_roi_filter:
            roi_filter = st.radio(
                "ROI type",
                ["Soma only", "Dendrite only", "All ROIs"],
                index=0,
                key="filter_roi_type",
            )
        else:
            roi_filter = "All ROIs"

    filtered = [
        s for s in sessions
        if s["celltype"] in sel_celltypes and s["animal_id"] in sel_animals
    ]

    # Apply ROI type filtering within each session
    if roi_filter != "All ROIs":
        import numpy as np
        target_code = 0 if roi_filter == "Soma only" else 1
        roi_filtered = []
        for s in filtered:
            roi_types = s.get("roi_types")
            if roi_types is not None and len(roi_types) == s["n_rois"]:
                mask = roi_types == target_code
                if mask.any():
                    s_copy = dict(s)
                    s_copy["dff"] = s["dff"][mask]
                    s_copy["roi_types"] = roi_types[mask]
                    s_copy["n_rois"] = int(mask.sum())
                    roi_filtered.append(s_copy)
            else:
                # No roi_types info — include as-is (assumed soma)
                roi_filtered.append(s)
        filtered = roi_filtered

    return filtered


@st.cache_data(ttl=300)
def get_s3_bucket_size(bucket: str) -> dict:
    """Get total size and file count for an S3 bucket (or prefix).

    Returns dict with ``"n_objects"``, ``"total_bytes"``, ``"total_gb"``.
    """
    s3 = get_s3_client()
    total_bytes = 0
    n_objects = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                total_bytes += obj["Size"]
                n_objects += 1
    except Exception:
        log.exception("Error listing bucket %s", bucket)
    return {
        "n_objects": n_objects,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / 1_073_741_824,
    }


@st.cache_data(ttl=300)
def get_s3_prefix_sizes(bucket: str, prefixes: list[str]) -> dict[str, dict]:
    """Get size per S3 prefix (stage).

    Returns dict[prefix -> {"n_objects", "total_bytes", "total_gb"}].
    """
    s3 = get_s3_client()
    result = {}
    for prefix in prefixes:
        total_bytes = 0
        n_objects = 0
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
                for obj in page.get("Contents", []):
                    total_bytes += obj["Size"]
                    n_objects += 1
        except Exception:
            pass
        result[prefix] = {
            "n_objects": n_objects,
            "total_bytes": total_bytes,
            "total_gb": total_bytes / 1_073_741_824,
        }
    return result


def download_s3_numpy(bucket: str, key: str):
    """Download and load a .npy file from S3."""
    import numpy as np

    data = download_s3_bytes(bucket, key)
    if data is None:
        return None
    return np.load(io.BytesIO(data), allow_pickle=True)
