"""Data access layer — loads metadata CSVs and S3 pipeline status.

Performance notes:
    - Heavy data (sync.h5, ca.h5) is cached in st.session_state for the
      lifetime of the browser session. This avoids re-downloading 100+ MB
      on every page navigation.
    - S3 byte downloads are cached with @st.cache_data (TTL 1800s / 30 min).
    - Filtering (celltype, animal, ROI type) operates on the cached data
      without triggering new S3 downloads.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import boto3
import streamlit as st

log = logging.getLogger("hm2p.frontend")

REGION = "ap-southeast-2"
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"
METADATA_DIR = Path(__file__).resolve().parent.parent / "metadata"


def sanitize_error(msg: str, max_length: int = 200) -> str:
    """Sanitize error message for UI display. Strip paths, tracebacks, and truncate."""
    if not msg:
        return "Unknown error"
    # Remove common path patterns
    msg = re.sub(r"(/[a-zA-Z0-9_./-]+)+", "<path>", msg)
    # Remove traceback blocks
    msg = re.sub(
        r"Traceback \(most recent call last\):.*?(?=\n\S|\Z)",
        "",
        msg,
        flags=re.DOTALL,
    )
    # Strip AWS account IDs (12-digit numbers)
    msg = re.sub(r"\b\d{12}\b", "<account>", msg)
    # Truncate
    msg = msg.strip()
    if len(msg) > max_length:
        msg = msg[:max_length] + "..."
    return msg or "Unknown error"

STAGE_PREFIXES = {
    "ca_extraction": "Stage 1 — Suite2p",
    "pose": "Stage 2 — DLC",
    "kinematics": "Stage 3 — Kinematics",
    "calcium": "Stage 4 — Calcium",
    "sync": "Stage 5 — Sync",
    "analysis": "Stage 6 — Analysis",
}

# ── Unified pipeline stage registry ─────────────────────────────────────
# Single source of truth for all pipeline status display.
# expected: how many sessions should have output (21 = excludes 5 bad behaviour)

PIPELINE_STAGES = {
    "ingest": {
        "label": "Stage 0 — Ingest",
        "short": "Ingest",
        "s3_prefix": None,  # rawdata bucket, not derivatives
        "expected": 26,
    },
    "ca_extraction": {
        "label": "Stage 1 — Suite2p",
        "short": "Suite2p",
        "s3_prefix": "ca_extraction",
        "expected": 26,
    },
    "pose": {
        "label": "Stage 2 — DLC",
        "short": "DLC",
        "s3_prefix": "pose",
        "expected": 26,
    },
    "kinematics": {
        "label": "Stage 3 — Kinematics",
        "short": "Kinematics",
        "s3_prefix": "kinematics",
        "expected": 21,
    },
    "calcium": {
        "label": "Stage 4 — Calcium",
        "short": "Calcium",
        "s3_prefix": "calcium",
        "expected": 26,
    },
    "sync": {
        "label": "Stage 5 — Sync",
        "short": "Sync",
        "s3_prefix": "sync",
        "expected": 21,
    },
    "analysis": {
        "label": "Stage 6 — Analysis",
        "short": "Analysis",
        "s3_prefix": "analysis",
        "expected": 21,
    },
    "kpms": {
        "label": "Stage 3b — MoSeq",
        "short": "MoSeq",
        "s3_prefix": "kinematics",  # syllables.npz lives under kinematics/
        "expected": 26,
    },
}


def get_stage_summary() -> dict[str, dict]:
    """Get unified pipeline status summary for all stages.

    Returns dict[stage_key -> {label, short, expected, done, status, color}].
    Uses cached pipeline_status from S3.
    """
    pipeline_status = get_pipeline_status()

    summary = {}
    for key, info in PIPELINE_STAGES.items():
        expected = info["expected"]

        if key == "kpms":
            # MoSeq: count syllables.npz files on S3
            done = _count_kpms_outputs()
        elif key == "ingest":
            # Ingest: count timestamps.h5 on rawdata bucket
            done = expected  # always 26/26 (already uploaded)
        else:
            done = sum(
                1 for s in pipeline_status.values() if s.get(key, False)
            )

        if done >= expected:
            status, color = "Complete", "green"
        elif done > 0:
            status, color = "In progress", "orange"
        else:
            status, color = "Not started", "red"

        summary[key] = {
            "label": info["label"],
            "short": info["short"],
            "expected": expected,
            "done": done,
            "status": status,
            "color": color,
        }

    return summary


@st.cache_data(ttl=120)
def _count_kpms_outputs() -> int:
    """Count syllables.npz files on S3."""
    try:
        s3 = get_s3_client()
        resp = s3.list_objects_v2(
            Bucket=DERIVATIVES_BUCKET, Prefix="kinematics/",
        )
        return sum(
            1 for obj in resp.get("Contents", [])
            if obj["Key"].endswith("syllables.npz")
        )
    except Exception:
        return 0


@st.cache_data(ttl=3600)
def load_experiments() -> list[dict[str, str]]:
    """Load experiments.csv into a list of dicts."""
    csv_path = METADATA_DIR / "experiments.csv"
    log.info("Loading experiments from %s", csv_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %d experiments", len(rows))
    return rows


@st.cache_data(ttl=3600)
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


@st.cache_data(ttl=120)
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


@st.cache_data(ttl=120)
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


@st.cache_data(ttl=1800)
def download_s3_bytes(bucket: str, key: str) -> bytes | None:
    """Download an S3 object as bytes. Cached for 30 minutes."""
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


# ── Session-state cached data loaders ──────────────────────────────────────
#
# These use st.session_state to cache heavy data (sync.h5, ca.h5) for the
# entire browser session. Data is downloaded once and reused across all
# page navigations. Call invalidate_session_cache() to force reload.


def _session_state_key(name: str) -> str:
    return f"_hm2p_cache_{name}"


def invalidate_session_cache(name: str | None = None) -> None:
    """Clear cached data from session state.

    Args:
        name: Cache key to clear ("sync_data", "ca_data"). If None, clears all.
    """
    if name is None:
        for k in list(st.session_state.keys()):
            if k.startswith("_hm2p_cache_"):
                del st.session_state[k]
    else:
        key = _session_state_key(name)
        if key in st.session_state:
            del st.session_state[key]


def load_all_sync_data() -> dict:
    """Load sync.h5 data for ALL sessions. Cached in session state.

    Returns dict with:
        ``"sessions"`` — list of dicts, each with keys:
            exp_id, sub, ses, animal_id, celltype, dff, hd_deg, speed_cm_s,
            light_on, active, bad_behav, n_rois, n_frames, frame_times,
            roi_types
        ``"n_sessions"`` — number of sessions loaded
        ``"n_total_rois"`` — total ROIs across all sessions
    """
    cache_key = _session_state_key("sync_data")
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    result = _fetch_all_sync_data()
    st.session_state[cache_key] = result
    return result


@st.cache_data(ttl=1800)
def _fetch_all_sync_data() -> dict:
    """Internal: download and parse all sync.h5 files from S3."""
    import h5py
    import numpy as np

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}
    sessions = []

    for exp in experiments:
        # Only include primary sessions in analysis
        if str(exp.get("primary_exp", "1")) != "1":
            continue

        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        animal_id = exp_id.split("_")[-1]
        animal_info = animal_map.get(animal_id, {})

        data = download_s3_bytes(DERIVATIVES_BUCKET, f"sync/{sub}/{ses}/sync.h5")
        if data is None:
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


def load_all_ca_data() -> list[dict]:
    """Load ca.h5 data for ALL sessions. Cached in session state.

    Returns list of dicts with: exp_id, animal_id, celltype, dff, fps,
    roi_types, n_rois, n_frames.
    """
    cache_key = _session_state_key("ca_data")
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    result = _fetch_all_ca_data()
    st.session_state[cache_key] = result
    return result


@st.cache_data(ttl=1800)
def _fetch_all_ca_data() -> list[dict]:
    """Internal: download and parse all ca.h5 files from S3."""
    import h5py
    import numpy as np

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}
    sessions = []

    for exp in experiments:
        # Only include primary sessions in analysis
        if str(exp.get("primary_exp", "1")) != "1":
            continue

        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        animal_id = exp_id.split("_")[-1]
        animal_info = animal_map.get(animal_id, {})

        data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if data is None:
            continue

        try:
            with h5py.File(io.BytesIO(data), "r") as f:
                dff = f["dff"][:]
                fps = float(f.attrs.get("fps_imaging", 30.0))
                roi_types = f["roi_types"][:] if "roi_types" in f else np.zeros(dff.shape[0], dtype=np.uint8)

            sessions.append({
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "animal_id": animal_id,
                "celltype": animal_info.get("celltype", "unknown"),
                "dff": dff,
                "fps": fps,
                "roi_types": roi_types,
                "n_rois": dff.shape[0],
                "n_frames": dff.shape[1],
            })
        except Exception:
            log.exception("Error reading ca.h5 for %s", exp_id)
            continue

    return sessions


def session_filter_sidebar(
    sessions: list[dict],
    show_roi_filter: bool = True,
    key_prefix: str = "filter",
) -> list[dict]:
    """Add optional sidebar filters for celltype, animal, and ROI type.

    Filtering operates on the already-cached session list — no new S3
    downloads are triggered by filter changes.

    When ``show_roi_filter`` is True, adds a soma/dendrite selector. If the
    user selects "Soma only" (default), each session's ``dff`` and
    ``roi_types`` are filtered to keep only soma ROIs.

    Args:
        sessions: List of session dicts from load_all_sync_data or load_all_ca_data.
        show_roi_filter: Whether to show ROI type radio.
        key_prefix: Streamlit widget key prefix (use unique per page to avoid
                    key collisions across pages).

    Returns:
        Filtered (and optionally ROI-subsetted) list.
    """
    if not sessions:
        return sessions

    import numpy as np

    celltypes = sorted(set(s["celltype"] for s in sessions))
    animals = sorted(set(s["animal_id"] for s in sessions))

    with st.sidebar:
        st.header("Filters")
        sel_celltypes = st.multiselect(
            "Cell type", celltypes, default=celltypes,
            key=f"{key_prefix}_celltype",
        )
        sel_animals = st.multiselect(
            "Animal", animals, default=animals,
            key=f"{key_prefix}_animal",
        )
        if show_roi_filter:
            roi_filter = st.radio(
                "ROI type",
                ["Soma only", "Dendrite only", "All ROIs"],
                index=0,
                key=f"{key_prefix}_roi_type",
            )
        else:
            roi_filter = "All ROIs"

    filtered = [
        s for s in sessions
        if s["celltype"] in sel_celltypes and s["animal_id"] in sel_animals
    ]

    # Apply ROI type filtering within each session
    if roi_filter != "All ROIs":
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
                roi_filtered.append(s)
        filtered = roi_filtered

    return filtered


@st.cache_data(ttl=600)
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


@st.cache_data(ttl=600)
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


@st.cache_data(ttl=1800)
def download_s3_numpy(bucket: str, key: str, *, allow_pickle: bool = False):
    """Download and load a .npy file from S3. Cached for 30 minutes.

    Parameters
    ----------
    allow_pickle : bool
        Only set True for Suite2p stat.npy / ops.npy which contain Python
        objects (lists of dicts).  All other .npy files (iscell, F, Fneu,
        spks) are plain numeric arrays and MUST use the default (False)
        to prevent arbitrary-code-execution via crafted .npy files.
    """
    import numpy as np

    data = download_s3_bytes(bucket, key)
    if data is None:
        return None
    return np.load(io.BytesIO(data), allow_pickle=allow_pickle)


# ── Suite2p spatial data loader ───────────────────────────────────────────


def load_all_suite2p_spatial() -> dict[str, dict]:
    """Load Suite2p stat.npy, ops.npy, iscell.npy for ALL sessions.

    Cached in session state (persists across page navigations, 1800s TTL
    via the underlying @st.cache_data fetcher).

    Returns dict keyed by exp_id with values containing:
        mean_img: np.ndarray or None
        shape_features: list of dicts (one per accepted ROI)
        accepted_ids: list of int (Suite2p global indices of accepted cells)
    """
    cache_key = _session_state_key("suite2p_spatial")
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    result = _fetch_all_suite2p_spatial()
    st.session_state[cache_key] = result
    return result


@st.cache_data(ttl=1800)
def _fetch_all_suite2p_spatial() -> dict[str, dict]:
    """Internal: download and parse Suite2p spatial files from S3."""
    import numpy as np

    experiments = load_experiments()
    result: dict[str, dict] = {}

    for exp in experiments:
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)

        s2p_prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0/"
        stat = download_s3_numpy(DERIVATIVES_BUCKET, s2p_prefix + "stat.npy", allow_pickle=True)
        ops = download_s3_numpy(DERIVATIVES_BUCKET, s2p_prefix + "ops.npy", allow_pickle=True)
        iscell = download_s3_numpy(DERIVATIVES_BUCKET, s2p_prefix + "iscell.npy")

        # Extract mean image from ops
        mean_img = None
        if ops is not None:
            ops_dict = ops.item() if isinstance(ops, np.ndarray) and ops.ndim == 0 else ops
            mean_img = ops_dict.get("meanImg")

        # Get accepted cell indices
        cell_mask = iscell[:, 0].astype(bool) if iscell is not None else None
        accepted_ids = list(np.flatnonzero(cell_mask)) if cell_mask is not None else None

        # Build per-ROI shape features from stat.npy
        shape_features: list[dict | None] = []
        if stat is not None and accepted_ids is not None:
            stat_list = list(stat)
            for global_idx in accepted_ids:
                if global_idx < len(stat_list):
                    s = stat_list[global_idx]
                    shape_features.append({
                        "aspect_ratio": float(s.get("aspect_ratio", 1.0)),
                        "radius": float(s.get("radius", 5.0)),
                        "compact": float(s.get("compact", 1.0)),
                        "npix": int(s.get("npix", 0)),
                        "skew": float(s.get("skew", 0.0)),
                        "med_y": int(s.get("med", [0, 0])[0]),
                        "med_x": int(s.get("med", [0, 0])[1]),
                        "ypix": s.get("ypix", np.array([], dtype=int)),
                        "xpix": s.get("xpix", np.array([], dtype=int)),
                    })
                else:
                    shape_features.append(None)

        result[exp_id] = {
            "mean_img": mean_img,
            "shape_features": shape_features,
            "accepted_ids": accepted_ids,
        }

    return result
