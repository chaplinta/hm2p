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

import configparser
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

# ── Signal type labels and selector ────────────────────────────────────────
SIGNAL_TYPE_LABELS: dict[str, str] = {
    "dff": "dF/F0",
    "deconv_norm": "Deconv (normalized)",
    "events": "Events (V&H)",
    "events_sd": "Events (SD threshold)",
    "spikes": "Spikes (CASCADE)",
}


def signal_type_selector(
    session: dict,
    key_prefix: str = "sig",
    default: str = "dff",
) -> tuple[str, np.ndarray | None]:
    """Show a radio button to select the calcium signal type.

    Only shows options that are available in the session data.
    Returns (signal_key, signal_array) where signal_array is
    (n_rois, n_frames) or None if unavailable.

    Call in the page body (not sidebar).
    """
    import numpy as np

    options = []
    for key, label in SIGNAL_TYPE_LABELS.items():
        if key == "dff" and "dff" in session:
            options.append((key, label))
        elif key == "deconv" and session.get("deconv") is not None:
            options.append((key, label))
        elif key == "deconv_norm" and session.get("deconv_norm") is not None:
            options.append((key, label))
        elif key == "events" and session.get("event_masks") is not None:
            options.append((key, label))
        elif key == "events_sd" and session.get("event_masks_sd") is not None:
            options.append((key, label))
        elif key == "spikes" and session.get("spikes") is not None:
            options.append((key, label))

    if not options:
        options = [("dff", "dF/F0")]

    keys = [k for k, _ in options]
    labels = [l for _, l in options]
    default_idx = keys.index(default) if default in keys else 0

    selected_label = st.radio(
        "Signal type",
        labels,
        index=default_idx,
        horizontal=True,
        key=f"{key_prefix}_signal_type",
    )
    selected_key = keys[labels.index(selected_label)]

    # Return the corresponding array
    _key_to_data = {
        "dff": "dff",
        "deconv": "deconv",
        "deconv_norm": "deconv_norm",
        "events": "event_masks",
        "events_sd": "event_masks_sd",
        "spikes": "spikes",
    }
    data_key = _key_to_data.get(selected_key, "dff")
    arr = session.get(data_key)

    return selected_key, arr


STAGE_PREFIXES = {
    "ca_extraction": "Stage 1 — Suite2p",
    "dlc_training": "Stage 2a — DLC Training",
    "pose": "Stage 2b — DLC Inference",
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
    "dlc_training": {
        "label": "Stage 2a — DLC Training",
        "short": "DLC Train",
        "s3_prefix": "dlc_training",
        "expected": 1,  # one trained model, not per-session
    },
    "pose": {
        "label": "Stage 2b — DLC Inference",
        "short": "DLC Infer",
        "s3_prefix": "pose",
        "expected": 26,
    },
    "kinematics": {
        "label": "Stage 3 — Kinematics",
        "short": "Kinematics",
        "s3_prefix": "kinematics",
        "expected": 26,
    },
    "kpms": {
        "label": "Stage 3b — MoSeq",
        "short": "MoSeq",
        "s3_prefix": "kinematics",  # syllables.npz lives under kinematics/
        "expected": 26,
    },
    "calcium": {
        "label": "Stage 4 — Calcium",
        "short": "Calcium",
        "s3_prefix": "calcium",
        "expected": 26,
    },
    "cascade": {
        "label": "Stage 4b — CASCADE",
        "short": "CASCADE",
        "s3_prefix": "calcium",  # spikes added to existing ca.h5
        "expected": 26,
    },
    "sync": {
        "label": "Stage 5 — Sync",
        "short": "Sync",
        "s3_prefix": "sync",
        "expected": 26,
    },
    "analysis": {
        "label": "Stage 6 — Analysis",
        "short": "Analysis",
        "s3_prefix": "analysis",
        "expected": 26,
    },
}


# Stages currently invalidated by a re-run. When a stage is re-running,
# all downstream stages are stale and should show as "pending re-run".
# This is checked by looking for a pipeline_rerun.json marker on S3.
DOWNSTREAM_DEPS: dict[str, list[str]] = {
    "dlc_training": ["pose", "kinematics", "kpms", "sync", "analysis"],
    "pose": ["kinematics", "kpms", "sync", "analysis"],
    "ca_extraction": ["calcium", "cascade", "sync", "analysis"],
    "kinematics": ["sync", "analysis"],
    "kpms": [],
    "calcium": ["sync", "analysis"],
    "cascade": [],  # CASCADE adds spikes to ca.h5; doesn't invalidate downstream
    "sync": ["analysis"],
    "analysis": [],
}


@st.cache_data(ttl=120)
def _get_rerun_status() -> dict:
    """Check S3 for pipeline_rerun.json marker indicating active re-runs.

    Also auto-detects running EC2 instances tagged hm2p-dlc-run or
    hm2p-suite2p as evidence of an active re-run, even if the marker
    file hasn't been uploaded yet.
    """
    result: dict = {}
    try:
        data = download_s3_bytes(DERIVATIVES_BUCKET, "pipeline_rerun.json")
        if data is not None:
            result = json.loads(data)
    except Exception:
        pass

    # Auto-detect from running EC2 instances if no marker exists.
    # Use the Name tag (inst["name"]) to identify instance type — the Project
    # tag value varies ("hm2p", "hm2p-dlc", "hm2p-suite2p") and cannot be
    # used reliably. Name tags are: "hm2p-dlc-retrain", "hm2p-dlc",
    # "hm2p-dlc-parallel-N", "hm2p-suite2p".
    if not result.get("rerunning"):
        try:
            instances = get_ec2_instances()
            for inst in instances:
                if inst["state"] != "running":
                    continue
                inst_name = inst.get("name", "").lower()
                if "dlc-retrain" in inst_name or "dlc_retrain" in inst_name:
                    # DLC retrain instance handles both training and inference.
                    # Check progress JSON to determine which phase.
                    _progress_data = download_s3_bytes(
                        DERIVATIVES_BUCKET, "dlc-retrain/_retrain_progress.json"
                    )
                    _in_inference = False
                    if _progress_data:
                        try:
                            _prog = json.loads(_progress_data)
                            _in_inference = "Inference" in _prog.get("status", "")
                        except Exception:
                            pass
                    if _in_inference:
                        result = {
                            "rerunning": ["pose"],
                            "reason": f"DLC inference running on {inst['id']}",
                        }
                    else:
                        result = {
                            "rerunning": ["dlc_training"],
                            "reason": f"DLC training running on {inst['id']}",
                        }
                    break
                if "dlc" in inst_name:
                    # DLC inference-only instance
                    result = {
                        "rerunning": ["pose"],
                        "reason": f"DLC inference running on {inst['id']}",
                    }
                    break
                if "suite2p" in inst_name:
                    result = {
                        "rerunning": ["ca_extraction"],
                        "reason": f"Suite2p running on {inst['id']}",
                    }
                    break
                if "downstream" in inst_name:
                    # Downstream CPU instance — kinematics, sync, analysis
                    result = {
                        "rerunning": ["kinematics", "sync", "analysis"],
                        "reason": f"Downstream pipeline running on {inst['id']}",
                    }
                    break
        except Exception:
            pass

    return result


def get_stage_summary() -> dict[str, dict]:
    """Get unified pipeline status summary for all stages.

    Returns dict[stage_key -> {label, short, expected, done, status, color}].
    Uses cached pipeline_status from S3. Checks for active re-runs and
    marks downstream stages as invalidated.
    """
    pipeline_status = get_pipeline_status()
    rerun = _get_rerun_status()

    summary = {}
    for key, info in PIPELINE_STAGES.items():
        expected = info["expected"]

        if key == "kpms":
            # MoSeq: count syllables.npz files on S3
            done = _count_kpms_outputs()
        elif key == "cascade":
            # CASCADE: count ca.h5 files that contain a 'spikes' dataset
            done = _count_cascade_outputs()
        elif key == "ingest":
            # Ingest: count timestamps.h5 on rawdata bucket
            done = expected  # always 26/26 (already uploaded)
        elif key == "dlc_training":
            # DLC Training: check for trained model weights on S3
            done = _count_dlc_training_outputs()
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

        # Check if this stage is invalidated by an upstream re-run
        invalidated = False
        rerunning_stages = rerun.get("rerunning", [])
        for rerun_stage in rerunning_stages:
            downstream = DOWNSTREAM_DEPS.get(rerun_stage, [])
            if key in downstream or key == rerun_stage:
                invalidated = True
                break

        if invalidated:
            if key in rerunning_stages:
                if key == "pose":
                    # Inference writes to pose-finetuned/, not pose/.
                    # Use the progress JSON for the real count.
                    _prog = download_s3_bytes(
                        DERIVATIVES_BUCKET, "dlc-retrain/_retrain_progress.json"
                    )
                    if _prog:
                        try:
                            _p = json.loads(_prog)
                            done = _p.get("completed", 0)
                        except Exception:
                            done = 0
                    else:
                        done = 0
                else:
                    rerun_started = rerun.get("started", "")
                    if rerun_started:
                        done = _count_new_outputs(key, rerun_started)
                status = f"Re-running ({done}/{expected})"
                color = "orange"
            else:
                status = "Stale (pending re-run)"
                color = "red"
                done = 0  # Show as 0 — data is stale

        summary[key] = {
            "label": info["label"],
            "short": info["short"],
            "expected": expected,
            "done": done,
            "status": status,
            "color": color,
            "invalidated": invalidated,
        }

    return summary


@st.cache_data(ttl=120)
def _count_new_outputs(stage_key: str, since_iso: str) -> int:
    """Count S3 outputs for a stage that were modified after a given timestamp."""
    from datetime import datetime, timezone

    try:
        since = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0

    prefix_map = {
        "dlc_training": "dlc_training/",
        "pose": "pose/",
        "kinematics": "kinematics/",
        "calcium": "calcium/",
        "sync": "sync/",
        "analysis": "analysis/",
        "ca_extraction": "ca_extraction/",
    }
    prefix = prefix_map.get(stage_key)
    if not prefix:
        return 0

    try:
        s3 = get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        sessions_done = set()
        for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".h5") and obj["LastModified"] > since:
                    parts = obj["Key"].split("/")
                    if len(parts) >= 3:
                        sessions_done.add(f"{parts[1]}/{parts[2]}")
        return len(sessions_done)
    except Exception:
        return 0


@st.cache_data(ttl=300)
def _count_cascade_outputs() -> int:
    """Count ca.h5 files that have CASCADE spikes by sampling one file.

    Downloading all 26 ca.h5 files to check for a key would be too slow.
    Instead, check a single sample file — CASCADE processes all sessions
    in one batch, so if one has spikes, all do.
    """
    import h5py as _h5py

    sample_key = "calcium/sub-1114353/ses-20210823T165950/ca.h5"
    try:
        data = download_s3_bytes(DERIVATIVES_BUCKET, sample_key)
        if data:
            with _h5py.File(io.BytesIO(data), "r") as f:
                if "deconv" in f or "spikes" in f:
                    return 26  # CASCADE runs all-or-nothing
        return 0
    except Exception:
        return 0


@st.cache_data(ttl=300)
def _count_dlc_training_outputs() -> int:
    """Check S3 for trained DLC model weights.

    Checks both dlc_training/models/ and dlc-retrain/models/ since the
    retrain script uploads to the latter. Returns 1 if any model or
    training-complete marker exists, 0 otherwise.
    """
    try:
        s3 = get_s3_client()
        model_suffixes = (".pt", ".pth", ".pb", ".index", ".data-00000-of-00001", ".pkl", ".json")
        for prefix in ("dlc_training/models/", "dlc-retrain/models/"):
            resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix)
            if any(
                obj["Key"].endswith(model_suffixes)
                for obj in resp.get("Contents", [])
            ):
                return 1
        return 0
    except Exception:
        return 0


@st.cache_data(ttl=60)
def _count_pose_finetuned_outputs() -> int:
    """Count sessions with finetuned pose output under pose-finetuned/ on S3."""
    try:
        s3 = get_s3_client()
        experiments = load_experiments()
        count = 0
        for exp in experiments:
            exp_id = exp["exp_id"]
            sub, ses = parse_session_id(exp_id)
            resp = s3.list_objects_v2(
                Bucket=DERIVATIVES_BUCKET,
                Prefix=f"pose-finetuned/{sub}/{ses}/",
                MaxKeys=1,
            )
            if resp.get("KeyCount", 0) > 0:
                count += 1
        return count
    except Exception:
        return 0


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


@st.cache_data
def get_mm_per_pix(sub: str, ses: str) -> float | None:
    """Return the mm-per-pixel scale factor for a session.

    Reads ``metadata/video_meta/{sub}_{ses}_meta.txt`` and parses the
    ``[scale] mm_per_pix`` key.  The result is cached indefinitely because
    these calibration files are static.

    Parameters
    ----------
    sub:
        Subject identifier in NeuroBlueprint format, e.g. ``"sub-1114353"``.
    ses:
        Session identifier in NeuroBlueprint format, e.g.
        ``"ses-20210823T165950"``.

    Returns
    -------
    float | None
        Scale factor in mm per pixel, or ``None`` if the file is not found or
        the key is absent.
    """
    meta_path = METADATA_DIR / "video_meta" / f"{sub}_{ses}_meta.txt"
    if not meta_path.exists():
        log.warning("No video meta file found: %s", meta_path)
        return None
    cfg = configparser.ConfigParser()
    cfg.read(meta_path)
    try:
        return float(cfg["scale"]["mm_per_pix"])
    except KeyError:
        log.warning("mm_per_pix key missing in %s", meta_path)
        return None


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
    """Get running/pending hm2p EC2 instances.

    Filters by Name tag prefix "hm2p-" to catch all project instances
    regardless of their Project tag value (which varies across scripts:
    "hm2p", "hm2p-dlc", "hm2p-suite2p").
    """
    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "instance-state-name", "Values": ["running", "pending"]},
                {"Name": "tag:Name", "Values": ["hm2p-*"]},
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
                        "name": tags.get("Name", ""),
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


def check_stale_data_warning(
    stages: list[str] | None = None,
    block: bool = False,
) -> bool:
    """Show a warning banner if any required stages have stale data.

    Call at the top of any page that loads sync/analysis data.

    Args:
        stages: List of pipeline stage keys this page depends on.
            Default: ["sync", "analysis"].
        block: If True, call st.stop() to prevent the page from rendering
            stale data. If False, show a warning but continue.

    Returns:
        True if data is stale.
    """
    rerun = _get_rerun_status()
    rerunning = rerun.get("rerunning", [])
    if not rerunning:
        return False

    if stages is None:
        stages = ["sync", "analysis"]

    invalidated = []
    for rerun_stage in rerunning:
        downstream = DOWNSTREAM_DEPS.get(rerun_stage, [])
        for s in stages:
            if s in downstream or s == rerun_stage:
                invalidated.append(s)

    if invalidated:
        reason = rerun.get("reason", "upstream stage re-running")
        affected = ", ".join(sorted(set(invalidated)))
        if block:
            st.error(
                f"**Data unavailable.** An upstream pipeline stage is re-running: "
                f"{reason}. This page depends on: {affected}. "
                f"Data will be available once the re-run completes and "
                f"downstream stages are re-processed."
            )
            st.stop()
        else:
            st.warning(
                f"**Data may be stale.** Pipeline re-run in progress: {reason}. "
                f"Affected stages: {affected}. "
                f"Results shown below are from the previous run."
            )
        return True
    return False


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

    If an upstream pipeline stage is re-running (detected via
    pipeline_rerun.json on S3 or running EC2 instances), shows an error
    and blocks the page — stale data should not be displayed.

    Returns dict with:
        ``"sessions"`` — list of dicts, each with keys:
            exp_id, sub, ses, animal_id, celltype, dff, hd_deg, speed_cm_s,
            light_on, active, bad_behav, n_rois, n_frames, frame_times,
            roi_types, deconv (or None), event_masks (or None)
        ``"n_sessions"`` — number of sessions loaded
        ``"n_total_rois"`` — total ROIs across all sessions
    """
    # Block page if upstream data is being re-processed
    check_stale_data_warning(stages=["sync"], block=True)

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
                # Suite2p deconvolved spikes
                deconv = f["deconv"][:] if "deconv" in f else None
                if deconv is None:
                    deconv = f["spks"][:] if "spks" in f else None
                # CASCADE calibrated spike rates (separate from deconv)
                spikes = f["spikes"][:] if "spikes" in f else None
                deconv_norm = f["deconv_norm"][:] if "deconv_norm" in f else None
                # Event masks (Voigts & Harnett binary events)
                event_masks = f["event_masks"][:] if "event_masks" in f else None
                # SD-threshold events (Zong et al. 2022 — more sensitive)
                event_masks_sd = f["event_masks_sd"][:] if "event_masks_sd" in f else None
                # Position and AHV (from kinematics, resampled to imaging rate)
                x_mm = f["x_mm"][:] if "x_mm" in f else None
                y_mm = f["y_mm"][:] if "y_mm" in f else None
                x_maze = f["x_maze"][:] if "x_maze" in f else None
                y_maze = f["y_maze"][:] if "y_maze" in f else None
                ahv_deg_s = f["ahv_deg_s"][:] if "ahv_deg_s" in f else None
                # Per-bodypart maze coordinates for skeleton visualisation
                bp_maze = {}
                for k in f.keys():
                    if k.startswith("bp_") and k.endswith("_x_maze"):
                        bp_name = k[3:-7]  # strip "bp_" and "_x_maze"
                        y_key = f"bp_{bp_name}_y_maze"
                        if y_key in f:
                            bp_maze[bp_name] = {
                                "x": f[k][:],
                                "y": f[y_key][:],
                            }

            sessions.append({
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "animal_id": animal_id,
                "celltype": animal_info.get("celltype", "unknown"),
                "exclude": str(exp.get("exclude", "0")).strip(),
                "primary_exp": str(exp.get("primary_exp", "1")).strip(),
                "dff": dff,
                "deconv": deconv,
                "deconv_norm": deconv_norm,
                "spikes": spikes,
                "event_masks": event_masks,
                "event_masks_sd": event_masks_sd,
                "hd_deg": hd_deg,
                "speed_cm_s": speed,
                "light_on": light_on,
                "active": active,
                "bad_behav": bad_behav,
                "roi_types": roi_types,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "x_maze": x_maze,
                "y_maze": y_maze,
                "ahv_deg_s": ahv_deg_s,
                "bp_maze": bp_maze if bp_maze else None,
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

    Returns list of dicts with: exp_id, sub, ses, animal_id, celltype, dff,
    fps, roi_types, n_rois, n_frames, event_masks (or None).
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
                event_masks = f["event_masks"][:] if "event_masks" in f else None
                event_masks_sd = f["event_masks_sd"][:] if "event_masks_sd" in f else None
                deconv_norm = f["deconv_norm"][:] if "deconv_norm" in f else None
                spikes_ca = f["spikes"][:] if "spikes" in f else None
                frame_times_ca = f["frame_times"][:] if "frame_times" in f else None

            sessions.append({
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "animal_id": animal_id,
                "celltype": animal_info.get("celltype", "unknown"),
                "dff": dff,
                "fps": fps,
                "roi_types": roi_types,
                "event_masks": event_masks,
                "event_masks_sd": event_masks_sd,
                "deconv_norm": deconv_norm,
                "spikes": spikes_ca,
                "frame_times": frame_times_ca,
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

    # Filters in main page body (not sidebar) — collapsible expander
    with st.expander("Filters", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)

        # Session inclusion filters
        has_exclude_info = any("exclude" in s for s in sessions)
        if has_exclude_info:
            with fc1:
                include_excluded = st.checkbox(
                    "Include excluded",
                    value=False,
                    key=f"{key_prefix}_include_excluded",
                )
                primary_only = st.checkbox(
                    "Primary only",
                    value=True,
                    key=f"{key_prefix}_primary_only",
                )
        else:
            include_excluded = True
            primary_only = False

        with fc2:
            sel_celltypes = st.multiselect(
                "Cell type", celltypes, default=celltypes,
                key=f"{key_prefix}_celltype",
            )
        with fc3:
            sel_animals = st.multiselect(
                "Animal", animals, default=animals,
                key=f"{key_prefix}_animal",
            )
        with fc4:
            if show_roi_filter:
                roi_filter = st.radio(
                    "ROI type",
                    ["Soma only", "Dendrite only", "All ROIs"],
                    index=0,
                    key=f"{key_prefix}_roi_type",
                )
            else:
                roi_filter = "All ROIs"

    filtered = []
    for s in sessions:
        if s["celltype"] not in sel_celltypes or s["animal_id"] not in sel_animals:
            continue
        if not include_excluded and s.get("exclude", "0") == "1":
            continue
        if primary_only and s.get("primary_exp", "1") != "1":
            continue
        filtered.append(s)

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
                    if s.get("deconv") is not None:
                        s_copy["deconv"] = s["deconv"][mask]
                    if s.get("deconv_norm") is not None:
                        s_copy["deconv_norm"] = s["deconv_norm"][mask]
                    if s.get("spikes") is not None:
                        s_copy["spikes"] = s["spikes"][mask]
                    if s.get("event_masks") is not None:
                        s_copy["event_masks"] = s["event_masks"][mask]
                    if s.get("event_masks_sd") is not None:
                        s_copy["event_masks_sd"] = s["event_masks_sd"][mask]
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

        # Extract mean and max images from ops
        mean_img = None
        max_img = None
        if ops is not None:
            ops_dict = ops.item() if isinstance(ops, np.ndarray) and ops.ndim == 0 else ops
            mean_img = ops_dict.get("meanImg")
            # True max projection (computed post-Suite2p from data.bin).
            # Falls back to meanImgE (contrast-enhanced mean) if max_proj not available.
            max_img = ops_dict.get("max_proj")
            if max_img is None:
                max_img = ops_dict.get("meanImgE")

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
            "max_img": max_img,
            "shape_features": shape_features,
            "accepted_ids": accepted_ids,
        }

    return result


# ── MoSeq data loaders ──────────────────────────────────────────────────


@st.cache_data(ttl=1800)
def load_kpms_summary() -> dict | None:
    """Load kpms_summary.json from S3."""
    data = download_s3_bytes(DERIVATIVES_BUCKET, "kinematics/kpms_summary.json")
    if data is None:
        return None
    try:
        return json.loads(data.decode())
    except Exception:
        log.exception("Error parsing kpms_summary.json")
        return None


@st.cache_data(ttl=300)
def list_syllable_sessions() -> list[dict]:
    """List sessions that have syllables.npz on S3.

    Returns list of dicts with keys: sub, ses, key, size.
    Uses paginator to handle >1000 objects.
    """
    try:
        s3 = get_s3_client()
        results = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix="kinematics/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("syllables.npz"):
                    parts = key.split("/")
                    sub = parts[1] if len(parts) > 1 else "---"
                    ses = parts[2] if len(parts) > 2 else "---"
                    results.append({"sub": sub, "ses": ses, "key": key, "size": obj["Size"]})
        return results
    except Exception as e:
        log.warning("Failed to list syllable sessions: %s", e)
        return []


@st.cache_data(ttl=1800)
def load_syllable_npz(s3_key: str) -> dict | None:
    """Load a syllables.npz from S3 and return numpy arrays as a dict.

    Returns dict with keys like 'syllable_id', 'syllable_prob'.
    """
    import numpy as np

    data = download_s3_bytes(DERIVATIVES_BUCKET, s3_key)
    if data is None:
        return None
    try:
        npz = np.load(io.BytesIO(data))
        return {k: npz[k] for k in npz.files}
    except Exception as e:
        log.warning("Failed to load syllable data from %s: %s", s3_key, e)
        return None


def load_all_syllable_data() -> dict:
    """Load syllable data for ALL sessions. Cached in session state.

    Returns dict with:
        ``"sessions"`` -- list of dicts with: sub, ses, key, animal_id,
            celltype, syllable_id, n_frames, n_syllables
        ``"n_sessions"`` -- number of sessions loaded
    """
    cache_key = _session_state_key("syllable_data")
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    result = _fetch_all_syllable_data()
    st.session_state[cache_key] = result
    return result


@st.cache_data(ttl=1800)
def _fetch_all_syllable_data() -> dict:
    """Internal: download and parse all syllables.npz from S3."""
    import numpy as np

    syl_sessions = list_syllable_sessions()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}

    sessions = []
    for ss in syl_sessions:
        npz = load_syllable_npz(ss["key"])
        if npz is None:
            continue
        syl_ids = npz.get("syllable_id", npz.get("syllable_ids"))
        if syl_ids is None:
            continue
        syl_ids = syl_ids.astype(int)
        animal_id = ss["sub"].replace("sub-", "")
        animal_info = animal_map.get(animal_id, {})
        unique_syls = np.unique(syl_ids)

        sessions.append({
            "sub": ss["sub"],
            "ses": ss["ses"],
            "key": ss["key"],
            "animal_id": animal_id,
            "celltype": animal_info.get("celltype", "unknown"),
            "syllable_id": syl_ids,
            "n_frames": len(syl_ids),
            "n_syllables": len(unique_syls),
        })

    return {
        "sessions": sessions,
        "n_sessions": len(sessions),
    }
