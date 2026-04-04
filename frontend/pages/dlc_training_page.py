"""DLC Training page (Stage 2a) — model training status and GPU monitoring."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    get_ec2_instances,
    get_s3_client,
    sanitize_error,
)

log = logging.getLogger("hm2p.frontend.dlc_training")

st.title("DLC Training (Stage 2a)")

st.markdown(
    "Fine-tunes the DeepLabCut SuperAnimal TopViewMouse model on manually labelled "
    "hm2p frames. GPU required (g5.xlarge, 24h maximum). DLC Inference (Stage 2b) "
    "depends on the trained model produced here."
)

# ── Training status from S3 ──────────────────────────────────────────────────
st.header("Training Status")

RETRAIN_PREFIX = "dlc-retrain"
TRAINING_MODEL_PREFIX = "dlc_training/models"


@st.cache_data(ttl=60)
def _load_retrain_progress() -> dict | None:
    """Load _retrain_progress.json from S3."""
    data = download_s3_bytes(DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/_retrain_progress.json")
    if data is None:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None


@st.cache_data(ttl=60)
def _load_gpu_monitor() -> list[dict] | None:
    """Load _gpu_monitor.csv from S3 and parse into list of dicts."""
    data = download_s3_bytes(DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/_gpu_monitor.csv")
    if data is None:
        return None
    import csv as _csv

    lines = data.decode(errors="replace").strip().split("\n")
    if len(lines) < 2:
        return None
    reader = _csv.DictReader(lines)
    rows = []
    for row in reader:
        try:
            gpu_col = row.get("utilization.gpu [%]", "0").strip().replace(" %", "")
            mem_used = row.get("memory.used [MiB]", "0").strip().replace(" MiB", "")
            mem_total = row.get("memory.total [MiB]", "0").strip().replace(" MiB", "")
            rows.append({
                "timestamp": row.get("timestamp", "").strip(),
                "gpu_util_pct": int(gpu_col),
                "mem_used_mb": int(mem_used),
                "mem_total_mb": int(mem_total),
            })
        except (ValueError, KeyError):
            continue
    return rows if rows else None


@st.cache_data(ttl=120)
def _check_model_exists() -> bool:
    """Check whether trained model weights exist on S3 under dlc_training/models/."""
    try:
        s3 = get_s3_client()
        resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=f"{TRAINING_MODEL_PREFIX}/")
        return any(
            obj["Key"].endswith((".pb", ".index", ".data-00000-of-00001", ".pkl"))
            for obj in resp.get("Contents", [])
        )
    except Exception:
        return False


with st.spinner("Checking S3 for training status..."):
    progress_data = _load_retrain_progress()
    gpu_data = _load_gpu_monitor()
    model_exists = _check_model_exists()

# Model completion status
if model_exists:
    st.success("Trained model weights found on S3 (dlc_training/models/).")
else:
    st.warning(
        "No trained model found. "
        "Run `scripts/launch_dlc_retrain_ec2.py` to start training."
    )

# Training progress
if progress_data:
    status = progress_data.get("status", "unknown")
    updated = progress_data.get("updated", "")
    col1, col2 = st.columns(2)
    col1.metric("Status", status)
    col2.metric("Last updated", updated[:19] if updated else "N/A")

    extra_keys = [k for k in progress_data if k not in {"status", "updated"}]
    if extra_keys:
        st.json({k: progress_data[k] for k in extra_keys})
else:
    st.info(
        "No training progress data on S3. "
        "Training has not been started or the log has not been uploaded yet."
    )

# ── GPU utilization ──────────────────────────────────────────────────────────
st.header("GPU Utilization")

if gpu_data:
    import pandas as pd

    df = pd.DataFrame(gpu_data)
    mean_util = df["gpu_util_pct"].mean()
    max_util = df["gpu_util_pct"].max()
    n_readings = len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean GPU util", f"{mean_util:.0f}%")
    col2.metric("Peak GPU util", f"{max_util}%")
    col3.metric("Readings", n_readings)

    st.line_chart(df.set_index("timestamp")["gpu_util_pct"], use_container_width=True)
else:
    st.info("No GPU monitoring data on S3 yet.")

# ── Running instances ────────────────────────────────────────────────────────
st.header("Active Instances")

try:
    instances = get_ec2_instances()
    retrain_instances = [
        i for i in instances
        if "dlc-retrain" in i.get("project", "").lower()
        or "dlc_retrain" in i.get("project", "").lower()
    ]
    if retrain_instances:
        for inst in retrain_instances:
            st.markdown(
                f"**{inst['id']}** — {inst['state']}  \n"
                f"Type: {inst.get('type', 'N/A')} | IP: {inst.get('ip', 'N/A')}"
            )
    else:
        st.info("No DLC training instances currently running.")
except Exception as exc:
    st.warning(f"Could not query EC2 instances: {sanitize_error(str(exc))}")

# ── Dependency note ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "DLC Inference (Stage 2b) depends on the model produced by this stage. "
    "After training completes, run `scripts/promote_finetuned_pose.py` to QC "
    "and promote results, then re-run Stage 2b."
)

# ── Launch instructions ──────────────────────────────────────────────────────
with st.expander("How to start training"):
    st.markdown(
        """
**Prerequisites:**
1. Label frames using `scripts/prepare_retrain_frames.py`
2. Upload labels: `uv run python scripts/upload_dlc_labels.py`

**Launch training (24h max, GPU enforced):**
```bash
uv run python scripts/launch_dlc_retrain_ec2.py
```

**Monitor progress:**
```bash
uv run python scripts/launch_dlc_retrain_ec2.py --progress
uv run python scripts/launch_dlc_retrain_ec2.py --status
```

**After training:**
- Review tracking quality on the Tracking QC page
- Promote fine-tuned results: `uv run python scripts/promote_finetuned_pose.py`
- Re-run DLC Inference (Stage 2b) on all sessions
        """
    )
