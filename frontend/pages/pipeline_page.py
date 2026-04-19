"""Pipeline monitoring page — active jobs, progress, stage completion matrix."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import logging

import streamlit as st

from frontend.data import (
    PIPELINE_STAGES,
    get_ec2_instances,
    get_pipeline_status,
    get_progress,
    get_stage_summary,
    invalidate_session_cache,
    load_experiments,
    sanitize_error,
)

log = logging.getLogger("hm2p.frontend")

st.title("Pipeline Status")

if st.button("Refresh"):
    st.cache_data.clear()
    invalidate_session_cache()

# ── Live activity ────────────────────────────────────────────────────────
import json

from frontend.data import DERIVATIVES_BUCKET, download_s3_bytes

_activity_items = []

# Check for running instances
_instances = get_ec2_instances()
_running = [i for i in _instances if i["state"] == "running"]

# DLC training/inference progress
_retrain_prog = download_s3_bytes(DERIVATIVES_BUCKET, "dlc-retrain/_retrain_progress.json")
if _retrain_prog:
    _rp = json.loads(_retrain_prog)
    _status = _rp.get("status", "")
    _updated = _rp.get("updated", "")[:19]
    if "Inference" in _status:
        _completed = _rp.get("completed", 0)
        _total = _rp.get("total", 26)
        _activity_items.append(("DLC Inference", f"{_completed}/{_total} sessions", _updated, _completed / max(_total, 1)))
    elif "Training" in _status:
        _activity_items.append(("DLC Training", _status, _updated, None))

# Downstream progress
_ds_prog = download_s3_bytes(DERIVATIVES_BUCKET, "dlc-retrain/_downstream_progress.json")
if _ds_prog:
    _dp = json.loads(_ds_prog)
    _ds_status = _dp.get("status", "")
    _ds_session = _dp.get("session", "")
    _ds_stage = _dp.get("stage", "")
    _ds_idx = _dp.get("session_idx", 0)
    _ds_total = _dp.get("total", 26)
    _ds_completed = _dp.get("completed", 0)
    _ds_failed = _dp.get("failed", 0)
    _ds_updated = _dp.get("updated", "")[:19]
    # Only show if a downstream instance is running or if it's recent
    _show_ds = any("downstream" in i.get("name", "").lower() for i in _running)
    if _show_ds:
        _activity_items.append((
            "Downstream Pipeline",
            f"Session {_ds_idx}/{_ds_total} ({_ds_stage}) — {_ds_session[:25]}",
            _ds_updated,
            _ds_completed / max(_ds_total, 1),
        ))

# Heartbeats
_gpu_hb = download_s3_bytes(DERIVATIVES_BUCKET, "dlc-retrain/_heartbeat.json")
_cpu_hb = download_s3_bytes(DERIVATIVES_BUCKET, "dlc-retrain/_downstream_heartbeat.json")

if _running or _activity_items:
    st.subheader("Live Activity")
    for name, status, updated, pct in _activity_items:
        st.markdown(f"**{name}** — {status}")
        if pct is not None:
            st.progress(min(pct, 1.0))
        st.caption(f"Updated: {updated}")

    # Heartbeats for running instances
    for inst in _running:
        inst_name = inst.get("name", "")
        if "dlc" in inst_name.lower() and _gpu_hb:
            hb = json.loads(_gpu_hb)
            st.caption(f"GPU: uptime {hb.get('uptime_s', 0) // 60:.0f} min, load {hb.get('load_avg_1m', '?')}")
        if "downstream" in inst_name.lower() and _cpu_hb:
            hb = json.loads(_cpu_hb)
            st.caption(f"CPU: uptime {hb.get('uptime_s', 0) // 60:.0f} min, load {hb.get('load_avg_1m', '?')}, disk {hb.get('disk_free_gb', '?')} GB free")

    st.markdown("---")
elif not _running:
    st.info("No active pipeline jobs.")
    st.markdown("---")

# ── Pipeline overview (from unified get_stage_summary) ───────────────────
st.subheader("Pipeline Overview")

with st.spinner("Checking pipeline status..."):
    stage_summary = get_stage_summary()

for key, info in stage_summary.items():
    st.markdown(
        f":{info['color']}[**{info['label']}**] — {info['status']}  \n"
        f"&nbsp;&nbsp;&nbsp;&nbsp;{info['done']}/{info['expected']} sessions"
    )

st.markdown("---")

# ── Live S3 completion counts ─────────────────────────────────────────────
st.subheader("S3 Completion Counts")
st.caption(
    "Live count of sessions with output in hm2p-derivatives (from get_stage_summary)."
)

# Show core stages (exclude ingest — rawdata bucket)
core_stages = [k for k in PIPELINE_STAGES if k != "ingest"]
cols = st.columns(len(core_stages))
for i, key in enumerate(core_stages):
    info = stage_summary[key]
    with cols[i]:
        st.metric(info["short"], f"{info['done']}/{info['expected']}")
        if info["expected"] > 0:
            st.progress(min(info["done"] / info["expected"], 1.0))

st.markdown("---")

# ── Active EC2 instances ──────────────────────────────────────────────────
st.subheader("Active EC2 Instances")
instances = get_ec2_instances()
if instances:
    for inst in instances:
        st.markdown(
            f"**{inst['project']}** | `{inst['id']}` | "
            f"{inst['type']} | {inst['state']} | "
            f"IP: {inst['ip']} | Since: {inst['launch_time']}"
        )
else:
    st.info("No running hm2p instances.")

# ── Progress JSON per stage ───────────────────────────────────────────────
st.subheader("Stage Progress (from _progress.json)")
for key, stage_info in PIPELINE_STAGES.items():
    if key == "ingest":
        continue  # ingest has no _progress.json
    s3_prefix = stage_info.get("s3_prefix", key)
    label = stage_info["label"]
    progress = get_progress(s3_prefix)
    if progress:
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        skipped = progress.get("skipped", 0)
        status = progress.get("status", "")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{label}** — {status}")
            if total > 0:
                pct = min((completed + skipped) / total, 1.0)
                st.progress(
                    pct,
                    text=f"{completed}/{total} done, {failed} failed, {skipped} skipped",
                )
        with col2:
            st.text(f"Updated: {progress.get('updated', '?')}")

        if failed > 0:
            errors = progress.get("failed_errors", {})
            with st.expander(f"Failed sessions ({failed})"):
                for sess in progress.get("failed_sessions", []):
                    err = errors.get(sess, "")
                    st.text(f"  {sess}: {sanitize_error(err)}" if err else f"  {sess}")
    else:
        st.text(f"{label}: no progress data")

# ── Stage completion matrix ───────────────────────────────────────────────
st.subheader("Stage Completion Matrix")

pipeline_status = get_pipeline_status()
experiments = load_experiments()

# Use core stages only. Skip ingest (rawdata bucket) and kpms/cascade —
# those share s3_prefix with kinematics/calcium and cannot be distinguished
# per-session without downloading HDF5 files. They are tracked at the
# aggregate level via _count_kpms_outputs() and _count_cascade_outputs().
_PER_SESSION_SKIP = {"ingest", "kpms", "cascade"}
_matrix_stages = {k: v for k, v in PIPELINE_STAGES.items() if k not in _PER_SESSION_SKIP}
header = "| Session | Animal | " + " | ".join(v["short"] for v in _matrix_stages.values()) + " |"
separator = "|---|---|" + "|".join(["---"] * len(_matrix_stages)) + "|"
rows = [header, separator]

for exp in experiments:
    exp_id = exp["exp_id"]
    animal = exp_id.split("_")[-1]
    status = pipeline_status.get(exp_id, {})
    cells = []
    for key, stage_info in _matrix_stages.items():
        prefix = stage_info.get("s3_prefix", key)
        done = status.get(prefix, False)
        cells.append("Y" if done else "-")
    rows.append(f"| {exp_id[:8]} | {animal} | " + " | ".join(cells) + " |")

st.markdown("\n".join(rows))
