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
            # nvidia-smi CSV has leading spaces in headers and values
            vals = {k.strip(): v.strip() for k, v in row.items()}
            gpu_col = vals.get("utilization.gpu [%]", "0").replace(" %", "").replace("%", "")
            mem_used = vals.get("memory.used [MiB]", "0").replace(" MiB", "").replace("MiB", "")
            mem_total = vals.get("memory.total [MiB]", "0").replace(" MiB", "").replace("MiB", "")
            rows.append({
                "timestamp": vals.get("timestamp", ""),
                "gpu_util_pct": int(gpu_col),
                "mem_used_mb": int(mem_used),
                "mem_total_mb": int(mem_total),
            })
        except (ValueError, KeyError):
            continue
    return rows if rows else None


@st.cache_data(ttl=120)
def _check_model_exists() -> bool:
    """Check whether trained model weights exist on S3.

    Checks both dlc_training/models/ and dlc-retrain/models/ since the
    retrain script uploads to the latter.
    """
    try:
        s3 = get_s3_client()
        model_suffixes = (".pt", ".pth", ".pb", ".index", ".data-00000-of-00001", ".pkl", ".json")
        for prefix in (f"{TRAINING_MODEL_PREFIX}/", f"{RETRAIN_PREFIX}/models/"):
            resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix)
            if any(
                obj["Key"].endswith(model_suffixes)
                for obj in resp.get("Contents", [])
            ):
                return True
        return False
    except Exception:
        return False


@st.cache_data(ttl=120)
def _parse_training_curves() -> list[dict] | None:
    """Parse epoch-level train/valid loss from the run log on S3."""
    import re

    data = download_s3_bytes(DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/_run_log.txt")
    if data is None:
        return None
    text = data.decode(errors="replace")
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\(lr=([\d.e+-]+)\),\s+train loss\s+([\d.]+)"
        r"(?:,\s+valid loss\s+([\d.]+))?"
    )
    rows = []
    for m in pattern.finditer(text):
        epoch = int(m.group(1))
        total = int(m.group(2))
        lr = float(m.group(3))
        train_loss = float(m.group(4))
        valid_loss = float(m.group(5)) if m.group(5) else None
        rows.append({
            "epoch": epoch,
            "total_epochs": total,
            "lr": lr,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        })
    return rows if rows else None


with st.spinner("Checking S3 for training status..."):
    progress_data = _load_retrain_progress()
    gpu_data = _load_gpu_monitor()
    model_exists = _check_model_exists()

# Model completion status
if model_exists:
    st.success("Trained model weights found on S3.")
else:
    # Check if training completed but model upload failed
    _curves = _parse_training_curves()
    if _curves and len(_curves) > 0:
        last_epoch = _curves[-1]
        if last_epoch["epoch"] == last_epoch["total_epochs"]:
            st.warning(
                f"Training completed ({last_epoch['epoch']} epochs) but model weights "
                f"are not on S3. The upload may have failed. Re-run with `--train-only` "
                f"or manually upload from the instance."
            )
        else:
            st.info(
                f"Training in progress or interrupted at epoch "
                f"{last_epoch['epoch']}/{last_epoch['total_epochs']}."
            )
    else:
        st.info(
            "No trained model or training log found. "
            "Run `scripts/launch_dlc_finetune_ec2.py` to start training."
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

# ── Training curves ─────────────────────────────────────────────────────────
st.header("Training Curves")

curve_data = _parse_training_curves()

if curve_data:
    import pandas as pd

    df_curves = pd.DataFrame(curve_data).set_index("epoch")
    total_epochs = curve_data[-1]["total_epochs"]
    final_train = curve_data[-1]["train_loss"]
    valid_rows = [r for r in curve_data if r["valid_loss"] is not None]
    best_valid = min(valid_rows, key=lambda r: r["valid_loss"]) if valid_rows else None

    import plotly.graph_objects as go

    final_train = curve_data[-1]["train_loss"]
    valid_rows = [r for r in curve_data if r["valid_loss"] is not None]
    best_valid = min(valid_rows, key=lambda r: r["valid_loss"]) if valid_rows else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Epochs completed", f"{len(curve_data)}/{total_epochs}")
    col2.metric("Final train loss", f"{final_train:.5f}")
    if best_valid:
        col3.metric("Best valid loss", f"{best_valid['valid_loss']:.5f}")
        col4.metric("Best checkpoint", f"Epoch {best_valid['epoch']}")

    # Plot train + valid loss with plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_curves.index,
        y=df_curves["train_loss"],
        mode="lines",
        name="Train loss",
        line=dict(color="#1f77b4", width=1.5),
    ))
    if valid_rows:
        fig.add_trace(go.Scatter(
            x=[r["epoch"] for r in valid_rows],
            y=[r["valid_loss"] for r in valid_rows],
            mode="lines+markers",
            name="Valid loss",
            line=dict(color="#d62728", width=2),
            marker=dict(size=6),
        ))
        if best_valid:
            fig.add_trace(go.Scatter(
                x=[best_valid["epoch"]],
                y=[best_valid["valid_loss"]],
                mode="markers",
                name=f"Best (epoch {best_valid['epoch']})",
                marker=dict(size=12, color="#2ca02c", symbol="star"),
                showlegend=True,
            ))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss (MSE)",
        height=400,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(x=0.7, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Overfitting warning
    if best_valid and valid_rows:
        last_valid = valid_rows[-1]
        if last_valid["valid_loss"] > best_valid["valid_loss"] * 1.2:
            st.warning(
                f"Validation loss increased from {best_valid['valid_loss']:.5f} "
                f"(epoch {best_valid['epoch']}) to {last_valid['valid_loss']:.5f} "
                f"(epoch {last_valid['epoch']}). The model is overfitting — "
                f"DLC selected the best checkpoint at epoch {best_valid['epoch']}."
            )

    # LR schedule
    with st.expander("Learning rate schedule"):
        lr_fig = go.Figure()
        lr_fig.add_trace(go.Scatter(
            x=df_curves.index, y=df_curves["lr"],
            mode="lines", name="Learning rate",
        ))
        lr_fig.update_layout(
            xaxis_title="Epoch", yaxis_title="LR", yaxis_type="log",
            height=250, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(lr_fig, use_container_width=True)
else:
    st.info("No training log on S3. Training curves will appear after training starts.")

# ── GPU utilization ──────────────────────────────────────────────────────────
st.header("GPU Utilization")

if gpu_data:
    import pandas as pd
    import plotly.graph_objects as go  # noqa: may be imported above

    df = pd.DataFrame(gpu_data)
    # Parse timestamps for proper x-axis
    df["time"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    active_mask = df["gpu_util_pct"] > 0
    mean_all = df["gpu_util_pct"].mean()
    mean_active = df.loc[active_mask, "gpu_util_pct"].mean() if active_mask.any() else 0
    max_util = df["gpu_util_pct"].max()
    n_readings = len(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean GPU util (all)", f"{mean_all:.0f}%")
    col2.metric("Mean GPU util (active)", f"{mean_active:.0f}%")
    col3.metric("Peak GPU util", f"{max_util}%")
    col4.metric("Readings", n_readings)

    gpu_fig = go.Figure()
    gpu_fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["gpu_util_pct"],
        mode="lines",
        name="GPU %",
        line=dict(color="#ff7f0e", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 127, 14, 0.2)",
    ))
    gpu_fig.update_layout(
        xaxis_title="Time",
        yaxis_title="GPU Utilization (%)",
        yaxis=dict(range=[0, 105]),
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(
            dtick=600_000,  # tick every 10 minutes
            tickformat="%H:%M",
        ),
    )
    st.plotly_chart(gpu_fig, use_container_width=True)

    # Memory usage
    with st.expander("GPU memory usage"):
        mem_fig = go.Figure()
        mem_fig.add_trace(go.Scatter(
            x=df["time"], y=df["mem_used_mb"],
            mode="lines", name="Used (MiB)",
        ))
        mem_fig.update_layout(
            xaxis_title="Time", yaxis_title="GPU Memory (MiB)",
            height=250, margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(dtick=600_000, tickformat="%H:%M"),
        )
        st.plotly_chart(mem_fig, use_container_width=True)
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
uv run python scripts/launch_dlc_finetune_ec2.py
```

**Monitor progress:**
```bash
uv run python scripts/launch_dlc_finetune_ec2.py --progress
uv run python scripts/launch_dlc_finetune_ec2.py --status
```

**After training:**
- Review tracking quality on the Tracking QC page
- Promote fine-tuned results: `uv run python scripts/promote_finetuned_pose.py`
- Re-run DLC Inference (Stage 2b) on all sessions
        """
    )
