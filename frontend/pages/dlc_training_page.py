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
    """Parse training metrics from learning_stats.csv on S3.

    Falls back to parsing the run log if learning_stats.csv is unavailable.
    Returns pixel-level RMSE when available (from learning_stats.csv),
    otherwise raw heatmap loss (from run log).
    """
    import io

    import pandas as pd

    # Try learning_stats.csv first (has pixel RMSE)
    for shuffle in ("trainset80shuffle1", "trainset95shuffle1"):
        key = f"{RETRAIN_PREFIX}/models/iteration-0/hm2p-retrainMar20-{shuffle}/train/learning_stats.csv"
        csv_data = download_s3_bytes(DERIVATIVES_BUCKET, key)
        if csv_data is not None:
            df = pd.read_csv(io.BytesIO(csv_data))
            rows = []
            for _, r in df.iterrows():
                epoch = int(r["step"])
                train_loss = r.get("losses/train.total_loss", float("nan"))
                valid_loss = r.get("losses/eval.total_loss", float("nan"))
                rmse = r.get("metrics/test.rmse", None)
                rmse_pcut = r.get("metrics/test.rmse_pcutoff", None)
                mAP = r.get("metrics/test.mAP", None)
                rows.append({
                    "epoch": epoch,
                    "total_epochs": int(df["step"].max()),
                    "lr": float("nan"),
                    "train_loss": float(train_loss),
                    "valid_loss": float(valid_loss) if pd.notna(valid_loss) else None,
                    "rmse_px": float(rmse) if pd.notna(rmse) else None,
                    "rmse_pcutoff_px": float(rmse_pcut) if pd.notna(rmse_pcut) else None,
                    "mAP": float(mAP) if pd.notna(mAP) else None,
                })
            if rows:
                return rows

    # Fallback: parse run log
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
        rows.append({
            "epoch": int(m.group(1)),
            "total_epochs": int(m.group(2)),
            "lr": float(m.group(3)),
            "train_loss": float(m.group(4)),
            "valid_loss": float(m.group(5)) if m.group(5) else None,
            "rmse_px": None,
            "rmse_pcutoff_px": None,
            "mAP": None,
        })
    return rows if rows else None


@st.cache_data(ttl=300)
def _load_per_bodypart_eval() -> dict | None:
    """Load per-bodypart evaluation results from S3."""
    import io

    for shuffle in ("trainset80shuffle1", "trainset95shuffle1"):
        prefix = f"{RETRAIN_PREFIX}/models/evaluation-results/"
        data = download_s3_bytes(DERIVATIVES_BUCKET, prefix)
        if data is not None:
            break

    # Try to find any CSV in evaluation-results/
    try:
        import boto3 as _boto3

        s3 = _boto3.client("s3", region_name="ap-southeast-2")
        resp = s3.list_objects_v2(
            Bucket=DERIVATIVES_BUCKET,
            Prefix=f"{RETRAIN_PREFIX}/models/evaluation-results/",
            MaxKeys=20,
        )
        csv_keys = [
            o["Key"]
            for o in resp.get("Contents", [])
            if o["Key"].endswith(".csv")
        ]
        if not csv_keys:
            return None

        obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=csv_keys[0])
        import pandas as _pd

        df = _pd.read_csv(io.BytesIO(obj["Body"].read()))

        # DLC evaluation CSV has columns like: bodyparts, RMSE, RMSE_pcutoff, etc.
        # Format varies by DLC version — try to extract what we can.
        if "bodyparts" in df.columns and "RMSE" in df.columns:
            result = {
                "bodyparts": df["bodyparts"].tolist(),
                "rmse": df["RMSE"].tolist(),
                "rmse_pcutoff": df.get("RMSE_pcutoff", df["RMSE"]).tolist(),
                "mAP_per_bp": df["mAP"].tolist() if "mAP" in df.columns else None,
            }
            return result

        # Fallback: try to parse whatever columns exist
        return None
    except Exception:
        return None


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
    # Convert UTC to Perth time (UTC+8)
    if updated:
        from datetime import datetime, timedelta, timezone
        try:
            utc_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            perth_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
            updated_local = perth_dt.strftime("%Y-%m-%d %H:%M AWST")
        except Exception:
            updated_local = updated[:19]
    else:
        updated_local = "N/A"
    st.markdown(f"**Status:** {status}")
    st.caption(f"Last updated: {updated_local}")

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

with st.expander("Understanding the metrics"):
    st.markdown("""
**Valid RMSE (all, px)** — Root Mean Square Error in pixels on the
validation set (20% held-out frames), computed across ALL predicted
bodypart locations. Includes predictions where the model is uncertain.
Lower = better. A value of 10 px on an 832×608 image means the average
prediction is ~10 pixels from the labelled ground truth (~1.2% of
image width).

**Valid RMSE (confident, px)** — Same as above but only for predictions
where the model's confidence exceeds the p-cutoff threshold. This
excludes uncertain predictions (e.g. occluded bodyparts) and is
typically lower than the all-points RMSE. This is the more relevant
metric for downstream analysis since low-confidence predictions are
filtered out by the kinematics pipeline.

**mAP (mean Average Precision)** — A detection metric from the COCO
object detection benchmark. For each bodypart, it computes Average
Precision: the area under the precision-recall curve at multiple
distance thresholds (how close the prediction must be to count as
correct). mAP is the mean across all bodyparts. Range 0–100%.

- **0%** = model cannot find any bodypart
- **~30%** = model finds bodyparts but with poor localisation
- **~60%** = model reliably detects most bodyparts with reasonable accuracy
- **~80%+** = publication-quality tracking

mAP is more informative than RMSE because it accounts for both
*detection* (did the model find the bodypart at all?) and
*localisation* (how close is the prediction to the true position?).
A model with low RMSE but low mAP is only accurate on the easy frames
and misses the hard ones.

**Training loss (heatmap + locref)** — The optimisation objective
during training. Combines two components:
1. *Heatmap loss*: MSE between predicted and target Gaussian heatmaps
   at 1/4 resolution. Each bodypart produces a 2D probability map;
   the loss measures how well the predicted peak matches the label.
2. *Location refinement (locref) loss*: subpixel offset prediction
   to refine the heatmap peak to full resolution accuracy.

This loss is NOT in pixels — it's in normalised heatmap space and
cannot be directly compared to RMSE. Use it to monitor convergence
(decreasing = learning) and overfitting (train decreasing but valid
increasing), but interpret the absolute value as pixel RMSE from the
validation metrics above.
""")

curve_data = _parse_training_curves()

if curve_data:
    import pandas as pd

    df_curves = pd.DataFrame(curve_data).set_index("epoch")
    total_epochs = curve_data[-1]["total_epochs"]

    import plotly.graph_objects as go

    # Check if pixel RMSE is available
    rmse_rows = [r for r in curve_data if r.get("rmse_px") is not None]
    has_pixel_metrics = bool(rmse_rows)

    if has_pixel_metrics:
        # Show pixel RMSE (from learning_stats.csv)
        best_rmse = min(rmse_rows, key=lambda r: r["rmse_pcutoff_px"] or 999)
        last_rmse = rmse_rows[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Epochs", f"{len(curve_data)}/{total_epochs}")
        col2.metric("Valid RMSE (all)", f"{last_rmse['rmse_px']:.1f} px")
        col3.metric("Valid RMSE (confident)", f"{last_rmse['rmse_pcutoff_px']:.1f} px")
        col4.metric("Best checkpoint", f"Epoch {best_rmse['epoch']}")

        if last_rmse.get("mAP") is not None:
            st.caption(f"mAP: {last_rmse['mAP']:.1f}%")

        # Plot pixel RMSE
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[r["epoch"] for r in rmse_rows],
            y=[r["rmse_px"] for r in rmse_rows],
            mode="lines+markers",
            name="Valid RMSE (all, px)",
            line=dict(color="#d62728", width=2),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=[r["epoch"] for r in rmse_rows],
            y=[r["rmse_pcutoff_px"] for r in rmse_rows],
            mode="lines+markers",
            name="Valid RMSE (confident, px)",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=[best_rmse["epoch"]],
            y=[best_rmse["rmse_pcutoff_px"]],
            mode="markers",
            name=f"Best ({best_rmse['rmse_pcutoff_px']:.1f} px, epoch {best_rmse['epoch']})",
            marker=dict(size=12, color="#2ca02c", symbol="star"),
        ))
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="RMSE (pixels)",
            height=400,
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(x=0.5, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Also show heatmap loss in expander
        with st.expander("Training loss (heatmap + locref)"):
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(
                x=df_curves.index, y=df_curves["train_loss"],
                mode="lines", name="Train loss",
                line=dict(color="#1f77b4", width=1.5),
            ))
            valid_rows = [r for r in curve_data if r["valid_loss"] is not None]
            if valid_rows:
                loss_fig.add_trace(go.Scatter(
                    x=[r["epoch"] for r in valid_rows],
                    y=[r["valid_loss"] for r in valid_rows],
                    mode="lines+markers", name="Valid loss",
                    line=dict(color="#d62728", width=2), marker=dict(size=5),
                ))
            loss_fig.update_layout(
                xaxis_title="Epoch", yaxis_title="Loss",
                height=300, margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(loss_fig, use_container_width=True)

        # mAP over epochs
        mAP_rows = [r for r in curve_data if r.get("mAP") is not None and r["mAP"] > 0]
        if mAP_rows:
            with st.expander("mAP over epochs"):
                mAP_fig = go.Figure()
                mAP_fig.add_trace(go.Scatter(
                    x=[r["epoch"] for r in mAP_rows],
                    y=[r["mAP"] for r in mAP_rows],
                    mode="lines+markers", name="mAP (%)",
                    line=dict(color="#ff7f0e", width=2), marker=dict(size=5),
                ))
                mAP_fig.update_layout(
                    xaxis_title="Epoch", yaxis_title="mAP (%)",
                    height=250, margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(mAP_fig, use_container_width=True)

        # Per-bodypart evaluation (from evaluation-results CSV on S3)
        _eval_data = _load_per_bodypart_eval()
        if _eval_data is not None:
            with st.expander("Per-bodypart RMSE"):
                import plotly.graph_objects as go  # noqa

                bp_names = _eval_data["bodyparts"]
                bp_rmse = _eval_data["rmse"]
                bp_rmse_pcut = _eval_data["rmse_pcutoff"]

                fig_bp = go.Figure()
                fig_bp.add_trace(go.Bar(
                    x=bp_names, y=bp_rmse,
                    name="RMSE (all, px)",
                    marker_color="#d62728",
                ))
                if bp_rmse_pcut:
                    fig_bp.add_trace(go.Bar(
                        x=bp_names, y=bp_rmse_pcut,
                        name="RMSE (confident, px)",
                        marker_color="#2ca02c",
                    ))
                fig_bp.update_layout(
                    xaxis_title="Bodypart", yaxis_title="RMSE (pixels)",
                    height=350, margin=dict(l=40, r=20, t=20, b=40),
                    barmode="group",
                )
                st.plotly_chart(fig_bp, use_container_width=True)

                if _eval_data.get("mAP_per_bp"):
                    st.markdown("**Per-bodypart mAP:**")
                    for bp, val in zip(bp_names, _eval_data["mAP_per_bp"]):
                        st.caption(f"  {bp}: {val:.1f}%")

    else:
        # Fallback: show raw heatmap loss (no pixel metrics available)
        final_train = curve_data[-1]["train_loss"]
        valid_rows = [r for r in curve_data if r["valid_loss"] is not None]
        best_valid = min(valid_rows, key=lambda r: r["valid_loss"]) if valid_rows else None

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Epochs", f"{len(curve_data)}/{total_epochs}")
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
            name="Train loss (heatmap MSE)",
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
