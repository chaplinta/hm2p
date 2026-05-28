"""DLC Training page (Stage 2a) — model training status, evaluation, and GPU monitoring.

Displays training artifacts from ``s3://hm2p-derivatives/dlc-retrain/``:
champion model info, aggregate evaluation metrics, training curves,
per-bodypart RMSE, per-frame error heatmaps, and GPU utilization.

All data comes from training-time artifacts. No inference outputs
(``pose/`` prefix) are needed.
"""

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
    "Trains a DLC HRNet-W32 model (ImageNet or SuperAnimal pretrained "
    "backbone) on manually labelled hm2p frames. 8 bodyparts mapped to "
    "SuperAnimal TopViewMouse keypoints. GPU required (g4dn.xlarge, 24h "
    "maximum). DLC Inference (Stage 2b) depends on the trained model "
    "produced here."
)

RETRAIN_PREFIX = "dlc-retrain"

# ── Bodypart display constants ──────────────────────────────────────────

BODYPARTS = [
    "nose_tip",
    "left_ear",
    "right_ear",
    "head_midpoint",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]

BP_COLORS: dict[str, str] = {
    "nose_tip": "#7F00FF",
    "left_ear": "#376DF8",
    "right_ear": "#12C7E5",
    "head_midpoint": "#5AF8C7",
    "neck": "#A4F89E",
    "mid_back": "#ECC76E",
    "mouse_center": "#FF6D38",
    "tail_base": "#FF0000",
}


# ═══════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=300)
def _load_champion_info() -> dict | None:
    """Load dlc-champion.json from S3."""
    data = download_s3_bytes(DERIVATIVES_BUCKET, "dlc-champion.json")
    if data is None:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None


@st.cache_data(ttl=120)
def _load_eval_results() -> dict | None:
    """Load _eval_results.json from S3 (aggregate train/test metrics)."""
    data = download_s3_bytes(DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/_eval_results.json")
    if data is None:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_per_bodypart_eval() -> dict | None:
    """Load _per_bodypart_eval.json from S3 (per-bodypart + per-frame)."""
    data = download_s3_bytes(
        DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/models/_per_bodypart_eval.json"
    )
    if data is None:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None


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
            vals = {k.strip(): v.strip() for k, v in row.items()}
            gpu_col = vals.get("utilization.gpu [%]", "0").replace(" %", "").replace("%", "")
            mem_used = vals.get("memory.used [MiB]", "0").replace(" MiB", "").replace("MiB", "")
            mem_total = vals.get("memory.total [MiB]", "0").replace(" MiB", "").replace("MiB", "")
            rows.append(
                {
                    "timestamp": vals.get("timestamp", ""),
                    "gpu_util_pct": int(gpu_col),
                    "mem_used_mb": int(mem_used),
                    "mem_total_mb": int(mem_total),
                }
            )
        except (ValueError, KeyError):
            continue
    return rows if rows else None


@st.cache_data(ttl=120)
def _check_model_exists() -> bool:
    """Check whether trained model weights exist on S3."""
    try:
        s3 = get_s3_client()
        model_suffixes = (
            ".pt",
            ".pth",
            ".pb",
            ".index",
            ".data-00000-of-00001",
            ".pkl",
            ".json",
        )
        for prefix in (f"{RETRAIN_PREFIX}/models/",):
            resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=prefix)
            if any(obj["Key"].endswith(model_suffixes) for obj in resp.get("Contents", [])):
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
        key = (
            f"{RETRAIN_PREFIX}/models/iteration-0/"
            f"hm2p-retrainMar20-{shuffle}/train/learning_stats.csv"
        )
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
                rows.append(
                    {
                        "epoch": epoch,
                        "total_epochs": int(df["step"].max()),
                        "lr": float("nan"),
                        "train_loss": float(train_loss),
                        "valid_loss": (float(valid_loss) if pd.notna(valid_loss) else None),
                        "rmse_px": (float(rmse) if pd.notna(rmse) else None),
                        "rmse_pcutoff_px": (float(rmse_pcut) if pd.notna(rmse_pcut) else None),
                        "mAP": float(mAP) if pd.notna(mAP) else None,
                    }
                )
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
        rows.append(
            {
                "epoch": int(m.group(1)),
                "total_epochs": int(m.group(2)),
                "lr": float(m.group(3)),
                "train_loss": float(m.group(4)),
                "valid_loss": (float(m.group(5)) if m.group(5) else None),
                "rmse_px": None,
                "rmse_pcutoff_px": None,
                "mAP": None,
            }
        )
    return rows if rows else None


# ═══════════════════════════════════════════════════════════════════════
# 1. Champion Info
# ═══════════════════════════════════════════════════════════════════════

st.header("Champion Model")

champ = _load_champion_info()
if champ:
    # Show promoted_at with time if available, fall back to training_date
    _promoted = champ.get("promoted_at", champ.get("training_date", "?"))
    if "T" in str(_promoted):
        try:
            from datetime import datetime, timezone, timedelta
            _dt = datetime.fromisoformat(str(_promoted).replace("Z", "+00:00"))
            _perth = _dt.astimezone(timezone(timedelta(hours=8)))
            _promoted = _perth.strftime("%Y-%m-%d %H:%M AWST")
        except Exception:
            _promoted = str(_promoted)[:19]
    st.info(
        f"**Champion:** {champ.get('champion_id', '?')}  \n"
        f"Architecture: {champ.get('architecture', '?')} | "
        f"Snapshot: {champ.get('snapshot', '?')} | "
        f"Promoted: {_promoted}"
    )
else:
    st.info(
        "No champion model declared yet. Train a model and run "
        "`scripts/declare_champion.py` to promote it."
    )

# ═══════════════════════════════════════════════════════════════════════
# 2. Aggregate Eval Metrics
# ═══════════════════════════════════════════════════════════════════════

st.header("Evaluation Metrics")

eval_data = _load_eval_results()

if eval_data:
    col_train, col_test, col_prev = st.columns(3)
    with col_train:
        st.metric(
            "Train RMSE (px)",
            f"{eval_data['train']['rmse']:.2f}",
        )
        st.metric(
            "Train mAP",
            f"{eval_data['train']['mAP']:.1f}%",
        )
    with col_test:
        st.metric(
            "Test RMSE (px)",
            f"{eval_data['test']['rmse']:.2f}",
        )
        st.metric(
            "Test mAP",
            f"{eval_data['test']['mAP']:.1f}%",
        )
    with col_prev:
        prev = eval_data.get("previous_champion", {})
        if prev and prev.get("train_rmse") is not None:
            rmse_delta = eval_data["train"]["rmse"] - prev["train_rmse"]
            map_delta = eval_data["train"]["mAP"] - prev.get("train_mAP", 0)
            st.metric(
                "Prev RMSE",
                f"{prev['train_rmse']:.2f}",
                delta=f"{rmse_delta:+.2f}",
                delta_color="inverse",
            )
            st.metric(
                "Prev mAP",
                f"{prev.get('train_mAP', '?')}",
                delta=f"{map_delta:+.1f}",
            )
        else:
            st.caption("No previous champion for comparison.")

    with st.expander("Training details"):
        st.markdown(
            f"- **Labeled frames:** {eval_data.get('n_labeled_frames', '?')}\n"
            f"- **Train/test split:** {eval_data.get('training_fraction', '?')}\n"
            f"- **Best epoch:** {eval_data.get('best_epoch', '?')} / "
            f"{eval_data.get('total_epochs', '?')}\n"
            f"- **Train mAR:** {eval_data['train'].get('mAR', '?')}  |  "
            f"**Test mAR:** {eval_data['test'].get('mAR', '?')}"
        )
else:
    st.info(
        "No evaluation results on S3. Results appear after training "
        "completes and `evaluate_network` runs."
    )

# ── Training status / progress ──────────────────────────────────────────

progress_data = _load_retrain_progress()
model_exists = _check_model_exists()

if model_exists:
    st.success("Trained model weights found on S3.")
elif progress_data:
    status = progress_data.get("status", "unknown")
    updated = progress_data.get("updated", "")
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
    _curves = _parse_training_curves()
    if _curves and len(_curves) > 0:
        last_epoch = _curves[-1]
        if last_epoch["epoch"] == last_epoch["total_epochs"]:
            st.warning(
                f"Training completed ({last_epoch['epoch']} epochs) but "
                f"model weights are not on S3. The upload may have failed."
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


# ═══════════════════════════════════════════════════════════════════════
# 3. Training Curves
# ═══════════════════════════════════════════════════════════════════════

st.header("Training Curves")

with st.expander("Understanding the metrics"):
    st.markdown("""
**Valid RMSE (all, px)** -- Root Mean Square Error in pixels on the
validation set (20% held-out frames), computed across ALL predicted
bodypart locations. Includes predictions where the model is uncertain.
Lower = better. A value of 10 px on an 832x608 image means the average
prediction is ~10 pixels from the labelled ground truth (~1.2% of
image width).

**Valid RMSE (confident, px)** -- Same as above but only for predictions
where the model's confidence exceeds the p-cutoff threshold. This
excludes uncertain predictions (e.g. occluded bodyparts) and is
typically lower than the all-points RMSE. This is the more relevant
metric for downstream analysis since low-confidence predictions are
filtered out by the kinematics pipeline.

**mAP (mean Average Precision)** -- A detection metric from the COCO
object detection benchmark. For each bodypart, it computes Average
Precision: the area under the precision-recall curve at multiple
distance thresholds (how close the prediction must be to count as
correct). mAP is the mean across all bodyparts. Range 0--100%.

- **0%** = model cannot find any bodypart
- **~30%** = model finds bodyparts but with poor localisation
- **~60%** = model reliably detects most bodyparts with reasonable accuracy
- **~80%+** = publication-quality tracking

mAP is more informative than RMSE because it accounts for both
*detection* (did the model find the bodypart at all?) and
*localisation* (how close is the prediction to the true position?).
A model with low RMSE but low mAP is only accurate on the easy frames
and misses the hard ones.

**Training loss (heatmap + locref)** -- The optimisation objective
during training. Combines two components:
1. *Heatmap loss*: MSE between predicted and target Gaussian heatmaps
   at 1/4 resolution. Each bodypart produces a 2D probability map;
   the loss measures how well the predicted peak matches the label.
2. *Location refinement (locref) loss*: subpixel offset prediction
   to refine the heatmap peak to full resolution accuracy.

This loss is NOT in pixels -- it's in normalised heatmap space and
cannot be directly compared to RMSE. Use it to monitor convergence
(decreasing = learning) and overfitting (train decreasing but valid
increasing), but interpret the absolute value as pixel RMSE from the
validation metrics above.
""")

curve_data = _parse_training_curves()

if curve_data:
    import pandas as pd
    import plotly.graph_objects as go

    df_curves = pd.DataFrame(curve_data).set_index("epoch")
    total_epochs = curve_data[-1]["total_epochs"]

    # Check if pixel RMSE is available
    rmse_rows = [r for r in curve_data if r.get("rmse_px") is not None]
    has_pixel_metrics = bool(rmse_rows)

    if has_pixel_metrics:
        last_rmse = rmse_rows[-1]
        mAP_rows = [r for r in rmse_rows if r.get("mAP") is not None and r["mAP"] > 0]
        best_mAP = max(mAP_rows, key=lambda r: r["mAP"]) if mAP_rows else last_rmse

        # Find the actual best snapshot DLC saved (from S3 filename)
        _best_epoch_actual = None
        try:
            import re as _re

            _s3_check = get_s3_client()
            _resp = _s3_check.list_objects_v2(
                Bucket=DERIVATIVES_BUCKET,
                Prefix=f"{RETRAIN_PREFIX}/models/",
                MaxKeys=100,
            )
            _best_files = [
                o["Key"] for o in _resp.get("Contents", []) if "snapshot-best" in o["Key"]
            ]
            if _best_files:
                _epochs_found = []
                for bf in _best_files:
                    m = _re.search(r"snapshot-best-(\d+)", bf)
                    if m:
                        _epochs_found.append(int(m.group(1)))
                if _epochs_found:
                    _best_epoch_actual = max(_epochs_found)
        except Exception:
            pass

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Epochs", f"{len(curve_data)}/{total_epochs}")
        col2.metric(
            "Valid RMSE (all)",
            (f"{last_rmse['rmse_px']:.1f} px" if last_rmse.get("rmse_px") is not None else "N/A"),
        )
        col3.metric(
            "Best mAP",
            (f"{best_mAP['mAP']:.1f}%" if best_mAP.get("mAP") is not None else "N/A"),
        )
        if _best_epoch_actual is not None:
            col4.metric("Selected model", f"Epoch {_best_epoch_actual}")
        else:
            col4.metric("Best by mAP", f"Epoch {best_mAP['epoch']}")

        # Plot pixel RMSE
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[r["epoch"] for r in rmse_rows],
                y=[r["rmse_px"] for r in rmse_rows],
                mode="lines+markers",
                name="Valid RMSE (all, px)",
                line=dict(color="#d62728", width=2),
                marker=dict(size=5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r["epoch"] for r in rmse_rows],
                y=[r["rmse_pcutoff_px"] for r in rmse_rows],
                mode="lines+markers",
                name="Valid RMSE (confident, px)",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=5),
            )
        )
        # Star on the actual selected model epoch
        _star_epoch = _best_epoch_actual or best_mAP["epoch"]
        _star_row = next((r for r in rmse_rows if r["epoch"] == _star_epoch), None)
        if _star_row and _star_row.get("rmse_px") is not None:
            _star_label = f"Selected model (epoch {_star_epoch}"
            if _star_row.get("mAP") is not None:
                _star_label += f", mAP {_star_row['mAP']:.1f}%"
            _star_label += ")"
            fig.add_trace(
                go.Scatter(
                    x=[_star_epoch],
                    y=[_star_row["rmse_px"]],
                    mode="markers",
                    name=_star_label,
                    marker=dict(size=12, color="#ff7f0e", symbol="star"),
                )
            )
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="RMSE (pixels)",
            height=400,
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(x=0.5, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap loss in expander
        with st.expander("Training loss (heatmap + locref)"):
            loss_fig = go.Figure()
            loss_fig.add_trace(
                go.Scatter(
                    x=df_curves.index,
                    y=df_curves["train_loss"],
                    mode="lines",
                    name="Train loss",
                    line=dict(color="#1f77b4", width=1.5),
                )
            )
            valid_rows = [r for r in curve_data if r["valid_loss"] is not None]
            if valid_rows:
                loss_fig.add_trace(
                    go.Scatter(
                        x=[r["epoch"] for r in valid_rows],
                        y=[r["valid_loss"] for r in valid_rows],
                        mode="lines+markers",
                        name="Valid loss",
                        line=dict(color="#d62728", width=2),
                        marker=dict(size=5),
                    )
                )
            loss_fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=300,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(loss_fig, use_container_width=True)

        # mAP over epochs
        if mAP_rows:
            with st.expander("mAP over epochs"):
                mAP_fig = go.Figure()
                mAP_fig.add_trace(
                    go.Scatter(
                        x=[r["epoch"] for r in mAP_rows],
                        y=[r["mAP"] for r in mAP_rows],
                        mode="lines+markers",
                        name="mAP (%)",
                        line=dict(color="#ff7f0e", width=2),
                        marker=dict(size=5),
                    )
                )
                mAP_fig.update_layout(
                    xaxis_title="Epoch",
                    yaxis_title="mAP (%)",
                    height=250,
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(mAP_fig, use_container_width=True)

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

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_curves.index,
                y=df_curves["train_loss"],
                mode="lines",
                name="Train loss (heatmap MSE)",
                line=dict(color="#1f77b4", width=1.5),
            )
        )
        if valid_rows:
            fig.add_trace(
                go.Scatter(
                    x=[r["epoch"] for r in valid_rows],
                    y=[r["valid_loss"] for r in valid_rows],
                    mode="lines+markers",
                    name="Valid loss",
                    line=dict(color="#d62728", width=2),
                    marker=dict(size=6),
                )
            )
        if best_valid:
            fig.add_trace(
                go.Scatter(
                    x=[best_valid["epoch"]],
                    y=[best_valid["valid_loss"]],
                    mode="markers",
                    name=f"Best (epoch {best_valid['epoch']})",
                    marker=dict(size=12, color="#2ca02c", symbol="star"),
                    showlegend=True,
                )
            )
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
                    f"Validation loss increased from "
                    f"{best_valid['valid_loss']:.5f} "
                    f"(epoch {best_valid['epoch']}) to "
                    f"{last_valid['valid_loss']:.5f} "
                    f"(epoch {last_valid['epoch']}). The model is "
                    f"overfitting -- DLC selected the best checkpoint "
                    f"at epoch {best_valid['epoch']}."
                )

else:
    st.info("No training log on S3. Training curves will appear after training starts.")


# ═══════════════════════════════════════════════════════════════════════
# 4. Per-Bodypart Evaluation
# ═══════════════════════════════════════════════════════════════════════

st.header("Per-Bodypart Evaluation")

bp_eval = _load_per_bodypart_eval()

if bp_eval:
    import numpy as _np
    import plotly.graph_objects as go  # noqa: F811

    bp_info = bp_eval.get("bodyparts", {})
    active_bps = [bp for bp in BODYPARTS if bp_info.get(bp, {}).get("rmse") is not None]

    if active_bps:
        # Summary table
        pcutoff = bp_info[active_bps[0]].get("pcutoff", 0.6)
        st.markdown(f"**Confidence threshold (pcutoff):** {pcutoff}")

        import pandas as _pd

        rows = []
        for bp in active_bps:
            info = bp_info[bp]
            rows.append({
                "Bodypart": bp,
                "Median (px)": f"{info.get('median_error', 0):.2f}",
                "P95 (px)": f"{info.get('p95_error', 0):.1f}",
                "Max (px)": f"{info.get('max_error', 0):.0f}",
                "PCK@10": f"{info.get('pck_10', 0):.1f}%",
                "% Above Cutoff": f"{info.get('pct_above_cutoff', 0):.1f}%",
                "Median (filtered)": f"{info.get('median_filtered', 0):.2f}" if info.get("n_filtered", 0) > 0 else "—",
                "P95 (filtered)": f"{info.get('p95_filtered', 0):.1f}" if info.get("n_filtered", 0) > 0 else "—",
                "n": info["n"],
            })
        st.dataframe(_pd.DataFrame(rows).set_index("Bodypart"), use_container_width=True)

        # Median + P95 bar chart (grouped)
        fig_rmse = go.Figure()
        fig_rmse.add_trace(
            go.Bar(
                x=active_bps,
                y=[bp_info[bp].get("median_error", 0) for bp in active_bps],
                marker_color=[BP_COLORS.get(bp, "#888") for bp in active_bps],
                name="Median error",
            )
        )
        fig_rmse.add_trace(
            go.Bar(
                x=active_bps,
                y=[bp_info[bp].get("p95_error", 0) for bp in active_bps],
                marker_color=[BP_COLORS.get(bp, "#888") for bp in active_bps],
                opacity=0.4,
                name="95th percentile",
            )
        )
        fig_rmse.update_layout(
            xaxis_title="Bodypart",
            yaxis_title="Error (pixels)",
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
            barmode="overlay",
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

        # Confidence: % above cutoff per bodypart
        fig_conf = go.Figure()
        fig_conf.add_trace(
            go.Bar(
                x=active_bps,
                y=[bp_info[bp].get("pct_above_cutoff", 0) for bp in active_bps],
                marker_color=[BP_COLORS.get(bp, "#888") for bp in active_bps],
                text=[f"{bp_info[bp].get('pct_above_cutoff', 0):.1f}%" for bp in active_bps],
                textposition="outside",
            )
        )
        fig_conf.update_layout(
            xaxis_title="Bodypart",
            yaxis_title=f"% Predictions Above pcutoff={pcutoff}",
            yaxis_range=[0, 105],
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        # PCK curves
        with st.expander("PCK curves"):
            _thresholds = [5, 10, 15, 20]
            fig_pck = go.Figure()
            for bp in active_bps:
                info = bp_info[bp]
                pck_vals = [info.get(f"pck_{t}", 0) for t in _thresholds]
                fig_pck.add_trace(
                    go.Scatter(
                        x=[str(t) for t in _thresholds],
                        y=pck_vals,
                        mode="lines+markers",
                        name=bp,
                        line=dict(color=BP_COLORS.get(bp, "#888")),
                    )
                )
            fig_pck.update_layout(
                xaxis_title="Threshold (pixels)",
                yaxis_title="PCK (%)",
                yaxis_range=[0, 105],
                height=350,
                margin=dict(l=40, r=20, t=20, b=40),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_pck, use_container_width=True)

        st.caption(
            f"From {bp_eval.get('n_total_matched', '?')} matched "
            f"frame-bodypart pairs. "
            f"Filtered metrics use predictions with likelihood > {pcutoff}. "
            "Median and P95 are more informative than RMSE when outliers "
            "are present (detector failures on a few frames)."
        )
    else:
        st.info("No per-bodypart data available in the evaluation JSON.")
else:
    st.info(
        "Per-bodypart evaluation not yet computed. It runs automatically after training completes."
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. Per-Frame Error Heatmap
# ═══════════════════════════════════════════════════════════════════════

st.header("Per-Frame Error Heatmap")

if bp_eval and bp_eval.get("per_frame"):
    import numpy as _np
    import plotly.graph_objects as go  # noqa: F811

    per_frame = bp_eval["per_frame"]
    bp_info = bp_eval.get("bodyparts", {})
    active_bps = [bp for bp in BODYPARTS if bp_info.get(bp, {}).get("rmse") is not None]

    if not active_bps:
        st.info("No bodypart data for per-frame heatmap.")
    else:
        # Separate train and test frames
        train_frames = [f for f in per_frame if f.get("split") == "train"]
        test_frames = [f for f in per_frame if f.get("split") == "test"]
        unknown_frames = [f for f in per_frame if f.get("split") not in ("train", "test")]

        col_threshold, _ = st.columns([1, 3])
        with col_threshold:
            threshold_px = st.number_input(
                "Error threshold (px)",
                min_value=1,
                max_value=200,
                value=20,
                step=1,
                help=("Frames with any bodypart error above this value are highlighted."),
            )

        def _render_heatmap(
            frames: list[dict],
            title: str,
            bps: list[str],
        ) -> None:
            """Render an error heatmap for a set of frames."""
            if not frames:
                st.caption(f"{title}: no frames.")
                return

            # Build error matrix (rows=frames, cols=bodyparts)
            z = []
            labels = []
            for f in frames:
                row = []
                for bp in bps:
                    err = f.get("errors", {}).get(bp, float("nan"))
                    row.append(err)
                z.append(row)
                labels.append(f.get("frame_id", "?"))

            z_arr = _np.array(z)

            # Sort by mean error (worst first)
            mean_errs = _np.nanmean(z_arr, axis=1)
            sort_idx = _np.argsort(-mean_errs)
            z_arr = z_arr[sort_idx]
            labels = [labels[i] for i in sort_idx]

            colorscale = [
                [0.0, "#FFFFFF"],
                [0.3, "#FFFACD"],
                [0.6, "#FFA500"],
                [1.0, "#CC0000"],
            ]

            fig_hm = go.Figure(
                go.Heatmap(
                    z=z_arr,
                    x=bps,
                    y=labels,
                    colorscale=colorscale,
                    zmin=0,
                    zmax=max(
                        threshold_px * 2,
                        float(_np.nanmax(z_arr)) if z_arr.size > 0 else 40,
                    ),
                    colorbar={"title": "Error (px)"},
                    hovertemplate=(
                        "Frame: %{y}<br>Body part: %{x}<br>Error: %{z:.1f} px<extra></extra>"
                    ),
                )
            )
            fig_hm.update_layout(
                title=title,
                xaxis_title="Body part",
                yaxis_title="Frame",
                height=max(250, len(labels) * 20 + 80),
                margin={"t": 50, "b": 40},
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            # Count worst frames
            n_above = (
                int((_np.nanmax(z_arr, axis=1) > threshold_px).sum()) if z_arr.size > 0 else 0
            )
            mean_all = float(_np.nanmean(z_arr)) if z_arr.size > 0 else 0
            st.caption(
                f"{len(frames)} frames, "
                f"mean error {mean_all:.1f} px, "
                f"{n_above} frame(s) with any bodypart > {threshold_px} px."
            )

        if test_frames:
            _render_heatmap(test_frames, "Test frames", active_bps)
        if train_frames:
            _render_heatmap(train_frames, "Train frames", active_bps)
        if unknown_frames:
            _render_heatmap(unknown_frames, "Frames (split unknown)", active_bps)

        # Worst frames table
        with st.expander("Worst frames (highest mean error)"):
            import pandas as pd

            rows_table = []
            for f in per_frame:
                errs = f.get("errors", {})
                if not errs:
                    continue
                err_vals = [v for v in errs.values() if not _np.isnan(v)]
                if not err_vals:
                    continue
                row = {
                    "frame": f.get("frame_id", "?"),
                    "split": f.get("split", "?"),
                    "mean_error": round(_np.mean(err_vals), 1),
                }
                for bp in active_bps:
                    row[bp] = round(errs.get(bp, float("nan")), 1)
                rows_table.append(row)

            if rows_table:
                df_worst = (
                    pd.DataFrame(rows_table)
                    .sort_values("mean_error", ascending=False)
                    .head(20)
                    .reset_index(drop=True)
                )
                st.dataframe(
                    df_worst.style.format(
                        {col: "{:.1f}" for col in ["mean_error"] + active_bps}
                    ).background_gradient(subset=["mean_error"], cmap="YlOrRd"),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No frame data available.")
else:
    st.info("Per-frame error data not yet available. It is generated during training evaluation.")


# ═══════════════════════════════════════════════════════════════════════
# 6. GPU Utilization
# ═══════════════════════════════════════════════════════════════════════

st.header("GPU Utilization")

gpu_data = _load_gpu_monitor()

if gpu_data:
    import pandas as pd
    import plotly.graph_objects as go  # noqa: F811

    df = pd.DataFrame(gpu_data)
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
    gpu_fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["gpu_util_pct"],
            mode="lines",
            name="GPU %",
            line=dict(color="#ff7f0e", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 127, 14, 0.2)",
        )
    )
    gpu_fig.update_layout(
        xaxis_title="Time",
        yaxis_title="GPU Utilization (%)",
        yaxis=dict(range=[0, 105]),
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(dtick=600_000, tickformat="%H:%M"),
    )
    st.plotly_chart(gpu_fig, use_container_width=True)

    # Memory usage
    with st.expander("GPU memory usage"):
        mem_fig = go.Figure()
        mem_fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["mem_used_mb"],
                mode="lines",
                name="Used (MiB)",
            )
        )
        mem_fig.update_layout(
            xaxis_title="Time",
            yaxis_title="GPU Memory (MiB)",
            height=250,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(dtick=600_000, tickformat="%H:%M"),
        )
        st.plotly_chart(mem_fig, use_container_width=True)
else:
    st.info("No GPU monitoring data on S3 yet.")


# ═══════════════════════════════════════════════════════════════════════
# Active instances
# ═══════════════════════════════════════════════════════════════════════

st.header("Active Instances")

try:
    instances = get_ec2_instances()
    retrain_instances = [
        i
        for i in instances
        if "dlc-retrain" in i.get("project", "").lower()
        or "dlc_retrain" in i.get("project", "").lower()
    ]
    if retrain_instances:
        for inst in retrain_instances:
            st.markdown(
                f"**{inst['id']}** -- {inst['state']}  \n"
                f"Type: {inst.get('type', 'N/A')} | "
                f"IP: {inst.get('ip', 'N/A')}"
            )
    else:
        st.info("No DLC training instances currently running.")
except Exception as exc:
    st.warning(f"Could not query EC2 instances: {sanitize_error(str(exc))}")


# ═══════════════════════════════════════════════════════════════════════
# Dependency note + instructions
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "DLC Inference (Stage 2b) depends on the model produced by this "
    "stage. After training completes, run "
    "`scripts/promote_finetuned_pose.py` to QC and promote results, "
    "then re-run Stage 2b."
)

with st.expander("How to add more labeled frames"):
    st.markdown(
        """
**1. Scan sessions to see labeling status and difficulty:**
```bash
uv run python scripts/select_hard_frames.py --scan
```

**2. Extract outlier frames (DLC's jump + uncertainty detection):**
```bash
# All sessions, DLC defaults:
uv run python scripts/select_hard_frames.py

# Limit to 8 new frames per session:
uv run python scripts/select_hard_frames.py --per-session 8

# Limit to 200 total across all sessions:
uv run python scripts/select_hard_frames.py --total 200

# One session only:
uv run python scripts/select_hard_frames.py --session 20220804_11_21

# Adjust thresholds:
uv run python scripts/select_hard_frames.py --jump-threshold 15 --p-bound 0.05
```

**3. Label frames:**
```bash
uv run python scripts/interactive_label.py
```

**4. Commit labels and upload to S3:**
```bash
git add sourcedata/trackers/dlc/*/labeled-data/*/CollectedData_*
git commit -m "feat: add N labeled frames for DLC retraining"
uv run python scripts/upload_dlc_labels.py
```
        """
    )

with st.expander("How to start training"):
    st.markdown(
        """
**Launch training (24h max, GPU enforced):**
```bash
# SA fine-tune (recommended, 120 epochs):
uv run python scripts/launch_dlc_finetune_ec2.py --sa-finetune

# ImageNet HRNet (400 epochs):
uv run python scripts/launch_dlc_finetune_ec2.py
```

**Monitor progress:**
```bash
uv run python scripts/launch_dlc_finetune_ec2.py --progress
uv run python scripts/launch_dlc_finetune_ec2.py --status
```

**After training:**
- Review tracking quality on the Tracking QC page
- Run inference: \
`uv run python scripts/launch_dlc_finetune_ec2.py --infer-only`
- Compare models: `uv run python scripts/compare_models.py`
        """
    )
