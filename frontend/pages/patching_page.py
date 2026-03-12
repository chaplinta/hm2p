"""Patching electrophysiology analysis page.

Displays per-cell passive and active electrophysiology metrics,
morphology summary, and Penk+ vs non-Penk group comparisons.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from frontend.data import DERIVATIVES_BUCKET, download_s3_bytes

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "patching" / "analysis"
S3_PREFIX = "patching/analysis"


@st.cache_data(ttl=300)
def _load_csv(name: str) -> pd.DataFrame | None:
    """Load a patching CSV from S3, falling back to local file."""
    # Try S3 first
    data = download_s3_bytes(DERIVATIVES_BUCKET, f"{S3_PREFIX}/{name}")
    if data is not None:
        return pd.read_csv(io.BytesIO(data))

    # Fall back to local file
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _nice_metric_name(col: str) -> str:
    """Convert column name like 'ephys_passive_RMP' to 'RMP'."""
    parts = col.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return col


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Patching Electrophysiology")

metrics = _load_csv("metrics.csv")
if metrics is None:
    st.warning("No patching results found. Run `python scripts/run_patching.py` first.")
    st.stop()

summary = _load_csv("summary_stats.csv")
mannwhitney = _load_csv("mannwhitney.csv")

# --- Overview ---
n_cells = len(metrics)
n_penk = (metrics["cell_type"] == "penkpos").sum()
n_nonpenk = (metrics["cell_type"] == "penkneg").sum()
n_animals = metrics["animal_id"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total cells", n_cells)
c2.metric("Penk+", n_penk)
c3.metric("Non-Penk", n_nonpenk)
c4.metric("Animals", n_animals)

# --- Passive electrophysiology ---
st.header("Passive Properties")

passive_cols = [c for c in metrics.columns if c.startswith("ephys_passive_")]
passive_with_data = [c for c in passive_cols if metrics[c].notna().any()]

if passive_with_data:
    display_df = metrics[["cell_index", "animal_id", "cell_type", "area", "layer"] + passive_with_data].copy()
    display_df.columns = ["Cell", "Animal", "Type", "Area", "Layer"] + [
        _nice_metric_name(c) for c in passive_with_data
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No passive electrophysiology data available.")

# --- Group comparison ---
st.header("Penk+ vs Non-Penk Comparison")

if mannwhitney is not None:
    sig = mannwhitney[mannwhitney["significant"] == True]  # noqa: E712
    has_data = mannwhitney[mannwhitney["p_value"].notna()]

    if len(sig) > 0:
        st.success(f"**{len(sig)}** significant result(s) after FDR correction (q < 0.05):")
        for _, row in sig.iterrows():
            name = _nice_metric_name(row["metric"])
            st.markdown(f"- **{name}**: U = {row['statistic']:.1f}, p = {row['p_value']:.4f}, p_FDR = {row['p_fdr']:.4f}")
    else:
        st.info("No significant differences after FDR correction.")

    if len(has_data) > 0:
        st.subheader("All tested metrics")
        show_mw = has_data[["metric", "statistic", "p_value", "p_fdr", "significant"]].copy()
        show_mw["metric"] = show_mw["metric"].apply(_nice_metric_name)
        show_mw.columns = ["Metric", "U statistic", "p-value", "p (FDR)", "Significant"]
        st.dataframe(show_mw, use_container_width=True, hide_index=True)
else:
    st.info("No Mann-Whitney comparison results. Run with `--stats`.")

# --- Summary statistics ---
if summary is not None:
    st.header("Descriptive Statistics by Group")

    # Filter to metrics with actual data
    summary_with_data = summary[summary["penkneg_n"].notna() & (summary["penkneg_n"] > 0)]
    if len(summary_with_data) > 0:
        summary_display = summary_with_data.copy()
        summary_display["metric"] = summary_display["metric"].apply(_nice_metric_name)

        # Show Penk+ and non-Penk side by side
        tab_penk, tab_nonpenk = st.tabs(["Penk+ (penkpos)", "Non-Penk (penkneg)"])

        with tab_penk:
            penk_cols = ["metric"] + [c for c in summary_display.columns if c.startswith("penkpos_")]
            df_p = summary_display[penk_cols].copy()
            df_p.columns = ["Metric", "n", "Mean", "Median", "Std", "SEM", "Min", "Max"]
            st.dataframe(df_p, use_container_width=True, hide_index=True)

        with tab_nonpenk:
            nonpenk_cols = ["metric"] + [c for c in summary_display.columns if c.startswith("penkneg_")]
            df_np = summary_display[nonpenk_cols].copy()
            df_np.columns = ["Metric", "n", "Mean", "Median", "Std", "SEM", "Min", "Max"]
            st.dataframe(df_np, use_container_width=True, hide_index=True)

# --- Active properties (placeholder until efel data available) ---
st.header("Active (Spike) Properties")

active_cols = [c for c in metrics.columns if c.startswith("ephys_active_")]
active_with_data = [c for c in active_cols if metrics[c].notna().any()]

if active_with_data:
    display_active = metrics[["cell_index", "cell_type"] + active_with_data].copy()
    display_active.columns = ["Cell", "Type"] + [_nice_metric_name(c) for c in active_with_data]
    st.dataframe(display_active, use_container_width=True, hide_index=True)
else:
    st.info("No active electrophysiology data yet. Spike feature extraction requires efel.")

# --- Morphology (placeholder until SWC data available) ---
st.header("Morphology")

morph_cols = [c for c in metrics.columns if c.startswith("morph_")]
morph_with_data = [c for c in morph_cols if metrics[c].notna().any() and not (metrics[c].dropna() == 0).all()]

if morph_with_data:
    display_morph = metrics[["cell_index", "cell_type"] + morph_with_data].copy()
    display_morph.columns = ["Cell", "Type"] + [_nice_metric_name(c) for c in morph_with_data]
    st.dataframe(display_morph, use_container_width=True, hide_index=True)
else:
    st.info("No morphology data yet. SWC tracing files (confocal-raw) not yet mounted.")

# --- Download ---
st.header("Download")
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download metrics.csv",
        metrics.to_csv(index=False).encode(),
        file_name="patching_metrics.csv",
        mime="text/csv",
    )
with col_b:
    if mannwhitney is not None:
        st.download_button(
            "Download Mann-Whitney results",
            mannwhitney.to_csv(index=False).encode(),
            file_name="patching_mannwhitney.csv",
            mime="text/csv",
        )
