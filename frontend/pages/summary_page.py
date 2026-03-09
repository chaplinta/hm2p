"""Summary Dashboard — key metrics at a glance across ALL sessions.

Shows all cells pooled across sessions by default, with optional
filtering by celltype or animal.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hm2p.analysis.classify import classify_population, classification_summary_table
from hm2p.analysis.gain import population_gain_modulation
from hm2p.analysis.speed import speed_modulation_index
from hm2p.analysis.stability import drift_per_epoch
from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

log = logging.getLogger("hm2p.frontend.summary")

st.title("Population Summary")
st.caption(
    "At-a-glance overview of ALL cells across ALL sessions. "
    "Filter by cell type or animal in the sidebar."
)


# ── Data loading ────────────────────────────────────────────────────────────

def _try_load_real_data():
    """Attempt to load real sync.h5 data from S3."""
    try:
        from frontend.data import load_all_sync_data, session_filter_sidebar
        all_data = load_all_sync_data()
        if all_data["n_sessions"] > 0:
            sessions = session_filter_sidebar(all_data["sessions"])
            return sessions, True
    except Exception as e:
        log.debug("Could not load real data: %s", e)
    return None, False


# ── Try to load real data ──────────────────────────────────────────────────

real_sessions, has_real = _try_load_real_data()

if not (has_real and real_sessions):
    st.warning("No sync data available yet. This page will populate automatically "
               "when Stage 5 (sync) completes.")
    st.stop()

st.success(
    f"Loaded {len(real_sessions)} sessions, "
    f"{sum(s['n_rois'] for s in real_sessions)} total cells"
)

# Pooled data for analysis — run per session, aggregate results
all_cells_info = []

for ses_data in real_sessions:
    signals = ses_data["dff"]
    hd = ses_data["hd_deg"]
    mask = ses_data["active"] & ~ses_data["bad_behav"]
    light_on = ses_data["light_on"]
    speed = ses_data["speed_cm_s"]
    n_rois = ses_data["n_rois"]
    exp_id = ses_data["exp_id"]
    celltype = ses_data["celltype"]

    # Classify per session
    pop = classify_population(
        signals, hd, mask, n_shuffles=200,
        rng=np.random.default_rng(42),
    )
    table = classification_summary_table(pop)
    for row in table:
        row["exp_id"] = exp_id
        row["celltype"] = celltype
        row["animal_id"] = ses_data["animal_id"]
        all_cells_info.append(row)

    # Gain + speed per cell
    gains = population_gain_modulation(signals, hd, mask, light_on)
    for i in range(n_rois):
        smi = speed_modulation_index(signals[i], speed, mask)
        all_cells_info[-n_rois + i]["gain_index"] = gains[i]["gain_index"]
        all_cells_info[-n_rois + i]["smi"] = smi["speed_modulation_index"]

# Build DataFrame
df = pd.DataFrame(all_cells_info)
n_total = len(df)
n_hd = df["is_hd"].sum()


# ── Top metrics ─────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Cells", n_total)
col2.metric("HD Cells", int(n_hd))
col3.metric("Non-HD", int(n_total - n_hd))
col4.metric("HD Fraction", f"{n_hd / n_total:.0%}" if n_total > 0 else "N/A")

hd_mvls = df[df["is_hd"]]["mvl"].values
col5.metric("Mean HD MVL", f"{np.mean(hd_mvls):.3f}" if len(hd_mvls) > 0 else "N/A")

n_sessions = len(real_sessions)
celltypes = df["celltype"].value_counts()
ct_str = " | ".join(f"{ct}: {n}" for ct, n in celltypes.items())
st.caption(f"{n_sessions} sessions | {ct_str}")

st.markdown("---")

# ── Classification table ────────────────────────────────────────────────────

st.subheader("Cell Classification")
df_display = df[["cell", "is_hd", "grade", "mvl", "p_value", "reliability", "mi",
                  "preferred_direction"]].copy()
df_display.insert(0, "session", df["exp_id"])
df_display.insert(1, "celltype", df["celltype"])

df_display["mvl"] = df_display["mvl"].apply(lambda x: f"{x:.3f}")
df_display["p_value"] = df_display["p_value"].apply(lambda x: f"{x:.4f}")
df_display["reliability"] = df_display["reliability"].apply(lambda x: f"{x:.3f}")
df_display["mi"] = df_display["mi"].apply(lambda x: f"{x:.4f}")
df_display["preferred_direction"] = df_display["preferred_direction"].apply(lambda x: f"{x:.1f}°")

st.dataframe(df_display, use_container_width=True, hide_index=True, height=300)

st.markdown(
    "**Grades:** A = strong HD (MVL≥0.4, reliability≥0.8) · "
    "B = moderate HD (MVL≥0.25) · C = weak HD · D = non-HD"
)

# ── Key metrics panels ──────────────────────────────────────────────────────

st.subheader("Population Metrics")
col_mvl, col_gain, col_speed = st.columns(3)

with col_mvl:
    st.markdown("**MVL Distribution**")
    fig = go.Figure()
    for ct in df["celltype"].unique():
        subset = df[df["celltype"] == ct]
        fig.add_trace(go.Histogram(x=subset["mvl"], name=ct, opacity=0.7))
    fig.update_layout(height=250, xaxis_title="MVL", yaxis_title="Count",
                       barmode="overlay", margin=dict(t=10, b=30))
    st.plotly_chart(fig, use_container_width=True, key="mvl_hist")

with col_gain:
    st.markdown("**Gain Modulation**")
    if "gain_index" in df.columns:
        mean_gmi = df["gain_index"].mean()
        st.metric("Mean GMI", f"{mean_gmi:.3f}")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["gain_index"], marker_color="orange"))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=200, xaxis_title="GMI", margin=dict(t=10, b=30))
        st.plotly_chart(fig, use_container_width=True, key="gmi_hist")

with col_speed:
    st.markdown("**Speed Modulation**")
    if "smi" in df.columns:
        mean_smi = df["smi"].mean()
        st.metric("Mean SMI", f"{mean_smi:.3f}")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["smi"], marker_color="green"))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=200, xaxis_title="SMI", margin=dict(t=10, b=30))
        st.plotly_chart(fig, use_container_width=True, key="smi_hist")

# ── Grade breakdown by celltype ─────────────────────────────────────────────

if "celltype" in df.columns:
    st.markdown("---")
    st.subheader("Classification by Cell Type")
    for ct in df["celltype"].unique():
        subset = df[df["celltype"] == ct]
        n_ct = len(subset)
        n_ct_hd = subset["is_hd"].sum()
        grade_counts = subset["grade"].value_counts().to_dict()
        grades_str = " | ".join(f"{g}: {n}" for g, n in sorted(grade_counts.items()))
        st.markdown(
            f"**{ct}** — {n_ct} cells, {n_ct_hd} HD ({n_ct_hd/n_ct:.0%}) — {grades_str}"
        )

# Footer
st.markdown("---")

with st.expander("Methods & References"):
    st.markdown("""
**dF/F baseline:** Rolling Gaussian smooth + min + max filter
(Pachitariu et al. 2017, doi:10.1101/061507).
[Suite2p GitHub](https://github.com/MouseLand/suite2p)

**Event detection:** Percentile-based noise model with CDF thresholding
(Voigts & Harnett 2020, doi:10.1016/j.neuron.2019.10.016).
[GitHub](https://github.com/jvoigts/cell_labeling_bhv)

**HD tuning curves:** Occupancy-normalized spike/calcium rate per angular bin
(Taube et al. 1990, doi:10.1523/JNEUROSCI.10-02-00420.1990).

**Mean vector length (MVL):** Resultant vector length of tuning curve
(Skaggs et al. 1996, doi:10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K).

**Spatial information:** Skaggs information rate in bits/spike
(Skaggs et al. 1993, doi:10.1162/neco.1996.8.6.1345).

**Significance testing:** Circular time-shift shuffle
(Muller et al. 1987, doi:10.1523/JNEUROSCI.07-07-01951.1987).

**Soma/dendrite classification:** Aspect ratio heuristic from Suite2p stat.npy
(aspect_ratio > 2.5 = dendrite). Analysis defaults to **soma only**.
""")

st.caption(
    "Summary combines classification (MVL + shuffle + reliability), "
    "gain modulation index, and speed modulation index across all cells. "
    "Use individual analysis pages for detailed exploration."
)
