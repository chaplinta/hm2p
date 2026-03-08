"""Summary Dashboard — key metrics at a glance across ALL sessions.

Shows all cells pooled across sessions by default, with optional
filtering by celltype or animal. Falls back to synthetic data if
no sync.h5 data is available yet.
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


def _make_synthetic_session(n_cells=10, n_frames=12000, kappa=3.0, noise=0.15,
                            dark_drift=20.0, dark_gain=0.8, speed_gain=0.3,
                            cycle_frames=1800, seed=42):
    """Generate a full synthetic session with all behavioural variables."""
    rng = np.random.default_rng(seed)
    hd = np.cumsum(rng.normal(0, 5, n_frames)) % 360.0
    speed = np.abs(rng.normal(10, 5, n_frames))
    theta = np.deg2rad(hd)
    prefs = np.linspace(0, 360, n_cells, endpoint=False)

    light_on = np.zeros(n_frames, dtype=bool)
    for start in range(0, n_frames, 2 * cycle_frames):
        light_on[start:min(start + cycle_frames, n_frames)] = True

    signals = np.zeros((n_cells, n_frames))
    n_hd = n_cells * 2 // 3

    for i in range(n_cells):
        if i < n_hd:
            k = np.clip(rng.normal(kappa, 0.5), 0.5, 10.0)
            current_pref = prefs[i]
            drift_per_frame = dark_drift / cycle_frames
            cell_dark_gain = np.clip(rng.normal(dark_gain, 0.1), 0.2, 1.5)
            cell_speed_gain = np.clip(rng.normal(speed_gain, 0.1), -0.1, 0.8)

            for j in range(n_frames):
                if not light_on[j]:
                    current_pref += drift_per_frame
                else:
                    current_pref = prefs[i]
                signals[i, j] = 0.1 + np.exp(k * np.cos(theta[j] - np.deg2rad(current_pref)))

            signals[i] /= signals[i].max()
            signals[i][~light_on] *= cell_dark_gain
            signals[i] *= (1 + cell_speed_gain * speed / np.max(speed))
        else:
            signals[i] = np.abs(rng.normal(1, 0.5, n_frames))

        signals[i] += rng.normal(0, noise, n_frames)
        signals[i] = np.clip(signals[i], 0, None)

    mask = np.ones(n_frames, dtype=bool)
    return signals, hd, speed, mask, light_on


# ── Try real data, fall back to synthetic ───────────────────────────────────

real_sessions, has_real = _try_load_real_data()

if has_real and real_sessions:
    st.success(
        f"Loaded {len(real_sessions)} sessions, "
        f"{sum(s['n_rois'] for s in real_sessions)} total cells"
    )
    use_synthetic = False

    # Pooled data for analysis — run per session, aggregate results
    all_cells_info = []
    session_labels = []

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

else:
    use_synthetic = True
    st.info("No sync data available yet — showing synthetic demo. "
            "Real data will load automatically when Stage 5 (sync) completes.")

    # Synthetic fallback
    n_cells = st.sidebar.slider("Total cells", 5, 20, 10, 1, key="sum_n")
    kappa = st.sidebar.slider("κ", 0.5, 8.0, 3.0, 0.5, key="sum_kappa")
    dark_drift = st.sidebar.slider("Dark drift (°)", 0.0, 60.0, 20.0, 5.0, key="sum_drift")
    dark_gain = st.sidebar.slider("Dark gain", 0.3, 1.2, 0.8, 0.05, key="sum_dgain")

    signals, hd, speed, mask, light_on = _make_synthetic_session(
        n_cells=n_cells, kappa=kappa, dark_drift=dark_drift, dark_gain=dark_gain,
    )

    with st.spinner("Running analysis..."):
        pop = classify_population(
            signals, hd, mask, n_shuffles=200,
            rng=np.random.default_rng(42),
        )

    table = classification_summary_table(pop)
    gains = population_gain_modulation(signals, hd, mask, light_on)
    for i, row in enumerate(table):
        smi = speed_modulation_index(signals[i], speed, mask)
        row["gain_index"] = gains[i]["gain_index"]
        row["smi"] = smi["speed_modulation_index"]
        row["exp_id"] = "synthetic"
        row["celltype"] = "demo"
        row["animal_id"] = "demo"

    df = pd.DataFrame(table)
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

if not use_synthetic:
    n_sessions = len(real_sessions)
    celltypes = df["celltype"].value_counts()
    ct_str = " | ".join(f"{ct}: {n}" for ct, n in celltypes.items())
    st.caption(f"{n_sessions} sessions | {ct_str}")

st.markdown("---")

# ── Classification table ────────────────────────────────────────────────────

st.subheader("Cell Classification")
df_display = df[["cell", "is_hd", "grade", "mvl", "p_value", "reliability", "mi",
                  "preferred_direction"]].copy()
if not use_synthetic:
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
    if not use_synthetic and "celltype" in df.columns:
        for ct in df["celltype"].unique():
            subset = df[df["celltype"] == ct]
            fig.add_trace(go.Histogram(x=subset["mvl"], name=ct, opacity=0.7))
    else:
        fig.add_trace(go.Histogram(x=df["mvl"], name="All", marker_color="royalblue"))
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

if not use_synthetic and "celltype" in df.columns:
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
st.caption(
    "Summary combines classification (MVL + shuffle + reliability), "
    "gain modulation index, and speed modulation index across all cells. "
    "Use individual analysis pages for detailed exploration."
)
