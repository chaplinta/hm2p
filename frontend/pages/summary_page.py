"""Summary Dashboard — key metrics at a glance for a single session.

Combines HD tuning, drift, gain, anchoring, and speed metrics
into a single-page overview with colour-coded quality indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hm2p.analysis.anchoring import anchoring_speed, anchoring_time_course
from hm2p.analysis.classify import classify_population, classification_summary_table
from hm2p.analysis.gain import population_gain_modulation
from hm2p.analysis.speed import speed_modulation_index
from hm2p.analysis.stability import drift_per_epoch
from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length

st.title("Session Summary")
st.caption(
    "At-a-glance overview combining classification, drift, gain, "
    "anchoring, and speed metrics for a synthetic session."
)


def _make_session(n_cells=10, n_frames=12000, kappa=3.0, noise=0.15,
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
    n_hd = n_cells * 2 // 3  # 2/3 are HD-tuned

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


# Sidebar
n_cells = st.sidebar.slider("Total cells", 5, 20, 10, 1, key="sum_n")
kappa = st.sidebar.slider("κ", 0.5, 8.0, 3.0, 0.5, key="sum_kappa")
dark_drift = st.sidebar.slider("Dark drift (°)", 0.0, 60.0, 20.0, 5.0, key="sum_drift")
dark_gain = st.sidebar.slider("Dark gain", 0.3, 1.2, 0.8, 0.05, key="sum_dgain")

signals, hd, speed, mask, light_on = _make_session(
    n_cells=n_cells, kappa=kappa, dark_drift=dark_drift, dark_gain=dark_gain,
)

# Run classification
with st.spinner("Running analysis..."):
    pop = classify_population(
        signals, hd, mask, n_shuffles=200,
        rng=np.random.default_rng(42),
    )

# === Top metrics ===
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Cells", n_cells)
col2.metric("HD Cells", pop["n_hd"])
col3.metric("Non-HD", pop["n_non_hd"])
col4.metric("HD Fraction", f"{pop['fraction_hd']:.0%}")

# Mean MVL of HD cells
hd_mvls = [pop["cells"][i]["mvl"] for i in pop["hd_indices"]]
col5.metric("Mean HD MVL", f"{np.mean(hd_mvls):.3f}" if hd_mvls else "N/A")

st.markdown("---")

# === Classification table ===
st.subheader("Cell Classification")
table = classification_summary_table(pop)
df = pd.DataFrame(table)
df["cell"] = df["cell"].apply(lambda x: f"Cell {x+1}")
df_display = df.copy()
df_display["mvl"] = df["mvl"].apply(lambda x: f"{x:.3f}")
df_display["p_value"] = df["p_value"].apply(lambda x: f"{x:.4f}")
df_display["reliability"] = df["reliability"].apply(lambda x: f"{x:.3f}")
df_display["mi"] = df["mi"].apply(lambda x: f"{x:.4f}")
df_display["preferred_direction"] = df["preferred_direction"].apply(lambda x: f"{x:.1f}°")
st.dataframe(df_display, use_container_width=True, hide_index=True, height=200)

# === Key analysis panels ===
st.subheader("Key Metrics")

col_drift, col_gain, col_speed = st.columns(3)

with col_drift:
    st.markdown("**Drift Analysis**")
    if pop["hd_indices"]:
        # Use first HD cell for drift analysis
        idx0 = pop["hd_indices"][0]
        dr = drift_per_epoch(signals[idx0], hd, mask, light_on)
        if dr["n_epochs"] > 0 and len(dr["cumulative_drift"]) > 1:
            max_drift = float(np.max(np.abs(dr["cumulative_drift"])))
            st.metric("Max drift", f"{max_drift:.1f}°")
            if max_drift > 45:
                st.warning("Significant drift detected")
            else:
                st.success("Stable PD")
        else:
            st.info("Insufficient epochs")
    else:
        st.info("No HD cells")

with col_gain:
    st.markdown("**Gain Modulation**")
    gains = population_gain_modulation(signals, hd, mask, light_on)
    gmi_values = [g["gain_index"] for g in gains]
    mean_gmi = np.mean(gmi_values)
    st.metric("Mean GMI", f"{mean_gmi:.3f}")
    if abs(mean_gmi) > 0.15:
        st.info(f"{'Light > Dark' if mean_gmi > 0 else 'Dark > Light'}")
    else:
        st.success("Similar gain")

with col_speed:
    st.markdown("**Speed Modulation**")
    smis = []
    for i in range(n_cells):
        r = speed_modulation_index(signals[i], speed, mask)
        smis.append(r["speed_modulation_index"])
    mean_smi = np.mean(smis)
    st.metric("Mean SMI", f"{mean_smi:.3f}")
    if abs(mean_smi) > 0.1:
        st.info("Speed-modulated population")
    else:
        st.success("Speed-independent")

# === Population polar plot ===
st.subheader("HD Cell Tuning Overview")
if pop["hd_indices"]:
    fig = go.Figure()
    for i, idx in enumerate(pop["hd_indices"][:8]):  # Limit to 8
        tc, bc = compute_hd_tuning_curve(signals[idx], hd, mask)
        theta_plot = np.concatenate([np.deg2rad(bc), [np.deg2rad(bc[0])]])
        r_plot = np.concatenate([tc / tc.max(), [tc[0] / tc.max()]])  # Normalise
        fig.add_trace(go.Scatterpolar(
            r=r_plot, theta=np.rad2deg(theta_plot),
            mode="lines", name=f"Cell {idx+1}",
            line=dict(width=2),
        ))
    fig.update_layout(
        height=350,
        title="Normalised HD Tuning Curves (HD Cells)",
        polar=dict(radialaxis=dict(showticklabels=False)),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No HD cells detected.")

# Footer
st.markdown("---")
st.caption(
    "Summary combines classification (MVL + shuffle + reliability), "
    "drift tracking, gain modulation, and speed modulation. "
    "Use individual analysis pages for detailed exploration."
)
