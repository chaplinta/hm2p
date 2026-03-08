"""HD Tuning Explorer — interactive head direction tuning curve analysis.

Dedicated page for visualizing and analysing head direction tuning:
polar tuning curves, MVL distributions, preferred direction maps,
tuning width, and significance testing with circular shuffle.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

log = logging.getLogger("hm2p.frontend.hd_tuning")

st.title("HD Tuning Explorer")
st.caption(
    "Interactive head direction tuning curve analysis — polar plots, MVL, "
    "preferred direction, tuning width, and significance testing."
)

import plotly.express as px
import plotly.graph_objects as go

from hm2p.analysis.tuning import (
    compute_hd_tuning_curve,
    mean_vector_length,
    peak_to_trough_ratio,
    preferred_direction,
    tuning_width_fwhm,
)

# --- Synthetic HD cell generator ---


def _generate_synthetic_hd_cell(
    n_frames: int = 5000,
    preferred_deg: float = 180.0,
    concentration: float = 2.0,
    noise_level: float = 0.3,
    baseline: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic HD cell data (signal, hd_deg, mask).

    Uses a von Mises tuning function with additive noise.
    """
    rng = np.random.default_rng(seed)
    # Random head direction trajectory (smoothed random walk)
    hd_steps = rng.normal(0, 5, n_frames)
    hd_deg = np.cumsum(hd_steps) % 360.0

    # Von Mises tuning: signal peaks at preferred_deg
    theta = np.deg2rad(hd_deg)
    pref_rad = np.deg2rad(preferred_deg)
    signal = baseline + np.exp(concentration * np.cos(theta - pref_rad))
    signal /= np.max(signal)  # Normalise to [0, 1]-ish range
    signal += rng.normal(0, noise_level, n_frames)
    signal = np.clip(signal, 0, None)

    # All frames valid
    mask = np.ones(n_frames, dtype=bool)
    return signal, hd_deg, mask


# --- Tabs ---
tab_single, tab_population, tab_significance, tab_params = st.tabs([
    "Single Cell", "Population", "Significance", "Parameter Explorer",
])

# --- Single Cell Demo ---
with tab_single:
    st.subheader("Single Cell Tuning Curve")

    col_pref, col_conc, col_noise = st.columns(3)
    with col_pref:
        pref_deg = st.slider("Preferred direction (°)", 0, 359, 180, 5, key="hd_pref")
    with col_conc:
        concentration = st.slider("Concentration (κ)", 0.1, 10.0, 2.0, 0.1, key="hd_kappa")
    with col_noise:
        noise = st.slider("Noise level", 0.0, 1.0, 0.3, 0.05, key="hd_noise")

    n_bins = st.select_slider("Number of bins", options=[12, 18, 24, 36, 72], value=36,
                               key="hd_nbins")
    sigma = st.slider("Smoothing σ (°)", 0.0, 30.0, 6.0, 1.0, key="hd_sigma")

    signal, hd_deg, mask = _generate_synthetic_hd_cell(
        preferred_deg=pref_deg, concentration=concentration,
        noise_level=noise, seed=42,
    )
    tc, bin_centers = compute_hd_tuning_curve(
        signal, hd_deg, mask, n_bins=n_bins, smoothing_sigma_deg=sigma,
    )

    mvl = mean_vector_length(tc, bin_centers)
    pd_deg = preferred_direction(tc, bin_centers)
    fwhm = tuning_width_fwhm(tc, bin_centers)
    ptr = peak_to_trough_ratio(tc)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MVL", f"{mvl:.3f}")
    col2.metric("Preferred dir", f"{pd_deg:.1f}°")
    col3.metric("FWHM", f"{fwhm:.1f}°")
    col4.metric("Peak/Trough", f"{ptr:.2f}" if not np.isnan(ptr) else "—")

    # Polar plot
    col_polar, col_linear = st.columns(2)
    with col_polar:
        # Close the polar curve by appending first point
        theta_plot = np.concatenate([np.deg2rad(bin_centers), [np.deg2rad(bin_centers[0])]])
        r_plot = np.concatenate([tc, [tc[0]]])
        r_plot = np.where(np.isnan(r_plot), 0, r_plot)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_plot, theta=np.rad2deg(theta_plot),
            mode="lines", fill="toself",
            fillcolor="rgba(65, 105, 225, 0.3)",
            line=dict(color="royalblue", width=2),
            name="Tuning curve",
        ))
        # Preferred direction arrow
        fig.add_trace(go.Scatterpolar(
            r=[0, np.nanmax(tc) * mvl], theta=[pd_deg, pd_deg],
            mode="lines",
            line=dict(color="red", width=3),
            name=f"MVL vector ({mvl:.3f})",
        ))
        fig.update_layout(
            height=400,
            polar=dict(
                radialaxis=dict(visible=True),
                angularaxis=dict(direction="clockwise", rotation=90),
            ),
            title="Polar Tuning Curve",
            showlegend=True,
            legend=dict(x=0, y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_linear:
        fig = go.Figure()
        tc_plot = np.where(np.isnan(tc), 0, tc)
        fig.add_trace(go.Bar(
            x=bin_centers, y=tc_plot,
            marker_color="royalblue",
            name="Tuning curve",
        ))
        fig.add_vline(x=pd_deg, line_color="red", line_dash="dash",
                      annotation_text=f"PD={pd_deg:.0f}°")
        fig.update_layout(
            height=400,
            title="Linear Tuning Curve",
            xaxis_title="Head Direction (°)",
            yaxis_title="Mean signal",
            xaxis=dict(range=[0, 360]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Raw data scatter
    with st.expander("Raw Data"):
        # Subsample for performance
        n_show = min(2000, len(signal))
        idx = np.linspace(0, len(signal) - 1, n_show, dtype=int)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=hd_deg[idx], y=signal[idx],
            mode="markers", marker=dict(size=2, opacity=0.3, color="gray"),
            name="Raw data",
        ))
        fig.add_trace(go.Scatter(
            x=bin_centers, y=tc_plot,
            mode="lines", line=dict(color="royalblue", width=3),
            name="Tuning curve",
        ))
        fig.update_layout(
            height=300, title="Signal vs Head Direction",
            xaxis_title="HD (°)", yaxis_title="Signal",
            xaxis=dict(range=[0, 360]),
        )
        st.plotly_chart(fig, use_container_width=True)


# --- Population Demo ---
with tab_population:
    st.subheader("Population HD Tuning")
    st.markdown(
        "Simulating a population of HD cells with uniformly distributed "
        "preferred directions and varying tuning sharpness."
    )

    n_cells = st.slider("Number of cells", 5, 50, 20, 5, key="pop_n")
    pop_kappa = st.slider("Population κ (mean)", 0.5, 8.0, 3.0, 0.5, key="pop_kappa")
    pop_noise = st.slider("Population noise", 0.05, 0.8, 0.2, 0.05, key="pop_noise")

    # Generate population
    rng_pop = np.random.default_rng(123)
    prefs = np.linspace(0, 360, n_cells, endpoint=False)
    kappas = np.clip(rng_pop.normal(pop_kappa, 1.0, n_cells), 0.5, 15.0)

    pop_data = []
    all_tcs = []
    for i in range(n_cells):
        sig, hd, msk = _generate_synthetic_hd_cell(
            n_frames=3000, preferred_deg=prefs[i],
            concentration=kappas[i], noise_level=pop_noise, seed=i * 17,
        )
        tc_i, bc_i = compute_hd_tuning_curve(sig, hd, msk, n_bins=36)
        mvl_i = mean_vector_length(tc_i, bc_i)
        pd_i = preferred_direction(tc_i, bc_i)
        fwhm_i = tuning_width_fwhm(tc_i, bc_i)
        pop_data.append({
            "Cell": i + 1, "Pref Dir (°)": f"{pd_i:.0f}",
            "MVL": f"{mvl_i:.3f}", "FWHM (°)": f"{fwhm_i:.0f}",
            "κ (input)": f"{kappas[i]:.1f}",
        })
        all_tcs.append(tc_i)

    # MVL histogram
    mvls = [mean_vector_length(tc_i, bc_i) for tc_i in all_tcs]
    col_hist, col_rose = st.columns(2)
    with col_hist:
        fig = go.Figure(data=[go.Histogram(x=mvls, nbinsx=15, marker_color="royalblue")])
        fig.update_layout(
            height=300, title="MVL Distribution",
            xaxis_title="Mean Vector Length", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_rose:
        # Preferred direction rose plot
        pds = [preferred_direction(tc_i, bc_i) for tc_i in all_tcs]
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=[1] * len(pds), theta=pds,
            marker_color=mvls,
            marker_colorscale="Viridis",
            marker_showscale=True,
            marker_colorbar=dict(title="MVL"),
            width=360 / n_cells * 0.8,
        ))
        fig.update_layout(
            height=300,
            polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
            title="Preferred Directions",
        )
        st.plotly_chart(fig, use_container_width=True)

    # All tuning curves heatmap
    tc_matrix = np.array([np.where(np.isnan(t), 0, t) for t in all_tcs])
    # Normalise each row
    row_max = tc_matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    tc_norm = tc_matrix / row_max

    # Sort by preferred direction
    sort_idx = np.argsort(pds)
    tc_sorted = tc_norm[sort_idx]

    fig = px.imshow(
        tc_sorted,
        x=[f"{b:.0f}" for b in bc_i],
        y=[f"Cell {sort_idx[i]+1}" for i in range(n_cells)],
        labels=dict(x="HD (°)", y="Cell", color="Norm. rate"),
        color_continuous_scale="Hot",
        title="Population Tuning Curves (sorted by preferred direction)",
        aspect="auto",
    )
    fig.update_layout(height=max(300, n_cells * 15 + 100))
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    with st.expander("Cell Details"):
        st.dataframe(pd.DataFrame(pop_data), hide_index=True)


# --- Significance Testing ---
with tab_significance:
    st.subheader("Circular Shuffle Significance Test")
    st.markdown(
        "Tests whether observed HD tuning is significantly greater than "
        "expected by chance, using circular time-shift shuffles "
        "(Skaggs et al., 1993)."
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sig_pref = st.slider("Preferred dir (°)", 0, 359, 90, 10, key="sig_pref")
    with col_s2:
        sig_kappa = st.slider("κ (tuning strength)", 0.0, 5.0, 1.5, 0.1, key="sig_kappa")
    with col_s3:
        n_shuffles = st.select_slider("Shuffles", [100, 500, 1000], value=500, key="sig_n")

    signal_s, hd_s, mask_s = _generate_synthetic_hd_cell(
        preferred_deg=sig_pref, concentration=sig_kappa,
        noise_level=0.3, seed=77,
    )

    with st.spinner("Running shuffle test..."):
        from hm2p.analysis.significance import hd_tuning_significance

        result = hd_tuning_significance(
            signal_s, hd_s, mask_s, n_shuffles=n_shuffles,
            metric="mvl", rng=np.random.default_rng(42),
        )

    p_val = result["p_value"]
    obs_mvl = result["observed"]
    shuf_dist = result["shuffle_distribution"]

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Observed MVL", f"{obs_mvl:.4f}")
    col_p2.metric("p-value", f"{p_val:.4f}")
    col_p3.metric("Significant?", "Yes" if p_val < 0.05 else "No")

    # Shuffle distribution histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=shuf_dist, nbinsx=30,
        marker_color="lightgray", name="Shuffle dist",
    ))
    fig.add_vline(x=obs_mvl, line_color="red", line_width=3,
                  annotation_text=f"Observed={obs_mvl:.4f}")
    percentile_95 = np.percentile(shuf_dist, 95)
    fig.add_vline(x=percentile_95, line_color="orange", line_dash="dash",
                  annotation_text="95th pctile")
    fig.update_layout(
        height=350,
        title=f"Shuffle Distribution (n={n_shuffles}, p={p_val:.4f})",
        xaxis_title="MVL", yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show the tuning curve
    tc_sig = result["tuning_curve"]
    bc_sig = result["bin_centers"]
    theta_s = np.concatenate([np.deg2rad(bc_sig), [np.deg2rad(bc_sig[0])]])
    r_s = np.concatenate([tc_sig, [tc_sig[0]]])
    r_s = np.where(np.isnan(r_s), 0, r_s)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_s, theta=np.rad2deg(theta_s),
        mode="lines", fill="toself",
        fillcolor="rgba(65, 105, 225, 0.3)" if p_val < 0.05 else "rgba(200, 200, 200, 0.3)",
        line=dict(color="royalblue" if p_val < 0.05 else "gray", width=2),
    ))
    fig.update_layout(
        height=350,
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
        title=f"Tuning Curve (p={p_val:.4f})",
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Parameter Explorer ---
with tab_params:
    st.subheader("How Parameters Affect Tuning Metrics")
    st.markdown(
        "Sweep a single parameter while holding others fixed to understand "
        "how concentration (κ), noise, and bin count affect MVL and FWHM."
    )

    sweep_param = st.radio(
        "Sweep parameter", ["κ (concentration)", "Noise level", "Number of bins"],
        horizontal=True, key="sweep_param",
    )

    sweep_results = []
    if sweep_param == "κ (concentration)":
        values = np.arange(0.5, 8.1, 0.5)
        for v in values:
            s, h, m = _generate_synthetic_hd_cell(concentration=v, seed=42)
            t, b = compute_hd_tuning_curve(s, h, m)
            sweep_results.append({
                "Value": v,
                "MVL": mean_vector_length(t, b),
                "FWHM": tuning_width_fwhm(t, b),
            })
        x_label = "κ"
    elif sweep_param == "Noise level":
        values = np.arange(0.05, 1.01, 0.05)
        for v in values:
            s, h, m = _generate_synthetic_hd_cell(noise_level=v, seed=42)
            t, b = compute_hd_tuning_curve(s, h, m)
            sweep_results.append({
                "Value": v,
                "MVL": mean_vector_length(t, b),
                "FWHM": tuning_width_fwhm(t, b),
            })
        x_label = "Noise"
    else:
        values = [6, 12, 18, 24, 36, 48, 72, 90]
        for v in values:
            s, h, m = _generate_synthetic_hd_cell(seed=42)
            t, b = compute_hd_tuning_curve(s, h, m, n_bins=v)
            sweep_results.append({
                "Value": v,
                "MVL": mean_vector_length(t, b),
                "FWHM": tuning_width_fwhm(t, b),
            })
        x_label = "n_bins"

    sweep_df = pd.DataFrame(sweep_results)

    col_sw1, col_sw2 = st.columns(2)
    with col_sw1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_df["Value"], y=sweep_df["MVL"],
            mode="lines+markers", marker_color="royalblue",
        ))
        fig.update_layout(
            height=300, title=f"MVL vs {x_label}",
            xaxis_title=x_label, yaxis_title="MVL",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_sw2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_df["Value"], y=sweep_df["FWHM"],
            mode="lines+markers", marker_color="orange",
        ))
        fig.update_layout(
            height=300, title=f"FWHM vs {x_label}",
            xaxis_title=x_label, yaxis_title="FWHM (°)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sweep_df, hide_index=True)

# --- Footer ---
st.markdown("---")
st.caption(
    "HD tuning analysis uses occupancy-normalised circular histograms with "
    "Nadaraya-Watson Gaussian smoothing. Significance via circular time-shift "
    "shuffle (Skaggs et al., 1993). MVL = mean vector length (Rayleigh, 1919)."
)
