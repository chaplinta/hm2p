"""Population Decoder — PVA HD decoding from population activity.

Decodes head direction from the activity of a population of HD cells
using the Population Vector Average (PVA) method.

References
----------
Georgopoulos et al. 1986. "Neuronal population coding of movement direction."
    Science. doi:10.1126/science.3749885
Peyrache et al. 2015. "Internally organized mechanisms of the head direction
    sense." Nature Neuroscience. doi:10.1038/nn.3968

Requires real sync.h5 data from S3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

st.title("Population Decoder")
st.caption(
    "Population Vector Average (PVA) head direction decoding from population activity. "
    "Cross-validated decoding accuracy from real HD cell populations."
)

import plotly.express as px
import plotly.graph_objects as go

from hm2p.analysis.decoder import (
    build_decoder,
    cross_validated_decode,
    decode_error,
    decode_hd,
)


# --- Data loading ---

def _try_load_real():
    """Attempt to load real sync.h5 data."""
    try:
        from frontend.data import load_all_sync_data, session_filter_sidebar
        all_data = load_all_sync_data()
        if all_data["n_sessions"] > 0:
            sessions = session_filter_sidebar(all_data["sessions"])
            return sessions, True
    except Exception:
        pass
    return None, False


real_sessions, has_real = _try_load_real()

if not has_real or not real_sessions:
    st.warning(
        "No data available yet. This page will populate when the relevant "
        "pipeline stage completes."
    )
    st.stop()

st.success(
    f"Loaded {len(real_sessions)} sessions, "
    f"{sum(s['n_rois'] for s in real_sessions)} total cells"
)

# Session selection
session_labels = [s["exp_id"] for s in real_sessions]
sel_session_idx = st.selectbox("Session", range(len(session_labels)),
                                format_func=lambda i: session_labels[i],
                                key="dec_session")
ses_data = real_sessions[sel_session_idx]

signals = ses_data["dff"]
hd = ses_data["hd_deg"]
mask = ses_data["active"] & ~ses_data["bad_behav"]
n_cells = ses_data["n_rois"]
n_frames = signals.shape[1]

tab_decode, tab_cv = st.tabs(["Decode", "Cross-Validation"])

# --- Single decode ---
with tab_decode:
    st.subheader("Frame-by-Frame Decoding")

    dec = build_decoder(signals, hd, mask)
    decoded, confidence = decode_hd(signals, dec)
    errs = decode_error(decoded, hd % 360.0)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean abs error", f"{errs['mean_abs_error']:.1f} deg")
    col2.metric("Median abs error", f"{errs['median_abs_error']:.1f} deg")
    col3.metric("Circular std", f"{errs['circular_std_error']:.1f} deg")
    col4.metric("Cells used", n_cells)

    # Decoded vs actual (time series)
    n_show = min(500, n_frames)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=hd[:n_show] % 360, mode="lines",
        line=dict(color="gray", width=1), name="Actual HD",
    ))
    fig.add_trace(go.Scatter(
        y=decoded[:n_show], mode="markers",
        marker=dict(size=2, color="royalblue", opacity=0.5), name="Decoded HD",
    ))
    fig.update_layout(
        height=300, title=f"Decoded vs Actual HD (first {n_show} frames)",
        xaxis_title="Frame", yaxis_title="HD (deg)",
        yaxis=dict(range=[0, 360]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Error distribution
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig = go.Figure(data=[go.Histogram(
            x=errs["errors_deg"], nbinsx=36,
            marker_color="royalblue",
        )])
        fig.add_vline(x=0, line_color="red", line_dash="dash")
        fig.update_layout(
            height=300, title="Error Distribution",
            xaxis_title="Error (deg)", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_e2:
        # Decoded vs actual scatter
        subsample = np.linspace(0, n_frames - 1, min(1000, n_frames), dtype=int)
        fig = go.Figure(data=[go.Scattergl(
            x=hd[subsample] % 360, y=decoded[subsample],
            mode="markers", marker=dict(size=2, opacity=0.3, color="royalblue"),
        )])
        fig.add_trace(go.Scatter(
            x=[0, 360], y=[0, 360], mode="lines",
            line=dict(color="red", dash="dash"), name="Perfect",
        ))
        fig.update_layout(
            height=300, title="Decoded vs Actual",
            xaxis_title="Actual HD (deg)", yaxis_title="Decoded HD (deg)",
            xaxis=dict(range=[0, 360]), yaxis=dict(range=[0, 360]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Decode confidence time series
    with st.expander("Decode Confidence"):
        n_conf = min(500, n_frames)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=confidence[:n_conf], mode="lines",
            line=dict(color="darkorange", width=1), name="Confidence",
        ))
        fig.update_layout(
            height=300,
            title=f"PVA Confidence (resultant vector length) -- first {n_conf} frames",
            xaxis_title="Frame", yaxis_title="Confidence (0-1)",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Mean confidence", f"{np.mean(confidence):.3f}")
        st.metric("Median confidence", f"{np.median(confidence):.3f}")


# --- Cross-validation ---
with tab_cv:
    st.subheader("Cross-Validated Decoding")

    n_folds = st.select_slider("Number of folds", [3, 5, 10], value=5, key="cv_folds")

    with st.spinner("Running cross-validation..."):
        cv_result = cross_validated_decode(
            signals, hd, mask, n_folds=n_folds,
            rng=np.random.default_rng(42),
        )

    cv_errs = cv_result["errors"]
    col1, col2, col3 = st.columns(3)
    col1.metric("CV mean abs error", f"{cv_errs['mean_abs_error']:.1f} deg")
    col2.metric("CV median abs error", f"{cv_errs['median_abs_error']:.1f} deg")
    col3.metric("Folds", n_folds)

    # Comparison: train vs CV
    st.markdown(
        f"**Train error:** {errs['mean_abs_error']:.1f} deg --- "
        f"**CV error:** {cv_errs['mean_abs_error']:.1f} deg --- "
        f"**Overfit gap:** {cv_errs['mean_abs_error'] - errs['mean_abs_error']:.1f} deg"
    )

    # CV error histogram
    fig = go.Figure(data=[go.Histogram(
        x=cv_errs["errors_deg"], nbinsx=36, marker_color="orange",
    )])
    fig.add_vline(x=0, line_color="red", line_dash="dash")
    fig.update_layout(
        height=300, title="CV Error Distribution",
        xaxis_title="Error (deg)", yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)

    # CV confidence
    cv_conf = cv_result["confidence"]
    st.markdown(
        f"**CV mean confidence:** {np.mean(cv_conf):.3f} --- "
        f"**CV median confidence:** {np.median(cv_conf):.3f}"
    )


# --- Methods & References ---
with st.expander("Methods & References"):
    st.markdown("""
**Population Vector Average (PVA) Decoder**

The PVA decoder computes the decoded head direction as the circular mean of
each cell's preferred direction, weighted by its current activity and tuning
strength (mean vector length):

```
decoded_angle = atan2(sum(w_i * sin(PD_i)), sum(w_i * cos(PD_i)))
```

where `w_i = z_scored_activity_i * MVL_i`.

This approach is model-free (no distributional assumptions on neural activity),
works directly with continuous dF/F signals, and naturally handles the circular
nature of head direction.

**References:**

- Georgopoulos, A. P., Schwartz, A. B. & Kettner, R. E. 1986. "Neuronal
  population coding of movement direction." *Science*.
  [doi:10.1126/science.3749885](https://doi.org/10.1126/science.3749885)

- Peyrache, A., Lacber, M. M. & Bhatt, D. 2015. "Internally organized
  mechanisms of the head direction sense." *Nature Neuroscience*.
  [doi:10.1038/nn.3968](https://doi.org/10.1038/nn.3968)

- Ajabi, Z. et al. 2023. "Population dynamics of head-direction neurons
  during drift and reorientation." *Nature*. [doi:10.1038/s41586-023-06086-7](https://doi.org/10.1038/s41586-023-06086-7)
""")


# --- Footer ---
st.markdown("---")
st.caption(
    "Population Vector Average (PVA) decoder: weights each cell's preferred "
    "direction by its current activity and mean vector length. Model-free, "
    "works with continuous dF/F, naturally circular. Cross-validation uses "
    "k-fold with shuffled frame assignment. "
    "Georgopoulos et al. (1986); Peyrache et al. (2015)."
)
