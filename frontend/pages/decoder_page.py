"""Population Decoder — PVA and template matching HD decoding.

Decodes head direction from the activity of a population of HD cells
using Population Vector Average (PVA) or template matching.

References
----------
Georgopoulos et al. 1986. "Neuronal population coding of movement direction."
    Science. doi:10.1126/science.3749885
Peyrache et al. 2015. "Internally organized mechanisms of the head direction
    sense." Nature Neuroscience. doi:10.1038/nn.3968
Wilson & McNaughton 1993. "Dynamics of the hippocampal ensemble code for
    space." Science. doi:10.1126/science.8351520

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
    "Decode head direction from population activity using PVA or template matching. "
    "Both methods work well with small cell counts (3-25 cells)."
)

import plotly.graph_objects as go

from hm2p.analysis.decoder import (
    build_decoder,
    cross_validated_decode,
    decode_error,
    decode_hd,
    pva_decode,
    template_decode,
    template_decode_cv,
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
    except Exception as e:
        st.warning(f"Could not load sync data: {e}")
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

# --- Method selection ---
method = st.radio(
    "Decoding method",
    ["Population Vector Average (PVA)", "Template Matching"],
    index=0,
    horizontal=True,
    help=(
        "**PVA**: Weights each cell's preferred direction by its current activity. "
        "Classic Georgopoulos (1986) approach. Works with 3+ HD cells.\n\n"
        "**Template Matching**: Correlates population activity with mean activity "
        "templates at each HD bin. Non-parametric, works with any N."
    ),
)

use_pva = method == "Population Vector Average (PVA)"


def _fmt_deg(v: float) -> str:
    return f"{v:.1f} deg" if np.isfinite(v) else "N/A"


tab_decode, tab_cv, tab_compare = st.tabs(["Decode", "Cross-Validation", "Compare Methods"])


# --- Single decode ---
with tab_decode:
    st.subheader("Frame-by-Frame Decoding")

    if use_pva:
        # Build decoder to get PDs and MVL, then use pva_decode
        dec = build_decoder(signals, hd, mask)
        decoded, confidence = pva_decode(
            signals, dec["preferred_directions"],
            mask=mask, mvl_weights=dec["mvl"],
        )
        # For error computation, only use valid frames
        valid = mask & np.isfinite(decoded)
        errs = decode_error(decoded[valid], hd[valid] % 360.0)
    else:
        decoded, confidence = template_decode(signals, hd, mask)
        valid = mask & np.isfinite(decoded)
        errs = decode_error(decoded[valid], hd[valid] % 360.0)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean abs error", _fmt_deg(errs["mean_abs_error"]))
    col2.metric("Median abs error", _fmt_deg(errs["median_abs_error"]))
    col3.metric("Circular std", _fmt_deg(errs["circular_std_error"]))
    col4.metric("Cells used", n_cells)

    if not np.isfinite(errs["mean_abs_error"]):
        st.warning(
            "Decoder returned NaN -- likely too few HD-tuned cells or insufficient "
            "HD sampling. Try selecting a different session with more HD cells."
        )

    # Decoded vs actual (time series) — only show valid frames
    valid_idx = np.where(valid)[0]
    n_show = min(500, len(valid_idx))
    show_idx = valid_idx[:n_show]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=hd[show_idx] % 360, mode="lines",
        line=dict(color="gray", width=1), name="Actual HD",
    ))
    fig.add_trace(go.Scatter(
        y=decoded[show_idx], mode="markers",
        marker=dict(size=2, color="royalblue", opacity=0.5), name="Decoded HD",
    ))
    fig.update_layout(
        height=300, title=f"Decoded vs Actual HD (first {n_show} valid frames)",
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
        subsample = np.linspace(0, len(valid_idx) - 1, min(1000, len(valid_idx)), dtype=int)
        sub_idx = valid_idx[subsample]
        fig = go.Figure(data=[go.Scattergl(
            x=hd[sub_idx] % 360, y=decoded[sub_idx],
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
        n_conf = min(500, len(valid_idx))
        conf_idx = valid_idx[:n_conf]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=confidence[conf_idx], mode="lines",
            line=dict(color="darkorange", width=1), name="Confidence",
        ))
        method_label = "PVA resultant vector length" if use_pva else "Template correlation"
        fig.update_layout(
            height=300,
            title=f"Confidence ({method_label}) -- first {n_conf} valid frames",
            xaxis_title="Frame", yaxis_title="Confidence (0-1)",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

        valid_conf = confidence[valid]
        st.metric("Mean confidence", f"{np.mean(valid_conf):.3f}")
        st.metric("Median confidence", f"{np.median(valid_conf):.3f}")


# --- Cross-validation ---
with tab_cv:
    st.subheader("Cross-Validated Decoding")

    n_folds = st.select_slider("Number of folds", [3, 5, 10], value=5, key="cv_folds")

    with st.spinner("Running cross-validation..."):
        if use_pva:
            cv_result = cross_validated_decode(
                signals, hd, mask, n_folds=n_folds,
                rng=np.random.default_rng(42),
            )
        else:
            cv_result = template_decode_cv(
                signals, hd, mask, n_folds=n_folds,
                rng=np.random.default_rng(42),
            )

    cv_errs = cv_result["errors"]
    col1, col2, col3 = st.columns(3)
    col1.metric("CV mean abs error", _fmt_deg(cv_errs["mean_abs_error"]))
    col2.metric("CV median abs error", _fmt_deg(cv_errs["median_abs_error"]))
    col3.metric("Folds", n_folds)

    # Comparison: train vs CV
    train_e = errs["mean_abs_error"]
    cv_e = cv_errs["mean_abs_error"]
    if np.isfinite(train_e) and np.isfinite(cv_e):
        st.markdown(
            f"**Train error:** {train_e:.1f} deg --- "
            f"**CV error:** {cv_e:.1f} deg --- "
            f"**Overfit gap:** {cv_e - train_e:.1f} deg"
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


# --- Method comparison ---
with tab_compare:
    st.subheader("PVA vs Template Matching Comparison")
    st.markdown(
        "Compare both decoding methods on this session. "
        "Uses 5-fold cross-validation for fair comparison."
    )

    with st.spinner("Running both decoders..."):
        cv_pva = cross_validated_decode(
            signals, hd, mask, n_folds=5,
            rng=np.random.default_rng(42),
        )
        cv_tmpl = template_decode_cv(
            signals, hd, mask, n_folds=5,
            rng=np.random.default_rng(42),
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Population Vector Average (PVA)**")
        st.metric("CV MAE", _fmt_deg(cv_pva["errors"]["mean_abs_error"]))
        st.metric("CV Median AE", _fmt_deg(cv_pva["errors"]["median_abs_error"]))
        st.metric("CV Circ Std", _fmt_deg(cv_pva["errors"]["circular_std_error"]))
        st.metric("Mean Confidence", f"{np.mean(cv_pva['confidence']):.3f}")
    with col2:
        st.markdown("**Template Matching**")
        st.metric("CV MAE", _fmt_deg(cv_tmpl["errors"]["mean_abs_error"]))
        st.metric("CV Median AE", _fmt_deg(cv_tmpl["errors"]["median_abs_error"]))
        st.metric("CV Circ Std", _fmt_deg(cv_tmpl["errors"]["circular_std_error"]))
        st.metric("Mean Confidence", f"{np.mean(cv_tmpl['confidence']):.3f}")

    # Overlay error distributions
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=cv_pva["errors"]["abs_errors_deg"], nbinsx=36,
        marker_color="royalblue", opacity=0.5, name="PVA",
    ))
    fig.add_trace(go.Histogram(
        x=cv_tmpl["errors"]["abs_errors_deg"], nbinsx=36,
        marker_color="orange", opacity=0.5, name="Template",
    ))
    fig.update_layout(
        barmode="overlay",
        height=300, title="CV Absolute Error Distribution",
        xaxis_title="Absolute Error (deg)", yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary
    pva_better = cv_pva["errors"]["mean_abs_error"] < cv_tmpl["errors"]["mean_abs_error"]
    better_method = "PVA" if pva_better else "Template Matching"
    diff = abs(cv_pva["errors"]["mean_abs_error"] - cv_tmpl["errors"]["mean_abs_error"])
    st.info(
        f"**{better_method}** performs better on this session by {diff:.1f} deg MAE. "
        f"With {n_cells} cells, both methods should be viable."
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
works directly with continuous dF/F0 signals, and naturally handles the circular
nature of head direction. Works with as few as 3 HD-tuned cells.

**Template Matching Decoder**

Builds a mean population activity template at each HD bin, then decodes each
time point by finding the HD bin whose template has the highest Pearson
correlation with the observed population activity vector.

Non-parametric, no tuning curve fitting required. Works with any number of cells.

**References:**

- Georgopoulos, A. P., Schwartz, A. B. & Kettner, R. E. 1986. "Neuronal
  population coding of movement direction." *Science*.
  [doi:10.1126/science.3749885](https://doi.org/10.1126/science.3749885)

- Peyrache, A., Lacber, M. M. & Bhatt, D. 2015. "Internally organized
  mechanisms of the head direction sense." *Nature Neuroscience*.
  [doi:10.1038/nn.3968](https://doi.org/10.1038/nn.3968)

- Wilson, M. A. & McNaughton, B. L. 1993. "Dynamics of the hippocampal
  ensemble code for space." *Science*.
  [doi:10.1126/science.8351520](https://doi.org/10.1126/science.8351520)

- Ajabi, Z. et al. 2023. "Population dynamics of head-direction neurons
  during drift and reorientation." *Nature*. [doi:10.1038/s41586-023-06086-7](https://doi.org/10.1038/s41586-023-06086-7)
""")


# --- Footer ---
st.markdown("---")
st.caption(
    "Population decoders for small HD cell populations. "
    "PVA: weights preferred directions by activity (Georgopoulos 1986). "
    "Template matching: correlates population vectors with HD-binned templates "
    "(Wilson & McNaughton 1993). Both work with 3-25 cells."
)
