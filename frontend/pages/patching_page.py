"""Patching electrophysiology & morphology analysis page.

Displays per-cell passive and active electrophysiology metrics,
morphology summary, and Penk+ vs non-Penk group comparisons with
box plots and Mann-Whitney U tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.data import DERIVATIVES_BUCKET, download_s3_bytes
from hm2p.constants import HEX_NONPENK, HEX_PENK

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "patching" / "analysis"
S3_PREFIX = "patching/analysis"

# Readable names for metrics
METRIC_LABELS: dict[str, str] = {
    # Passive ephys
    "ephys_passive_RMP": "RMP (mV)",
    "ephys_passive_rin": "Input resistance (MΩ)",
    "ephys_passive_tau": "Membrane tau (ms)",
    "ephys_passive_incap": "Input capacitance (pF)",
    "ephys_passive_sag": "Sag ratio",
    "ephys_passive_rhreo": "Rheobase (pA)",
    "ephys_passive_maxsp": "Max spike rate (Hz)",
    # Active ephys
    "ephys_active_minVm": "AP trough (mV)",
    "ephys_active_peakVm": "AP peak (mV)",
    "ephys_active_maxVmSlope": "Max dV/dt (mV/ms)",
    "ephys_active_halfVm": "Half-amplitude Vm (mV)",
    "ephys_active_amplitude": "AP amplitude (mV)",
    "ephys_active_maxAHP": "AHP (mV)",
    "ephys_active_halfWidth": "Half-width (ms)",
    # Morph — apical
    "morph_api_len": "Apical total length (μm)",
    "morph_api_max_plen": "Apical max path length (μm)",
    "morph_api_bpoints": "Apical branch points",
    "morph_api_mpeucl": "Apical path/Euclidean ratio",
    "morph_api_maxbo": "Apical max branch order",
    "morph_api_mblen": "Apical mean branch length (μm)",
    "morph_api_mplen": "Apical mean path length (μm)",
    "morph_api_mbo": "Apical mean branch order",
    "morph_api_width": "Apical width (μm)",
    "morph_api_height": "Apical height (μm)",
    "morph_api_depth": "Apical depth (μm)",
    "morph_api_wh": "Apical width/height",
    "morph_api_wd": "Apical width/depth",
    "morph_api_shlpeakcr": "Apical Sholl peak crossings",
    "morph_api_shlpeakcrdist": "Apical Sholl peak distance (μm)",
    "morph_api_ext_super": "Apical extension superficial (μm)",
    "morph_api_ext_deep": "Apical extension deep (μm)",
    # Morph — basal
    "morph_bas_len": "Basal total length (μm)",
    "morph_bas_max_plen": "Basal max path length (μm)",
    "morph_bas_bpoints": "Basal branch points",
    "morph_bas_mpeucl": "Basal path/Euclidean ratio",
    "morph_bas_maxbo": "Basal max branch order",
    "morph_bas_mblen": "Basal mean branch length (μm)",
    "morph_bas_mplen": "Basal mean path length (μm)",
    "morph_bas_mbo": "Basal mean branch order",
    "morph_bas_width": "Basal width (μm)",
    "morph_bas_height": "Basal height (μm)",
    "morph_bas_depth": "Basal depth (μm)",
    "morph_bas_wh": "Basal width/height",
    "morph_bas_wd": "Basal width/depth",
    "morph_bas_ntrees": "Basal tree count",
    "morph_bas_shlpeakcr": "Basal Sholl peak crossings",
    "morph_bas_shlpeakcrdist": "Basal Sholl peak distance (μm)",
    "morph_bas_ext_super": "Basal extension superficial (μm)",
    "morph_bas_ext_deep": "Basal extension deep (μm)",
}


def _label(col: str) -> str:
    """Return a readable label for a metric column."""
    return METRIC_LABELS.get(col, col)


@st.cache_data(ttl=300)
def _load_csv(name: str) -> pd.DataFrame | None:
    """Load a patching CSV from S3, falling back to local file."""
    data = download_s3_bytes(DERIVATIVES_BUCKET, f"{S3_PREFIX}/{name}")
    if data is not None:
        return pd.read_csv(io.BytesIO(data))
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _comparison_box(
    df: pd.DataFrame,
    col: str,
    mw_row: pd.Series | None = None,
) -> go.Figure:
    """Box + strip plot comparing Penk+ vs non-Penk for a single metric."""
    penk = df.loc[df["cell_type"] == "penkpos", col].dropna()
    nonpenk = df.loc[df["cell_type"] == "penkneg", col].dropna()

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=penk, name="Penk+", marker_color=HEX_PENK,
        boxpoints="all", jitter=0.3, pointpos=-1.5,
        marker=dict(size=5, opacity=0.7),
    ))
    fig.add_trace(go.Box(
        y=nonpenk, name="Penk\u207bCamKII+", marker_color=HEX_NONPENK,
        boxpoints="all", jitter=0.3, pointpos=-1.5,
        marker=dict(size=5, opacity=0.7),
    ))

    label = _label(col)
    title = label
    if mw_row is not None and pd.notna(mw_row.get("p_value")):
        p = mw_row["p_value"]
        p_fdr = mw_row.get("p_fdr", np.nan)
        sig = mw_row.get("significant", False)
        star = " *" if sig else ""
        title = f"{label}<br><sub>p = {_format_p(p)}, q = {_format_p(p_fdr)}{star}</sub>"

    fig.update_layout(
        title=dict(text=title, font_size=13),
        yaxis_title=label,
        showlegend=False,
        height=350,
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


def _render_comparison_grid(
    df: pd.DataFrame,
    cols: list[str],
    mannwhitney: pd.DataFrame | None,
    ncols: int = 3,
) -> None:
    """Render a grid of box plots for the given metric columns."""
    mw_lookup = {}
    if mannwhitney is not None:
        for _, row in mannwhitney.iterrows():
            mw_lookup[row["metric"]] = row

    for i in range(0, len(cols), ncols):
        batch = cols[i : i + ncols]
        grid = st.columns(ncols)
        for j, col in enumerate(batch):
            with grid[j]:
                fig = _comparison_box(df, col, mw_lookup.get(col))
                st.plotly_chart(fig, use_container_width=True, key=f"box_{col}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Patching Electrophysiology & Morphology")

metrics = _load_csv("metrics.csv")
if metrics is None:
    st.warning("No patching results found. Run `python scripts/run_patching.py` first.")
    st.stop()

mannwhitney = _load_csv("mannwhitney.csv")
summary = _load_csv("summary_stats.csv")

# --- Overview ---
n_cells = len(metrics)
n_penk = (metrics["cell_type"] == "penkpos").sum()
n_nonpenk = (metrics["cell_type"] == "penkneg").sum()
n_animals = metrics["animal_id"].nunique()
n_with_morph = metrics[[c for c in metrics.columns if c.startswith("morph_")]].notna().any(axis=1).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total cells", n_cells)
c2.metric("Penk+", n_penk)
c3.metric("Penk\u207bCamKII+", n_nonpenk)
c4.metric("With morphology", n_with_morph)

# --- Significant findings ---
if mannwhitney is not None:
    sig = mannwhitney[mannwhitney["significant"] == True].sort_values("p_fdr")  # noqa: E712
    if len(sig) > 0:
        st.success(f"**{len(sig)} significant difference(s)** (FDR q < 0.05)")
        for _, row in sig.iterrows():
            label = _label(row["metric"])
            st.markdown(
                f"- **{label}**: U = {row['statistic']:.0f}, "
                f"p = {_format_p(row['p_value'])}, "
                f"q = {_format_p(row['p_fdr'])}"
            )

# --- Tabs ---
tab_passive, tab_active, tab_morph_api, tab_morph_bas, tab_data, tab_refs = st.tabs([
    "Passive Ephys", "Active (Spike)", "Morphology: Apical", "Morphology: Basal",
    "Data Tables", "Methods & References",
])

# --- Passive ephys ---
with tab_passive:
    st.header("Passive Electrophysiology")
    passive_cols = [c for c in metrics.columns if c.startswith("ephys_passive_") and metrics[c].notna().any()]
    if passive_cols:
        _render_comparison_grid(metrics, passive_cols, mannwhitney)
    else:
        st.info("No passive electrophysiology data available.")

# --- Active ephys ---
with tab_active:
    st.header("Active (Spike) Properties")
    active_cols = [c for c in metrics.columns if c.startswith("ephys_active_") and metrics[c].notna().any()]
    if active_cols:
        _render_comparison_grid(metrics, active_cols, mannwhitney)
    else:
        st.info("No active electrophysiology data available.")

# --- Morphology: Apical ---
with tab_morph_api:
    st.header("Apical Dendrite Morphology")
    api_cols = [c for c in metrics.columns if c.startswith("morph_api_") and metrics[c].notna().any()]
    if api_cols:
        _render_comparison_grid(metrics, api_cols, mannwhitney)
    else:
        st.info("No apical morphology data available.")

# --- Morphology: Basal ---
with tab_morph_bas:
    st.header("Basal Dendrite Morphology")
    bas_cols = [c for c in metrics.columns if c.startswith("morph_bas_") and metrics[c].notna().any()]
    if bas_cols:
        _render_comparison_grid(metrics, bas_cols, mannwhitney)
    else:
        st.info("No basal morphology data available.")

# --- Data tables ---
with tab_data:
    st.header("Per-cell Metrics")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    if summary is not None:
        st.header("Descriptive Statistics by Group")
        summary_with_data = summary[summary.iloc[:, 1].notna()]
        if len(summary_with_data) > 0:
            display = summary_with_data.copy()
            display["metric"] = display["metric"].apply(_label)
            st.dataframe(display, use_container_width=True, hide_index=True)

    if mannwhitney is not None:
        st.header("Mann-Whitney U Results")
        mw_display = mannwhitney[mannwhitney["p_value"].notna()].copy()
        mw_display["metric"] = mw_display["metric"].apply(_label)
        mw_display = mw_display.sort_values("p_value")
        st.dataframe(mw_display, use_container_width=True, hide_index=True)

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

# --- Methods & References ---
with tab_refs:
    st.header("Methods & References")
    with st.expander("Electrophysiology", expanded=True):
        st.markdown("""
**Passive properties** extracted from sub-threshold current injection sweeps
(hyperpolarising steps for sag/tau/Rin, ramp for rheobase).

**Active properties** extracted from the first spike at rheobase current using
waveform analysis (peak, trough, half-width, AHP, max dV/dt).

**Statistical comparison:** Mann-Whitney U test (two-sided, unpaired) with
Benjamini-Hochberg FDR correction (q < 0.05).
""")
    with st.expander("Morphology"):
        st.markdown("""
Dendritic morphology traced in MATLAB using the **TREES toolbox** and stored
as `morph_data.mat` files. Metrics extracted per compartment (apical, basal).

**Sholl analysis:** Counts dendritic crossings at concentric shells from the soma.

**References:**
- Cuntz et al. 2010. "One rule to grow them all: a general theory of neuronal
  branching and its practical application." *PLoS Comput Biol*.
  doi:10.1371/journal.pcbi.1000877.
  GitHub: https://github.com/cuntzlab/treestoolbox
- Sholl 1953. "Dendritic organization in the neurons of the visual and motor
  cortices of the cat." *J Anat* 87(4):387-406.
""")
