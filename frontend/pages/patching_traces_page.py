"""Patching Trace Viewer — raw electrophysiology sweeps from WaveSurfer H5 files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure src/ is on the path for hm2p imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import plotly.graph_objects as go

st.title("Patching Trace Viewer")
st.caption("View raw electrophysiology traces from WaveSurfer recordings.")

PATCHING_ROOT = Path("/data/patching")
EPHYS_DIR = PATCHING_ROOT / "ephys"
CELLS_CSV = PATCHING_ROOT / "metadata" / "cells.csv"


@st.cache_data(ttl=600)
def load_cells():
    """Load cells.csv metadata."""
    import pandas as pd

    if not CELLS_CSV.exists():
        return None
    return pd.read_csv(CELLS_CSV, encoding="latin-1")


@st.cache_data(ttl=600)
def build_cell_directory_map() -> dict[str, Path]:
    """Scan all date directories to build ephys_id -> directory mapping."""
    mapping: dict[str, Path] = {}
    if not EPHYS_DIR.exists():
        return mapping
    for date_dir in sorted(EPHYS_DIR.iterdir()):
        if date_dir.is_dir():
            for cell_dir in sorted(date_dir.iterdir()):
                if cell_dir.is_dir():
                    mapping[cell_dir.name] = cell_dir
    return mapping


@st.cache_data(ttl=600)
def load_traces(h5_path: str):
    """Load WaveSurfer H5 file and return sweep data."""
    from hm2p.patching.io import get_sweep_traces, load_wavesurfer

    ws_data = load_wavesurfer(Path(h5_path))

    # Get sampling rate from header
    header = ws_data.get("header", {})
    acq = header.get("Acquisition", {})
    sample_rate = acq.get(
        "SampleRate", header.get("AcquisitionSampleRate", 20000)
    )
    if isinstance(sample_rate, np.ndarray):
        sample_rate = float(sample_rate.flat[0])
    else:
        sample_rate = float(sample_rate)

    # Extract individual sweeps using sweep keys
    sweep_keys = sorted(
        k for k in ws_data if k.startswith("sweep_") or k.startswith("trial_")
    )
    sweeps = []
    for sk in sweep_keys:
        s = ws_data[sk]
        if isinstance(s, dict) and "analogScans" in s:
            arr = np.asarray(s["analogScans"], dtype=np.float64)
            if arr.ndim == 2:
                sweeps.append(arr[:, 0])  # voltage channel
            else:
                sweeps.append(arr)

    return {"sweeps": sweeps, "sample_rate": sample_rate, "n_sweeps": len(sweeps)}


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

cells = load_cells()
if cells is None:
    st.warning(f"No cells.csv found at {CELLS_CSV}.")
    st.stop()

if not EPHYS_DIR.exists():
    st.warning(f"Patching data directory not found at {EPHYS_DIR}.")
    st.stop()

cell_dir_map = build_cell_directory_map()

# Cell selector
cell_labels = []
for _, row in cells.iterrows():
    ctype = row.get("cell_type", "?")
    label = (
        f"Cell {row['cell_index']} — {row['animal_id']} — {ctype} ({row['ephys_id']})"
    )
    cell_labels.append(label)

selected_idx = st.selectbox(
    "Cell",
    range(len(cell_labels)),
    format_func=lambda i: cell_labels[i],
    key="patch_trace_cell",
)
selected_cell = cells.iloc[selected_idx]

# Find cell directory from pre-built map
ephys_id = selected_cell["ephys_id"]
cell_dir = cell_dir_map.get(ephys_id)
if cell_dir is None:
    st.warning(f"No ephys directory found for {ephys_id}.")
    st.stop()

# List H5 files (including subdirectories like IV/)
h5_files = sorted(cell_dir.glob("*.h5"))
for subdir in sorted(cell_dir.iterdir()):
    if subdir.is_dir() and subdir.name != "Images":
        h5_files.extend(sorted(subdir.glob("*.h5")))

if not h5_files:
    st.warning(f"No H5 files found in {cell_dir}.")
    st.stop()

# Protocol selector — show relative path from cell dir for clarity
protocol_labels = []
for f in h5_files:
    rel = f.relative_to(cell_dir)
    protocol_labels.append(str(rel))

selected_file_idx = st.selectbox(
    "Protocol",
    range(len(protocol_labels)),
    format_func=lambda i: protocol_labels[i],
    key="patch_trace_protocol",
)
selected_file = h5_files[selected_file_idx]

# Cell info
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cell", selected_cell["cell_index"])
col2.metric("Type", selected_cell.get("cell_type", "?"))
col3.metric("Animal", selected_cell["animal_id"])
col4.metric("Area", selected_cell.get("area", "?"))

# Load traces
with st.spinner("Loading traces..."):
    try:
        data = load_traces(str(selected_file))
    except Exception as e:
        st.error(f"Could not load {selected_file.name}: {e}")
        st.stop()

sweeps = data["sweeps"]
fs = data["sample_rate"]

if not sweeps:
    st.warning("No sweeps found in this file.")
    st.stop()

st.success(
    f"Loaded {data['n_sweeps']} sweeps at {fs / 1000:.0f} kHz "
    f"from {protocol_labels[selected_file_idx]}"
)

# ---------------------------------------------------------------------------
# Overlay all sweeps
# ---------------------------------------------------------------------------
st.subheader("All Sweeps (overlay)")

fig = go.Figure()
for i, sweep in enumerate(sweeps):
    time_ms = np.arange(len(sweep)) / fs * 1000
    fig.add_trace(
        go.Scatter(
            x=time_ms,
            y=sweep,
            mode="lines",
            name=f"Sweep {i + 1}",
            opacity=0.6,
            line=dict(width=1),
        )
    )
fig.update_layout(
    height=450,
    xaxis_title="Time (ms)",
    yaxis_title="Voltage (mV)",
    title=f"{protocol_labels[selected_file_idx]} — {data['n_sweeps']} sweeps",
    showlegend=data["n_sweeps"] <= 20,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Individual sweep selector
# ---------------------------------------------------------------------------
st.subheader("Single Sweep")

if data["n_sweeps"] == 1:
    sweep_idx = 0
    st.text("Sweep 1 of 1")
else:
    sweep_idx = st.slider("Sweep", 1, data["n_sweeps"], 1, key="patch_trace_sweep") - 1
sweep = sweeps[sweep_idx]
time_ms = np.arange(len(sweep)) / fs * 1000

fig_single = go.Figure()
fig_single.add_trace(
    go.Scatter(
        x=time_ms,
        y=sweep,
        mode="lines",
        line=dict(color="royalblue", width=1.5),
        name=f"Sweep {sweep_idx + 1}",
    )
)
fig_single.update_layout(
    height=350,
    xaxis_title="Time (ms)",
    yaxis_title="Voltage (mV)",
    title=f"Sweep {sweep_idx + 1} of {data['n_sweeps']}",
)
st.plotly_chart(fig_single, use_container_width=True)

# ---------------------------------------------------------------------------
# Sweep statistics
# ---------------------------------------------------------------------------
with st.expander("Sweep Statistics"):
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Min (mV)", f"{np.min(sweep):.1f}")
    s2.metric("Max (mV)", f"{np.max(sweep):.1f}")
    s3.metric("Mean (mV)", f"{np.mean(sweep):.1f}")
    s4.metric("Duration (ms)", f"{len(sweep) / fs * 1000:.0f}")
