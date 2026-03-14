"""QC Report — automated quality control report for any session."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.qc")

st.title("QC Report")
st.caption("Automated quality control report — one-page summary of session data quality.")

# --- Session selector ---
experiments = load_experiments()
animals = load_animals()
animal_map = {a["animal_id"]: a for a in animals}

exp_ids = [e["exp_id"] for e in experiments]
selected = st.selectbox(
    "Session",
    exp_ids,
    format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
    key="qc_session",
)

if not selected:
    st.stop()

sub, ses = parse_session_id(selected)
animal_id = selected.split("_")[-1]
animal = animal_map.get(animal_id, {})
celltype = animal.get("celltype", "?")
exp = next(e for e in experiments if e["exp_id"] == selected)


@st.cache_data(ttl=300)
def load_qc_data(sub: str, ses: str) -> dict | None:
    """Load ca.h5 for QC analysis."""
    import h5py

    data = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
    if data is None:
        return None

    f = h5py.File(io.BytesIO(data), "r")
    result = {
        "dff": f["dff"][:],
        "fps": float(f.attrs.get("fps_imaging", 9.8)),
    }
    if "event_masks" in f:
        result["event_masks"] = f["event_masks"][:]
    if "spks" in f:
        result["spks"] = f["spks"][:]
    if "noise_probs" in f:
        result["noise_probs"] = f["noise_probs"][:]
    f.close()
    return result


with st.spinner("Loading session data..."):
    data = load_qc_data(sub, ses)

if data is None:
    st.warning("No calcium data found.")
    st.stop()

dff = data["dff"]
n_rois, n_frames = dff.shape
fps = data["fps"]
duration_s = n_frames / fps

# --- Compute QC metrics ---
snrs = []
skewnesses = []
mean_dffs = []
max_dffs = []
std_dffs = []
event_counts = []
event_rates = []
active_fracs = []
noise_scores = []

for i in range(n_rois):
    trace = dff[i]
    baseline_std = np.std(trace[trace < np.percentile(trace, 50)])
    peak = np.percentile(trace, 95)
    snr = peak / baseline_std if baseline_std > 0 else 0
    snrs.append(snr)
    mean_dffs.append(float(np.nanmean(trace)))
    max_dffs.append(float(np.nanmax(trace)))
    std_dffs.append(float(np.nanstd(trace)))

    if trace.std() > 0:
        skewnesses.append(float(((trace - trace.mean()) ** 3).mean() / trace.std() ** 3))
    else:
        skewnesses.append(0.0)

    if "event_masks" in data:
        em = data["event_masks"][i].astype(bool)
        onsets = np.flatnonzero(em[1:] & ~em[:-1])
        n_events = len(onsets) + (1 if em[0] else 0)
        event_counts.append(n_events)
        event_rates.append(n_events / (duration_s / 60))
        active_fracs.append(float(em.mean()))
    else:
        event_counts.append(0)
        event_rates.append(0.0)
        active_fracs.append(0.0)

    if "noise_probs" in data:
        val = data["noise_probs"][i]
        arr = np.asarray(val)
        noise_scores.append(float(arr.mean()) if arr.size > 1 else float(arr.item()))

snrs = np.array(snrs)
skewnesses = np.array(skewnesses)

# --- QC Checks ---
checks = []

# 1. SNR check
median_snr = np.median(snrs)
n_good_snr = (snrs >= 3).sum()
if median_snr >= 5:
    checks.append(("SNR", "PASS", f"Median SNR = {median_snr:.1f} (excellent)"))
elif median_snr >= 3:
    checks.append(("SNR", "PASS", f"Median SNR = {median_snr:.1f} (good)"))
elif median_snr >= 2:
    checks.append(("SNR", "WARN", f"Median SNR = {median_snr:.1f} (marginal)"))
else:
    checks.append(("SNR", "FAIL", f"Median SNR = {median_snr:.1f} (poor)"))

# 2. Duration check
if duration_s >= 600:
    checks.append(("Duration", "PASS", f"{duration_s:.0f}s ({duration_s/60:.1f} min)"))
elif duration_s >= 300:
    checks.append(("Duration", "WARN", f"{duration_s:.0f}s — short session"))
else:
    checks.append(("Duration", "FAIL", f"{duration_s:.0f}s — very short"))

# 3. ROI count
if n_rois >= 10:
    checks.append(("ROI count", "PASS", f"{n_rois} ROIs detected"))
elif n_rois >= 5:
    checks.append(("ROI count", "WARN", f"Only {n_rois} ROIs — limited statistical power"))
else:
    checks.append(("ROI count", "FAIL", f"Only {n_rois} ROIs"))

# 4. Skewness — good calcium traces should be right-skewed
median_skew = np.median(skewnesses)
if median_skew >= 1.0:
    checks.append(("Skewness", "PASS", f"Median skew = {median_skew:.2f} (good transients)"))
elif median_skew >= 0.5:
    checks.append(("Skewness", "WARN", f"Median skew = {median_skew:.2f} (weak transients)"))
else:
    checks.append(("Skewness", "FAIL", f"Median skew = {median_skew:.2f} (poor transients)"))

# 5. Event detection
if event_rates:
    median_rate = np.median(event_rates)
    if median_rate >= 1.0:
        checks.append(("Event rate", "PASS", f"Median = {median_rate:.1f} events/min"))
    elif median_rate > 0:
        checks.append(("Event rate", "WARN", f"Median = {median_rate:.1f} events/min (low)"))
    else:
        checks.append(("Event rate", "FAIL", "No events detected"))

# 6. Exclude flag
if exp.get("exclude") == "1":
    checks.append(("Exclude flag", "FAIL", "Session marked as excluded in experiments.csv"))
else:
    checks.append(("Exclude flag", "PASS", "Not excluded"))

# 7. Bad behaviour
bad_behav = exp.get("bad_behav_times", "")
if bad_behav and bad_behav != "0" and bad_behav.strip():
    checks.append(("Bad behaviour", "WARN", f"Bad behav times: {bad_behav}"))
else:
    checks.append(("Bad behaviour", "PASS", "No bad behaviour periods"))

# --- Display report ---
st.markdown("---")

# Header card
st.subheader(f"Session: {selected}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cell type", celltype)
col2.metric("ROIs", n_rois)
col3.metric("Duration", f"{duration_s/60:.1f} min")
col4.metric("FPS", f"{fps:.1f}")

# Overall grade
n_pass = sum(1 for _, s, _ in checks if s == "PASS")
n_warn = sum(1 for _, s, _ in checks if s == "WARN")
n_fail = sum(1 for _, s, _ in checks if s == "FAIL")

if n_fail == 0 and n_warn == 0:
    grade = "A"
    grade_color = "green"
elif n_fail == 0 and n_warn <= 2:
    grade = "B"
    grade_color = "blue"
elif n_fail <= 1:
    grade = "C"
    grade_color = "orange"
else:
    grade = "D"
    grade_color = "red"

st.markdown(f"### Overall Grade: :{grade_color}[{grade}]")
st.markdown(f"{n_pass} pass, {n_warn} warnings, {n_fail} failures")

# QC checks table
st.subheader("Quality Checks")
for name, status, msg in checks:
    if status == "PASS":
        st.markdown(f":green[**{name}**]: {msg}")
    elif status == "WARN":
        st.markdown(f":orange[**{name}**]: {msg}")
    else:
        st.markdown(f":red[**{name}**]: {msg}")

# --- Visualizations ---
st.markdown("---")
st.subheader("Quality Distributions")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=["SNR", "Skewness", "Event Rate (events/min)", "Max dF/F0"],
)

fig.add_trace(
    go.Histogram(x=snrs, nbinsx=20, marker_color="steelblue", name="SNR"),
    row=1, col=1,
)
fig.add_vline(x=3, line_dash="dash", line_color="red", row=1, col=1)

fig.add_trace(
    go.Histogram(x=skewnesses, nbinsx=20, marker_color="steelblue", name="Skewness"),
    row=1, col=2,
)

if event_rates:
    fig.add_trace(
        go.Histogram(x=event_rates, nbinsx=20, marker_color="steelblue", name="Event Rate"),
        row=2, col=1,
    )

fig.add_trace(
    go.Histogram(x=max_dffs, nbinsx=20, marker_color="steelblue", name="Max dF/F0"),
    row=2, col=2,
)

fig.update_layout(height=500, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Per-ROI quality scatter
st.subheader("Per-ROI Quality")
fig = px.scatter(
    x=snrs,
    y=skewnesses,
    color=event_rates if event_rates else None,
    labels={"x": "SNR", "y": "Skewness", "color": "Event rate"},
    title="SNR vs Skewness (colored by event rate)",
    opacity=0.6,
    color_continuous_scale="Viridis",
)
fig.add_vline(x=3, line_dash="dash", line_color="red", opacity=0.3)
fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.3)
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Good vs bad ROIs summary
st.subheader("ROI Classification")
snr_thresh = 3.0
skew_thresh = 0.5
good_mask = (snrs >= snr_thresh) & (np.array(skewnesses) >= skew_thresh)
n_good = good_mask.sum()

col1, col2, col3 = st.columns(3)
col1.metric("Good ROIs (SNR>=3, skew>=0.5)", f"{n_good}/{n_rois}")
col2.metric("SNR>=3 only", f"{(snrs >= 3).sum()}/{n_rois}")
col3.metric("Fraction good", f"{n_good/n_rois*100:.0f}%")

# Metadata
st.markdown("---")
st.subheader("Session Metadata")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Experiment:**")
    for key in ["lens", "orientation", "fibre", "primary_exp", "bad_2p_frames", "bad_behav_times", "exclude"]:
        st.text(f"  {key}: {exp.get(key, '')}")
with col2:
    st.markdown("**Animal:**")
    for key in ["celltype", "strain", "gcamp", "virus_id", "hemisphere", "sex"]:
        st.text(f"  {key}: {animal.get(key, '')}")
