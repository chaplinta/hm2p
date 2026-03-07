# hm2p Frontend Design

## Overview

A web-based dashboard for monitoring pipeline jobs, browsing raw and processed data,
and performing scientific analysis — all organized around the experiment/session structure
defined in `experiments.csv` and `animals.csv`.

---

## Requirements

### R1 — Pipeline Monitoring (real-time)

View logs and job status for running and completed pipeline stages.

- Which sessions are queued / running / done / failed, per stage
- Live-updating progress (polls S3 `_progress.json` or CloudWatch)
- Log streaming for active EC2 instances (CloudWatch or SSH tail)
- Cost tracking: instance uptime, estimated spend
- Ability to trigger reruns or terminate instances

**Data sources:** S3 progress JSON, CloudWatch Logs, EC2 describe-instances, `experiments.csv`

### R2 — Raw Data Browser

View raw experimental data for any session, without downloading to local machine.

- **TIFF stacks**: Mean projection, max projection, frame slider (don't need every frame —
  load a subsampled set or summary images). Show metadata from `.meta.txt`.
- **Overhead video**: Embedded video player (stream from S3 or load first/last N frames).
  Show video metadata (fps, resolution, duration).
- **TDMS / DAQ**: Timing pulse visualization — camera triggers, 2P frame triggers,
  light on/off transitions. Show sync quality (jitter, dropped frames).
- **Session metadata**: Everything from `experiments.csv` for this session — orientation,
  lens, fibre, bad frames, bad behav times, notes, exclude flag.

**Data sources:** `s3://hm2p-rawdata/rawdata/{sub}/{ses}/`, `metadata/experiments.csv`

### R3 — Session Navigation via Metadata

Use `experiments.csv` and `animals.csv` as the primary navigation structure.

- **Session table**: Sortable/filterable table of all 26 sessions. Columns: animal, date,
  celltype (penk/nonpenk), lens, orientation, primary_exp, exclude, pipeline status per stage.
- **Animal table**: Group sessions by animal. Show celltype, implant date, number of sessions,
  total cells detected.
- **Filters**: By animal, celltype, date range, exclude status, pipeline completion.
- **Click-through**: Click a session row → session detail page with all raw + processed data.
- **Color coding**: Green = complete, yellow = in progress, red = failed, grey = excluded.

**Data sources:** `metadata/experiments.csv`, `metadata/animals.csv`, S3 listing

### R4 — Processed Data Viewer

View outputs from each pipeline stage for any session.

- **Suite2p (Stage 1)**:
  - Mean image + ROI contours (cell vs non-cell)
  - Cell map colored by index or by metric (SNR, probability, size)
  - Classification probability histogram
  - dF/F traces for selected cells (click ROI → show trace)
  - Registration quality: correlation with reference, x/y shifts over time
  - Summary stats: N ROIs, N cells, mean SNR

- **DLC / Pose (Stage 2)**:
  - Overlay keypoints on video frames (ear-left, ear-right, back-upper, back-middle, back-tail)
  - Likelihood heatmap per bodypart over time
  - Trajectory plot (x/y position over session)
  - Flag low-confidence frames

- **Kinematics (Stage 3)**:
  - Head direction over time (circular plot + time series)
  - Position heatmap (occupancy)
  - Speed trace
  - AHV trace
  - Light on/off shading on all time series
  - Bad behaviour intervals highlighted

- **Calcium (Stage 4)**:
  - dF/F heatmap (all cells x time)
  - Per-cell dF/F trace with neuropil
  - CASCADE spike rates (when available)
  - SNR distribution

- **Sync (Stage 5)**:
  - Aligned neural + behavioural timeseries
  - HD tuning curves per cell (polar plots)
  - Population vector visualization

**Data sources:** `s3://hm2p-derivatives/{stage}/{sub}/{ses}/`

### R5 — Analysis (future)

Scientific analysis tools, built on top of Stages 1–5 outputs.

- **HD tuning**: Polar tuning curves, Rayleigh vector length, preferred direction
- **Light on vs off**: Compare tuning in light-on vs light-off epochs
- **Population decoding**: Bayesian or linear decoder for HD from population activity
- **Cross-session comparison**: Same cells across sessions (if tracked), or population-level
  stats across animals/celltypes
- **Penk+ vs nonPenk**: Side-by-side comparison of all metrics by celltype
- **Export**: Download figures, tables, filtered datasets as CSV/HDF5

**Data sources:** `sync.h5`, `kinematics.h5`, `ca.h5`, `metadata/*.csv`

---

## Architecture

### Technology Choice: TBD

Options under consideration (decision deferred):

| Framework | Pros | Cons |
| --- | --- | --- |
| **Streamlit** | Fastest to build, Python-native, good for data scientists | Re-runs script on widget change, limited layout, no linked brushing |
| **Panel (HoloViz)** | Linked brushing, Jupyter-native, handles large data via Datashader | Steeper learning curve, more boilerplate |
| **Dash (Plotly)** | Good interactive plots, React-like callbacks | More boilerplate than Streamlit |
| **NWB Widgets / Neurosift** | Free NWB viewers with ROI + trace support | Requires NWB export first |

Key factors for decision:
- Linked views (click ROI → see trace) strongly favors Panel
- Speed of initial development favors Streamlit
- Future analysis complexity favors Panel or Dash

### Data Access Pattern

All data lives on S3. Two access strategies:

1. **On-demand from S3**: Load data when a session is selected. Works for small files
   (metadata, iscell.npy, kinematics.h5). Too slow for TIFFs and videos.

2. **Pre-computed summaries**: For heavy data (TIFFs, videos), pre-compute summary images
   and thumbnails during pipeline execution. Store as lightweight PNGs/JSONs on S3.
   The frontend loads these instead of raw data.

Recommended: Hybrid. Metadata + numpy arrays loaded on demand. TIFFs/videos served as
pre-computed summaries or via S3 presigned URLs for streaming.

### Deployment

Options:
- **Local only**: `streamlit run app.py` or `panel serve app.py` on laptop
- **EC2**: Small instance (t3.micro) running the app, accessible via browser
- **Streamlit Cloud / HuggingFace Spaces**: Free hosting for Streamlit/Panel apps
- **GitHub Pages + static export**: For read-only summary dashboards

For a single-user research project, local or a small EC2 instance is simplest.

### Page Structure (draft)

```
/ (Home)
├── Session Table (R3) — filterable list of all sessions
│   └── /session/{sub}/{ses} — single session detail page
│       ├── Raw Data tab (R2) — TIFF summary, video player, TDMS
│       ├── Suite2p tab (R4) — ROIs, traces, classification
│       ├── Pose tab (R4) — keypoints, trajectories, likelihood
│       ├── Kinematics tab (R4) — HD, position, speed, AHV
│       ├── Calcium tab (R4) — dF/F, spikes, SNR
│       └── Sync tab (R4) — aligned data, tuning curves
├── Animals — grouped view by animal/celltype
├── Pipeline — job monitoring dashboard (R1)
│   ├── Active jobs (EC2 instances, progress)
│   ├── Stage completion matrix (sessions x stages)
│   └── Logs viewer
└── Analysis (R5, future)
    ├── HD Tuning
    ├── Population Decoding
    └── Penk+ vs nonPenk comparison
```

---

## Open Questions

1. **Do we need multi-user access?** Or is this just for you? Affects auth and deployment.
2. **How important is video playback?** Streaming from S3 adds complexity. Could start with
   keyframe thumbnails and add full playback later.
3. **Should the frontend trigger pipeline runs?** (R1 mentions "ability to trigger reruns")
   This adds complexity — could start read-only and add triggers later.
4. **What analysis visualizations matter most?** HD tuning curves? Population vectors?
   This determines what to build first in R5.
5. **Offline support?** Should the app work without S3 access (e.g. on a plane)?
   Could cache session data locally.

---

## Implementation Plan (suggested phases)

### Phase 1 — Navigation + Pipeline Monitoring
- Session table from `experiments.csv` + `animals.csv`
- Pipeline status matrix (which stages complete per session, from S3 listing)
- Progress polling for active jobs
- Framework decision made here

### Phase 2 — Processed Data Viewer
- Suite2p visualization (mean image, ROIs, traces, classification)
- Kinematics plots (HD, position, speed)
- Load numpy/HDF5 from S3 on demand

### Phase 3 — Raw Data Browser
- TIFF summary images (pre-computed during Stage 1)
- Video thumbnails / keyframe viewer
- TDMS timing visualization

### Phase 4 — Analysis
- HD tuning curves
- Light on/off comparisons
- Population-level summaries by celltype
