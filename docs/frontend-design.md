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

### R5 — Analysis (implemented)

Scientific analysis tools, built on top of Stages 1–5 outputs and Stage 6 analysis.h5.

- **HD tuning**: Polar tuning curves, Rayleigh vector length, preferred direction
- **Light on vs off**: Compare tuning in light-on vs light-off epochs
- **Population decoding**: Bayesian HD decoder with cross-validation
- **Cross-session comparison**: Population-level stats across animals/celltypes
- **Penk+ vs nonPenk**: Side-by-side comparison of all metrics by celltype
- **Stability/Drift/Gain/Anchoring**: Temporal stability, drift rate, gain modulation
- **AHV/Speed/Info Theory**: Angular head velocity tuning, speed tuning, spatial information
- **Cell classification**: Automated HD cell classification with summary tables
- **Export**: Download figures, tables, filtered datasets as CSV/HDF5

**Data sources:** `analysis.h5`, `sync.h5`, `kinematics.h5`, `ca.h5`, `metadata/*.csv`

---

## Architecture

### Technology Choice: Streamlit

**Streamlit** selected for initial implementation — fastest to build, Python-native.
Can migrate to Panel or Dash later if linked brushing becomes critical.

Run locally from devcontainer:

```bash
streamlit run frontend/app.py
```

Deployment (deferred): Cloudflare Tunnel + Access for private Google OAuth,
or Tailscale for zero-config private access.

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

### Page Structure

The frontend has 43+ pages organised in 5 navigation sections. See `frontend/pages/`
for the full list. Key sections:

- **Overview**: Home, Summary, Sessions, Animals, Pipeline, Batch
- **Pipeline**: Suite2p, Calcium, DLC Pose, Tracking QC, Sync, Z-Drift, Anatomy
- **Explore**: Explorer, Timeline, ROI Gallery, Events, Correlations, Trace Compare
- **Analysis**: HD Tuning, Decoder, Population, Light/Dark, Stability, Drift, Gain,
  Anchoring, Speed, AHV, Info Theory, Classify, Signal Quality, QC Report, Maze, etc.
- **System**: AWS, Costs, Changelog

---

## Open Questions

1. **Do we need multi-user access?** Single-user for now; Google OAuth (streamlit-google-auth)
   is implemented but only whitelists one email. Multi-user deferred.
2. **How important is video playback?** Deferred — keyframe thumbnails and pose trajectory
   overlays are implemented; full streaming not yet needed.
3. **Should the frontend trigger pipeline runs?** Read-only for now. Pipeline runs are
   triggered via CLI scripts. May add trigger capability later.
4. ~~**What analysis visualizations matter most?**~~ Resolved — all analysis pages are
   implemented (HD tuning, decoder, population, stability, etc.).
5. **Offline support?** Not implemented. All data loads from S3. Could add local caching
   in future.

---

## Implementation Status

### Phase 1 — Navigation + Pipeline Monitoring (done)
- Session table from `experiments.csv` + `animals.csv` with filters
- Pipeline status matrix (S3 listing per stage per session)
- Active EC2 instance display
- Progress polling from `_progress.json`
- Failed session error details

### Phase 2 — Suite2p Viewer (done)
- Mean image + ROI contours (cell/non-cell overlay)
- dF/F traces with neuropil and deconvolved options
- Classification probability histogram
- Registration shift plots + reference image
- TIFF summary images from ops.npy (mean, max-proj, enhanced, reference)
- Registered frame viewer (S3 range request on data.bin)
- Raw TIFF file listing from S3

### Phase 3 — DLC / Pose & Calcium Viewer (done)
- DLC monitoring page: live progress, EC2 status, per-session pose trajectory viewer,
  likelihood box plots, QC metrics
- Calcium data viewer: 4-tab interface (Overview heatmap + correlation matrix + per-ROI stats,
  Trace Viewer with event/deconv overlays, Event Detection analysis, Cell Drill-down)

### Phase 4 — Cross-Session & Population Analysis (done)
- Cross-session comparison page: Penk vs non-Penk, Mann-Whitney U tests, per-animal grouping
- Population overview: aggregate 391 ROIs across 26 sessions, SNR/skewness/event rate distributions,
  quality filtering with adjustable thresholds, full table with CSV export
- Batch overview: at-a-glance quality metrics, ROI counts, SNR bars, color-coded status table

### Phase 5 — Analysis & Exploration (done)
- Multi-signal analysis page: 6 tabs (Signal Comparison, Activity, HD Tuning, Place Tuning,
  Robustness, Population Summary). Cross-signal MVL scatter, significance agreement (Jaccard).
- Data Explorer: unified session drill-down with calcium traces, event overlays, light cycle
  overlay, timestamps, pose trajectories, S3 file browser
- Session Timeline: temporal overview with light cycles, speed, population dF/F heatmap,
  event rate, per-ROI trace browser
- ROI Gallery: grid view of all ROIs with mini traces, event overlays, sortable by SNR/event
  rate/max dF/F, quality filtering
- Event Browser: individual calcium transient analysis with waveform gallery, aligned mean
  trace, event statistics (duration, peak, AUC, IEI), population raster
- Correlations & Ensembles: pairwise correlation matrix with hierarchical clustering, PCA
  dimensionality analysis, population co-activation and ensemble detection

### Phase 6 — Kinematics & Tuning (done)
- HD tuning curves with light on/off comparison (from sync.h5 + analysis.h5)
- Full condition-split analysis (movement x light x celltype)
- Place tuning maps
- Robustness analysis across parameter grids
- Population decoder page
- Stability, drift, gain, anchoring, speed, AHV, info theory pages
- Classify page (automated HD cell classification)
- Signal quality and QC report pages
- All analysis features load real data from analysis.h5 on S3
