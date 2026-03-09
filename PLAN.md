# hm2p Cloud Pipeline — Project Plan

## Overview

Ground-up rewrite of the hm2p neuroscience analysis pipeline. All data lives in the cloud,
all compute runs in the cloud, with full CPU and GPU support. Local processing is also
supported for all CPU-bound stages; GPU stages (pose estimation, 2P extraction) require
cloud or a local machine with a suitable GPU.

The experiment combines two-photon calcium imaging with overhead behavioural video tracking
in freely-moving mice. The primary recording target is **retrosplenial cortex (RSP)** and nearby cortex — a
region with prominent head-direction cells. Two genetically-defined cell populations are
studied: **Penk+ RSP neurons** (Penk-Cre + ADD3 virus — Cre-ON labelling) and **non-Penk
CamKII+ RSP neurons** (virus 344 — Cre-OFF/intersectional: Cre expression in Penk+ cells
blocks expression, so only non-Penk CamKII+ cells are labelled). Both use GCaMP7f.

The overhead room lights follow a **1 min on / 1 min off** cycle. Light off = **total
darkness** = complete visual cue removal. This tests whether RSP HD cells can maintain
directional tuning without visual landmarks. The primary scientific goals are: (1) HD
tuning characterisation in each cell population, (2) population HD decoding, (3) whether
HD cell activity is anchored to visual vs idiothetic (path integration) cues, (4) whether
this differs between Penk+ and CamKII+ populations.

---

## 1. Raw Data Definition

### 1.1 Experimental Sessions

Every session is identified by a canonical **session ID**:

```text
YYYYMMDD_HH_MM_SS_<animal_id>
e.g. 20220804_13_52_02_1117646
```

In NeuroBlueprint folder naming this becomes `ses-{YYYYMMDD}T{HHMMSS}` (e.g. `ses-20220804T135202`).
Multiple sessions per day per animal exist, so the full timestamp is required to avoid collisions.

Ground-truth session registry lives in two flat CSV files:

| File | Contents |
| --- | --- |
| `metadata/animals.csv` | Animal IDs, genotype, sex, surgery details |
| `metadata/experiments.csv` | Session IDs, animal ID, experiment type, `extractor`, `tracker` columns |

### 1.2 Raw Data Types

#### A — Two-Photon Calcium Imaging

| Property | Detail |
| --- | --- |
| Acquisition system | SciScan (resonant scanner) |
| Raw format | `.raw` (SciScan proprietary) → converted to `.tif` stacks |
| Resolution | 512 × 512 px (typical) |
| Frame rate | ~30 Hz |
| Channels | Green (GCaMP functional) + Red (anatomical reference) |
| Planes | **Single plane** — soma and dendrite ROIs co-exist within the same imaging plane and are distinguished post-hoc by shape |
| Sidecar metadata | `.meta.txt` per session — DAQ settings, frame counts, timing |
| Current volume | ~280 GB extraction outputs across 29 sessions |

The pipeline is **extractor-agnostic**. **roiextractors** (CatalystNeuro) provides a unified
`SegmentationExtractor` API across multiple ROI extraction backends, so all downstream
signal processing uses the same interface regardless of which tool ran upstream.

Currently supported extractors:

| Extractor | Notes |
| --- | --- |
| **Suite2p** (default) | Existing classifiers reused; neuropil trace (Fneu) available |
| **CaImAn** | CNMF-based; neuropil modelled internally, no separate Fneu |

The `extractor` field in `experiments.csv` records which tool was used per session.

Raw `.raw` files are converted to TIFF via `raw2tif`, then passed to the chosen extractor.

#### B — Behavioural Video

| Property | Detail |
| --- | --- |
| Camera | Basler acA1300-200um |
| Format | `.mp4` (H.264) |
| Frame rate | ~100 fps, synchronised to imaging via DAQ trigger |
| Content | Overhead view of mouse in maze (rose-maze / open field / linear track) |
| Calibration | Lens-specific `.npz` files (4 mm and 6 mm lenses) |
| Per-session metadata | `meta/meta.txt` — crop region, scale (mm/pixel), maze ROI corners |
| Current volume | ~900 MB raw video across 30 sessions |

#### C — Pose Tracking

The pipeline is **tracker-agnostic**. **movement** (neuroinformatics.dev) provides a unified
`load_poses.from_file(file=filepath, source_software=...)` call that returns the same `xarray.Dataset`
regardless of which tracker was used, so downstream kinematics code never changes when
swapping backends.

Currently supported trackers (all tested in `movement`):

| Tracker | Format | Notes |
| --- | --- | --- |
| **DeepLabCut** | `.h5` or `.csv` | Current default; existing model reused |
| **SLEAP** | `.h5` (analysis) or `.slp` | Drop-in alternative to DLC |
| **LightningPose** | `.csv` | GPU-efficient alternative |
| **Anipose** | `.csv` | Multi-camera 3D triangulation |
| **NWB** | `.nwb` (ndx-pose) | Standardised neuroscience format |

Current default: **DeepLabCut 3.0 (PyTorch) — SuperAnimal TopViewMouse + HRNet-W32 + FasterRCNN detector**.

| Property | Detail |
| --- | --- |
| Body parts | `left_ear`, `right_ear`, `mid_back`, `mouse_center`, `tail_base` |
| Output format | `.h5` per video: `(frame × body_part) → (x, y, likelihood)` |
| Current volume | ~55 GB (model weights + labeled frames + outputs) |

The `tracker` field in `experiments.csv` records which tracker was used per session.
Pose outputs (from any tracker) feed into Stage 3 without modification.

---

## 2. Data Organisation Standard

We adopt the **NeuroBlueprint** folder specification
(<https://neuroblueprint.neuroinformatics.dev>), a BIDS-inspired standard for systems
neuroscience. **DataShuttle** (<https://datashuttle.neuroinformatics.dev>) handles transfer
and validation between local machines and cloud storage.

```text
hm2p/
├── rawdata/
│   └── sub-{animal_id}/
│       └── ses-{YYYYMMDD}/
│           ├── funcimg/          # 2P TIFF stacks + .meta.txt
│           └── behav/            # .mp4 video + meta/ folder
│
├── sourcedata/                   # Original unmodified assets
│   ├── trackers/                 # Tracker models + labeled data (DLC, SLEAP, etc.)
│   └── metadata/                 # animals.csv, experiments.csv
│
└── derivatives/
    ├── ca_extraction/
    │   └── sub-{animal_id}/ses-{date}/
    │       └── <extractor-native files>  # Suite2p folder, CaImAn .hdf5, etc.
    ├── pose/
    │   └── sub-{animal_id}/ses-{date}/
    │       └── <tracker-native file>     # *DLC_resnet50*.h5, *.slp, *.csv, etc.
    ├── movement/
    │   └── sub-{animal_id}/ses-{date}/
    │       └── kinematics.h5     # HD, position, speed + higher-level (camera rate)
    ├── calcium/
    │   └── sub-{animal_id}/ses-{date}/
    │       └── ca.h5             # dF/F0, events, SNR per ROI (imaging rate)
    └── sync/
        └── sub-{animal_id}/ses-{date}/
            └── sync.h5           # neural + behavioural aligned to imaging frames
```

---

## 3. Processing Pipeline

### Stage 0 — Data Ingest, Validation & DAQ Parsing

**Input:** Raw files from acquisition machine (currently on Dropbox)

1. Validate session registry — check all expected raw files exist for each session
2. Upload raw data to cloud storage (`s3://hm2p-rawdata/`) via DataShuttle
3. Convert `.raw` imaging files → `.tif` using `raw2tif`
4. Validate video metadata (crop, scale, ROI) is complete for each session
5. **Parse TDMS DAQ files** → save `timestamps.h5` per session:
   - Camera trigger times → `frame_times_camera` (N,) float64
   - SciScan line clock → `frame_times_imaging` (T,) float64
   - Lighting on/off pulse times
   - This isolates the `nptdms` dependency in one place; all downstream stages read clean HDF5

**Tools:** DataShuttle, nptdms, custom validator
**Compute:** CPU only, minimal — can run locally or in cloud

---

### Stage 1 — 2P Preprocessing & ROI Extraction (pluggable)

**Input:** TIFF stacks per session (`rawdata/.../funcimg/`)

The extractor is swappable — `experiments.csv` records which was used per session via the
`extractor` column. All downstream code reads via the **roiextractors** unified API:

```python
from roiextractors import Suite2pSegmentationExtractor  # swap class to change extractor
seg    = Suite2pSegmentationExtractor(folder_path=...)
F      = seg.get_traces(name="raw")
Fneu   = seg.get_traces(name="neuropil")   # None for CaImAn
iscell = seg.get_accepted_list()
masks  = seg.get_roi_image_masks()
```

Extractor-specific steps:

| Extractor | Steps | GPU |
| --- | --- | --- |
| Suite2p | Bad-frame detection, motion correction, ROI detection, F/Fneu extraction | recommended |
| CaImAn | Motion correction (NoRMCorre), CNMF segmentation, trace extraction | recommended |

**Soma vs dendrite ROI classification:**
There is a single imaging plane. Within this plane both compact soma and elongated dendrite
segment ROIs are detected by Suite2p (this is a cortical slice — dendritic processes of
deeper/shallower neurons pass through the focal plane). Suite2p is run **once** with
parameters broad enough to detect both types. ROIs are then classified post-hoc using
shape statistics from `stat.npy` (aspect ratio, radius, compactness) and the existing
`classifier_soma.npy` / `classifier_dend.npy` files. Each ROI is labelled `soma`, `dend`,
or `artefact` — no separate Suite2p run required.

**Common output:** extractor-native files in `derivatives/ca_extraction/.../`
plus `bad_frames.npy` for bad-frame masking in Stage 4.

**Tools:** Suite2p 1.0+ (default), CaImAn; roiextractors as unified read layer
**Compute:** GPU strongly recommended for motion correction; local GPU acceptable

---

### Stage 2 — Pose Estimation (pluggable tracker)

**Input:** Pre-processed behavioural `.mp4` video per session

> **Note:** All 26 sessions have pre-processed videos (undistorted then cropped
> by the legacy pipeline). The `.mp4` files in `rawdata/.../behav/` are ready
> for direct tracker inference — no runtime preprocessing is needed.
> Utility functions for undistortion and cropping are retained in
> `pose/preprocess.py` for future sessions.

Tracker-specific inference:

| Tracker | Entrypoint | GPU |
| --- | --- | --- |
| DeepLabCut (default) | `deeplabcut.analyze_videos(...)` | required |
| SLEAP | `sleap.load_model()` + `predict()` | required |
| LightningPose | `predict_single_video()` | required |

**Output:** tracker-native pose file in `derivatives/pose/sub-{id}/ses-{date}/`.
Stage 3 discovers the file automatically via glob (`*.h5`, `*.csv`, `*.slp`).

**Tools:** tracker-specific (DLC 3.x default)
**Compute:** GPU required — cloud EC2 g4dn or local GPU machine

---

### Stage 3 — Behavioural Kinematics: movement

**Input:** Raw pose file from Stage 2 (any supported tracker format)

`movement` provides a single unified load call regardless of tracker:

```python
from movement.io import load_poses
ds = load_poses.from_file(file=pose_file, source_software="DeepLabCut")
# swap source_software="SLEAP" / "LightningPose" etc. — output is identical
```

All backends return an `xarray.Dataset` with dimensions `(time × keypoints × space)`
plus a `confidence` data variable. Downstream code never inspects which tracker was used.

**Primary kinematic outputs (required for all sessions):**

1. Load pose dataset via `load_poses.from_file(file=..., source_software=<tracker>)`
2. Apply `orientation` rotation (from `experiments.csv`) to all keypoint coordinates —
   a per-session correction for camera placement variation, ensures HD is referenced
   consistently across sessions. Rotation is in degrees; applied as a 2D rotation matrix.
3. Filter low-confidence detections (confidence < 0.9 → NaN)
4. Interpolate short gaps (≤ 5 frames) and smooth
5. Compute via `movement.kinematics`:
   - **Head direction (HD):** forward vector from `left_ear` → `right_ear`, unwrapped (deg)
   - **Position:** centroid of body keypoints, converted pixel → mm via scale calibration
   - **Speed:** `kinematics.compute_speed()` on position (cm/s)
   - **Angular head velocity (AHV):** `kinematics.compute_velocity()` on HD (deg/s)
   - **Movement state:** binary active/inactive threshold on speed
6. Align light on/off events from DAQ timestamps (`timestamps.h5`):
   - Light follows a **periodic 1 min on / 1 min off** cycle throughout the session
   - Overhead room lights — light off = **total darkness** / full visual cue removal
   - Per-frame `light_on` boolean stored in `kinematics.h5` and `sync.h5`
7. Compute maze-coordinate positions (7 × 5 unit rose-maze grid):
   - Map pixel position → mm → maze units via scale calibration + ROI metadata
   - Clip out-of-bounds positions to nearest maze boundary point using shapely Polygon
   - Maze polygon (7 × 5 units) defined from `get_maze_poly()` in legacy code

**Higher-level behavioural analyses — Stage 3b (planned, not in core scope yet):**

> **Note:** keypoint-MoSeq, VAME, and DLC2Action are **not** in `pyproject.toml` due to
> incompatible numpy/scikit-learn pins. They must be installed in separate environments.
> See [docs/manual-installs.md](docs/manual-installs.md).

The pipeline is designed to support **minimal-labelling behavioural analysis** — discovering
behavioural structure from pose kinematics without manually labelling thousands of frames.
Both primary tools consume the `movement` xarray output (our Stage 3 output format) directly.

| Tool | Approach | Labels needed | Input | Output | Recommendation |
| --- | --- | --- | --- | --- | --- |
| **keypoint-MoSeq** (Datta lab, v0.6.8) | AR-HMM; learns stereotyped motifs + durations | **Zero** | DLC `.h5` / SLEAP `.h5` | Syllable ID per frame, transition matrix | **Gold standard for freely-moving mice** |
| **VAME** (EthoML, v0.12+) | VAE on pose timeseries; clusters latent space | **Zero** | movement xarray (native) | Motif ID per frame + UMAP + NWB export | Good; native movement integration |
| **DLC2Action** | Transformer + TCN; active learning | 10–100 frames | DLC `.h5` | Action class per frame | When target categories are known |
| ~~B-SOiD~~ | ~~UMAP+GMM~~ | — | — | — | **Not recommended** — stale since 2021 |
| ~~MotionMapper~~ | ~~Wavelet + t-SNE~~ | — | — | — | **Not recommended** — MATLAB; stale 2020 |

**Primary recommendation:** keypoint-MoSeq is the current gold standard for freely-moving
mice (Weinreb et al. 2024, *Nature Methods*; v0.6.8 released Feb 2026). It reads DLC `.h5`
directly and uses a JAX-based AR-HMM that models egocentric heading + motion trajectories.

**VAME** (EthoML v0.12+, `pip install vame-py`) is a strong alternative, especially because
v0.7+ natively accepts the `movement` xarray format (VAME issue #111) — zero conversion needed
from Stage 3 outputs:

```python
from movement.io import load_poses
import vame

ds = load_poses.from_file(file=pose_file, source_software="DeepLabCut")
# ds is an xr.Dataset — VAME v0.7+ accepts this natively
vame.create_project(..., pose_estimation_format="movement")
```

keypoint-MoSeq reads DLC `.h5` directly (the same files that feed `movement`):

```python
import keypoint_moseq as kpms
config = kpms.load_config(project_dir)
data = kpms.load_keypoints(dlc_files, format="deeplabcut")
model = kpms.fit_model(data, config)  # ~GPU or CPU
syllables = model.get_syllables()     # (n_frames,) int array per session
```

These analyses produce optional columns appended to `kinematics.h5`:

```text
/syllable_id    (N,) int16   — VAME / keypoint-MoSeq syllable index per camera frame
/syllable_prob  (N, S) float32  — posterior probability over S syllables (optional)
```

- **Ethogram:** classify frames into named behaviours (grooming, running, still, turning)
  using supervised or semi-supervised labelling. Added to `kinematics.h5` as `/ethogram`.
- These are **optional outputs** — the pipeline runs without them; syllable analysis
  is a post-hoc step on top of Stage 3.

**Known behavioural artefact — head-mount constraint:** Mice can get stuck on the HM2P
head-mounted scope fibre and wires, creating artefactual immobility periods. These are
logged in `experiments.csv` as `bad_behav_times` (mm:ss–mm:ss format). The pipeline
applies this mask to exclude these periods from all behavioural analyses. The per-session
mask is stored as `kinematics.h5:/bad_behav` (N,) bool.

**Tools:** `movement`, `shapely`, NumPy, SciPy
**Compute:** CPU only — can run locally or in cloud
**Output:** `derivatives/movement/sub-{id}/ses-{date}/kinematics.h5`

---

### Stage 4 — Calcium Signal Processing

**Input:** roiextractors extractor object (from Stage 1 outputs) + `timestamps.h5` +
`bad_frames.npy`

All steps operate on the unified roiextractors API — extractor-agnostic.

#### 4a — Neuropil Subtraction (two options, set in `config/pipeline.yaml`)

| Method | When to use |
| --- | --- |
| **Fixed coefficient** (default) | `F_corr = F − 0.7 × Fneu` — fast, Suite2p only; CaImAn handles neuropil internally |
| **FISSA** (optional) | Spatial ICA on ROI masks + raw movie — more accurate in densely labelled tissue; extractor-agnostic |

#### 4b — Baseline & dF/F0

1. Baseline F0: sliding window min of Gaussian-smoothed trace (Suite2p method)
2. Compute `dF/F0 = (F_corr − F0) / F0`

#### 4c — Calibrated Spike Inference via CASCADE

**CASCADE** (Rupprecht et al. 2021, *Nature Neuroscience*) is the primary spike inference
method ([manual install](docs/manual-installs.md) — pins `tensorflow==2.3`, Python 3.8 only).
Unlike threshold-based methods it outputs a continuous **spike rate in spikes/s**
(calibrated physical units) using pre-trained deep-learning models matched to the GCaMP
indicator and imaging frame rate. This replaces the OASIS deconvolution and Voigts & Harnett
threshold methods as the primary output.

```python
from cascade2p import checks, cascade
# model selected by indicator (e.g. 'GCaMP6s') + fps (e.g. ~10 Hz)
spike_rates = cascade.predict(model_name, dff_traces)  # shape: (n_rois, n_frames)
```

The Voigts & Harnett threshold method is retained as a **fallback** and for comparison.

#### 4d — Per-ROI Statistics

- SNR: mean event amplitude / std of non-event periods
- `n_spikes`, `spike_rate` (spikes/min, bad frames excluded)
- `mean_dff_amp`: mean peak dF/F0 during events

**Tools:** roiextractors, CASCADE (`cascade2p`, [manual install](docs/manual-installs.md)), FISSA ([manual install](docs/manual-installs.md)), NumPy, SciPy
**Compute:** CPU only (CASCADE inference is fast on CPU) — can run locally or in cloud
**Output:** `derivatives/calcium/sub-{id}/ses-{date}/ca.h5`

---

### Stage 5 — Neural–Behavioural Synchronisation

**Input:** `kinematics.h5` + `ca.h5` + `.meta.txt` (imaging frame timestamps)

1. Extract precise 2P frame timestamps from DAQ `.meta.txt`
2. Resample behavioural kinematics from camera rate (~100 Hz) to imaging rate (~30 Hz)
   via linear interpolation at each imaging frame timestamp
3. Merge into single synchronised DataFrame indexed by frame number

**Tools:** Pandas, SciPy interp1d
**Compute:** CPU only — can run locally or in cloud
**Output:** `derivatives/sync/sub-{id}/ses-{date}/sync.h5`

---

---

## 4. Analysis Framework Design

The HDF5 outputs from Stages 3–5 are designed to be directly loadable into a standard
analysis stack. This section defines the target design — implementation is future scope.

### 4.1 pynapple — Unified Timeseries Interface

**pynapple** (Peyrache lab) is the standard Python interface for joint neural + behavioural
analysis. Every HDF5 output is designed to load directly into pynapple data structures:

```python
import pynapple as nap
import h5py

# Load calcium traces
with h5py.File("ca.h5") as f:
    dff  = nap.TsdFrame(t=f["frame_times_imaging"][:], d=f["dff"][:].T)
    spks = nap.TsdFrame(t=f["frame_times_imaging"][:], d=f["spikes"][:].T)

# Load kinematics
with h5py.File("kinematics.h5") as f:
    hd    = nap.Tsd(t=f["frame_times_camera"][:], d=f["hd"][:])
    speed = nap.Tsd(t=f["frame_times_camera"][:], d=f["speed"][:])

# Time-aware alignment — pynapple handles resampling automatically
active = nap.IntervalSet(start=..., end=...)  # from movement state
dff_active = dff.restrict(active)             # automatic timestamp-based slicing
```

### 4.2 NEMOS — GLM Encoding Models

**NEMOS** (Flatiron Institute) fits Poisson GLMs relating neural activity to behavioural
variables. Pynapple-native. Answers: *"which behavioural variables drive this ROI?"*

```python
import nemos as nmo
model = nmo.glm.PopulationGLM()
model.fit(X=basis.compute_features(hd, speed), y=spks)
```

### 4.3 CEBRA — Population Latent Embeddings

> **Note:** CEBRA is a [manual install](docs/manual-installs.md) — it pins `numpy<2.0`
> on some platforms, conflicting with other pipeline dependencies.

**CEBRA** (Schneider et al. 2023, *Nature*; v0.6.0, Jan 2026; Apache 2.0) learns a
low-dimensional embedding of population dF/F that is consistent with a specified behavioural
variable. Two operating modes:

- **Hypothesis-driven (label-conditioned):** HD, position, or DLC keypoints guide the
  embedding → population structure driven by the behavioural variable (e.g. a ring-shaped
  manifold for HD cells in RSC)
- **Discovery-driven (time-contrastive):** no behavioural labels required → unsupervised
  population embedding

Both modes take numpy arrays `(time, neurons)` for neural data and `(time, features)` for
behaviour. `movement` xarray → numpy is one line (`ds.position.values`).

```python
import cebra
import numpy as np

# dff_matrix: (T, R) float32   — from sync.h5 ["dff"].T
# hd_array:   (T,) float32     — from sync.h5 ["hd"]

model = cebra.CEBRA(model_architecture="offset10-model", max_iterations=10000)
model.fit(dff_matrix, conditional=hd_array)          # or fit(dff_matrix) for time-contrastive
embedding = model.transform(dff_matrix)              # → (T, d) low-dimensional
```

Particularly suited to RSP data: a ring-shaped manifold in CEBRA space is
a hallmark signature of HD population coding.

### 4.4 NWB — Archive Format

**neuroconv** converts roiextractors outputs directly to NWB, making all pipeline outputs
shareable on DANDI and readable by any NWB-compatible tool (pynapple, MNE, etc.):

```python
from neuroconv import NWBConverter
# wrap Suite2pSegmentationExtractor → NWBFile with RoiResponseSeries, TimeSeries
```

### 4.5 HDF5 → Analysis Path Design Principles

All HDF5 schemas are designed so that:

- Arrays are indexed by **time first** (C-contiguous for fast row slicing in pynapple)
- Timestamps are stored as float64 seconds since session start
- No index columns are needed — timestamps are the index
- Schema is self-describing (session_id, fps_*, units stored as HDF5 attributes)

---

## 5. Local vs Cloud Processing

| Stage | Local (CPU-only machine) | Local (GPU machine) | Cloud EC2 |
| --- | --- | --- | --- |
| 0 — Ingest | ✓ | ✓ | ✓ |
| 1 — 2P extraction | slow (CPU fallback) | ✓ | ✓ (recommended) |
| 2 — Pose estimation | ✗ (no GPU) | ✓ | ✓ (recommended) |
| 3 — Kinematics | ✓ | ✓ | ✓ |
| 4 — Calcium processing | ✓ | ✓ | ✓ |
| 5 — Sync | ✓ | ✓ | ✓ |

The Snakemake pipeline detects compute environment via a profile (`local`, `local-gpu`,
`aws-batch`) set in `config/compute.yaml`. Switching between local and cloud requires only
changing this profile — all file paths use the same relative structure, with S3 paths
substituted automatically when running in cloud mode.

---

## 5. Cloud Architecture

### 5.1 Recommended Provider: AWS

- Widest GPU instance selection (A10G, V100, A100 via g4dn/p3/p4)
- Native Snakemake support via AWS Batch
- S3 is the de-facto standard for scientific data; Suite2p and DLC both support it
- Strong cost-optimisation via Spot Instances and S3 Intelligent-Tiering
- Mature IAM for multi-user lab access control

**Alternative: GCP Vertex AI** — good ML tooling but less neuroscience community adoption.

**Alternative: Institutional HPC** (e.g. UCL Myriad/Kathleen with SLURM) — best
cost-efficiency if available; use S3 for storage regardless.

### 5.2 AWS Layout

```text
S3 Buckets
  s3://hm2p-rawdata/        ← raw TIFFs, videos (Infrequent Access tier)
  s3://hm2p-derivatives/    ← ca_extraction, pose, movement, calcium, sync (Standard)

Compute (Spot Instances)
  g4dn.xlarge   (~$0.16/hr spot)  ← DLC inference, Suite2p/CaImAn GPU
  g4dn.2xlarge  (~$0.30/hr spot)  ← DLC training
  c5.4xlarge    (~$0.27/hr spot)  ← kinematics, calcium processing, sync

Orchestration
  Snakemake + AWS Batch     ← managed job queue
  (or local Snakemake → EC2 SSH for simpler start)

Notebooks
  EC2 + JupyterLab          ← interactive analysis
```

### 5.3 Rough Cost Estimates

| Resource | Estimate |
| --- | --- |
| S3 ~600 GB rawdata (Infrequent Access) | ~$7/month |
| S3 ~150 GB derivatives (Standard, growing) | ~$3/month |
| GPU compute, 30 sessions (one-time) | ~$150–350 total |
| CPU compute, 30 sessions (one-time) | ~$30 total |
| Data upload ~600 GB (one-time) | ~$54 |

---

## 6. Technology Stack

### 6.1 Core Pipeline

| Layer | Tool | Notes |
| --- | --- | --- |
| Data standard | NeuroBlueprint | BIDS-inspired folder spec |
| Data transfer | DataShuttle | Upload/validate to S3 |
| DAQ parsing | nptdms | TDMS → timestamps.h5 (Stage 0 only) |
| 2P extraction | Suite2p 1.0+ (default), CaImAn | Pluggable via `extractor` field |
| Extraction API | **roiextractors** (CatalystNeuro) | Unified SegmentationExtractor |
| Neuropil subtraction | Fixed coefficient (default), **FISSA** (spatial ICA) | FISSA: more accurate in dense tissue; [manual install](docs/manual-installs.md) |
| Spike inference | **CASCADE** (Rupprecht et al. 2021) | Calibrated spikes/s; pre-trained GCaMP models; [manual install](docs/manual-installs.md) |
| Pose estimation | DeepLabCut 3.x (default), SLEAP, LightningPose | Pluggable via `tracker` field |
| Kinematics | **movement** (neuroinformatics.dev) | Unified xarray.Dataset regardless of tracker |
| Behavioural syllables | **VAME** v0.12+ (EthoML) | Zero-label unsupervised; accepts movement xarray natively; [manual install](docs/manual-installs.md) |
| Behavioural syllables (alt) | **keypoint-MoSeq** (Datta lab) | AR-HMM; zero-label; Nature Methods 2024; [manual install](docs/manual-installs.md) |
| NWB conversion | **neuroconv** (CatalystNeuro) | roiextractors + movement → NWB archive |
| Analysis interface | **pynapple** (Peyrache lab) | Unified TsdFrame for dF/F + behaviour |
| Encoding models | **NEMOS** (Flatiron Institute) | GLM, pynapple-native, JAX backend |
| Population embeddings | **CEBRA** (Schneider et al. 2023) | Contrastive latent spaces w/ behaviour; [manual install](docs/manual-installs.md) |
| Orchestration | Snakemake 8.x+ | DAG-based, supports local + AWS Batch profiles |
| Storage | AWS S3 | All persistent data |
| GPU compute | AWS EC2 g4dn Spot (or local GPU) | Pose tracking + 2P extraction |
| CPU compute | AWS EC2 c5 Spot (or local) | Kinematics, calcium processing, sync |
| Notebooks | JupyterLab on EC2 or local | Interactive analysis |
| Packaging | uv + conda (GPU envs) | Reproducible environments |
| Containers | Docker | Reproducible compute |

### 6.2 Code Quality & Best Practice Tooling

| Tool | Role |
| --- | --- |
| **ruff** | Fast linting + formatting (replaces black + flake8 + isort) |
| **mypy** | Static type checking — catches type errors at development time |
| **pytest** + **pytest-cov** | Unit testing + coverage reporting (target ≥ 90%) |
| **hypothesis** | Property-based testing — auto-generates adversarial inputs for numerical functions |
| **pandera** | Runtime schema validation for pandas DataFrames and xarray Datasets |
| **pre-commit** | Runs ruff, mypy, nbstripout automatically before every git commit |
| **nbstripout** | Strips notebook outputs before committing — keeps git history clean |
| **mkdocs** + Material theme | Documentation site auto-built from docstrings |
| **DVC** (Data Version Control) | Tracks data and model artifacts alongside git commits |

---

## 7. What Is Reused Unchanged

- Trained DeepLabCut model — copied to `sourcedata/trackers/dlc/`
- Suite2p classifiers (`classifier_soma.npy`, `classifier_dend.npy`) — used for post-hoc ROI labelling
- Camera calibration files (`.npz`)
- Calcium event detection algorithm (Voigts & Harnett) — retained as fallback; CASCADE is primary
- Metadata CSVs (`animals.csv`, `experiments.csv`)

---

## 8. Implementation Status

### Completed

1. ✅ **Project skeleton** — `pyproject.toml`, `src/hm2p/`, `tests/` (1038+ tests, 92%
   coverage), pre-commit, GitHub Actions CI/lint, `uv` venv, `metadata/` CSVs.
2. ✅ **HDF5 schemas** — pandera schema validation in `io/hdf5.py` for all output files.
3. ✅ **S3 data upload** — 26 sessions (91.4 GiB, 503 objects) uploaded to
   `s3://hm2p-rawdata/rawdata/` in NeuroBlueprint layout. Verified with
   `scripts/verify_s3_upload.sh`.
4. ✅ **Stage 0 — Ingest** — TDMS parser → `timestamps.h5`; fully unit-tested.
5. ✅ **Stage 1 — Suite2p extraction** — `extraction/suite2p.py` (extractor class +
   post-hoc soma/dendrite ROI classification), `extraction/run_suite2p.py` (wraps
   `suite2p.run_s2p()`), `extraction/base.py` (abstract interface). CaImAn extractor
   also implemented. All tested with synthetic data.
6. ✅ **Stage 3 — Kinematics** — `kinematics/compute.py` using `movement` library.
   HD, position, speed, AHV, movement state, light on/off, bad_behav masking.
7. ✅ **Stage 4 — Calcium processing** — neuropil subtraction (fixed coefficient),
   dF/F0 baseline, per-ROI stats. CASCADE spike inference deferred (needs conda env).
8. ✅ **Stage 5 — Sync** — `sync/align.py` resamples behaviour → imaging frame times.
9. ✅ **Snakemake DAG** — `workflow/Snakefile` + 6 stage rules (`workflow/rules/*.smk`)
   with resource specs. Three profiles: `local`, `local-gpu`, `aws-batch`.
10. ✅ **Docker images defined** — `docker/gpu.Dockerfile` (CUDA 12.1 + Suite2p + DLC),
    `docker/cpu.Dockerfile` (CPU-only stages).
11. ✅ **AWS S3 infrastructure** — buckets created (`hm2p-rawdata`, `hm2p-derivatives`),
    versioning enabled, lifecycle policy (Standard → IA after 30 days).
12. ✅ **Legacy pipeline reference** — copied into `old-pipeline/` (read-only, never modify).
13. ✅ **EC2 launch script** — `scripts/launch_suite2p_ec2.py` launches a g4dn.xlarge
    instance via boto3, bootstraps Suite2p, processes all 26 sessions, uploads results
    to S3, and self-terminates. Supports `--status`, `--terminate`, `--dry-run`.

### In Progress / Recently Completed

14. ✅ **Suite2p cloud run (Stage 1)** — All 26 sessions processed on EC2 g4dn.xlarge
    (GPU) with custom soma classifier and legacy parameters. Results uploaded to
    `s3://hm2p-derivatives/ca_extraction/{sub}/{ses}/suite2p/`. Local validation on
    `sub-1117788/ses-20221018T105617`: 99 ROIs, 25 cells (custom classifier), 14577 frames.
    Key fixes: Suite2p 1.0 API (`run_s2p(db=..., settings=...)`), `sparsedetect` mode()
    bug patch, dpkg lock wait for Ubuntu DLAMI, S3 progress tracking.
15. ✅ **Visualization script** — `scripts/viz_suite2p.py` generates multi-panel figure
    (mean image + ROIs, cell map, classification histogram, top-N dF/F traces).
    Example output: `docs/figures/suite2p_example_sub-1117788.png`.

### Cloud Infrastructure

- `scripts/launch_suite2p_ec2.py` — boto3 script to launch g4dn.xlarge, process all
  sessions, upload results to S3, self-terminate. Supports `--status`/`--progress`/`--terminate`.
- S3 progress tracking: `--progress` reads `_progress.json` from S3.
- Custom classifier at `s3://hm2p-derivatives/config/suite2p/classifier_soma.npy`.
- EC2 key pair (`hm2p-suite2p`) and security group (`hm2p-suite2p-sg`) already created.
- G/VT On-Demand vCPU quota: 4 (sufficient for g4dn.xlarge).
- `scripts/aws_batch_setup.sh` — creates Batch compute environments + job queues.
- `scripts/ecr_push.sh` — builds and pushes GPU/CPU Docker images to ECR.
- `workflow/profiles/aws-batch/config.yaml` — Snakemake AWS Batch profile.

### In Progress

16. 🔄 **DLC pose estimation (Stage 2)** — IN PROGRESS. 13/26 sessions completed on
    2x g5.xlarge EC2 On-Demand (parallel shards). DLC 3.0 PyTorch backend with
    SuperAnimal TopViewMouse + HRNet-W32 + FasterRCNN detector. Videos subsampled
    to 30 fps before inference.

### Recently Completed

17. ✅ **Stage 4 — Calcium processing (cloud)** — 26/26 sessions processed. 391 total
    ROIs across 26 sessions. ca.h5 files uploaded to S3. Soma/dendrite/artefact
    classification via Suite2p stat.npy shape metrics.
18. ✅ **Frontend / dashboard** — Streamlit app with 43+ pages organized in 5 navigation
    sections (Overview, Pipeline, Explore, Analysis, System). Loads real data from S3;
    shows clear message if unavailable. No synthetic data.
19. ✅ **Analysis framework** — 16 analysis modules: activity, tuning, significance,
    comparison, decoder, stability, population, ahv, information, classify, gain,
    anchoring, speed, run, save, plus maze analysis module.
20. ✅ **keypoint-MoSeq** — Docker container + orchestration script implemented for
    zero-label behavioural syllable extraction.
21. ✅ **Security tooling** — bandit, checkov, detect-secrets, pip-audit, vulture
    integrated in CI + pre-commit hooks.

### Remaining

22. ⬜ **CASCADE spike inference** — requires separate conda env (tensorflow==2.3,
    Python 3.8 only). See `docs/manual-installs.md`. Can run on CPU after Stage 4
    dF/F0 is computed.
23. ⬜ **FISSA neuropil subtraction** — optional, more accurate than fixed coefficient.
    Requires separate env (scikit-learn<1.2). See `docs/manual-installs.md`.
24. ⬜ **neuroconv NWB export** — write NWB files from ca.h5 + kinematics.h5 for DANDI
    archiving. Stub only.
25. ⬜ **Rotate hm2p-agent S3 credentials** — current access key was exposed in EC2
    user-data script. Rotate after cloud runs complete.
