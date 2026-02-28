# Agent Instructions — hm2p-v2

## Critical Rules

**NEVER modify or delete files in these directories:**

- `/Users/tristan/Neuro/hm2p-analysis` — legacy code (read-only reference only)
- `/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/hm2p` — all data (read-only)

You **may copy files from these directories into `/Users/tristan/Neuro/hm2p-v2`** (e.g. to
bring in metadata CSVs, calibration files, or model weights). Do not delete or modify
anything outside of `hm2p-v2`.

All new code goes in `/Users/tristan/Neuro/hm2p-v2`, connected to `github.com/chaplinta/hm2p` (private).

---

## Design Philosophy

This is a **ground-up redesign**, not a port of the old code. The legacy pipeline in
`hm2p-analysis` is a useful reference for understanding what computations are needed, but
the new code must be:

- **Clean and well-structured** — proper modules, clear separation of concerns
- **Fully unit-tested** — every processing function has tests; no untested logic
- **Extractor/tracker-agnostic** — pluggable backends for calcium extraction and pose tracking
- **Cloud-first, locally runnable** — CPU stages can run locally; GPU stages require a GPU
- **Data-standard compliant** — NeuroBlueprint folder layout throughout
- **Modern** — always use the latest stable versions of all libraries (see Versions below)

Do not copy-paste logic from the old pipeline. Read it to understand the computation,
then reimplement cleanly with tests.

---

## Versions

Always use the **latest stable release** of each tool at time of implementation.
Do not pin to old versions without a documented compatibility reason.

| Tool | Role |
| --- | --- |
| Suite2p (latest) | 2P motion correction + ROI extraction (default extractor) |
| CaImAn (latest) | Alternative 2P extractor |
| roiextractors (latest) | Unified read API across all calcium extractors |
| CASCADE / `cascade2p` (latest) | Calibrated spike inference — primary event detection (replaces OASIS/V&H) |
| FISSA (latest) | Spatial ICA neuropil subtraction — optional, more accurate than fixed coefficient |
| DeepLabCut 3.x+ (latest) | Pose estimation (default tracker) |
| SLEAP (latest) | Alternative pose tracker |
| LightningPose (latest) | Alternative pose tracker |
| movement (latest) | Unified kinematics from any pose tracker |
| keypoint-MoSeq (latest, ≥ 0.6) | Zero-label AR-HMM syllables; gold standard for freely-moving mice; Nature Methods 2024 |
| VAME / EthoML (latest, ≥ 0.12, `vame-py`) | Zero-label VAE syllables; movement xarray native input; NWB export |
| DLC2Action (latest) | Semi-supervised action recognition with active learning; 10–100 labels |
| pynapple (latest) | Unified timeseries interface — load dF/F + behaviour for analysis |
| NEMOS (latest) | GLM encoding models, pynapple-native, JAX backend |
| CEBRA (latest) | Contrastive population embeddings with behavioural supervision |
| neuroconv (latest) | roiextractors + movement → NWB export for archiving |
| nptdms (latest) | Parse NI TDMS DAQ files → timestamps.h5 |
| Snakemake 8.x+ (latest) | Pipeline orchestration |
| uv (latest) | Python package management |
| Docker (latest) | Reproducible compute environments |
| pytest + pytest-cov (latest) | Unit testing + coverage |
| hypothesis (latest) | Property-based testing for numerical functions |
| pandera (latest) | Runtime DataFrame / xarray / HDF5 schema validation |
| mypy (latest) | Static type checking |
| ruff (latest) | Linting + formatting (replaces black + flake8 + isort) |
| pre-commit (latest) | Auto-runs ruff, mypy, nbstripout before every commit |
| DVC (latest) | Data and model artifact versioning alongside git |

---

## Project Context

**Experiment:** freely-moving mouse in rose-maze / open field / linear track.
**Brain region:** Retrosplenial cortex (RSP) and nearby cortex — HD cells. NOT subiculum or postsubiculum.
**Cell types:** Two non-overlapping RSP populations — (1) **Penk+** (Penk-Cre mouse + ADD3 virus, Cre-ON); (2) **non-Penk CamKII+** (virus 344, Cre-OFF intersectional: Cre in Penk+ cells blocks expression). Column `celltype` in `animals.csv`: `"penk"` or `"nonpenk"`.
**Imaging:** Single plane per session — soma and dendrite ROIs coexist in one plane; classified post-hoc by shape. No second dendrite plane.
**Lights:** Overhead room lights, 1 min on / 1 min off. Light off = **total darkness** = complete visual cue removal. Tests idiothetic vs visual HD anchoring. Tracked via TDMS timestamps → `light_on` bool in `kinematics.h5` and `sync.h5`.
**Behavioural artefact:** Mice can get stuck on HM2P fibre/wires → artefactual immobility. Logged in `experiments.csv` as `bad_behav_times`; stored as `bad_behav` bool in HDF5. Must exclude these frames.
**serial2p:** Whole-brain z-stack per animal for anatomical localisation. Not part of this pipeline (used manually).
**Primary science goal:** Compare HD tuning, population HD decoding, and visual cue dependence between Penk+ and CamKII+ RSP neurons. Test whether each population anchors HD to visual vs path-integration cues.
**Neural recording:** two-photon GCaMP calcium imaging (~30 Hz, single or dual plane).
**Behaviour:** overhead camera (~100 fps, Basler acA1300-200um), DAQ-synchronised to imaging.
**Body parts tracked:** `ear-left`, `ear-right`, `back-upper`, `back-middle`, `back-tail`.
**Session ID format:** `YYYYMMDD_HH_MM_SS_<animal_id>` (e.g. `20220804_13_52_02_1117646`).
**NeuroBlueprint session name:** `ses-{YYYYMMDD}T{HHMMSS}` (e.g. `ses-20220804T135202`) — full timestamp required as multiple sessions per day exist.
**Ground-truth registry:** `metadata/animals.csv`, `metadata/experiments.csv`.
**Experiment types:** All sessions are rose-maze only. Side camera (`_side_left.camera.mp4`) is never used — ignore it.
**orientation column:** Per-session rotation angle (degrees) in `experiments.csv` to correct for camera placement variation. Applied as a 2D rotation to all keypoint coordinates before HD computation.
**New columns needed in experiments.csv:** `extractor` (default `"suite2p"`) and `tracker` (default `"dlc"`) — to be added when setting up the project skeleton (deferred).
**Data volume:** ~550 GB across 29–30 sessions.

---

## Protected Data Locations (read-only reference only)

| What | Path |
| --- | --- |
| Legacy analysis code | `/Users/tristan/Neuro/hm2p-analysis/` |
| All raw + processed data | `/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/hm2p/` |

---

## Key Tools

| Purpose | Tool |
| --- | --- |
| DAQ parsing | **nptdms** → `timestamps.h5` (Stage 0) |
| 2P preprocessing + extraction | Suite2p (default), CaImAn — pluggable via `extractor` field |
| Unified extraction API | **roiextractors** — same interface regardless of extractor |
| Neuropil subtraction | Fixed coefficient (default) or **FISSA** (spatial ICA) |
| Spike inference | **CASCADE** — calibrated spikes/s from dF/F (primary); V&H threshold (fallback) |
| Pose estimation | DeepLabCut (default), SLEAP, LightningPose — pluggable via `tracker` field |
| Unified kinematics | **movement** (neuroinformatics.dev) — same xarray.Dataset regardless of tracker |
| Behavioural syllables | **VAME** (v0.7+ movement-native, zero labels) or **keypoint-MoSeq** (AR-HMM) |
| Analysis interface | **pynapple** — `TsdFrame` for dF/F/spikes; `Tsd` for behaviour |
| Encoding models | **NEMOS** — GLM; pynapple-native |
| Population embeddings | **CEBRA** — contrastive latent spaces |
| NWB archiving | **neuroconv** — HDF5 outputs → NWB → DANDI |
| Data organisation standard | NeuroBlueprint (neuroblueprint.neuroinformatics.dev) |
| Data transfer to cloud | DataShuttle (datashuttle.neuroinformatics.dev) |
| Pipeline orchestration | Snakemake (local + AWS Batch profiles) |
| Cloud storage | AWS S3 |
| GPU compute | AWS EC2 g4dn Spot, or local GPU machine |
| CPU compute | AWS EC2 c5 Spot, or local machine |

---

## Required Behavioural Outputs (Stage 3)

Primary (required for all sessions):

- **Head direction (HD)** — angle from ear vector, unwrapped, degrees
- **Position** — x/y in mm (body centroid + scale calibration)
- **Speed** — cm/s, smoothed
- **Angular head velocity (AHV)** — deg/s
- **Movement state** — binary active/inactive

Optional Stage 3b outputs (exploratory — deferred until Stages 0–5 complete):

- **Behavioural syllables** — zero-label unsupervised segmentation via **keypoint-MoSeq** (gold standard for freely-moving mice) or **VAME** (v0.12+ movement native). Output: `/syllable_id (N,) int16` in kinematics.h5.
- **Ethogram** — semi-supervised with **DLC2Action** (active learning; 10–100 labeled clips).
- **Avoid:** B-SOiD (stale since 2021), MotionMapper (MATLAB, stale 2020).
- VAME v0.12 (`pip install vame-py`) natively accepts the `movement` xarray Dataset. keypoint-MoSeq reads DLC `.h5` directly (same source files as movement).
- CEBRA (v0.6+, Apache 2.0): joint neural + behaviour embeddings. Two modes: (1) HD/position-guided (for RSC HD cell population analysis); (2) time-contrastive (zero labels). Input is numpy `(T, R)` — exactly what `sync.h5["dff"].T` gives.

---

## Local vs Cloud Compute

| Stage | Local CPU | Local GPU | Cloud |
| --- | --- | --- | --- |
| 0 — Ingest | ✓ | ✓ | ✓ |
| 1 — 2P extraction | slow | ✓ | ✓ |
| 2 — Pose estimation | ✗ | ✓ | ✓ |
| 3 — Kinematics | ✓ | ✓ | ✓ |
| 4 — Calcium processing | ✓ | ✓ | ✓ |
| 5 — Sync | ✓ | ✓ | ✓ |

Compute profile set in `config/compute.yaml`: `local`, `local-gpu`, or `aws-batch`.

---

## Processing Pipeline (summary)

```text
Stage 0  Ingest + validate + DAQ parse  CPU    DataShuttle → S3; TDMS → timestamps.h5
Stage 1  2P extraction (pluggable)      GPU    TIFF → ca_extraction/ via roiextractors
Stage 2  Pose estimation (pluggable)    GPU    .mp4 → pose/ (DLC / SLEAP / LP)
Stage 3  Kinematics (movement)          CPU    pose → kinematics.h5 (HD, position, speed)
Stage 4  Calcium processing             CPU    roiextractors → FISSA → CASCADE → ca.h5
Stage 5  Sync                           CPU    kinematics + ca → sync.h5
```

Full details in [PLAN.md](PLAN.md). Architecture in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Unit Testing Requirements

Unit tests are **mandatory at every opportunity** — no exceptions.

- **Every function** (public and private) must have at least one unit test
- Tests use small **synthetic arrays only** — never read real data files
- Use `hypothesis` for numerical functions (dF/F0, HD computation, spike inference wrappers)
  to auto-generate adversarial inputs and find edge cases
- Use `pandera` to validate HDF5 schemas in tests — test that outputs conform to schema
- Framework: `pytest` + `pytest-cov`; tests live in `tests/` mirroring `src/` structure
- CI runs tests on every push (GitHub Actions); PRs blocked if coverage drops below 90%
- Coverage target: ≥ 90% — hard requirement, not a guideline
- Prefer many small focused tests over few large integration tests

---

## Data Standard (NeuroBlueprint)

```text
rawdata/sub-{animal_id}/ses-{YYYYMMDD}/funcimg/    ← 2P TIFFs + .meta.txt
rawdata/sub-{animal_id}/ses-{YYYYMMDD}/behav/      ← video + meta/
sourcedata/trackers/                                ← DLC / SLEAP models + labeled data
sourcedata/calibration/                             ← camera .npz files
sourcedata/metadata/                                ← animals.csv, experiments.csv
derivatives/ca_extraction/...                       ← extractor-native files
derivatives/pose/...                                ← tracker-native files
derivatives/movement/...                            ← kinematics.h5
derivatives/calcium/...                             ← ca.h5
derivatives/sync/...                                ← sync.h5
```

---

## What Is Reused From the Legacy Pipeline

| Asset | New location |
| --- | --- |
| Trained DLC model weights | `sourcedata/trackers/dlc/` |
| Suite2p classifiers | `sourcedata/trackers/suite2p/` |
| Camera calibration `.npz` files | `sourcedata/calibration/` |
| Metadata CSVs | `sourcedata/metadata/` |

The calcium event detection algorithm (Voigts & Harnett) is **ported verbatim** into a
new, unit-tested module — not copied as-is from the old code.
