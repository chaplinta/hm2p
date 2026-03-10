# hm2p — Head-Direction & Two-Photon Analysis Pipeline

Cloud-based analysis pipeline for freely-moving mouse experiments combining
**two-photon calcium imaging** with **overhead behavioural video tracking**.

---

## What This Is

Mice explore a rose-maze (or open field / linear track) while we record:

- **2P calcium imaging** (~9.6 Hz, GCaMP) of **retrosplenial cortex (RSP)** and nearby cortex — a region with prominent head-direction cells
- **Overhead video** (~100 fps) of the mouse, synchronised to imaging via DAQ
- **Periodic light stimulus** — 1 min on / 1 min off visual landmark manipulation throughout each session

Two genetically-defined, non-overlapping RSP populations are compared:

| Population | Labelling strategy | `celltype` |
| --- | --- | --- |
| **Penk+ RSP neurons** | Penk-Cre mouse + ADD3 virus (Cre-ON: only Penk+ cells express GCaMP7f) | `"penk"` |
| **CamKII+ non-Penk RSP neurons** | Penk-Cre mouse + virus 344 (Cre-OFF intersectional: Cre blocks expression in Penk+ cells, leaving only non-Penk CamKII+ cells labelled) | `"nonpenk"` |

The overhead room lights cycle **1 min on / 1 min off**. Light off = **total darkness** —
all visual landmarks are removed. This tests whether RSP HD cells use visual vs
path-integration cues to maintain their directional tuning.

The primary scientific goals are:

1. Characterise **HD tuning** in Penk+ vs CamKII+ RSP neurons
2. Determine whether each population anchors HD to **visual or idiothetic cues**
3. Decode HD at population level and compare coding strategies between cell types

From these raw recordings, this pipeline extracts:

| Output | Where used |
| --- | --- |
| Head direction (HD) | Head-direction cell characterisation |
| Position (x, y mm) | Place cell mapping |
| Speed, AHV | Movement-state gating |
| Per-ROI dF/F0 | Calcium signal quality control |
| Calibrated spike rates (CASCADE) | Neural-behavioural correlation |
| Synchronised neural + behav dataset | All downstream analyses |

This is a **ground-up rewrite** of the legacy `hm2p-analysis` pipeline.
The new pipeline is cloud-first, fully unit-tested, and uses pluggable
backends for both calcium extraction and pose tracking.

---

## Repository Map

```text
hm2p-v2/
├── README.md          ← you are here
├── PLAN.md            ← full pipeline design (Stages 0–5)
├── ARCHITECTURE.md    ← code layout, HDF5 schemas, interface contracts
├── CLAUDE.md          ← coding standards, tool versions, rules for AI agents (auto-loaded by Claude Code)
├── old-pipeline/      ← legacy pipeline code (read-only reference — never modify)
├── frontend/         ← Streamlit dashboard (43 pages)
│   ├── app.py
│   ├── data.py       ← S3 data loading + caching
│   └── pages/        ← analysis, pipeline QC, system pages
├── docs/
│   ├── data-guide.md  ← raw data formats, file structures, legacy processing
│   ├── aws-setup.md   ← AWS account, IAM, S3 bucket setup
│   ├── analysis-plan.md
│   └── research-landscape.md
├── src/hm2p/          ← pipeline source code
├── tests/             ← unit tests (≥ 90% coverage required)
├── workflow/          ← Snakemake DAG + compute profiles
├── config/            ← pipeline parameters
└── docker/            ← reproducible compute environments
```

---

## Pipeline at a Glance

```text
Raw Data (Dropbox → S3)
  │
  ▼ Stage 0 — Ingest & Validate
  │  Check files, upload to S3, parse TDMS DAQ → timestamps.h5
  │
  ├──────────────────────────────────┐
  ▼                                  ▼
Stage 1 — 2P Extraction          Stage 2 — Pose Estimation
  Suite2p (default) or CaImAn      DeepLabCut (default) / SLEAP / LightningPose
  ROI detection, F traces           Keypoint tracking (5 body parts)
  → derivatives/ca_extraction/      → derivatives/pose/
  │                                  │
  ▼ Stage 4 — Calcium Processing   Stage 3 — Kinematics
  │  Neuropil subtraction             movement library
  │  dF/F0, CASCADE spike inference   HD, position, speed, AHV
  │  → derivatives/calcium/ca.h5     → derivatives/movement/kinematics.h5
  │                                  │
  └──────────────┬───────────────────┘
                 ▼
            Stage 5 — Synchronisation
              Resample behaviour → imaging frame times
              → derivatives/sync/sync.h5
                 │
                 ▼
            Analysis (done — 16 modules)
              pynapple · NEMOS · CEBRA†
              † CEBRA requires separate env
```

All stages are **pluggable**: swap the calcium extractor or pose tracker by
changing one field in `experiments.csv` — downstream code is unchanged.

---

## Key Design Decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Calcium extraction API | **roiextractors** | Unified interface for Suite2p, CaImAn, etc. |
| Spike inference | **CASCADE** (Rupprecht et al. 2021) | Calibrated spikes/s; pre-trained GCaMP models |
| Neuropil subtraction | Fixed 0.7×Fneu (default) or **FISSA** | FISSA more accurate in dense tissue ([manual install](docs/manual-installs.md)) |
| Kinematics API | **movement** (SWC/UCL) | Unified xarray output; all trackers supported |
| Behavioural syllables | **keypoint-MoSeq** / **VAME** | Zero-label; separate envs required ([manual install](docs/manual-installs.md)) |
| Data standard | **NeuroBlueprint** | BIDS-inspired; DataShuttle support |
| Intermediate format | **HDF5** | Fast random access; pynapple-native |
| Analysis interface | **pynapple** | Timestamp-aware slicing across neural + behav |
| Pipeline orchestration | **Snakemake** | Local + AWS Batch with same DAG |

---

## Data Organisation

We follow the **NeuroBlueprint** data standard:

```text
hm2p/
├── rawdata/
│   └── sub-{animal_id}/
│       └── ses-{YYYYMMDD}T{HHMMSS}/     ← full timestamp (multiple sessions/day)
│           ├── funcimg/                  ← 2P TIFF stacks + .meta.txt
│           └── behav/                    ← .mp4 video + meta/ folder
│
├── sourcedata/
│   ├── trackers/                         ← DLC model weights + labeled frames
│   ├── calibration/                      ← camera .npz lens calibrations
│   └── metadata/                         ← animals.csv, experiments.csv
│
└── derivatives/
    ├── ca_extraction/                    ← Suite2p / CaImAn native outputs
    ├── pose/                             ← DLC / SLEAP / LP outputs
    ├── movement/                         ← kinematics.h5 (HD, position, speed, AHV)
    ├── calcium/                          ← ca.h5 (dF/F0, spike rates)
    └── sync/                             ← sync.h5 (neural + behav aligned)
```

Data lives on **AWS S3** (`s3://hm2p-rawdata/`, `s3://hm2p-derivatives/`).
Transfer and validation is handled by **DataShuttle**.

---

## Compute Requirements

| Stage | CPU-only laptop | Local GPU machine | AWS (recommended) |
| --- | --- | --- | --- |
| 0 — Ingest | ✓ | ✓ | ✓ |
| 1 — 2P extraction | slow | ✓ | ✓ (g4dn Spot ~$0.16/hr) |
| 2 — Pose estimation | ✗ | ✓ | ✓ (g4dn Spot) |
| 3 — Kinematics | ✓ | ✓ | ✓ (c5 Spot ~$0.27/hr) |
| 4 — Calcium processing | ✓ | ✓ | ✓ |
| 5 — Sync | ✓ | ✓ | ✓ |

Switch between local and cloud by changing one line in `config/compute.yaml`:

```yaml
profile: local        # or local-gpu, aws-batch
```

One-time processing cost for all 30 sessions: ~$180–380 (AWS Spot).
Ongoing storage: ~$10/month (600 GB on S3 Infrequent Access).

---

## Neuroscientist's Guide

### What you need to run a session

1. Raw session folder on Dropbox with:
   - `*_XYT.tif` (2P TIFF stack, from SciScan → raw2tif)
   - `*.meta.txt` (experiment metadata)
   - `*-di.tdms` (DAQ timing pulses)
   - `*.mp4` (overhead behavioural video)
   - `meta/meta.txt` (video crop/scale/ROI)

2. Session registered in `metadata/experiments.csv` with:
   - `exp_id` (e.g. `20220804_13_52_02_1117646`)
   - `animal_id`, `extractor` (`suite2p`), `tracker` (`dlc`)

3. Animal registered in `metadata/animals.csv`

See [docs/data-guide.md](docs/data-guide.md) for a full description of all
raw file formats, field meanings, and the mapping to NeuroBlueprint paths.

### Outputs you get per session

After running all 5 stages you have:

| File | Contents |
| --- | --- |
| `timestamps.h5` | Precise camera + imaging frame times from DAQ |
| `kinematics.h5` | HD, position (mm), speed (cm/s), AHV (deg/s), movement state |
| `ca.h5` | dF/F0, CASCADE spike rates (spikes/s), SNR per ROI |
| `sync.h5` | All of the above aligned to imaging frames — the analysis-ready file |

The `sync.h5` file loads directly into **pynapple** for analysis:

```python
import pynapple as nap, h5py

with h5py.File("sync.h5") as f:
    t      = f["frame_time"][:]
    spikes = nap.TsdFrame(t=t, d=f["spikes"][:].T)   # ROIs × time → TsdFrame
    hd     = nap.Tsd(t=t, d=f["hd"][:])
    speed  = nap.Tsd(t=t, d=f["speed"][:])

# Time-aware restriction to active periods only:
active_ep     = nap.IntervalSet(...)
spikes_active = spikes.restrict(active_ep)
```

### Running on AWS

See [docs/aws-setup.md](docs/aws-setup.md) for account creation, IAM roles,
S3 bucket setup, and how to launch EC2 Spot instances for each stage.

---

## For Developers / AI Agents

- See [CLAUDE.md](CLAUDE.md) for coding standards, version pins, unit testing
  requirements (≥ 90% coverage), and rules for reading/writing data.
- See [ARCHITECTURE.md](ARCHITECTURE.md) for source layout, HDF5 schemas, and
  interface contracts between modules.
- See [PLAN.md](PLAN.md) for the full pipeline design including detailed stage
  descriptions, pluggable backend design, and technology stack.

### Quick rules

- **Never modify** `old-pipeline/`, `/Users/tristan/Neuro/hm2p-analysis/`, or Dropbox data
- All new code goes in `src/hm2p/`; all tests in `tests/`
- Tests use **synthetic data only** — never read real data files in tests
- Use `hypothesis` for numerical functions, `pandera` for schema validation
- Run `pre-commit` before committing (ruff + mypy + nbstripout)

### Claude Code scientific skills

18 curated skills from [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills)
are available for Claude Code agents working on this project. Skills provide domain-specific
knowledge for matplotlib, seaborn, plotly, scikit-learn, statsmodels, pymc, and more.

See the **Claude Code Scientific Skills** section in [CLAUDE.md](CLAUDE.md) for setup
instructions (local macOS and devcontainer).

---

## Status

**Implementation phase** — core pipeline code and tests are written. 1038+ tests passing, 92% coverage.

| Component | Status |
| --- | --- |
| Project skeleton (pyproject.toml, pre-commit, CI) | Done |
| HDF5 schema validation | Done |
| Stage 0 — TDMS ingest (`ingest/daq.py`) | Done |
| **Stage 1 — Suite2p cloud run (all 26 sessions)** | **Done** |
| Stage 2 — DLC pose estimation | In progress (13/26 sessions) |
| Stage 3 — Kinematics (`kinematics/compute.py`) | Done |
| Stage 4 — Calcium processing (`calcium/`) | Done — 26/26 sessions (CASCADE deferred) |
| Stage 5 — Sync (`sync/align.py`) | Done |
| Suite2p extractor (`extraction/suite2p.py`) | Done |
| CaImAn extractor (`extraction/caiman.py`) | Done |
| S3 data upload (26 sessions) | Done |
| EC2 cloud run infrastructure | Done (`scripts/launch_suite2p_ec2.py`) |
| Snakemake DAG | Pending — rules defined, shell commands needed |
| Docker images for cloud | Done (gpu, cpu, kpms Dockerfiles) |
| Frontend dashboard (43 pages) | Done |
| Analysis framework (16 modules) | Done |
| keypoint-MoSeq Docker integration | Done |
| NWB export (neuroconv) | Pending — stub only |

---

## References

- **CASCADE**: Rupprecht et al. (2021) *Nature Neuroscience* — calibrated spike inference ([manual install](docs/manual-installs.md))
- **FISSA**: Keemink et al. (2018) *Scientific Reports* — neuropil subtraction ([manual install](docs/manual-installs.md))
- **keypoint-MoSeq**: Weinreb et al. (2024) *Nature Methods* — gold standard zero-label AR-HMM behaviour segmentation ([manual install](docs/manual-installs.md))
- **VAME**: Luxem et al. (2022) *Nature Communications* — zero-label VAE syllables ([manual install](docs/manual-installs.md))
- **CEBRA**: Schneider et al. (2023) *Nature* — joint neural-behavioural embeddings ([manual install](docs/manual-installs.md))
- **NEMOS**: Flatiron Institute — GLM encoding models for neural data
- **pynapple**: Viejo et al. — unified timeseries interface for systems neuroscience
- **movement**: Neuroinformatics.dev (SWC/UCL) — unified kinematics from pose tracking
- **roiextractors**: CatalystNeuro — unified API for calcium imaging extractors
- **NeuroBlueprint**: neuroblueprint.neuroinformatics.dev
- **DataShuttle**: datashuttle.neuroinformatics.dev
