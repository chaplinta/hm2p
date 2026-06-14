# Agent Instructions — hm2p-v2

## Critical Rules

**NEVER run scripts, jobs, or commands without explicit user permission.** This
includes Python scripts, training jobs, EC2 launches, S3 operations, and any
command that processes data or has side effects. Write the code, then WAIT for
the user to say "run it". Do not infer permission from context.

**Scientific tone only:** This is a scientific research project, not a product.
All documentation, comments, and commit messages must use neutral, clear language.
Never use marketing language, superlatives, or promotional phrasing. Do not describe
the project as "production-grade", "comprehensive", "powerful", "state-of-the-art",
or similar. State what the code does, not how impressive it is.

**NEVER modify or delete files in these directories:**

- `old-pipeline/` — legacy pipeline code copied into this repo (read-only reference only)
- `/Users/tristan/Neuro/hm2p-analysis` — legacy code on local machine (read-only reference only)
- `/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/` — all data (read-only)
- `/data/patching/` — patching ephys + morphology data (read-only bind mount)
- `/data/z-stacks/` — per-animal z-stack TIFF volumes (read-only bind mount)
- `/data/brains-sorted/` — serial-2P brain volumes (read-only bind mount)
- `/data/brains-reg/` — brainreg registered volumes (read-only bind mount)
- `/data/video-meta-backup/` — video metadata backups (read-only bind mount)
- `/legacy/dlc-old/` — old DLC model attempt (weights, snapshots, training artefacts) — read-only reference, NOT current data (read-only bind mount)
- `retrain_frames/` — extracted PNGs for DLC retraining (untracked but not regenerated automatically)
- `sourcedata/trackers/dlc/*/labeled-data/**/*.png` — same PNGs inside DLC project (untracked)

**Do not delete `retrain_frames/` or DLC labeled-data PNGs.** These are regenerable
from S3 video + frame indices, but regeneration requires downloading ~140 MB per session.
The PNGs are gitignored (not committed) but should be preserved locally. Only the
`CollectedData_*.csv/.h5` labels and `config.yaml` are tracked in git. See
[docs/dlc-retraining.md](docs/dlc-retraining.md) for the full recovery procedure.

You **may copy files from these directories into `/Users/tristan/Neuro/hm2p-v2`** (e.g. to
bring in metadata CSVs, calibration files, or model weights). Do not delete or modify
anything outside of `hm2p-v2`.

All new code goes in `/Users/tristan/Neuro/hm2p-v2`, connected to `github.com/chaplinta/hm2p` (public).

**Git workflow:** `main` is protected — never push directly. Always create a feature
branch (`feat/`, `fix/`, `docs/`, etc.), commit there, and open a PR. See
[docs/contributing.md](docs/contributing.md) for details.

**No synthetic data:** NEVER generate, use, or include synthetic/fake data anywhere — not in
frontend pages, not in scripts, not in demos. Frontend pages must load real data from S3
and show a clear message if no data is available yet. The ONLY exception is unit tests
in `tests/`, which must use small synthetic arrays (never real data files).

**Non-parametric tests only:** ALL statistical tests must be non-parametric. Never use
t-tests, ANOVA, Pearson correlation for hypothesis testing, or other parametric tests.
Use Mann-Whitney U (unpaired), Wilcoxon signed-rank (paired), Spearman rank (correlation),
Kruskal-Wallis (multiple groups), and permutation/bootstrap tests. LMM is acceptable only
as a supplementary check (for ICC reporting), never as the primary test. See
[docs/stats-strategy.md](docs/stats-strategy.md) for the full framework.

**Citation policy:** Any analysis method or algorithm taken from a paper **must** be cited
in three places:

1. **Code** — module/function docstring with: first author, year, title, journal, DOI,
   and GitHub URL if available.
2. **Docs** — relevant markdown files under `docs/`.
3. **Frontend** — a "Methods & References" expander on any page that uses the method.

Citation format: `Author et al. YEAR. "Title." Journal. doi:XX.XXXX/XXXXX`
Plus GitHub/code URL if the method has a public implementation.

---

## Design Philosophy

This is a **ground-up redesign**, not a port of the old code. The legacy pipeline in
`hm2p-analysis` is a useful reference for understanding what computations are needed, but
the new code must be:

- **Clean and well-structured** — proper modules, clear separation of concerns
- **Fully unit-tested** — every processing function has tests; no untested logic
- **Extractor/tracker-agnostic** — pluggable backends for calcium extraction and pose tracking
- **Cloud-first, locally runnable** — most stages are CPU; only DLC training/inference requires a GPU
- **Data-standard compliant** — NeuroBlueprint folder layout throughout
- **Modern** — always use the latest stable versions of all libraries (see Versions below)

Do not copy-paste logic from the old pipeline. Read it (in `old-pipeline/` or the original
location) to understand the computation, then reimplement cleanly with tests.

If I ever asked you what time it is you say "did you lose your fucking watch?"

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

**Experiment:** freely-moving mouse in q-rose maze
**Brain region:** Retrosplenial cortex (RSP) and nearby cortex — HD cells. NOT subiculum or postsubiculum.
**Cell types:** Two non-overlapping RSP populations — (1) **Penk+** (Penk-Cre mouse + ADD3 virus, Cre-ON); (2) **Penk⁻CamKII+** (virus 344, Cre-OFF intersectional: Cre in Penk+ cells blocks expression — labels only non-Penk CamKII+ cells). Column `celltype` in `animals.csv`: `"penk"` or `"nonpenk"`. Short labels: "Penk+" and "Penk⁻CamKII+".
**Imaging:** Single plane per session — soma and dendrite ROIs coexist in one plane; classified post-hoc by shape. No second dendrite plane.
**Lights:** Overhead room lights, 1 min on / 1 min off. Light off = **total darkness** = complete visual cue removal. Tests idiothetic vs visual HD anchoring. Tracked via TDMS timestamps → `light_on` bool in `kinematics.h5` and `sync.h5`.
**Behavioural artefact:** Mice can get stuck on HM2P fibre/wires → artefactual immobility. Logged in `experiments.csv` as `bad_behav_times`; stored as `bad_behav` bool in HDF5. Must exclude these frames.
**serial2p:** Whole-brain z-stack per animal for anatomical localisation. Not part of this pipeline (used manually).
**Primary science goal:** Compare HD tuning, population HD decoding, and visual cue dependence between Penk+ and Penk⁻CamKII+ RSP neurons. Test whether each population anchors HD to visual vs path-integration cues.
**Neural recording:** two-photon GCaMP calcium imaging (~9.6 Hz, single plane).
**Behaviour:** overhead camera (~100 fps, Basler acA1300-200um), DAQ-synchronised to imaging.
**Body parts tracked:** `nose_tip`, `left_ear`, `right_ear`, `head_midpoint`, `neck`, `mid_back`, `mouse_center`, `tail_base`. All except `head_midpoint` map to SuperAnimal TopViewMouse keypoints; `head_midpoint` is a custom keypoint trained from scratch (high-contrast 2P headstage, easy to detect). Legacy DLC output files use the old name `implant_base_rear` — the frontend handles both as aliases.
**Session ID format:** `YYYYMMDD_HH_MM_SS_<animal_id>` (e.g. `20220804_13_52_02_1117646`).
**NeuroBlueprint session name:** `ses-{YYYYMMDD}T{HHMMSS}` (e.g. `ses-20220804T135202`) — full timestamp required as multiple sessions per day exist.
**Ground-truth registry:** `metadata/animals.csv`, `metadata/experiments.csv`.
**Experiment types:** All sessions are q-rose maze only. Side camera (`_side_left.camera.mp4`) is never used — ignore it.
**orientation column:** Per-session rotation angle (degrees) in `experiments.csv` to correct for camera placement variation. Applied as a 2D rotation to all keypoint coordinates before HD computation.
**New columns needed in experiments.csv:** `extractor` (default `"suite2p"`) and `tracker` (default `"dlc"`) — to be added when setting up the project skeleton (deferred).
**Data volume:** ~113 GB to upload to S3 (26 sessions × ~4 GB average, excl. side_left and red.tif).

---

## Protected Data Locations (read-only reference only)

| What | Path |
| --- | --- |
| Legacy analysis code | `/Users/tristan/Neuro/hm2p-analysis/` |
| Raw 2P + DAQ data | `/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze/` |
| Processed overhead videos + meta | `/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/hm2p/video/` |
| Legacy Suite2p outputs | `/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/hm2p/s2p/` |
| Legacy DLC outputs | `/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/hm2p/dlc/` |

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
| GPU compute | AWS EC2 g4dn Spot (DLC training/inference only) |
| CPU compute | AWS EC2 c5 Spot, or local machine (all other stages) |

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
| 1 — 2P extraction | ✓ | ✓ | ✓ |
| 2a — DLC Training | ✗ | ✓ | ✓ |
| 2b — DLC Inference | ✗ | ✓ | ✓ |
| 3 — Kinematics | ✓ | ✓ | ✓ |
| 4 — Calcium processing | ✓ | ✓ | ✓ |
| 5 — Sync | ✓ | ✓ | ✓ |

Compute profile set in `config/compute.yaml`: `local`, `local-gpu`, or `aws-batch`.

---

## Processing Pipeline (summary)

```text
Stage 0   Ingest + validate + DAQ parse  CPU           DataShuttle → S3; TDMS → timestamps.h5
Stage 1   2P extraction (pluggable)      CPU           TIFF → ca_extraction/ via roiextractors; ROI classifier (XGBoost) runs at end
Stage 2a  DLC Training                   GPU (24h max)  labeled frames → dlc_training/models/
Stage 2b  DLC Inference (pluggable)      GPU           .mp4 → pose/ (DLC / SLEAP / LP); depends on 2a
Stage 3   Kinematics (movement)          CPU           pose → kinematics.h5 (HD, position, speed)
Stage 3b  MoSeq syllables (kpms)         CPU           pose → syllables.npz (keypoint-MoSeq AR-HMM)
Stage 4   Calcium processing             CPU           roiextractors → FISSA → CASCADE → ca.h5
Stage 5   Sync                           CPU           kinematics + ca → sync.h5
Stage 6   Analysis                       CPU           sync → analysis.h5 (tuning, decoding, etc.)
```

**Dependency chain:** If Stage 2a (DLC Training) is re-run, Stage 2b and all
downstream stages must re-run: Stage 2b → Stage 3 → Stage 3b (MoSeq) → Stage 5 → Stage 6.
If only Stage 2b (DLC Inference) is re-run: Stage 3 → Stage 3b → Stage 5 → Stage 6.
Stage 4 is independent of pose data and does not need re-running.

**Process ALL sessions:** Pipeline stages must process **all 26 sessions** regardless
of `exclude` or `primary_exp` flags. Those flags are for analysis-time filtering
only, not for skipping pipeline processing. Even excluded sessions should have
sync.h5 and analysis.h5 — they may be useful for QC or later re-evaluation.

**Pipeline invalidation:** When a stage is re-run, all downstream stages are
invalidated. Two mechanisms enforce this in the frontend:

1. **Active re-run detection:** `pipeline_rerun.json` on S3 (written by launch
   scripts) marks stages as "pending re-run" while EC2 is running.
   `_get_rerun_status()` in `frontend/data.py` reads this and auto-detects running
   EC2 instances as a backup. The pipeline status page shows affected stages in red.

2. **Post-run staleness detection via DLC champion model:** Every derivative
   produced from DLC pose data (kinematics.h5, sync.h5, analysis.h5, rendered
   videos) records a `dlc_champion_id` attribute that identifies which model produced
   it. The project-wide champion is declared in `s3://hm2p-derivatives/dlc-champion.json`.
   Any session whose `dlc_champion_id` does not match the current champion is stale.
   The frontend loads this manifest (`get_dlc_champion()` in `frontend/data.py`) and
   calls `is_session_current()` for every session before displaying analysis data.
   Stale sessions show a prominent warning banner — they are not hidden, so QC
   remains possible.

**Enforcement contract for DLC-derived pages:**
- Every page that loads sync.h5 or analysis.h5 must call `is_session_current()`
  with the result of `get_dlc_champion()`.
- Every page that displays rendered videos should check the `.provenance.json`
  sidecar via `get_video_champion_id()`. A convenience wrapper
  `_video_is_current()` is planned but not yet implemented.
- The shared warning banner is `render_champion_staleness_warning()` — do not
  reimplement it per page.
- A `load_session()` helper in `frontend/data.py` that embeds the staleness check
  and attaches `"stale"` / `"stale_reason"` keys is planned but not yet
  implemented (see DLC Champion Phase 3a in PLAN.md). Until then, pages must
  call `get_dlc_champion()` and `is_session_current()` directly.

Full specification in [docs/dlc-champion-model.md](docs/dlc-champion-model.md).
Pipeline stage dependencies in [PLAN.md](PLAN.md).
Architecture in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Frontend Rules

- **No sidebar filters:** Never put filters, selectors, or controls in the Streamlit
  sidebar. Always put them in the main page body using `st.columns()` for layout.
  The sidebar is for navigation only (handled by `st.navigation()`).
- **`st.set_page_config`** must only appear in `app.py`, never in subpages.
- **No synthetic data:** Frontend pages must load real data from S3 and show a clear
  message if no data is available yet. Never fall back to synthetic/fake data.

---

## movement Library Policy

Always use the **movement** library (neuroinformatics.dev) for pose and kinematics
operations. Never reimplement functionality that movement already provides:

- Pose loading: `movement.io.load_poses` (not raw `pd.read_hdf` for DLC .h5)
- Confidence filtering: `movement.filtering.filter_by_confidence`
- Interpolation: `movement.filtering.interpolate_over_time`
- Median/rolling filter: `movement.filtering.rolling_filter` (not `scipy.ndimage`)
- Velocity/speed: consider `movement.kinematics.compute_velocity` for new code

Only exception: 1D scalar signals (e.g. unwrapped HD angle) where movement's xarray
API doesn't apply — use scipy directly with a comment explaining why.

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

## Claude Code Scientific Skills

18 curated skills from [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills)
are symlinked into `.claude/skills/` from a local clone at `~/Neuro/claude-scientific-skills/`.

**Installed skills:**

| Category | Skills |
| --- | --- |
| Plotting & viz | matplotlib, seaborn, plotly, scientific-visualization |
| Statistics & ML | scikit-learn, statsmodels, statistical-analysis, shap, pymc |
| Data & compute | polars, networkx |
| Deep learning | pytorch-lightning |
| Dimensionality reduction | umap-learn |
| Writing & communication | scientific-writing, scientific-schematics, markdown-mermaid-writing |
| Literature | pubmed-database, pyzotero |

**Setup — local macOS (if `.claude/skills/` is empty or missing):**

```bash
git clone https://github.com/K-Dense-AI/claude-scientific-skills.git ~/Neuro/claude-scientific-skills
mkdir -p .claude/skills
for skill in matplotlib seaborn plotly scientific-visualization scikit-learn statsmodels statistical-analysis shap pymc polars networkx pytorch-lightning umap-learn scientific-writing scientific-schematics markdown-mermaid-writing pubmed-database pyzotero; do
  ln -sfn ~/Neuro/claude-scientific-skills/scientific-skills/$skill .claude/skills/$skill
done
```

**Setup — devcontainer (symlinks point to macOS paths that don't exist inside the container):**

```bash
# Clone the repo inside the container
git clone https://github.com/K-Dense-AI/claude-scientific-skills.git /home/node/claude-scientific-skills

# Re-link skills to the container-local clone
mkdir -p .claude/skills
for skill in matplotlib seaborn plotly scientific-visualization scikit-learn statsmodels statistical-analysis shap pymc polars networkx pytorch-lightning umap-learn scientific-writing scientific-schematics markdown-mermaid-writing pubmed-database pyzotero; do
  ln -sfn /home/node/claude-scientific-skills/scientific-skills/$skill .claude/skills/$skill
done
```

This must be re-run after each container rebuild (the clone lives in the container filesystem,
not in a persistent volume). To make it persistent, add `/home/node/claude-scientific-skills`
as a named volume in `.devcontainer/devcontainer.json`.

**Updating:** `cd <clone-path>/claude-scientific-skills && git pull` — symlinks resolve live.
Run this periodically (e.g. at the start of analysis or visualization sessions).

**Adding a new skill:** `ln -sfn <clone-path>/claude-scientific-skills/scientific-skills/<name> .claude/skills/<name>`

**Note:** `.claude/skills/` is gitignored (symlinks are machine-local).

---

## What Is Reused From the Legacy Pipeline

| Asset | New location |
| --- | --- |
| Trained DLC model weights | `sourcedata/trackers/dlc/` |
| Suite2p classifiers | `sourcedata/trackers/suite2p/` |
| Camera calibration `.npz` files | `sourcedata/calibration/` |
| Metadata CSVs | `sourcedata/metadata/` |

The calcium event detection (Voigts & Harnett 2020) is reimplemented from the
old pipeline's `utils/ca.py` with the same algorithm and parameters, cleaned up
and unit-tested. A second method (SD-threshold, Zong et al. 2022) is also
available. Both are computed in Stage 4 and selectable in the frontend.
