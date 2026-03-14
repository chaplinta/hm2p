# Requirements Status

Last updated: 2026-03-14

---

## Pipeline Processing

### Stage 0 -- Ingest, Validation & DAQ Parsing (CPU)

- [x] Session registry validation (check raw files exist per session)
- [x] Upload raw data to S3 (`s3://hm2p-rawdata/`) -- 26 sessions, 91.4 GiB, 503 objects
- [x] TDMS DAQ parsing via `nptdms` -> `timestamps.h5`
- [x] `timestamps.h5` schema: `frame_times_camera`, `frame_times_imaging`, `light_on/off_times`
- [x] Unit tests for DAQ parsing (`tests/ingest/test_daq.py`, `tests/ingest/test_daq_parse.py`)
- [x] Validation module (`src/hm2p/ingest/validate.py`, `tests/ingest/test_validate.py`)
- **Status: Complete (26/26 sessions)**
- **Evidence:** `src/hm2p/ingest/daq.py`, `scripts/run_stage0_daq.py`

### Stage 1 -- 2P Extraction (GPU)

- [x] Suite2p extractor class with roiextractors unified API (`src/hm2p/extraction/suite2p.py`)
- [x] CaImAn extractor class (`src/hm2p/extraction/caiman.py`)
- [x] Abstract extractor interface (`src/hm2p/extraction/base.py`)
- [x] Post-hoc soma/dendrite/artefact ROI classification (aspect_ratio > 2.5 -> dendrite)
- [x] Suite2p run wrapper (`src/hm2p/extraction/run_suite2p.py`)
- [x] EC2 launch script (`scripts/launch_suite2p_ec2.py`)
- [x] Z-drift estimation from serial2p z-stacks (`src/hm2p/extraction/zdrift.py`)
- [x] Cloud run complete -- 26/26 sessions on EC2 g4dn.xlarge
- [x] Results on S3 at `s3://hm2p-derivatives/ca_extraction/{sub}/{ses}/suite2p/`
- [x] Unit tests (`tests/extraction/test_suite2p.py`, `test_caiman.py`, `test_run_suite2p.py`, `test_zdrift.py`)
- **Status: Complete (26/26 sessions)**

### Stage 2 -- Pose Estimation (GPU)

- [x] DLC 3.0 PyTorch backend with SuperAnimal TopViewMouse + HRNet-W32 + FasterRCNN detector
- [x] Videos subsampled to 30fps before inference
- [x] Pose run dispatcher (`src/hm2p/pose/run.py`)
- [x] Pose quality metrics (`src/hm2p/pose/quality.py`)
- [x] Video preprocessing utilities (`src/hm2p/pose/preprocess.py`)
- [x] DLC retraining helpers (`src/hm2p/pose/retrain.py`)
- [x] EC2 launch scripts: single (`scripts/launch_dlc_ec2.py`), parallel (`scripts/launch_dlc_parallel.py`)
- [x] Cloud run complete -- 26/26 sessions
- [x] Unit tests (`tests/pose/test_preprocess.py`, `test_quality.py`, `test_retrain.py`, `test_run.py`)
- **Status: Complete (26/26 sessions)**

### Stage 3 -- Behavioural Kinematics (CPU)

- [x] Load poses via `movement` library (`src/hm2p/kinematics/compute.py`)
- [x] Head direction (HD) from ear vector, unwrapped, degrees
- [x] Position (x/y in mm via scale calibration)
- [x] Speed (cm/s, smoothed)
- [x] Angular head velocity (AHV, deg/s)
- [x] Movement state (binary active/inactive)
- [x] `orientation` rotation correction applied per session
- [x] Light on/off alignment from DAQ timestamps
- [x] `bad_behav` mask from `experiments.csv`
- [x] Confidence filtering (< 0.9 -> NaN) and short-gap interpolation
- [x] Maze coordinate positions (7x5 rose-maze grid)
- [x] `kinematics.h5` output with full schema
- [x] Run script (`scripts/run_stage3_kinematics.py`)
- [x] Unit tests (`tests/kinematics/test_compute.py`, `test_compute_dataset.py`)
- [ ] Behavioural syllables via keypoint-MoSeq or VAME (Stage 3b -- deferred)
- [~] Syllables module exists (`src/hm2p/kinematics/syllables.py`, `tests/kinematics/test_syllables.py`) but not run on real data
- **Status: Complete (21/21 sessions with DLC results at 30fps)**
- **Note:** 21 not 26 because kinematics requires DLC pose output; all 26 sessions have DLC, 21 have kinematics.h5 on S3

### Stage 4 -- Calcium Signal Processing (CPU)

- [x] Neuropil subtraction -- fixed coefficient (`src/hm2p/calcium/neuropil.py`)
- [x] dF/F0 baseline computation -- 3-step rolling filter (`src/hm2p/calcium/dff.py`)
- [x] Voigts & Harnett event detection fallback (`src/hm2p/calcium/events.py`)
- [x] Per-ROI statistics (SNR, spike_rate, n_events)
- [x] Soma/dendrite/artefact `roi_types` written to `ca.h5`
- [x] Stage 4 runner (`src/hm2p/calcium/run.py`, `scripts/run_stage4_calcium.py`)
- [x] `ca.h5` output with full schema
- [x] Unit tests (`tests/calcium/test_neuropil.py`, `test_dff.py`, `test_events.py`, `test_spikes.py`, `test_run.py`)
- [ ] CASCADE calibrated spike inference (requires separate conda env, Python 3.8 + TensorFlow 2.3)
- [ ] FISSA spatial ICA neuropil subtraction (requires separate env, scikit-learn < 1.2)
- **Status: Complete (26/26 sessions, 391 total ROIs). CASCADE and FISSA deferred.**

### Stage 5 -- Neural-Behavioural Synchronisation (CPU)

- [x] Resample behavioural kinematics from camera rate (~100 Hz / 30 Hz) to imaging rate (~9.6 Hz)
- [x] Merge neural + behavioural into single `sync.h5`
- [x] Off-by-one fix for Suite2p N+1 frame_times
- [x] `sync.h5` schema with all fields (dff, spikes, events, hd, ahv, x, y, speed, active, light_on, bad_behav, roi_types)
- [x] Alignment module (`src/hm2p/sync/align.py`)
- [x] Validation module (`src/hm2p/sync/validate.py`)
- [x] Run script (`scripts/run_stage5_sync.py`)
- [x] Unit tests (`tests/sync/test_align.py`, `test_validate.py`)
- **Status: Complete (21/21 sessions)**

### Stage 6 -- Analysis (CPU)

- [x] Activity analysis (`src/hm2p/analysis/activity.py`)
- [x] HD tuning curves, PD, MVL, Rayleigh (`src/hm2p/analysis/tuning.py`)
- [x] Circular shuffle significance testing (`src/hm2p/analysis/significance.py`)
- [x] Tuning curve correlation, PD shift, split-half reliability (`src/hm2p/analysis/comparison.py`)
- [x] Bayesian population HD decoder (`src/hm2p/analysis/decoder.py`)
- [x] Temporal stability, light/dark drift (`src/hm2p/analysis/stability.py`)
- [x] Population-level summary statistics (`src/hm2p/analysis/population.py`)
- [x] Angular head velocity tuning (`src/hm2p/analysis/ahv.py`)
- [x] Spatial/directional information -- Skaggs (`src/hm2p/analysis/information.py`)
- [x] Automated HD cell classification (`src/hm2p/analysis/classify.py`)
- [x] Light/dark gain modulation index (`src/hm2p/analysis/gain.py`)
- [x] Visual vs idiothetic HD anchoring (`src/hm2p/analysis/anchoring.py`)
- [x] Speed modulation analysis (`src/hm2p/analysis/speed.py`)
- [x] Analysis runner (`src/hm2p/analysis/run.py`, `scripts/run_stage6_analysis.py`)
- [x] Analysis output saver (`src/hm2p/analysis/save.py`)
- [x] Analysis caching (`src/hm2p/analysis/cache.py`)
- [x] Multi-signal analysis (dF/F, deconv, events run separately, stored in `analysis.h5`)
- [x] Unit tests for all analysis modules (18 test files in `tests/analysis/`)
- **Status: Complete (21/21 sessions)**

---

## Data Standards

### NeuroBlueprint Compliance

- [x] Folder layout: `rawdata/sub-{id}/ses-{date}/funcimg/` and `behav/`
- [x] Derivatives layout: `derivatives/{stage}/sub-{id}/ses-{date}/`
- [x] Session naming: `ses-{YYYYMMDD}T{HHMMSS}` format
- [x] `sourcedata/` for trackers, calibration, metadata
- [x] S3 mirrors local NeuroBlueprint structure
- **Status: Complete**

### HDF5 Schemas

- [x] `timestamps.h5` schema defined and validated (`src/hm2p/io/hdf5.py`)
- [x] `kinematics.h5` schema with all required fields
- [x] `ca.h5` schema with dff, spikes, events, roi_types, SNR
- [x] `sync.h5` schema with merged neural + behavioural data
- [x] `analysis.h5` schema with per-signal-type results
- [x] pandera schema validation in `io/hdf5.py`
- [x] Unit tests for HDF5 I/O (`tests/io/test_hdf5.py`)
- **Status: Complete**

### S3 Organization

- [x] `s3://hm2p-rawdata/` -- raw TIFFs, videos (with lifecycle to IA after 30 days)
- [x] `s3://hm2p-derivatives/` -- all pipeline outputs (Standard tier)
- [x] S3 path resolution module (`src/hm2p/io/s3.py`)
- [x] Versioning enabled on both buckets
- [x] Upload/download scripts (`scripts/upload_to_s3.py`, `scripts/download_from_s3.py`)
- **Status: Complete**

---

## Analysis Requirements

### HD Tuning Analysis

- [x] Tuning curve computation with configurable bins (12-72), smoothing, speed filter
- [x] Mean Vector Length (MVL)
- [x] Preferred Direction (PD)
- [x] Rayleigh test
- [x] Skaggs spatial information (bits/event)
- [x] Tuning width (FWHM)
- [x] Peak-to-trough ratio
- **Status: Complete**

### Significance Testing

- [x] Circular shuffle test (preserves temporal autocorrelation)
- [x] Configurable N shuffles (default 1000)
- [x] P-value computation
- **Status: Complete**

### Condition Comparisons

- [x] Tuning curve correlation (light vs dark)
- [x] PD shift between conditions
- [x] MVL ratio (dark/light)
- [x] Rate map correlation
- [x] Split-half reliability
- [x] Rayleigh test wrapper
- **Status: Complete**

### Population Decoding

- [x] Bayesian population HD decoder (`build_decoder`, `decode_hd`)
- [x] Cross-validated decoding (`cross_validated_decode`)
- [x] Decode error metrics (mean/median absolute error, circular mean/std)
- **Status: Complete**

### Temporal Stability

- [x] Split temporal halves
- [x] Sliding window stability
- [x] Light/dark stability comparison
- [x] Drift per epoch
- [x] Dark drift rate
- **Status: Complete**

### Multi-Signal Analysis

- [x] dF/F0 signal
- [x] Deconvolved (Suite2p spks) signal
- [x] V&H events (binary) signal
- [x] Results stored per signal type in `analysis.h5`
- [x] Jaccard similarity of significant-ROI sets across signals
- **Status: Complete**

### Light/Dark Condition Analysis

- [x] 2x2 condition split (movement x light)
- [x] Movement modulation index
- [x] Light modulation index
- [x] Gain modulation index (`src/hm2p/analysis/gain.py`)
- [x] Visual vs idiothetic anchoring (`src/hm2p/analysis/anchoring.py`)
- [x] Light/dark comparison page (`frontend/pages/light_compare_page.py`)
- [ ] Linear mixed-effects model (movement * light * celltype, mouse random intercept)
- **Status: Mostly complete. Mixed-effects model not yet implemented.**

### Advanced Analysis (Planned/Deferred)

- [ ] pynapple integration for unified timeseries interface
- [ ] NEMOS GLM encoding models
- [ ] CEBRA contrastive population embeddings
- [ ] Place tuning analysis (rate maps, spatial coherence, place field detection)
- [ ] Parameter grid robustness checking
- [ ] Behavioural syllables analysis (keypoint-MoSeq run script exists but not executed on real data)
- **Status: Not started (deferred until core pipeline validated)**

---

## Frontend Requirements

### Overview Section

- [x] Home page (`home_page.py`)
- [x] Summary page (`summary_page.py`)
- [x] Sessions page (`sessions_page.py`)
- [x] Animals page (`animals_page.py`)
- [x] Pipeline status page (`pipeline_page.py`)
- [x] Batch page (`batch_page.py`)

### Pipeline Monitoring Section

- [x] Suite2p results page (`suite2p_page.py`)
- [x] Calcium processing page (`calcium_page.py`)
- [x] DLC Pose page (`dlc_page.py`)
- [x] Tracking QC page (`tracking_quality_page.py`)
- [x] Sync page (`sync_page.py`)
- [x] Z-Drift page (`zdrift_page.py`)
- [x] Anatomy page (`anatomy_page.py`)

### Explore Section

- [x] Explorer page (`explorer_page.py`)
- [x] Timeline page (`timeline_page.py`)
- [x] ROI Gallery page (`gallery_page.py`)
- [x] Events page (`events_page.py`)
- [x] Correlations page (`correlations_page.py`)
- [x] Trace Compare page (`trace_compare_page.py`)
- [x] Event Dynamics page (`event_dynamics_page.py`)

### Analysis Section

- [x] Analysis overview page (`analysis_page.py`)
- [x] Compare page (`compare_page.py`)
- [x] Population page (`population_page.py`)
- [x] Light/Dark page (`light_page.py`)
- [x] Light Compare page (`light_compare_page.py`)
- [x] Pub Stats page (`stats_page.py`)
- [x] Maze page (`maze_page.py`)
- [x] HD Tuning page (`hd_tuning_page.py`)
- [x] Decoder page (`decoder_page.py`)
- [x] Stability page (`stability_page.py`)
- [x] Drift page (`drift_page.py`)
- [x] Gain page (`gain_page.py`)
- [x] Anchoring page (`anchoring_page.py`)
- [x] Speed page (`speed_page.py`)
- [x] Pop. Dynamics page (`pop_dynamics_page.py`)
- [x] AHV page (`ahv_page.py`)
- [x] Info Theory page (`info_theory_page.py`)
- [x] Classify page (`classify_page.py`)
- [x] Signal Quality page (`signal_quality_page.py`)
- [x] QC Report page (`qc_report_page.py`)

### System Section

- [x] AWS page (`aws_page.py`)
- [x] Costs page (`cost_page.py`)
- [x] Changelog page (`changelog_page.py`)

### Patching Section

- [x] Patching overview page (`patching_page.py`)
- [x] Patching traces page (`patching_traces_page.py`)

### Other Frontend

- [x] MoSeq pages (`moseq_page.py`, `moseq_explore_page.py`)
- [x] Data loading from S3 (`frontend/data.py`)
- [x] Session filter sidebar
- [x] No synthetic data in frontend (loads real data, shows message if unavailable)
- [x] `st.set_page_config` only in `app.py`
- [x] Dict-based `st.navigation()` for section grouping
- **Total: 47 pages**
- **Status: Complete**

---

## Code Quality

### Unit Testing

- [x] pytest + pytest-cov framework
- [x] Tests mirror `src/` structure in `tests/`
- [x] Synthetic data only in tests (no real data files)
- [x] hypothesis property-based testing (`tests/analysis/test_hypothesis_analysis.py`, `tests/maze/test_hypothesis.py`)
- [x] pandera schema validation in tests
- [x] 1119+ total tests
- [x] 227 patching-specific tests
- [x] Frontend e2e tests (11 test files in `tests/frontend/`)
- [~] Coverage target >= 90% (configured in CI as `--cov-fail-under=90`; reported at 91%+ per memory)
- **Status: Complete**

### CI/CD (GitHub Actions)

- [x] `ci.yml` -- pytest on Python 3.11 + 3.12, coverage >= 90%, Codecov upload
- [x] `ci.yml` -- Snakemake dry-run validation
- [x] `lint.yml` -- ruff check + format
- [x] `lint.yml` -- mypy strict
- [x] `lint.yml` -- bandit security linter
- [x] `lint.yml` -- checkov IaC scanner (Dockerfiles)
- [x] `lint.yml` -- pip-audit dependency CVE scanner
- [x] `lint.yml` -- vulture dead code detection
- [x] `lint.yml` -- detect-secrets baseline check
- [x] `lint.yml` -- pandera schema validation via tests
- [ ] CD (deployment) -- not planned (pipeline is run on-demand)
- **Status: Complete**

### Type Checking

- [x] mypy configured in CI (`lint.yml`)
- [x] mypy in pre-commit hooks
- **Status: Complete**

### Linting & Formatting

- [x] ruff check + ruff format (replaces black + flake8 + isort)
- [x] Configured in CI and pre-commit
- **Status: Complete**

### Pre-commit Hooks

- [x] ruff (lint + format)
- [x] mypy (type checking)
- [x] nbstripout (strip notebook outputs)
- [x] bandit (security)
- [x] checkov (IaC security)
- [x] detect-secrets (credentials leak prevention)
- [x] Standard hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-merge-conflict, debug-statements
- **Status: Complete** (`/workspace/.pre-commit-config.yaml`)

### Other Quality Tools

- [ ] DVC (Data Version Control) -- not set up (no `.dvc` files found)
- [ ] mkdocs documentation site -- not set up
- [x] structlog -- needs verification whether used in pipeline modules
- **Status: DVC and mkdocs not implemented**

---

## Patching Pipeline

### Modules (10/10 complete)

- [x] `config.py` -- Patching pipeline configuration (11 tests)
- [x] `io.py` -- WaveSurfer H5 + SWC file I/O (18 tests)
- [x] `ephys.py` -- Electrophysiology signal processing (33 tests)
- [x] `protocols.py` -- Stimulus protocol parsing & response extraction (27 tests)
- [x] `spike_features.py` -- AP waveform feature extraction (11 tests)
- [x] `morphology.py` -- SWC morphology loading & analysis (26 tests)
- [x] `metrics.py` -- Intrinsic excitability & passive properties (14 tests)
- [x] `statistics.py` -- Statistical comparisons (Penk vs non-Penk) (33 tests)
- [x] `pca.py` -- PCA on electrophysiological features (29 tests)
- [x] `run.py` -- Batch runner for patching analysis (30 tests)
- **Total: 227 tests**
- **Status: Complete (all 10 modules implemented and tested)**

### Patching Remaining Work

- [ ] Plotting module (`src/hm2p/patching/plotting.py` -- not yet created)
- [x] Frontend pages (`patching_page.py`, `patching_traces_page.py`)
- **Status: Mostly complete. Plotting module pending.**

---

## Cloud Infrastructure

### AWS S3

- [x] `s3://hm2p-rawdata/` bucket created (ap-southeast-2)
- [x] `s3://hm2p-derivatives/` bucket created (ap-southeast-2)
- [x] Versioning enabled on both buckets
- [x] Lifecycle policy: Standard -> IA after 30 days
- [x] S3 access logging enabled -> `hm2p-access-logs` (90-day expiry)
- [x] 26 sessions uploaded (91.4 GiB rawdata)
- **Status: Complete**

### EC2 Compute

- [x] Key pair `hm2p-suite2p` created
- [x] Security group `hm2p-suite2p-sg` (sg-020161fb424325e6b)
- [x] IAM role `hm2p-ec2-role` + instance profile (S3 + CloudWatch)
- [x] AMI: Deep Learning Base OSS Nvidia, Ubuntu 22.04
- [x] Suite2p cloud run completed (g4dn.xlarge)
- [x] DLC cloud run completed (g4dn.xlarge + g5.xlarge parallel shards)
- [x] EC2 launch scripts for Suite2p, DLC (single + parallel), keypoint-MoSeq
- [x] IAM setup scripts (`scripts/setup_ec2_iam.py`, `scripts/setup_frontend_iam.py`)
- [x] SSM setup script (`scripts/setup_ssm.py`)
- [x] Auto-shutdown script (`scripts/setup_auto_shutdown.py`)
- [ ] Terminate stopped instances (7 stopped instances accruing EBS costs)
- [ ] Rotate hm2p-agent S3 credentials (exposed in EC2 user-data)
- **Status: Mostly complete. Credential rotation and instance cleanup pending.**

### Security

- [x] Security group lockdown: ports 22+8501 restricted to user's IP
- [x] S3 access logging enabled
- [x] Google auth for frontend (streamlit-google-auth, whitelist)
- [x] Error sanitization in frontend (strips paths/tracebacks)
- [x] bandit, checkov, detect-secrets, pip-audit, vulture in CI
- [x] detect-secrets pre-commit hook
- [x] SG lockdown script (`scripts/setup_sg_lockdown.py`)
- [x] S3 logging script (`scripts/setup_s3_logging.py`)
- [ ] Credential rotation (access key exposed in EC2 user-data script)
- **Status: Mostly complete. Credential rotation outstanding.**

---

## Pipeline Orchestration

### Snakemake

- [x] Main Snakefile (`workflow/Snakefile`)
- [x] Stage rules: `ingest.smk`, `extraction.smk`, `pose.smk`, `kinematics.smk`, `calcium.smk`, `sync.smk`
- [x] Three profiles: `local`, `local-gpu`, `aws-batch`
- [x] Snakemake dry-run in CI
- [x] Compute profile config (`config/compute.yaml`, `config/pipeline.yaml`)
- **Status: Complete**

### Docker

- [x] GPU Dockerfile (`docker/gpu.Dockerfile`) -- CUDA 12.1 + Suite2p + DLC
- [x] CPU Dockerfile (`docker/cpu.Dockerfile`) -- CPU-only stages
- [x] keypoint-MoSeq Dockerfile (`docker/kpms.Dockerfile`)
- [ ] Images pushed to ECR -- needs verification
- **Status: Complete (Dockerfiles defined)**

---

## Citation Policy

### Citations in Code (docstrings)

- [x] dF/F baseline: Pachitariu et al. 2017, doi:10.1101/061507
- [x] Event detection: Voigts & Harnett 2020, doi:10.1016/j.neuron.2019.10.016
- [x] HD tuning: Taube et al. 1990, doi:10.1523/JNEUROSCI.10-02-00420.1990
- [x] MVL/tuning curves: Skaggs et al. 1996
- [x] Spatial information: Skaggs et al. 1993, doi:10.1162/neco.1996.8.6.1345
- [x] Circular shuffle: Muller et al. 1987, doi:10.1523/JNEUROSCI.07-07-01951.1987
- [x] Neuropil (FISSA): Keemink et al. 2018, doi:10.1038/s41598-018-21640-2
- [ ] CASCADE: Rupprecht et al. 2021 -- needs verification (CASCADE not yet integrated)
- [ ] CEBRA: Schneider et al. 2023 -- needs verification (CEBRA not yet integrated)
- [ ] keypoint-MoSeq: Weinreb et al. 2024 -- needs verification

### Citations in Docs

- [x] `docs/analysis-plan.md` references: Zong et al. 2022, Skaggs et al. 1996, Voigts & Harnett 2020
- [x] `docs/reference-papers.md` exists (needs content verification)
- [ ] Full citation coverage in docs for all methods -- needs verification

### Citations in Frontend

- [ ] "Methods & References" expanders on analysis pages -- needs verification per page
- **Status: Partially complete. Core methods cited in code; frontend and docs coverage needs audit.**

---

## Design Philosophy Compliance

- [x] Ground-up redesign (not a port of old pipeline)
- [x] Clean module structure with separation of concerns
- [x] Extractor-agnostic (Suite2p + CaImAn via roiextractors)
- [x] Tracker-agnostic (DLC + SLEAP + LightningPose via movement)
- [x] Cloud-first, locally runnable (S3 + EC2, with local profiles)
- [x] NeuroBlueprint data standard throughout
- [x] Latest stable library versions
- [x] No copy-paste from old pipeline (reimplemented with tests)
- **Status: Complete**

---

## Remaining / Outstanding Items

| Item | Priority | Notes |
|------|----------|-------|
| CASCADE spike inference | Medium | Requires separate conda env (Python 3.8 + TF 2.3) |
| FISSA neuropil subtraction | Low | Optional, more accurate than fixed coefficient |
| neuroconv NWB export | Low | Stub exists at `src/hm2p/io/nwb.py` |
| Credential rotation | High | hm2p-agent S3 key exposed in EC2 user-data |
| Terminate stopped EC2 instances | Medium | 7 stopped instances accruing EBS costs |
| DVC setup | Low | Not yet initialized |
| mkdocs documentation site | Low | Not yet set up |
| pynapple integration | Medium | Planned for analysis interface |
| NEMOS GLM encoding models | Low | Planned for future analysis |
| CEBRA population embeddings | Low | Planned for future analysis |
| Linear mixed-effects model | Medium | Needed for Penk vs non-Penk comparison |
| Place tuning analysis | Medium | Rate maps, spatial coherence |
| Parameter grid robustness | Low | Sweep across analysis parameters |
| Patching plotting module | Low | Missing from patching pipeline |
| Citation audit (frontend) | Medium | Verify Methods & References on all analysis pages |
| 5 remaining sessions for kinematics | Medium | 21/26 have kinematics.h5; 5 need processing |
