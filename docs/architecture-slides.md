# hm2p Architecture — Slide Deck

*For a neuroscientist audience. Precise terminology, file paths, module names.*
*Last updated: 2026-05-14.*

---

## Slide 1: The Experiment

- **Animal:** Freely-moving mouse in a Rosenberg q-rose maze (7 x 5 unit grid, 23 accessible cells, 6 dead-end arms)
- **Brain region:** Retrosplenial cortex (RSP) and adjacent cortex — prominent head-direction (HD) cells
- **Imaging:** Single-plane two-photon GCaMP7f calcium imaging at ~9.6 Hz (SciScan resonant scanner, 512 x 512 px)
- **Behaviour:** Overhead video at ~100 fps (Basler acA1300-200um), DAQ-synchronised to imaging
- **Light manipulation:** Room lights cycle 1 min on / 1 min off — light off = total darkness = complete visual cue removal
- **Two non-overlapping RSP populations:**
  - **Penk+**: Penk-Cre mouse + AAV-ADD3 (Cre-ON) — labels Penk-expressing neurons
  - **Penk-minus CamKII+**: Penk-Cre mouse + virus 344 (Cre-OFF intersectional) — Cre blocks expression in Penk+ cells; only non-Penk CamKII+ neurons labelled
- **Science questions:** (1) HD tuning in each population, (2) visual vs idiothetic anchoring, (3) population-level HD decoding, (4) cell-type differences
- **Dataset:** 26 sessions across 9 animals; ~113 GB raw data (excluding side camera and red channel)

**Speaker notes:**
The Rosenberg maze provides a complex spatial environment with dead-end arms that force the mouse to make directional decisions. The light manipulation is the key experimental variable — by removing all visual cues in total darkness, we test whether RSP HD cells maintain their tuning through path integration alone or require visual landmarks. The two cell populations are genetically defined and non-overlapping: Penk-Cre drives GCaMP in Penk+ cells via the ADD3 Cre-ON virus, while the Cre-OFF intersectional virus 344 blocks expression in Penk+ cells so only non-Penk CamKII+ neurons are labelled. Both use GCaMP7f. The `celltype` column in `metadata/animals.csv` is `"penk"` or `"nonpenk"`. Single-plane imaging means soma and dendrite ROIs coexist in one plane — they are separated post-hoc by morphological classification, not by a second imaging plane.

---

## Slide 2: Raw Data — What Comes Off the Rig

- **Two-photon TIFFs:** `.raw` (SciScan) converted to `.tif` stacks; one per session; green (functional) + red (anatomical reference)
- **DAQ timing file:** `.tdms` file with trigger pulses — camera frames, SciScan line clock, light on/off pulses
- **Overhead video:** `.mp4` (H.264), pre-processed (undistorted + cropped by legacy pipeline); ready for tracker inference
- **Video metadata:** `meta/meta.txt` per session — crop region, scale (mm/px), maze ROI corner coordinates
- **SciScan metadata:** `.meta.txt` — frame rate (`frames.p.sec`), DAQ channel map, imaging parameters
- **Camera calibration:** lens-specific `.npz` files (4 mm and 6 mm lenses) for undistortion
- **Z-stacks:** serial2p multi-page TIFFs for z-drift estimation (16 of 26 sessions; 13 unique z-stacks)
- **Whole-brain volumes:** post-mortem serial2p (25 um isotropic), registered to Allen CCFv3 via brainreg

**Speaker notes:**
The raw-to-TIFF conversion (`raw2tif`) happens once and the TIFFs are the starting point for the pipeline. The `.tdms` DAQ file is critical — it provides the only precise timing link between the camera and the microscope. All 26 sessions have pre-processed videos already (undistorted and cropped), so the `.mp4` files in `rawdata/.../behav/` can go directly to DLC. The side camera (`_side_left.camera.mp4`) is never used. Video metadata (`meta/meta.txt`) contains the pixel-to-mm scale factor and the maze ROI corners needed for coordinate transformation. The z-stacks are used for the optional z-drift analysis (Stage 1b, `extraction/zdrift.py`) — they let us quantify how much the focal plane shifts during a session.

---

## Slide 3: Data Flow — From Raw to Analysis-Ready

```text
Raw Data (Dropbox -> S3)
  |
  v Stage 0 -- Ingest & DAQ Parse (CPU)
  |  TDMS -> timestamps.h5 (frame times, light pulses)
  |
  +---------------------------+
  v                           v
Stage 1 -- 2P Extraction    Stage 2a -- DLC Training (GPU)
  Suite2p + Cellpose 3        Fine-tune SuperAnimal model
  -> ca_extraction/           -> dlc_training/models/
  |                           |
  |                           v
  |                         Stage 2b -- DLC Inference (GPU)
  |                           DLC / SLEAP / LightningPose
  |                           -> pose/ (8 bodyparts per frame)
  |                           |
  v Stage 4 -- Calcium        v Stage 3 -- Kinematics (CPU)
  |  Neuropil -> dF/F ->        movement library
  |  CASCADE spike rates        HD, position, speed, AHV
  |  -> calcium/ca.h5          -> movement/kinematics.h5
  |                           |
  +-------------+-------------+
                |
                v Stage 5 -- Sync (CPU)
                |  Resample behaviour -> imaging frame times
                |  -> sync/sync.h5
                |
                v Stage 6 -- Analysis (CPU)
                   HD tuning, decoding, anchoring,
                   stability, gain, maze, NaviGraph
                   -> analysis/analysis.h5
```

- **Parallel branches:** Stages 1 and 2 are independent — calcium extraction and pose estimation run in parallel
- **Convergence at Stage 5:** sync.h5 merges both branches at the imaging frame rate
- **Stage 4 is independent of pose:** re-running DLC does NOT require re-running calcium processing

**Speaker notes:**
The key insight in this data flow is the two independent branches. The calcium branch (Stages 1, 4) and the pose branch (Stages 2, 3) can run in any order. They converge at Stage 5 (Sync), which resamples the ~100 Hz behavioural kinematics down to the ~9.6 Hz imaging rate using linear interpolation at each 2P frame timestamp. This means if you re-run DLC (Stage 2), you need to re-run Stages 3, 5, and 6 — but NOT Stage 4. The `scripts/run_downstream_pipeline.py` script automates this: it runs Stages 3 -> 3b -> 5 -> 6 in sequence. All intermediate files are HDF5 format. The final `sync.h5` is the analysis-ready file that loads directly into pynapple.

---

## Slide 4: S3 Storage Layout

```text
s3://hm2p-rawdata/
  rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/
    funcimg/    -- 2P TIFF stacks + .meta.txt
    behav/      -- .mp4 video + meta/ folder + .tdms
  sourcedata/
    trackers/dlc/          -- DLC model weights + labeled data
    calibration/           -- camera .npz files
    metadata/              -- animals.csv, experiments.csv
    zstacks/{zstack_id}/   -- serial2p z-stacks

s3://hm2p-derivatives/
  derivatives/
    ca_extraction/{sub}/{ses}/suite2p/  -- Suite2p native output
    pose/{sub}/{ses}/                   -- DLC .h5 files
    movement/{sub}/{ses}/kinematics.h5  -- HD, position, speed
    calcium/{sub}/{ses}/ca.h5           -- dF/F, spikes, SNR
    sync/{sub}/{ses}/sync.h5            -- merged neural + behav
    analysis/{sub}/{ses}/analysis.h5    -- tuning, decoding
  dlc-retrain/                          -- training data + models
  dlc-champion.json                     -- current champion model
  dlc-champion-history/                 -- superseded champions
```

- **NeuroBlueprint** folder standard (BIDS-inspired, from SWC/UCL neuroinformatics)
- Session names: `ses-{YYYYMMDD}T{HHMMSS}` — full timestamp because multiple sessions per day exist
- Two S3 buckets: `hm2p-rawdata` (Infrequent Access) and `hm2p-derivatives` (Standard)
- Storage cost: ~$10/month (600 GB raw + ~150 GB derivatives)

**Speaker notes:**
NeuroBlueprint is a BIDS-inspired folder specification designed for systems neuroscience experiments. It uses `sub-` and `ses-` prefixes throughout. DataShuttle (also from the SWC neuroinformatics unit) handles upload and validation. The separation into two buckets allows different storage tiers — raw data rarely needs re-accessing so it goes to Infrequent Access ($0.0125/GB/month), while derivatives are accessed frequently by the frontend so they stay on Standard ($0.023/GB/month). The `dlc-champion.json` at the bucket root is the single source of truth for which DLC model is current — every derivative that depends on pose data records its `dlc_champion_id` attribute for staleness detection.

---

## Slide 5: Stage 0 — DAQ Parsing

- **Input:** `daq.tdms` per session (NI DAQ binary format)
- **Parser:** `nptdms` library — isolates the TDMS dependency to one module (`src/hm2p/ingest/daq.py`)
- **Output:** `timestamps.h5` with:
  - `frame_times_camera` (N,) float64 — camera trigger timestamps (seconds since session start)
  - `frame_times_imaging` (T,) float64 — 2P frame timestamps from SciScan line clock
  - `light_on_times` / `light_off_times` (L,) float64 — lighting pulse edges
  - `fps_camera`, `fps_imaging` attributes
- **Validation:** `ingest/validate.py` checks raw file completeness per session
- **Runner:** `scripts/run_stage0_daq.py`
- **Status:** 26/26 sessions complete

**Speaker notes:**
The DAQ file is the timing backbone of the entire pipeline. The SciScan microscope and the Basler camera are both triggered by the DAQ — their frame times are extracted from trigger pulse edges in the TDMS data. The camera runs at ~100 fps and the microscope at ~9.6 Hz. The light on/off times are extracted from a separate DAQ channel that records the room light control signal. All downstream stages read clean HDF5 timestamps rather than parsing TDMS directly. This is a deliberate isolation of the nptdms dependency — if the DAQ format changes, only `daq.py` needs updating.

---

## Slide 6: Stage 1 — Two-Photon Extraction

- **Input:** TIFF stacks from `rawdata/.../funcimg/`
- **Default extractor:** Suite2p (with Cellpose 3 anatomical prior, `anatomical_only=2`)
- **Alternative:** CaImAn (CNMF-based) — implemented in `extraction/caiman.py`
- **Unified API:** `roiextractors` (CatalystNeuro) — `seg.get_traces("raw")`, `seg.get_traces("neuropil")`, `seg.get_roi_image_masks()`
- **Cellpose 3 anatomical prior:** seeds ROI candidates from mean/max projection, biasing toward compact soma morphologies before activity statistics refine the set
- **Soma vs dendrite classification:** post-hoc from `stat.npy` shape statistics (aspect ratio, radius, compactness)
  - `extraction/soma_features.py` — per-ROI feature extraction
  - `extraction/soma_classifier.py` — rule-based scorer (default) or logistic regression (trainable via `scripts/train_soma_classifier.py`)
  - Output: `roi_types` array (0=soma, 1=dendrite, 2=artefact) + calibrated probabilities (`p_soma`, `p_dend`, `p_artefact`)
- **Manual ROI curation:** `extraction/curation.py` — append-only label CSV, latest-timestamp-wins resolution
- **Z-drift:** `extraction/zdrift.py` — register imaging frames against serial2p z-stacks
- **Status:** 26/26 sessions processed on EC2 g4dn.xlarge

**Speaker notes:**
The choice to use Cellpose 3 as an anatomical prior (mode 2) is deliberate. In single-plane RSP imaging, dendritic processes pass through the focal plane and look similar to small somata in activity space. The Cellpose prior biases detection toward compact, roughly circular morphologies, reducing false soma detections. Activity statistics then refine the candidate set. The alternative (mode 0, activity-only) is available as a fallback. The soma classifier framework allows both rule-based classification (reproducing the legacy hand-tuned thresholds) and a trained logistic regression model. Manual curation via the ROI Curation frontend page (`roi_curation_page.py`) writes to an append-only CSV — re-labelling never overwrites previous labels, so full provenance is preserved. The roiextractors API means all downstream calcium processing code is extractor-agnostic — switching from Suite2p to CaImAn requires only changing the `extractor` field in `experiments.csv`.

---

## Slide 7: Stage 2 — DLC Pose Tracking

- **Current model:** DeepLabCut 3.0 (PyTorch) + SuperAnimal TopViewMouse + HRNet-W32 + FasterRCNN detector
- **8 tracked bodyparts:** `nose_tip`, `left_ear`, `right_ear`, `head_midpoint`, `neck`, `mid_back`, `mouse_center`, `tail_base`
  - All except `head_midpoint` map to SuperAnimal TopViewMouse keypoints
  - `head_midpoint` is a custom keypoint (high-contrast 2P headstage visible on overhead video)
  - Legacy DLC files use the old name `implant_base_rear` — both names are handled
- **Stage 2a — Training (GPU, 24h max):**
  - Fine-tune SuperAnimal on manually labeled hm2p frames
  - Training data: 183 labeled frames across sessions in `sourcedata/trackers/dlc/*/labeled-data/`
  - Launched via `scripts/launch_dlc_finetune_ec2.py` -> EC2 g4dn/g5 instance
  - W&B logging for training metrics (`scripts/upload_runs_to_wandb.py`)
  - Memory replay patch (`scripts/patch_dlc_memory_replay.py`) fixes DLC 3.0rc13/rc14 bug
- **Stage 2b — Inference (GPU):**
  - `deeplabcut.analyze_videos()` on all 26 session videos
  - Output: `.h5` per session in `derivatives/pose/{sub}/{ses}/`
- **Pluggable:** SLEAP and LightningPose also supported via `movement` unified loader
- **Status:** 26/26 sessions complete

**Speaker notes:**
The SuperAnimal TopViewMouse model provides a strong prior for common mouse bodyparts, but it does not include the head midpoint keypoint (which corresponds to the 2P headstage mounting point). This keypoint is trained from scratch during fine-tuning and is easy to detect because the headstage is a high-contrast dark object on the mouse's head. The HRNet-W32 architecture was chosen over ResNet because it maintains high spatial resolution throughout the network. The FasterRCNN detector handles animal detection before pose estimation. Fine-tuning uses the SA transfer learning pathway. A known bug in DLC 3.0rc13/rc14 causes a KeyError when the detector finds no animal in a frame during memory replay — the `patch_dlc_memory_replay.py` script patches this at runtime on EC2. The 24-hour hard timeout and GPU utilization watchdog (terminates if 0% GPU for 5 consecutive minutes) prevent runaway costs.

---

## Slide 8: DLC Retraining Workflow

```text
1. Tracking QC page -> identify bad frames -> export frame indices
                                              |
2. Mac: scripts/prepare_retrain_frames.py     v
        Downloads video, extracts PNGs, creates DLC project
                                              |
3. Mac: deeplabcut.label_frames('...')        v
        Manually label bodyparts in napari GUI
                                              |
4. Mac: scripts/upload_dlc_labels.py          v
        Uploads labeled-data + config.yaml to S3
                                              |
5. Mac: scripts/launch_dlc_finetune_ec2.py    v
        Launches g4dn/g5 EC2 instance which runs:
          a. Download labels from S3
          b. Create training dataset (SA transfer weights)
          c. Train network (DLC 3.0 PyTorch, up to 50k iters)
          d. Re-run inference on all 26 sessions
          e. Write promoted.json per session
          f. Declare new DLC champion (dlc-champion.json)
          g. Self-terminate
                                              |
6. Frontend: compare fine-tuned vs previous in Tracking QC page
                                              |
7. Mac: scripts/run_downstream_pipeline.py    v
        Re-runs Stages 3 -> 3b -> 5 -> 6 for all 26 sessions
```

- **Frame selection tools:** `select_hard_frames.py` (PCA + k-means), `select_frames_image_clustering.py` (uncertainty-weighted clustering), `select_labelling_frames.py` (manual)
- **Per-bodypart RMSE:** `scripts/compute_bodypart_rmse.py` — compares predictions vs ground truth
- **Model comparison:** `scripts/compare_models.py` + `src/hm2p/pose/finetune.py` — non-parametric paired comparison with promotion gate

**Speaker notes:**
The DLC retraining workflow spans local Mac and cloud EC2. Frame selection uses PCA + k-means to find diverse, underrepresented frames rather than random selection. The `select_hard_frames.py` script clusters all video frames and picks centroids from clusters not yet represented in the training data. After labeling in napari, labeled data goes to S3 and triggers a cloud training run. The key safety feature is that EC2 instances self-terminate on completion (or after 24 hours, or if GPU utilisation drops to 0% for 5 minutes). The champion model system ensures that every derivative (kinematics.h5, sync.h5, analysis.h5) records which DLC model produced it, so the frontend can detect when data is stale after a model update.

---

## Slide 9: DLC Champion Model System

- **Single source of truth:** `s3://hm2p-derivatives/dlc-champion.json`
  - Contains: `champion_id` (deterministic hash of model name + architecture + snapshot), `model_name`, `architecture`, `snapshot`, `promoted_at`, `promoted_by_ec2_instance`, `promoted_by_git_sha`
- **Provenance chain:** every DLC-derived file stamps `dlc_champion_id` as an HDF5 attribute
  - `kinematics.h5` -> `sync.h5` -> `analysis.h5` (inherited through the pipeline)
  - Rendered videos get a `.provenance.json` sidecar
- **Frontend enforcement:**
  - `frontend/data.py::get_dlc_champion()` — loads manifest (cached 300s)
  - `frontend/data.py::is_session_current()` — compares session's stored champion_id vs current
  - `frontend/data.py::render_champion_staleness_warning()` — shared banner on all DLC-dependent pages
  - Pages display a warning when data is stale but never hide it — QC must remain possible
- **Key modules:** `src/hm2p/pose/select.py` (compute_champion_id, get_champion_manifest), `scripts/declare_dlc_champion.py`
- **History:** superseded champions archived to `dlc-champion-history/`

**Speaker notes:**
The champion model system solves a fundamental provenance problem: when you retrain DLC, all downstream derivatives become stale. Without tracking which model produced which output, you cannot know whether your analysis.h5 reflects the current or a previous model. The champion_id is a deterministic string computed from the model name, architecture, and snapshot — it changes only when the model itself changes. The frontend enforces this by checking every session's stored champion_id against the current manifest before displaying analysis data. Stale sessions show a prominent warning banner but are never hidden, because you may want to compare old vs new results during QC. The `scripts/declare_dlc_champion.py` script is called automatically at the end of a successful training run, so no manual step is needed.

---

## Slide 10: Stage 3 — Kinematics

- **Input:** Pose `.h5` from Stage 2 (any tracker format)
- **Loader:** `movement.io.load_poses.from_file(file=path, source_software="DeepLabCut")`
  - Returns `xarray.Dataset` with dimensions `(time, individuals, keypoints, space)` + `confidence`
  - Tracker-agnostic: swap `source_software` and downstream code is unchanged
- **Processing pipeline (`kinematics/compute.py`):**
  1. Apply per-session `orientation` rotation (from `experiments.csv`) to all keypoint coordinates
  2. Rename SuperAnimal bodypart names to project names (`nose` -> `nose_tip`, etc.)
  3. Filter low-confidence detections (< 0.9 -> NaN)
  4. Interpolate short gaps (up to 5 frames) and smooth
  5. Compute head direction from ear vector (primary) + 3 QC vectors (nose-head, nose-neck, head-neck)
  6. Fuse HD estimates with confidence weighting (`hd_confidence`)
  7. Compute position (mm), speed (cm/s), AHV (deg/s), movement state
  8. Align light on/off from `timestamps.h5`
  9. Apply bad_behav mask from `experiments.csv`
  10. Compute maze coordinates (7 x 5 grid) via scale calibration + Shapely polygon clipping
  11. Per-bodypart maze coordinates (`bp_{kp}_x_maze`, `bp_{kp}_y_maze`)
- **Perspective correction:** `kinematics/perspective.py` — projects bodypart heights to ground plane, removing parallax
- **Output:** `derivatives/movement/{sub}/{ses}/kinematics.h5`
- **Runner:** `scripts/run_stage3_kinematics.py`

**Speaker notes:**
The `movement` library from the SWC/UCL neuroinformatics unit provides the unified pose loading layer. It returns the same xarray Dataset regardless of which tracker produced the data, so switching from DLC to SLEAP would require changing one string. Head direction is computed from 4 independent bodypart vectors and fused with confidence weighting — this provides robustness against individual bodypart detection failures. The orientation rotation corrects for camera placement variation across sessions (the camera is not always mounted at exactly the same angle). Perspective correction is important because the overhead camera has a finite viewing angle — bodyparts that are elevated above the maze floor (e.g., the head with the 2P headstage) appear displaced from their true ground-plane position. The Rosenberg maze is 7 x 5 units; maze coordinates are computed by projecting pixel positions through scale calibration and cropping to the maze boundary polygon using Shapely.

---

## Slide 11: Stage 4 — Calcium Processing

- **Input:** Suite2p/CaImAn native files (via roiextractors) + `timestamps.h5` + `bad_frames.npy`
- **Processing pipeline (`calcium/run.py`):**
  1. **Neuropil subtraction** (`calcium/neuropil.py`): `F_corr = F - 0.7 * Fneu` (fixed coefficient, default) or FISSA (spatial ICA, optional)
  2. **Baseline & dF/F0** (`calcium/dff.py`): sliding window minimum of Gaussian-smoothed trace
  3. **Event detection** (`calcium/events.py`): Voigts & Harnett 2020 threshold method (primary fallback), SD-threshold (Zong et al. 2022)
  4. **CASCADE spike inference** (`calcium/spikes.py`): calibrated spike rates in spikes/s from pre-trained deep-learning models matched to GCaMP indicator + frame rate
  5. **Per-ROI QC** (`calcium/qc.py`): SNR, decay tau, neuropil-dF/F correlation, bleach slope, active fraction
  6. **Neuropil contamination analysis** (`calcium/neuropil_analysis.py`): QC metrics for neuropil subtraction quality
- **Output:** `derivatives/calcium/{sub}/{ses}/ca.h5`
  - `dff` (R, T) float32 — dF/F0 per ROI per frame
  - `spikes` (R, T) float32 — CASCADE spike rate (spikes/s)
  - `event_masks` (R, T) float32 — V&H binary events
  - `event_masks_sd` (R, T) float32 — SD-threshold events
  - `deconv` / `deconv_norm` (R, T) float32 — Suite2p deconvolved spikes
  - `roi_types` (R,) uint8, `snr` (R,) float32
- **Stage 4b:** CASCADE can be re-run independently (`scripts/run_cascade.py`) without repeating neuropil/dF/F steps
- **Status:** 26/26 sessions processed; CASCADE integration deferred (requires tensorflow==2.3, Python 3.8)

**Speaker notes:**
The calcium processing chain follows the standard approach: neuropil subtraction removes the contaminating neuropil fluorescence signal that is mixed into every ROI's trace, dF/F0 normalisation expresses the signal as a fraction of baseline fluorescence, and event/spike detection identifies neural activity from the normalised traces. CASCADE (Rupprecht et al. 2021, Nature Neuroscience) is the primary spike inference method because it outputs calibrated spike rates in physical units (spikes/s), trained on ground-truth simultaneous imaging + electrophysiology datasets. The V&H threshold method is retained as a fallback. The fixed neuropil coefficient of 0.7 is the Suite2p default — FISSA (spatial ICA) is more accurate in densely labelled tissue but requires a separate environment due to dependency conflicts. Stage 4b is separated because CASCADE can be re-run with a different model (e.g., matching a different GCaMP variant or frame rate) without repeating the neuropil subtraction and dF/F computation steps.

---

## Slide 12: Stage 5 — Neural-Behavioural Synchronisation

- **Input:** `kinematics.h5` (camera rate ~100 Hz) + `ca.h5` (imaging rate ~9.6 Hz) + `timestamps.h5`
- **Core operation (`sync/align.py`):**
  - Resample all kinematics signals from camera rate to imaging frame times using linear interpolation
  - Boolean signals (`active`, `light_on`, `bad_behav`) use nearest-neighbour resampling
  - Calcium data copied verbatim (already at imaging rate)
- **Sync diagnostics (`sync/diagnostics.py`):**
  - Per-channel scalars: median ISI, MAD, CV, drift slope
  - Cross-channel: start/end offset, temporal overlap
  - Light protocol: period, duty cycle, first state
  - Session classification into 7 `sync_status` tiers (config: `config/sync.yaml`)
- **Sync report (`sync/report.py`):**
  - Aggregates all sessions into `sync_report.parquet` (one row per session, attrs only — fast)
  - Frontend: `frontend/pages/sync_report_page.py` + `frontend/components/sync_diag.py`
- **Output:** `derivatives/sync/{sub}/{ses}/sync.h5`
  - All kinematics fields resampled to (T,) imaging rate
  - All calcium fields (R, T) copied unchanged
  - DLC provenance triplet: `dlc_model_name`, `dlc_snapshot`, `dlc_champion_id` (inherited from kinematics.h5)
- **Runner:** `scripts/run_stage5_sync.py`

**Speaker notes:**
The sync stage is where the two independent data streams (neural and behavioural) converge into a single time-aligned file. The key challenge is that the camera runs at ~100 fps while imaging runs at ~9.6 Hz — a ~10x difference. Linear interpolation at each imaging frame timestamp is appropriate for continuous signals like position and HD, while boolean signals use nearest-neighbour. The sync diagnostics module classifies each session into quality tiers based on non-parametric statistics (medians, MAD, CV — no parametric assumptions). Sessions with timing drift, missing frames, or poor overlap are flagged automatically. The sync report aggregates these diagnostics across all 26 sessions for at-a-glance quality assessment. The sync.h5 file is the final analysis-ready dataset — it's what loads into pynapple.

---

## Slide 13: Stage 6 — Analysis

- **Input:** `sync.h5`
- **19 analysis modules** in `src/hm2p/analysis/`:

| Module | What it computes |
|---|---|
| `tuning.py` | HD tuning curves, preferred direction (PD), mean vector length (MVL), Rayleigh test |
| `significance.py` | Circular shuffle tests for HD significance |
| `classify.py` | Automated HD cell classification |
| `comparison.py` | Tuning curve correlation, PD shift, split-half reliability |
| `decoder.py` | Bayesian population HD decoder (MAE in degrees) |
| `stability.py` | Temporal stability, light/dark drift analysis |
| `gain.py` | Light/dark gain modulation index |
| `anchoring.py` | Visual vs idiothetic HD anchoring |
| `ahv.py` | Angular head velocity tuning |
| `speed.py` | Speed modulation analysis |
| `information.py` | Spatial/directional information (Skaggs, bits/spike) |
| `activity.py` | Active-cell detection, firing rate statistics |
| `population.py` | Population-level summary statistics |
| `mixed_stats.py` | Cross-module Penk+ vs CamKII+ comparisons |
| `celltype_dynamics.py` | Time-resolved population dynamics by cell type |
| `rastermap_analysis.py` | Rastermap-based neural population visualisation |
| `cache.py` | Analysis result caching utilities |
| `run.py` | Stage 6 runner: full analysis pipeline |
| `save.py` | Write analysis.h5 outputs |

- **Multi-signal:** analyses run on dF/F, deconvolved spikes, and event masks
- **Non-parametric statistics only:** Mann-Whitney U, Wilcoxon, Spearman, Kruskal-Wallis, permutation/bootstrap
- **Output:** `derivatives/analysis/{sub}/{ses}/analysis.h5`

**Speaker notes:**
Every statistical test in the analysis module is non-parametric — this is a hard rule. The data from calcium imaging violates the assumptions of parametric tests (normality, homoscedasticity) in multiple ways: spike rates are not normally distributed, tuning curve shapes vary across cells, and the sample size per cell type (Penk+ vs CamKII+) is modest. We use Mann-Whitney U for unpaired comparisons, Wilcoxon signed-rank for paired, Spearman rank for correlations, and permutation/bootstrap tests for more complex hypotheses. The circular shuffle test for HD significance shuffles timestamps to break the temporal relationship between neural activity and head direction while preserving the autocorrelation structure of both signals. The Bayesian decoder uses the population vector of all HD cells to decode head direction from neural activity — the mean absolute error (MAE) in degrees is the primary output metric.

---

## Slide 14: Maze Analysis & NaviGraph

- **Maze topology (`maze/topology.py`):** Rosenberg maze as a graph — 7 x 5 grid, adjacency matrix, dead-end identification, corridor/junction classification
- **Discretization (`maze/discretize.py`):** continuous (x, y) -> maze cell assignment using Shapely polygon containment
- **Behavioural analysis (`maze/analysis.py`):** occupancy, exploration metrics, turn bias, movement sequences, dead-end visits
- **NaviGraph-inspired neural analyses (`maze/neural.py`):**
  1. **Light/dark graph annotation:** occupancy-normalised activity per ROI per maze cell, split by light condition
  2. **Decision-point HD tuning:** HD tuning curves split by location type (junction vs corridor vs dead-end)
  3. **Path familiarity:** activity change with repeated corridor traversals
  4. **Junction choice prediction:** cross-validated logistic decoding of turn choice from pre-junction population vectors
- **Citation:** Koren Iton A et al. 2025. "NaviGraph: A graph-based framework for multimodal analysis of spatial decision-making." bioRxiv. doi:10.1101/2025.05.18.654725
- **Frontend:** `maze_page.py` (topology + occupancy), `maze_animation_page.py` (canvas trajectory playback)

**Speaker notes:**
The NaviGraph-inspired analyses go beyond standard HD tuning curves by asking how neural activity relates to maze topology. For example, does HD tuning differ at junctions (where the mouse must choose a direction) vs corridors (where the path is constrained)? Do HD cells show different activity patterns at dead-ends (where the mouse must turn around)? The junction choice prediction analysis is particularly interesting for RSP — if pre-junction population vectors predict which way the mouse will turn, this suggests RSP HD activity participates in spatial decision-making, not just passive heading representation. All functions are pure numpy with no I/O — insufficient data returns NaN rather than raising. The maze animation page uses an HTML5 Canvas component (`frontend/components/maze_canvas.py`) that renders the mouse trajectory, skeleton, and HD arrow at 60 fps without Streamlit reruns.

---

## Slide 15: Canvas Maze Animation

- **Component:** `frontend/components/maze_canvas.py` — HTML5 Canvas + JavaScript animation
- **Renders at 60 fps** without Streamlit reruns (all playback logic in JS)
- **Visual elements:**
  - Mouse trajectory trail
  - Bodypart skeleton overlay
  - Head direction arrow
  - Light/dark state indicator
  - Maze grid with dead-end shading
- **Controls:** play/pause, speed slider, frame scrubber, zoom
- **Embedded via** `st.html(unsafe_allow_javascript=True)` — falls back to deprecated `st.components.v1.html()` on older Streamlit
- **Page:** `frontend/pages/maze_animation_page.py`

**Speaker notes:**
The canvas animation is a custom Streamlit component that bypasses the standard Streamlit re-render cycle. Streamlit's default behaviour is to re-run the entire Python script on every interaction, which is far too slow for real-time animation. Instead, the maze canvas embeds a self-contained HTML+JS+CSS block that runs at 60 fps in the browser. Data is serialised as JSON and injected into the JS code at render time. Each DOM element gets a unique suffix to avoid collisions when Streamlit re-renders the component. This provides fluid trajectory playback that you can scrub through to identify tracking failures, behavioural artefacts, or interesting behavioural events.

---

## Slide 16: Frontend Dashboard

- **Framework:** Streamlit with `st.navigation()` for multi-page layout
- **67 pages** registered in navigation across 5 sections:

| Section | Pages | Purpose |
|---|---|---|
| Overview (7) | Home, Sessions, Animals, Pipeline, Cell Summary, Literature, Methods | Project status, metadata, references |
| Pipeline (16) | Suite2p, Calcium, CASCADE, DLC Training, Training QC, Training Fit, Label Review, DLC Inference, DLC Viewer, Tracking QC, Perspective, MoSeq, Sync Report, Z-Drift, Anatomy, Illumination | Per-stage QC and diagnostics |
| Explore (16) | Explorer, Timeline, ROI Gallery, ROI Viewer, ROI Curation, Events, Event Dynamics, Correlations, Trace Compare, Pop. Activity, Neuropil, Rastermap, MoSeq Explore, MoSeq Exemplars, Behaviour, Maze Animation | Interactive data exploration |
| Analysis (23) | Hypotheses, Analysis, Compare, Population, Light/Dark, Light Compare, Pub Stats, Maze, HD Tuning, Place Tuning, Decoder, Stability, Drift, Gain, Anchoring, Speed, Pop. Dynamics, AHV, Info Theory, Classify, Signal Quality, QC Report, Patching (x3) | Scientific analysis and figures |
| System (3) | AWS, Costs, Changelog | Infrastructure monitoring |

- **Data loading:** `frontend/data.py` — S3 caching with `@st.cache_data`, session filtering, DLC champion checks
- **Authentication:** Google OAuth (optional; disabled in local dev)
- **No sidebar filters** — all controls in page body via `st.columns()`
- **No synthetic data** — pages load real data from S3 and show messages when unavailable

**Speaker notes:**
The frontend is the primary interface for QC and analysis review. It is not a publication-quality figure generator — it is a real-time dashboard for monitoring pipeline progress, inspecting individual sessions, and verifying analysis results before preparing manuscript figures in separate scripts. The 67 pages are organised into 5 navigation sections. The "Pipeline" section lets you inspect each stage's output for each session (e.g., Suite2p ROI masks, DLC tracking overlays, sync timing diagnostics). The "Explore" section provides interactive tools for deep-diving into individual ROIs, traces, and events. The "Analysis" section presents the scientific results — HD tuning curves, decoder performance, anchoring analysis, etc. Google OAuth restricts access when deployed to a public URL; in local development mode, auth is skipped. All data is loaded from S3 via boto3 with Streamlit's `@st.cache_data` decorator for performance.

---

## Slide 17: Analysis Loading — pynapple Interface

- **sync.h5 loads directly into pynapple** — no reshaping needed:
```python
import pynapple as nap, h5py

with h5py.File("sync.h5") as f:
    t       = f["frame_times"][:]
    spikes  = nap.TsdFrame(t=t, d=f["spikes"][:].T)    # (T, R)
    dff     = nap.TsdFrame(t=t, d=f["dff"][:].T)       # (T, R)
    hd      = nap.Tsd(t=t, d=f["hd_deg"][:])
    speed   = nap.Tsd(t=t, d=f["speed_cm_s"][:])
    active  = nap.Tsd(t=t, d=f["active"][:])

active_ep     = nap.IntervalSet(...)
spikes_active = spikes.restrict(active_ep)
```
- **Design:** arrays are time-first (C-contiguous), timestamps are float64 seconds since session start
- **Planned tools:**
  - **NEMOS** (Flatiron) — GLM encoding models, pynapple-native, JAX backend
  - **CEBRA** (Schneider et al. 2023) — contrastive population embeddings; ring manifold for HD
  - **neuroconv** — HDF5 -> NWB export for DANDI archiving
- **Dataset names:** `hd_deg` (not `hd`), `speed_cm_s` (not `speed`), `frame_times` (plural)

**Speaker notes:**
The HDF5 schemas are deliberately designed for zero-friction pynapple loading. Arrays are stored time-first so that slicing `f["dff"][:].T` gives you `(T, R)` which is what pynapple's TsdFrame expects. Timestamps are always float64 seconds since session start, which is pynapple's native time unit. The dataset names are explicit about units: `hd_deg` (degrees), `speed_cm_s` (cm/s), `ahv_deg_s` (deg/s) — this prevents unit confusion. NEMOS would let us fit GLMs to ask "which behavioural variables (HD, speed, position) drive this ROI's activity?" — it's pynapple-native. CEBRA would let us look for ring-shaped manifolds in population activity space, which is a hallmark signature of HD population coding.

---

## Slide 18: Infrastructure — AWS

- **Account:** 390897005556, region: ap-southeast-2 (Sydney)
- **S3 buckets:** `hm2p-rawdata` (Infrequent Access), `hm2p-derivatives` (Standard)
- **EC2 instances (Spot):**
  - g4dn.xlarge (~$0.16/hr) — DLC inference, Suite2p
  - g4dn.2xlarge / g5.xlarge (~$0.30/hr) — DLC training
  - c5.4xlarge (~$0.27/hr) — CPU stages (kinematics, calcium, sync)
- **Safety mechanisms:**
  - GPU watchdog: terminate if 0% utilisation for 5+ minutes
  - 24-hour hard timeout on all instances
  - Self-termination on completion (`InstanceInitiatedShutdownBehavior=terminate`)
  - Security group restricted to known IP (103.106.88.142/32)
  - SSM Session Manager for keyless SSH
  - S3 access logging enabled
- **Launch scripts:**
  - `scripts/launch_suite2p_ec2.py` — Suite2p processing
  - `scripts/launch_dlc_finetune_ec2.py` — DLC training + inference
  - `scripts/launch_dlc_parallel.py` — DLC inference across N parallel shards
  - `scripts/launch_downstream_cpu.py` — CPU stages after pose update
  - `scripts/launch_kpms_ec2.py` — keypoint-MoSeq on EC2
- **Cost tracking:** `scripts/aws_costs.py`, `src/hm2p/io/aws_cost.py`, `frontend/pages/cost_page.py`
- **One-time processing:** ~$180-380 (all 26 sessions). Ongoing storage: ~$10/month.

**Speaker notes:**
All EC2 instances are Spot — significantly cheaper than On-Demand but can be interrupted. The pipeline is designed to handle interruptions: partial outputs are detected and re-run. The GPU watchdog is important because DLC training can sometimes hang silently with 0% GPU utilisation — without the watchdog, you would accumulate charges indefinitely. The self-termination behaviour means the instance terminates (not just stops) when the user-data script finishes, so you never accidentally leave a running instance. The security group is locked to a single IP address, and SSM Session Manager provides keyless SSH access for debugging. All launch scripts support `--status`, `--progress`, `--terminate`, and `--dry-run` flags. Cost is reported in USD first, then AUD in brackets.

---

## Slide 19: Pipeline Orchestration

- **Snakemake 8.x+** — DAG-based workflow engine
- **Snakefile:** `workflow/Snakefile` (main DAG)
- **Stage rules:** `workflow/rules/` — `ingest.smk`, `extraction.smk`, `pose.smk`, `kinematics.smk`, `calcium.smk`, `sync.smk`, `sync_report.smk`
- **Three compute profiles:** `workflow/profiles/`
  - `local/` — CPU-only execution on laptop
  - `local-gpu/` — all stages on local GPU machine
  - `aws-batch/` — full cloud pipeline (managed job queue)
- **Profile selection:** `config/compute.yaml` (`profile: local`)
- **Docker images:** `docker/`
  - `gpu.Dockerfile` — CUDA 12.1 + Suite2p + DLC
  - `cpu.Dockerfile` — CPU-only stages (movement, calcium, sync)
  - `kpms.Dockerfile` — keypoint-MoSeq isolated environment
  - `cascade.Dockerfile` — CASCADE with tensorflow (isolated due to Python 3.8 / TF 2.3 constraint)
- **Pipeline runner scripts** (bypass Snakemake for direct execution):
  - `scripts/run_stage0_daq.py` through `scripts/run_stage6_analysis.py`
  - `scripts/run_downstream_pipeline.py` — runs Stages 3 -> 3b -> 5 -> 6 after pose update

**Speaker notes:**
Snakemake provides automatic dependency tracking — if you re-run Stage 2 (DLC), Snakemake knows to re-run Stages 3, 5, and 6 but not Stage 4. In practice, most pipeline runs use the direct runner scripts rather than the full Snakemake DAG, because the pipeline is relatively simple (6 stages, linear dependency) and the Snakemake rules primarily add overhead for this scale. The Docker images isolate incompatible dependencies: CASCADE requires Python 3.8 + tensorflow 2.3 (which conflicts with everything modern), keypoint-MoSeq has its own numpy/scikit-learn pins, and the GPU image needs CUDA. The `cascade.Dockerfile` is a standalone container for Stage 4b.

---

## Slide 20: Data Standards — HDF5 Schemas

- **Consistent indexing:** arrays are time-first (C-contiguous for fast row slicing)
- **Timestamps:** float64 seconds since session start
- **Units in dataset names:** `hd_deg`, `speed_cm_s`, `ahv_deg_s`, `x_mm`, `y_mm`
- **Validation:** `pandera` schemas in `src/hm2p/io/hdf5.py`
- **Key files and their shapes:**

| File | Key datasets | Shape | Rate |
|---|---|---|---|
| `timestamps.h5` | `frame_times_camera`, `frame_times_imaging`, `light_on_times` | (N,), (T,), (L,) | - |
| `kinematics.h5` | `hd_deg`, `x_mm`, `y_mm`, `speed_cm_s`, `ahv_deg_s`, `active`, `light_on`, `bad_behav` | (N,) | ~100 Hz |
| `ca.h5` | `dff`, `spikes`, `event_masks`, `roi_types`, `snr` | (R,T), (R,T), (R,T), (R,), (R,) | ~9.6 Hz |
| `sync.h5` | All kinematics (resampled) + all calcium (verbatim) | (T,), (R,T) | ~9.6 Hz |
| `analysis.h5` | `tuning_curves`, `pd`, `mvl`, `rayleigh_p`, `is_hd`, `decoder_error` | (R,B), (R,), (R,) | - |

- **Provenance attributes:** `session_id`, `fps_imaging`, `dlc_model_name`, `dlc_snapshot`, `dlc_champion_id`
- **Backward compatibility aliases:** `x_mm` = `x_body_mm`, `y_mm` = `y_body_mm`, `speed_cm_s` = `speed_body_cm_s`

**Speaker notes:**
The HDF5 schema design is intentional — every design choice serves the pynapple interface. Time-first arrays mean that slicing a single time point from `dff` is a contiguous memory read (C-order). Float64 timestamps avoid precision loss over long sessions (a 30-minute session at 10 Hz is only 18,000 frames — float32 would lose sub-millisecond precision). Dataset names include units to prevent the most common source of analysis bugs. Pandera validation in `io/hdf5.py` checks that outputs conform to the schema at write time — this catches bugs early rather than discovering them when loading stale data weeks later. Backward compatibility aliases exist because early versions of kinematics.h5 used different names; they are maintained so existing analysis code does not break.

---

## Slide 21: Code Quality & Testing

- **117 test files** in `tests/` mirroring `src/hm2p/` structure
- **Coverage target:** >= 90% (hard requirement)
- **Testing frameworks:**
  - `pytest` + `pytest-cov` — standard unit tests + coverage
  - `hypothesis` — property-based testing for numerical functions (auto-generates adversarial inputs)
  - `pandera` — runtime schema validation in tests
- **Pre-commit hooks:** ruff (format + lint), mypy (strict type checking), nbstripout, detect-secrets
- **CI:** GitHub Actions — `ci.yml` (pytest on Python 3.11 + 3.12), `lint.yml` (ruff + mypy + security)
- **Security tooling:** bandit, checkov, detect-secrets, pip-audit, vulture (dead code)
- **Test rules:**
  - Tests use **synthetic data only** — never read real data files
  - Every function (public and private) must have at least one test
  - Non-parametric statistics tests verify that parametric alternatives are never used

**Speaker notes:**
The testing philosophy is that every function must have at least one test, and tests must use synthetic data only. This means test data is small numpy arrays constructed in the test, not loaded from files. Hypothesis testing is particularly valuable for numerical functions like dF/F computation and HD angle calculations — it auto-generates edge cases (NaN inputs, zero-length arrays, large values) that would be tedious to write manually. The pre-commit hooks catch formatting issues, type errors, and accidentally committed secrets before they reach the repository. The CI pipeline runs on every push and blocks merges if coverage drops below 90%.

---

## Slide 22: Source Code Layout

```text
src/hm2p/
  __init__.py, cli.py, config.py, constants.py, plotting.py, session.py
  ingest/        -- Stage 0: validate.py, daq.py
  extraction/    -- Stage 1: base.py, suite2p.py, caiman.py, run_suite2p.py,
                    soma_features.py, soma_classifier.py, curation.py, zdrift.py
  pose/          -- Stage 2: run.py, preprocess.py, quality.py, retrain.py,
                    select.py, finetune.py, dedup.py
  kinematics/    -- Stage 3: compute.py, perspective.py, syllables.py
  calcium/       -- Stage 4: neuropil.py, neuropil_analysis.py, dff.py,
                    spikes.py, events.py, population.py, qc.py, run.py
  sync/          -- Stage 5: align.py, validate.py, diagnostics.py, report.py
  analysis/      -- Stage 6: 19 modules (tuning, decoder, etc.)
  maze/          -- Maze topology + discretization + behavioural + neural analysis
  anatomy/       -- brainreg registration + injection site rendering
  patching/      -- Patch-clamp electrophysiology pipeline (11 modules)
  io/            -- HDF5 I/O, S3 paths, NWB export, AWS costs
```

- **Total:** ~75 Python modules (excluding `__init__.py`)
- **Separation of concerns:** processing vs analysis vs frontend vs I/O
- **No circular imports:** strict layering (io -> processing -> analysis)

**Speaker notes:**
The module layout follows the pipeline stages, with each stage getting its own package. The `io/` package handles all file I/O and cloud interaction — no other module imports boto3 or writes HDF5 directly (they call io functions). The `analysis/` package depends on `sync/` output but never on `extraction/` or `pose/` directly. The `maze/` package is independent of the pipeline stages — it provides pure functions for maze topology and spatial analysis. The `patching/` package is a separate sub-pipeline for patch-clamp electrophysiology data (membrane properties, spike waveforms, morphology) that is independent of the calcium imaging pipeline.

---

## Slide 23: Patching Pipeline

- **Purpose:** Analyse patch-clamp electrophysiology + morphology data for the same cell populations (Penk+ vs CamKII+)
- **11 modules** in `src/hm2p/patching/`:
  - `config.py` — pipeline configuration (`config/patching.yaml`)
  - `io.py` — WaveSurfer H5 + SWC file I/O
  - `ephys.py` — electrophysiology signal processing
  - `protocols.py` — stimulus protocol parsing and response extraction
  - `spike_features.py` — AP waveform feature extraction
  - `morphology.py` — SWC morphology loading and analysis
  - `metrics.py` — intrinsic excitability and passive membrane properties
  - `statistics.py` — statistical comparisons (Penk vs non-Penk, non-parametric)
  - `pca.py` — PCA on electrophysiological features
  - `run.py` — batch runner
  - `plotting/morph_plots.py` — morphology visualisation
- **Frontend pages:** `patching_page.py`, `patching_traces_page.py`, `patching_morph_page.py`
- **Data:** read-only bind mount at `/data/patching/`
- **Status:** processing modules complete; frontend pages functional

**Speaker notes:**
The patching pipeline is complementary to the calcium imaging pipeline. While the imaging pipeline measures population-level HD tuning in behaving animals, the patching data provides single-cell biophysical characterisation of the same cell populations — membrane properties (input resistance, capacitance, resting potential), action potential waveforms, intrinsic excitability (f-I curves), and dendritic morphology. The key question is whether Penk+ and CamKII+ neurons differ in their intrinsic properties, which might explain differences in HD tuning observed in the imaging data. All statistical comparisons use non-parametric tests.

---

## Slide 24: Current Status

### Completed
| Component | Status |
|---|---|
| Project skeleton (pyproject.toml, CI, pre-commit) | Done |
| Stage 0 — DAQ parsing | 26/26 sessions |
| Stage 1 — Suite2p extraction | 26/26 sessions |
| Stage 2 — DLC pose estimation | 26/26 sessions (SA + HRNet) |
| Stage 3 — Kinematics | Code complete; awaiting DLC re-run for refresh |
| Stage 4 — Calcium processing | 26/26 sessions |
| Stage 5 — Sync | 21/21 sessions processed |
| Stage 6 — Analysis (19 modules) | Code complete; awaiting DLC re-run for refresh |
| Frontend (67 pages) | Operational |
| Patching pipeline (11 modules) | Complete |
| DLC champion model system | Phases 1-2 substantially implemented |
| AWS infrastructure | Operational |

### Pending
| Component | Status |
|---|---|
| CASCADE spike inference (Stage 4b) | Requires separate conda env (TF 2.3, Python 3.8) |
| FISSA neuropil subtraction | Requires separate env (scikit-learn < 1.2) |
| NWB export via neuroconv | Stub only |
| DLC champion Phase 3 (frontend enforcement) | Partially implemented |
| Stage 3b — MoSeq syllables | Docker container ready; awaiting DLC re-run |
| Snakemake shell commands | Rules defined; direct runner scripts used in practice |

**Speaker notes:**
The pipeline is functionally complete for all core stages. The main outstanding work is (1) CASCADE spike inference, which requires an isolated Python 3.8 environment due to tensorflow 2.3 constraints, (2) a DLC model re-run to refresh all pose-dependent derivatives with the current model, and (3) NWB export for data archiving on DANDI. The DLC champion model system is substantially implemented in code but the PLAN.md checklist has not been updated to reflect this. The frontend is operational with 67 pages and loads real data from S3. The Snakemake DAG has rules defined but in practice, pipeline stages are run via direct scripts (`run_stage3_kinematics.py`, etc.) rather than through the full Snakemake orchestration.

---

## Slide 25: Known Issues & Documentation Gaps

### Documentation inconsistencies (as of 2026-05-14)
- **Count mismatches:** README.md says "53 pages" and "17 modules"; ARCHITECTURE.md says "60 pages" and "20 modules"; actual counts are 67 pages and 19 modules
- **Body part list:** PLAN.md Section 1.2C still lists only 5 bodyparts; CLAUDE.md correctly lists 8
- **8 source modules missing from ARCHITECTURE.md source tree:** `pose/select.py`, `pose/finetune.py`, `pose/dedup.py`, `calcium/qc.py`, `maze/neural.py`, `sync/diagnostics.py`, `sync/report.py`, `extraction/curation.py`
- **37 scripts missing from ARCHITECTURE.md scripts section**
- **Undocumented features:** W&B integration, NaviGraph analyses, canvas maze animation, sync diagnostics system, ROI curation workflow, per-bodypart RMSE, frame selection tools
- **Stale champion model checklist:** PLAN.md shows Phases 1-4 all unchecked but Phase 1 and Phase 2 are substantially implemented

### Technical debt
- **CASCADE not yet running:** primary spike inference method not integrated due to env constraints
- **FISSA not yet running:** optional neuropil subtraction blocked by dependency conflicts
- **NWB export:** stub only — no data archived on DANDI yet
- **5 sessions missing sync.h5:** only 21/26 processed (likely timing issues in remaining 5)
- **README pynapple example:** uses wrong dataset names (`hd` instead of `hd_deg`, `speed` instead of `speed_cm_s`)

**Speaker notes:**
The documentation gaps are a maintenance issue, not a technical one — the code is ahead of the docs. The most critical gap is the bodypart list in PLAN.md, which lists only the original 5 SuperAnimal keypoints instead of the current 8 (which include nose_tip, head_midpoint, and neck). The README pynapple example has incorrect dataset names that would cause KeyError if copy-pasted. The CASCADE and FISSA integration are blocked by Python version and dependency conflicts — both require isolated environments (Docker containers) that are defined but not yet integrated into the main workflow. The 5 missing sync.h5 sessions need investigation — they likely have timing edge cases that the sync validation catches.

---

## Slide 26: Key File Paths Reference

| Purpose | Path |
|---|---|
| Pipeline source code | `src/hm2p/` |
| Tests | `tests/` |
| Frontend | `frontend/app.py`, `frontend/data.py`, `frontend/pages/`, `frontend/components/` |
| Pipeline runner scripts | `scripts/run_stage{0,3,4,5,6}_*.py` |
| DLC workflow scripts | `scripts/*dlc*.py`, `scripts/launch_dlc_*.py` |
| Config files | `config/pipeline.yaml`, `config/compute.yaml`, `config/sync.yaml`, `config/patching.yaml` |
| Metadata | `metadata/animals.csv`, `metadata/experiments.csv` |
| Snakemake | `workflow/Snakefile`, `workflow/rules/`, `workflow/profiles/` |
| Docker | `docker/gpu.Dockerfile`, `docker/cpu.Dockerfile`, `docker/kpms.Dockerfile`, `docker/cascade.Dockerfile` |
| Architecture docs | `ARCHITECTURE.md`, `PLAN.md`, `CLAUDE.md` |
| Topic docs | `docs/` (41 markdown files) |
| S3 rawdata | `s3://hm2p-rawdata/` |
| S3 derivatives | `s3://hm2p-derivatives/` |
| DLC champion manifest | `s3://hm2p-derivatives/dlc-champion.json` |
| Legacy code (read-only) | `old-pipeline/` |

**Speaker notes:**
This reference slide is for quick lookup during discussions. The key entry points are: `frontend/app.py` to run the dashboard (`streamlit run frontend/app.py`), the `scripts/run_stage*.py` files to run individual pipeline stages, and `CLAUDE.md` for the complete set of project rules and constraints. The `docs/` directory contains 41 topic-specific markdown files covering everything from DLC retraining procedures to statistical strategy to neuropil analysis literature review. The legacy pipeline code in `old-pipeline/` is read-only reference — it is never modified.
