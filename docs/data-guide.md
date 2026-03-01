# Data Guide — hm2p

Reference for raw data structure, file formats, and legacy processing logic.
Written from inspection of the legacy code (`hm2p-analysis/`) and raw data
(`Dropbox/Neuro/Margrie/hm2p/`). **Do not modify the source directories.**

---

## 1. Experiment Overview

**Experiment type:** Freely-moving mouse in a rose-maze (also open field and linear
track variants). Two-photon GCaMP calcium imaging recorded simultaneously with overhead
behavioural video.

**Brain region:** Retrosplenial cortex (RSP) and nearby cortex. **Not** subiculum or
postsubiculum. Injection coordinates: AP ~7.2–8.2, ML ~5.0–5.2, DV ~0.28–0.57 mm.

**Cell types recorded:**

| Cell type | Mouse line | Virus | `celltype` value |
| --- | --- | --- | --- |
| Penk+ RSP neurons | Penk-Cre | ADD3 (GCaMP7f) — Cre-ON | `"penk"` |
| Non-Penk CamKII+ RSP neurons | Penk-Cre | 344 (GCaMP7f) — Cre-OFF/intersectional | `"nonpenk"` |

Virus 344 uses a Cre-dependent OFF strategy: Cre expression in Penk+ cells **prevents**
GCaMP expression, so only non-Penk (CamKII+) cells are labelled. This is a clean intersectional
approach to label two non-overlapping populations in the same brain region.

Some animals also received Flp-dependent viruses (A160.1 + A83). GCaMP: `7f` for most;
`8f` for animal 1116663 only.

**Imaging planes:** There is a **single imaging plane** per session. Within this plane,
both soma (compact, round) and dendrite segment (elongated) ROIs are detected by Suite2p
and classified post-hoc using shape statistics. There is **no separate dendrite plane**.

**Light protocol:** Overhead room lights follow a **1 min on / 1 min off** periodic cycle.
Light off = **total darkness** — complete visual cue removal. This tests whether RSP HD cells
can maintain directional tuning without visual landmarks (idiothetic/path integration cues only).
The light state is recorded by a DAQ digital input channel and saved as pulse timestamps in the TDMS file.

**serial2p:** Each animal has a serial 2P z-stack of the whole brain for anatomical
localisation of the injection site. This is used manually for anatomy and is **not** part of this pipeline.

**Head-mount constraint:** Mice carry a head-mounted 2P scope (HM2P). The fibre/wires can
cause the mouse to get stuck against maze walls, creating artefactual immobility periods.
These periods are manually identified per session and logged in `experiments.csv` as
`bad_behav_times` (mm:ss-mm:ss format, semicolon-separated).

**Sessions:** 29–30 sessions across ~7 animals. Multiple sessions per animal; up to two
sessions per day. ~550 GB total data on Dropbox.

**Session ID format:**

```text
YYYYMMDD_HH_MM_SS_<animal_id>
e.g.  20220804_13_52_02_1117646
```

Note: two timestamps exist per session — the SciScan acquisition start
(`YYYYMMDD_HH_MM_SS`) and the DAQ/experiment start (`YYYYMMDD_HH_MM_SS`, usually
offset by a few seconds). The session folder name uses the **SciScan** timestamp; the
`.meta.txt` filename uses the **DAQ** timestamp.

---

## 2. Raw Data Locations

All paths below are under `/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/`.
**Read-only — never modify or delete anything here.**

| Location | Contents |
| --- | --- |
| `shared/lab-108/experiments/01 lights-maze/` | **Primary raw data** — one date folder per day, one session folder per recording |
| `hm2p/video/{session_id}/` | **Processed overhead videos** (undistorted + cropped MP4) + `meta.txt` — used as DLC input |
| `hm2p/video-meta-backup/` | Backup of per-session crop/scale/ROI metadata |
| `hm2p/s2p/` | Legacy Suite2p outputs |
| `hm2p/dlc/` | DeepLabCut model + tracked outputs |
| `hm2p/proc/` | Legacy processed HDF5 databases (behave, ca) |
| `hm2p/db/` | Legacy aggregated per-session HDF5 databases |
| `hm2p-analysis/metadata/` | Legacy `animals.csv`, `experiments.csv` (canonical copies now in repo `metadata/`) |
| `hm2p-analysis/cam-calibrations/` | Lens-specific camera calibration `.npz` files |

### S3 Upload Plan

Upload ~**113 GB** to `s3://hm2p-rawdata/` (once AWS credentials are working):

| Source | S3 destination | Size | Notes |
| --- | --- | --- | --- |
| `01 lights-maze/{date}/{session_id}/` | `rawdata/sub-{id}/ses-.../funcimg/` | ~96 GB | Exclude `_side_left.camera.mp4` and `_XYT.red.tif` |
| `hm2p/video/{session_id}/*-cropped.mp4` | `rawdata/sub-{id}/ses-.../behav/` | ~17 GB | Cropped video — DLC input; skip `*-undistort.mp4` |
| `hm2p/video/{session_id}/meta.txt` | `rawdata/sub-{id}/ses-.../behav/` | <1 MB | Crop ROI, scale, maze corners — required for Stage 3 |
| `hm2p/video/{session_id}/meta/` | `rawdata/sub-{id}/ses-.../behav/meta/` | <1 MB | Raw napari CSVs — provenance |
| `metadata/*.csv` | `sourcedata/metadata/` | <1 MB | Copy of git repo CSVs for cloud access |
| `hm2p-analysis/cam-calibrations/` | `sourcedata/calibration/` | <1 MB | Camera `.npz` files |

**Note:** The raw overhead `_overhead.camera.mp4` exists in the session folder for only 8/26 sessions.
For all 26 sessions, processed versions (undistorted + cropped) exist in `hm2p/video/` — these are
what DLC was trained on and what Stage 2 pose estimation should use.

---

## 3. Raw Session Directory

Each session lives under a date folder in the raw data root:

```text
shared/lab-108/experiments/01 lights-maze/
└── 2021_08_23/
    └── 20210823_16_59_50_1114353/         ← SciScan session folder
        ├── 20210823_16_59_50_1114353_XYT.raw      ← 2P raw imaging data (SciScan proprietary) [may be absent]
        ├── 20210823_16_59_50_1114353_XYT.tif      ← TIFF stack — primary imaging input for Suite2p
        ├── 20210823_16_59_50_1114353_XYT.ini      ← SciScan acquisition settings
        ├── 20210823_16_59_50_1114353_XYT.notes.txt
        ├── 20210823_16_59_50_1114353_OME.xml       ← OME metadata
        ├── 20210823_17_00_04_1114353_maze-rose.meta.txt    ← Experiment metadata (INI format)
        ├── 20210823_17_00_04_1114353_maze-rose-di.tdms     ← DAQ digital input signals (REQUIRED)
        ├── 20210823_17_00_04_1114353_maze-rose-di.tdms_index
        ├── [20210823_17_00_04_1114353_maze-rose-ai.tdms]   ← DAQ analog input (3 sessions only, keep)
        ├── 20210823_17_00_04_1114353_maze-rose_overhead.camera.mp4    ← Raw overhead video [only 8/26 sessions]
        ├── 20210823_17_00_04_1114353_maze-rose_side_left.camera.mp4   ← Side camera — NEVER USED, skip upload
        ├── 20210823_17_00_04_1114353_maze-rose_acA1300-200um_Arena_Output.pfs  ← Camera config
        ├── 20210823_17_00_04_1114353_maze-rose_acA1920-150um_Arena_Output.pfs
        ├── 20210823_17_00_04_1114353_maze-rose_Power rotator.xlsx     ← Laser power log
        └── NOTE.txt

hm2p/video/
└── 20210823_16_59_50_1114353/            ← Processed video for ALL 26 sessions
    ├── 20210823_17_00_04_1114353_maze-rose_overhead.camera-cropped.mp4    ← Cropped overhead
    ├── 20210823_17_00_04_1114353_maze-rose_overhead.camera-undistort.mp4  ← Undistorted overhead
    ├── meta.txt      ← Crop ROI, scale (mm/px), maze corners
    └── meta/         ← Legacy per-field CSVs (backup only)
```

**Imaging file notes:**

- Most sessions: both `_XYT.raw` (SciScan proprietary) and `_XYT.tif` (converted) exist. Suite2p reads the TIFF.
- A few sessions have only `_XYT.tif` — the .raw was never saved or was lost. This is fine; TIFF is sufficient.
- `_XYT.red.tif` exists for 5 sessions (red anatomical channel) — not used by the pipeline, skip upload.

**Video notes:**

- The raw `_overhead.camera.mp4` is present in the session folder for only 8/26 sessions.
- **DLC pose estimation uses the processed videos from `hm2p/video/`** (undistorted + cropped),
  since this is what the DLC model was trained on and the only format available for all sessions.
- The `meta.txt` in `hm2p/video/{session_id}/` contains the crop ROI, pixel scale, and maze corners
  needed for Stage 3 kinematics — this is the video metadata source for the new pipeline.

---

## 4. Experiment Metadata — `.meta.txt`

**Filename pattern:** `{DAQ_timestamp}_{animal_id}_{type}.meta.txt`
(e.g. `20210823_17_00_04_1114353_maze-rose.meta.txt`)

**Format:** INI file parsed with `configparser`. Sections:

### `[Experiment]`

| Field | Example | Notes |
| --- | --- | --- |
| `date` | `2021-08-23 17:00:04.633870` | DAQ wall clock at experiment start |
| `name` | `1114353` | Animal ID |
| `type` | `maze-rose` | Experiment type |
| `complete` | `True` | Did the experiment complete normally |
| `starttime` | `1629734404.9` | Unix timestamp, experiment start |
| `endtime` | `1629736283.5` | Unix timestamp, experiment end |
| `duration` | `1878.59` | Seconds |
| `camstarttime` | `1629734409.5` | Unix timestamp, camera acquisition start |
| `sciscanstopped` | `True` | Whether SciScan was stopped cleanly |

### `[Video]`

| Field | Example | Notes |
| --- | --- | --- |
| `fps` | `100` | Camera frame rate (Hz) |
| `numberofcameras` | `2` | Overhead + side_left |
| `daqcameratrigger` | `True` | Camera triggered by DAQ (hardware sync) |
| `exposuretime` | `9000` | Exposure in microseconds |

### `[Camera_1]` / `[Camera_2]`

| Field | Example | Notes |
| --- | --- | --- |
| `id` | `acA1300-200um` | Basler camera model (overhead) |
| `name` | `overhead` | Logical name |
| `daqphyschan` | `/port0/line2` | DAQ line for trigger |
| `resolution` | `(1280, 1024)` | Pixel resolution |

### `[DAQ]`

| Field | Example | Notes |
| --- | --- | --- |
| `devicename` | `ExpDAQ` | NI DAQ device |
| `file-di` | `...-di.tdms` | Digital input TDMS file |
| `sf` | `10000` | DAQ sample frequency (Hz) |
| `sciscanchanname` | `sci_sync` | Channel name for SciScan line clock |
| `cameratriggerchanname` | `cam_trigger` | Channel name for camera trigger |
| `uselights` | `True` | Whether lighting was used |
| `lightschanname` | `lights` | Channel name for lighting signal |

### `[Lights]`

| Field | Example | Notes |
| --- | --- | --- |
| `timeon` | `60030.72` | Light-on duration (ms) |
| `timeoff` | `60030.72` | Light-off duration (ms) |
| `lineclockpulseson` | `93798` | Number of SciScan line clock pulses while lights on |

### `[SciScan]`

| Field | Example | Notes |
| --- | --- | --- |
| `checkrunning` | `True` | Whether SciScan was expected to run |
| `imgfile` | `..._XYT.raw` | Path to raw imaging file |
| `inifile` | `..._XYT.ini` | Path to SciScan `.ini` settings |

---

## 5. SciScan Settings — `_XYT.ini`

INI file (single section `[_]`) containing all SciScan acquisition parameters:

| Field | Example | Notes |
| --- | --- | --- |
| `frames.p.sec` | `9.645` | Imaging frame rate (Hz) — varies per session |
| `ms.p.line` | `0.640` | Milliseconds per scan line |
| `x.pixels` | `320` | Image width in pixels |
| `y.pixels` | `162` | Image height in pixels |
| `x.pixel.sz` | `0.0000005` | Pixel size in metres (x-axis) |
| `no..of.frames.to.acquire` | `18000` | Total frames requested |
| `x.bidi.mode` | `TRUE` | Bidirectional scanning |

**Key computation:** imaging frame rate is derived from the SciScan line clock in the TDMS
file, not from `frames.p.sec` directly:

```text
sci_line_indexes = rising edges of line clock signal
sci_frame_indexes = sci_line_indexes[y_pixels - 1 :: y_pixels]  # every y_pix-th line
sci_frame_times = daq_time[sci_frame_indexes]  # when each frame finishes
```

Frame times are recorded **when each frame finishes scanning** (end of last line), not
when it starts.

---

## 6. DAQ Signals — `_maze-rose-di.tdms`

**Format:** NI TDMS (National Instruments Technical Data Management Streaming).
Parsed with `nptdms` (`TdmsFile.read()`).

**Sample rate:** 10,000 Hz (configurable; set in `DAQ.sf`).

**Key channels in the digital input file:**

| Channel name | Content |
| --- | --- |
| `cam_trigger` | Camera trigger pulses — one rising edge per camera frame |
| `sci_sync` | SciScan line clock — y_pixels pulses per imaging frame |
| `lights` | Lighting control signal — high when lights on |

**Camera frame timestamps** are extracted as the times of rising edges in `cam_trigger`:

```python
cam_trigger_indexes = rising_edges(abs(cam_trigger_data), threshold=0.9)
cam_trigger_times   = daq_time[cam_trigger_indexes]   # seconds
```

---

## 7. Two-Photon Imaging Data

### Raw Format

**File:** `YYYYMMDD_HH_MM_SS_<animal_id>_XYT.raw` — SciScan proprietary binary format.
Converted to multi-frame TIFF (`_XYT.tif`) for downstream processing.

**Dimensions:** `(n_frames, y_pixels, x_pixels)` — for the session above:
`(18000, 162, 320)` at ~9.6 Hz. Exact frame count may differ from the requested count.

**Channels:** Single channel (green GCaMP functional). Some sessions also record a red
anatomical reference channel in a separate file.

**Pixel size:** ~0.5 µm/pixel (x) × 0.7 µm/pixel (y) at ×10 objective.

### Suite2p Output Structure

```text
hm2p/s2p/{session_id}/
├── soma/
│   ├── bad_frames.npy          ← boolean (n_frames,) — frames with low image quality
│   ├── suite2p_soma/
│   │   └── plane0/
│   │       ├── F.npy           ← (n_rois, n_frames) float32 — raw fluorescence
│   │       ├── Fneu.npy        ← (n_rois, n_frames) float32 — neuropil fluorescence
│   │       ├── spks.npy        ← (n_rois, n_frames) float32 — OASIS deconvolved
│   │       ├── iscell.npy      ← (n_rois, 2) — col 0: is accepted (0/1); col 1: classifier prob
│   │       ├── stat.npy        ← (n_rois,) structured array — shape stats per ROI
│   │       └── ops.npy         ← dict — all Suite2p parameters + mean image
│   ├── images/
│   ├── movies/
│   └── regmetrics/
└── dend/
    └── suite2p_dend/
        └── plane0/             ← same structure as soma
```

**Bad frame detection:** Bad frames are detected by Spearman correlation of each frame
against the max z-projection of the imaging stack. Frames with correlation below threshold
(~0.1) are flagged. `bad_frames.npy` is a boolean array of length `n_frames`.

**Neuropil correction:** `F_corr = F − neucoeff × Fneu` where `neucoeff = 0.7` (Suite2p
default, stored in `ops["neucoeff"]`).

**dF/F0 computation:** F0 is a sliding-window estimate using a minimum filter followed by
a Gaussian-smoothed maximum filter — equivalent to Suite2p's baseline estimation.

### ROI Classification — Soma vs Dendrite

Currently Suite2p is run **twice** with different parameters. The new pipeline will run
Suite2p **once** and classify ROIs post-hoc from `stat.npy` shape statistics:

| `stat` field | Type | Notes |
| --- | --- | --- |
| `ypix`, `xpix` | int arrays | Pixel coordinates of ROI |
| `lam` | float array | Pixel weights |
| `radius` | float | Estimated cell radius (pixels) |
| `aspect_ratio` | float | Major / minor axis ratio — high = elongated (dendrite) |
| `compact` | float | Compactness metric |
| `footprint` | float | Spatial footprint size |
| `npix` | int | Number of pixels in ROI |

Soma ROIs: compact (`aspect_ratio` ≈ 1), `radius` ~5–15 px.
Dendrite ROIs: elongated (`aspect_ratio` >> 1), large footprint.

---

## 8. Behavioural Video and Pose Tracking

### Video Processing Pipeline (Legacy — already done for all sessions)

The raw overhead `.camera.mp4` was processed in three steps per session using the legacy
scripts in `hm2p-analysis/scripts/mov/`:

```text
raw overhead MP4  →  undistort (OpenCV remap)  →  manual annotation (napari)  →  crop + orient (ffmpeg)
```

1. **Undistort** (`mov_undistort.py`): loads camera calibration `.npz` for the session lens
   (`f4mm` or `f6mm`), computes optimal camera matrix via `cv2.getOptimalNewCameraMatrix()`
   then remaps every frame via `cv2.initUndistortRectifyMap()` + `cv2.remap()`.
   Output: `*-undistort.mp4` (H.264 CRF 17).

2. **Annotate** (`mov_annotate.py`, napari GUI): user manually draws three shape layers on
   the undistorted frame and saves as CSVs in `meta/`:
   - `meta/crop.csv` — one rectangle: bounding box of the visible arena
   - `meta/scale.csv` — several line pairs between air-table reference holes 25 mm apart
   - `meta/roi.csv` — one rectangle: maze boundary (can be rotated)

3. **Write metadata** (`write_mov_meta_data()` in `utils/behave.py`):
   - Reads the three CSVs
   - Pads crop to nearest 32-pixel multiple (H.264 codec requirement)
   - Computes `mm_per_pix = 25.0 / mean_dist_pix` from scale lines
   - Transforms ROI corners from undistorted frame → cropped frame (subtract crop offset,
     apply orientation rotation if 90° or 180°)
   - Writes consolidated `meta.txt` INI file

4. **Crop** (`mov_crop.py`): applies ffmpeg `.crop()` filter using `[crop]` section of
   `meta.txt`. Output: `*-cropped.mp4`.

**The `*-cropped.mp4` is the DLC input for all sessions.** This processing is already
complete for all 26 sessions in `hm2p/video/`. The new pipeline does not re-run it.

### Video Files

```text
hm2p/video/{session_id}/
├── {timestamp}_{animal_id}_maze-rose_overhead.camera-cropped.mp4    ← DLC input (copy to S3)
├── {timestamp}_{animal_id}_maze-rose_overhead.camera-undistort.mp4  ← intermediate (skip S3)
├── meta.txt                                                          ← crop/scale/roi (copy to S3)
├── meta/
│   ├── crop.csv     ← napari rectangle: crop bounding box (copy to S3)
│   ├── scale.csv    ← napari lines: scale reference holes (copy to S3)
│   ├── roi.csv      ← napari rectangle: maze boundary (copy to S3)
│   └── movie-frame.tif   ← representative frame (skip — large, redundant)
└── sum/             ← legacy summary plots (skip)
```

### Per-Session Video Metadata — `meta.txt`

**Format:** INI file parsed with `configparser`. Real example from session `20210823_16_59_50_1114353`:

```ini
[crop]
x = 108             ; top-left x of crop in undistorted frame (pixels)
y = 261             ; top-left y
width = 832         ; crop width (padded to 32-pixel multiple)
height = 608        ; crop height (padded to 32-pixel multiple)
width_diff = 20     ; pixels added to right edge for padding
height_diff = 5     ; pixels added to bottom edge for padding

[scale]
mm_per_pix = 0.8113         ; spatial calibration (pixels → mm)
mean_dist_pix = 30.813      ; mean pixel distance between scale reference holes
dist_mm = 25.0              ; known reference distance on air table (always 25.0 mm)

[roi]
; Maze corners in undistorted (pre-crop) pixel coordinates
x1_raw = 257  y1_raw = 333
x2_raw = 872  y2_raw = 343
x3_raw = 865  y3_raw = 770
x4_raw = 251  y4_raw = 761
; Maze corners in cropped-video pixel coordinates
x1 = 149.0   y1 = 72.0
x2 = 764.0   y2 = 82.0
x3 = 757.0   y3 = 509.0
x4 = 143.0   y4 = 500.0
width = 608.0       ; x3 - x1 in cropped frame (pixels)
height = 437.0      ; y3 - y1 in cropped frame (pixels)
rotation = -179.07  ; angle of top edge (atan2(y2-y1, x2-x1)), degrees
```

**Using `meta.txt` in the new pipeline (Stage 3):**

```python
import configparser
cfg = configparser.ConfigParser()
cfg.read(meta_txt_path)

mm_per_pix  = float(cfg["scale"]["mm_per_pix"])
crop_x      = int(cfg["crop"]["x"])
crop_y      = int(cfg["crop"]["y"])
# Maze corners in cropped-video pixels:
corners_px = np.array([[float(cfg["roi"][f"x{i}"]), float(cfg["roi"][f"y{i}"])]
                        for i in range(1, 5)])
# maze_width_px / maze_height_px in pixels → use for pixel→maze-unit mapping
maze_w_px = float(cfg["roi"]["width"])
maze_h_px = float(cfg["roi"]["height"])
```

### Camera Calibration

**Files:** `hm2p-analysis/cam-calibrations/` (copied to `sourcedata/calibration/` on S3)

| File | Lens | Camera |
| --- | --- | --- |
| `acA1300-200um_C125-0418-5M.npz` | 4 mm | Basler acA1300-200um |
| `acA1300-200um_C125-0618-5M.npz` | 6 mm | Basler acA1300-200um |

The lens used per session is in `experiments.csv` column `lens` (`f4mm` or `f6mm`).
`.npz` keys: `mtx` (3×3 camera matrix), `dist` (distortion coefficients),
`rvecs`, `tvecs` (not used for undistortion).

**Note:** Since all videos are already undistorted and cropped, the calibration files are
kept for reference and for processing any new sessions in future.

### DeepLabCut Pose Tracking

**Model:** ResNet-50, trained on hm2p-maze project.
**Model name:** `DLC_resnet50_hm2p-mazeFeb17shuffle1_950000`

**Body parts tracked:**

| Keypoint | Description |
| --- | --- |
| `ear-left` | Left ear tip |
| `ear-right` | Right ear tip |
| `back-upper` | Upper back / base of neck |
| `back-middle` | Mid-back |
| `back-tail` | Base of tail |

**Output file:** HDF5 per video, `{video_basename}{DLC_iter_name}.h5`.
Loaded with `pd.read_hdf()`.

**HDF5 structure:** MultiIndex columns `(scorer, bodypart, coord)` where `coord` ∈
`{x, y, likelihood}`. Shape: `(n_frames, n_keypoints × 3)`.

**DLC output directory:** `hm2p/dlc/hm2p-maze-tristan-2023-02-17/videos/`

---

## 9. Derived Behavioural Metrics

Computed in legacy `utils/behave.py` by `calc_behav()`. All metrics are per camera frame.

### Head Direction (HD)

```python
absolute_hd = arctan2(ear_left_x - ear_right_x, ear_left_y - ear_right_y)
hd_degrees  = 180 + absolute_hd * 180 / pi          # range 0–360
hd_unwrapped = np.unwrap(hd_rad, discont=pi) * 180/pi  # continuous
```

### Angular Head Velocity (AHV)

First-order gradient of unwrapped HD, in deg/s.
Filtered version uses Gaussian smoothing before gradient.

### Position

Head position: centroid of `ear-left` and `ear-right` in mm.
Back position: `back-upper` keypoint in mm.
Pixel coordinates × `mm_per_pix` from `[scale]` section of `meta.txt`.

### Speed

Instantaneous: Euclidean distance between consecutive head positions per frame.
Smoothed: Gaussian-filtered speed trace. Units: cm/s.

### Movement State — Active / Inactive

```text
Active if speed > ACTIVE_SPEED_THRESH_UP (0.5 cm/s)
         OR AHV > ACTIVE_AHV_THRESH_UP (10 deg/s)

Inactive if speed < ACTIVE_SPEED_THRESH_LO (0.1 cm/s)
           AND AHV < ACTIVE_AHV_THRESH_LO (2 deg/s)

Hysteresis: requires ≥ 10 consecutive inactive frames to switch state
```

### Head-Mount Stuck Artefact (`bad_behav`)

Mice can get stuck on the HM2P head-mounted scope fibre and wires, creating artefactual
periods of immobility (speed ≈ 0, AHV ≈ 0). These are manually identified per session
by inspection of the behavioural video and logged in `experiments.csv`:

```text
bad_behav_times: "1:32-2:15;5:40-6:01"   (mm:ss-mm:ss, semicolon-separated)
```

The legacy function `get_bad_behav_indexes()` converts these to a boolean array mask.
In the new pipeline this mask is stored as `kinematics.h5:/bad_behav (N,) bool`.
All downstream analyses should exclude frames where `bad_behav == True`.

### Light State (`light_on`)

The periodic 1 min on / 1 min off lighting cycle is reconstructed per camera frame
from DAQ timestamps:

```python
# Per camera frame: find nearest light_on and light_off event
i_on  = searchsorted(light_on_times,  cam_time, side='right') - 1
i_off = searchsorted(light_off_times, cam_time, side='right') - 1
light_on[frame] = abs(light_on_times[i_on] - cam_time) < abs(light_off_times[i_off] - cam_time)
```

Stored as `kinematics.h5:/light_on (N,) bool`. This is a key experimental variable —
analyses compare HD tuning under light vs dark conditions.

### Maze Coordinate System

Position is mapped from pixels → mm → maze units:

```python
x_maze = (x_mm - roi_x1_mm) / roi_width_mm * maze_square_w   # maze_square_w = 7.0
y_maze = (y_mm - roi_y1_mm) / roi_height_mm * maze_square_h   # maze_square_h = 5.0
```

Out-of-bounds positions are clipped to the nearest point inside the rose-maze boundary
using a `shapely.geometry.Polygon`:

```python
Polygon([(0,0),(3,0),(3,1),(2,1),(2,2),(5,2),(5,1),(4,1),(4,0),(7,0),(7,1),(6,1),
         (6,4),(7,4),(7,5),(4,5),(4,4),(5,4),(5,3),(4,3),(4,5),(3,5),(3,3),(2,3),
         (2,4),(3,4),(3,5),(0,5),(0,4),(1,4),(1,1),(0,1)])
```

This polygon encodes the rose-maze shape (a 7 × 5 unit grid with corridors). Positions
outside this polygon are clamped to the nearest boundary point.

---

## 10. Calcium Signal Processing (Legacy Logic)

Reference: `proc/proc_ca.py` and `utils/ca.py`.

### Neuropil Subtraction (Suite2p)

```python
F_corr = F - ops["neucoeff"] * Fneu   # neucoeff = 0.7
```

### Baseline F0

Sliding window estimate using min-filter followed by max-filter on Gaussian-smoothed trace:

```python
Flow = gaussian_filter(F_corr, sigma=[0, sig_baseline * fps])
Flow = minimum_filter1d(Flow, size=win_baseline * fps)
Flow = maximum_filter1d(Flow, size=win_baseline * fps)
F0   = Flow
dFoF0 = (F_corr - F0) / F0
```

### Event Detection — Voigts & Harnett Method

Parameters (from `utils/ca.py`):

| Parameter | Value | Notes |
| --- | --- | --- |
| `smooth_sigma` | 3 frames | Gaussian smoothing before noise estimation |
| `prc_mean` | 40th percentile | Estimate of signal mean for noise model |
| `prc_low` | 10th percentile | Lower bound of noise Gaussian |
| `prc_high` | 90th percentile | Upper bound of noise Gaussian |
| `prob_onset` | 0.2 | Noise probability threshold for event onset |
| `prob_offset` | 0.7 | Noise probability threshold for event offset |
| `alpha` | 1 (EVT_DET_ALPHA) | Significance threshold |

**Algorithm:**

1. Rectify dF/F0 (clip negatives to 0)
2. Gaussian-smooth and normalise to [0, 1]
3. Fit a Gaussian noise model from percentiles: `mean = prc_mean`, `std = prc_high − prc_low`
4. Compute noise probability: `p_noise = 1 − CDF(trace, mean, std)` × 2 (two-tailed fold)
5. Detect onset: `1 − p_noise` crosses `1 − prob_onset` (rising)
6. Detect offset: `p_noise` rises above `prob_offset`
7. Validate: event must reach `alpha` significance within its duration

**Outputs:** `onsets`, `offsets`, `masks` (bool array), `amps` (peak dF/F0 per event).

**Note for new pipeline:** CASCADE (Rupprecht et al. 2021) replaces this as the primary
spike inference method, providing calibrated spikes/s in physical units. V&H is retained
as a fallback for comparison.

### SNR Calculation

```python
signal = mean(event_amplitudes)          # mean peak dF/F0 per event
noise  = std(dFoF0 during non-events, excluding bad frames)
snr    = signal / noise
```

---

## 11. Metadata CSVs

### `experiments.csv`

Columns:

| Column | Type | Notes |
| --- | --- | --- |
| `exp_index` | int | Sequential index |
| `exp_id` | str | Session ID (`YYYYMMDD_HH_MM_SS_<animal_id>`) — all sessions are rose-maze type |
| `implant_date` | date | Surgery date |
| `zstack_id` | str | Reference z-stack session ID |
| `fibre` | str | Fibre type (e.g. `SFB`) |
| `lens` | str | Camera lens (`f4mm` or `f6mm`) |
| `orientation` | float | Rotation angle (degrees) applied to all keypoint coords to correct for per-session camera placement variation; ensures HD is referenced consistently across sessions |
| `maze_session_num` | int | Which visit to the maze for this animal |
| `bad_2p_frames` | str | Manual bad-frame ranges, format `start-end;start-end` (frame numbers) |
| `bad_behav_times` | str | Manual bad-behaviour ranges, format `MM:SS-MM:SS;…` |
| `primary_exp` | int | 1 = primary analysis session |
| `exclude` | int | 1 = excluded from analysis |
| `Notes` | str | Free-text notes |

**New columns to add:** `extractor` (default `"suite2p"`) and `tracker` (default `"dlc"`)
for pluggable backend selection. To be added when the project skeleton is set up.

**Bad 2P frames:** stored as frame-number ranges (`16840-18000`). `-1` or `end` means
until the last frame. Used to mask out PMT dropout artefacts that Suite2p did not catch.

**Bad behav times:** stored as `MM:SS-MM:SS` time ranges. Converted to frame indices
using `fps`. Used to exclude periods where tracking was lost or the mouse left the maze.

### `animals.csv`

| Column | Notes |
| --- | --- |
| `animal_id` | Numeric animal ID (e.g. `1114353`) |
| `short_id` | Human-readable shorthand (e.g. `7f-01`) |
| `dob` | Date of birth |
| `sex` | m/f |
| `strain` | Mouse strain — all animals are Penk-Cre line (some also Rbp4) |
| `gcamp` | GCaMP variant: `7f` = GCaMP7f (most), `8f` = GCaMP8f (animal 1116663) |
| `celltype` | `"penk"` = Penk+ RSP cells (virus ADD3); `"nonpenk"` = CamKII+ RSP cells (virus 344) |
| `virus_id` | Virus identifier — ADD3 (Penk-driven GCaMP7f), 344 (CamKII-driven GCaMP7f), A122, A160.1+A83 (Flp-dependent) |
| `hemisphere` | Hemisphere of injection (L/R) |
| `injection_date` | Virus injection date |
| `has_serial2p` | Whether a serial section 2P brain atlas exists |
| `inj_ap`, `inj_ml`, `inj_dv` | Injection coordinates (mm from bregma) |

---

## 12. Legacy Processed HDF5 Files

These are the legacy aggregated databases in `hm2p/db/`. Each file stores all sessions
as a pandas DataFrame keyed by `exp_id`. **Do not write to these files.** Documented
here as a reference for the new schema design.

| File | Contents |
| --- | --- |
| `behave.h5` | Per-frame behavioural metrics (HD, speed, AHV, position) at camera rate |
| `behave_frames.h5` | Behavioural metrics resampled to imaging frame rate |
| `behave_events.h5` | Lighting on/off event tables |
| `ca_rois.h5` | Per-ROI statistics (SNR, n_events, events/min, mean amplitude) |
| `ca.h5` | Per-frame calcium traces (dF/F0, events, deconvolved) at imaging rate |

---

## 13. Mapping Legacy → New NeuroBlueprint Layout

| Legacy path | New NeuroBlueprint path |
| --- | --- |
| `shared/lab-108/experiments/…/{session_id}/…_XYT.raw` | `rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/funcimg/…_XYT.raw` |
| `shared/lab-108/experiments/…/{session_id}/…_XYT.tif` | `rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/funcimg/…_XYT.tif` |
| `shared/lab-108/experiments/…/{session_id}/….meta.txt` | `rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/funcimg/….meta.txt` |
| `shared/lab-108/experiments/…/{session_id}/…-di.tdms` | `rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/funcimg/…-di.tdms` |
| `hm2p/video/{session_id}/….mp4` | `rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/behav/….mp4` |
| `hm2p/video/{session_id}/meta/` | `rawdata/sub-{animal_id}/ses-{YYYYMMDD}T{HHMMSS}/behav/meta/` |
| `hm2p/s2p/{session_id}/suite2p_soma/` | `derivatives/ca_extraction/sub-{animal_id}/ses-…/suite2p_soma/` |
| `hm2p/dlc/…/{video_base}DLC*.h5` | `derivatives/pose/sub-{animal_id}/ses-…/{video_base}DLC*.h5` |
| `hm2p-analysis/metadata/animals.csv` | `sourcedata/metadata/animals.csv` |
| `hm2p-analysis/metadata/experiments.csv` | `sourcedata/metadata/experiments.csv` |
| `hm2p-analysis/cam-calibrations/` | `sourcedata/calibration/` |
| `hm2p/dlc/hm2p-maze-tristan-2023-02-17/` | `sourcedata/trackers/dlc/` |
