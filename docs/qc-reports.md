# Data-quality reports

Five standalone HTML pages for checking the processing of every session. Each
page embeds its data and Plotly.js, so it opens offline from `file://`. The
pages link to each other and must stay in the same folder (`docs/qc/`).

| Page | Stage checked | Input on S3 (`hm2p-derivatives`) |
| --- | --- | --- |
| `qc-tracking.html` | 2b pose tracking | champion DLC `.h5` in `pose/{sub}/{ses}/` |
| `qc-movement.html` | 3 kinematics | `kinematics/{sub}/{ses}/kinematics.h5` |
| `qc-syllables.html` | 3b keypoint-MoSeq | `kinematics/{sub}/{ses}/syllables.npz` (+ provenance, `kinematics/kpms_model/`) |
| `qc-rois.html` | 1 ROI classification | `ca_extraction/{sub}/{ses}/suite2p/plane0/*.npy`, `calcium/{sub}/{ses}/ca.h5` |
| `qc-spikes.html` | 4 calcium processing | `calcium/{sub}/{ses}/ca.h5` |

All 26 sessions are included regardless of `exclude` / `primary_exp`; those
flags are shown as labels. A session whose input is missing is listed with the
error instead of being dropped.

## Building

```bash
python scripts/make_qc_reports.py                          # read S3, summarise, cache
python scripts/make_qc_reports.py --reports rois --classifier-reference   # + LOSO reference
python scripts/build_qc_reports_html.py                    # write docs/qc/*.html
```

`make_qc_reports.py` only reads from S3 (profile `hm2p-agent`, override with
`HM2P_AWS_PROFILE`). Per-session summaries are cached in
`results/qc/cache/{report}/` (gitignored), so an interrupted run resumes;
`--refresh` ignores the cache, `--sessions` restricts to given `exp_id`s. The
report data go to `results/qc/data/{report}.json`.

Code: per-session summaries in `src/hm2p/qc/` (pure functions, unit-tested);
templates in `scripts/templates/qc_*.html` with shared helpers in
`qc_common.css` / `qc_common.js`.

## What each page checks

### Tracking

Raw tracker output before filtering. DLC 3 (PyTorch) likelihoods are not
calibrated probabilities (typically 0.1–0.5 for correct detections), so no
fixed likelihood cut-off is used to score a session. The page shows per-keypoint
likelihood distributions with the 25th percentile (the kinematics stage's
default `quantile:0.25` cut-off), a 10 s timeline, the lowest-confidence
stretches to check in the video, and checks from `hm2p.pose.quality`:

| Check | Definition | Warning / problem |
| --- | --- | --- |
| Jumps | frame-to-frame displacement > 150 cm/s | > 0.5 % / > 2 % of frames |
| Ear swaps | left ear on the minority side of the nose→neck axis within the session | > 1 % / > 5 % |
| Left ear side | fraction of frames with the left ear on the positive side; must fall on the same side of 50 % as most sessions (catches whole-session left/right mislabelling) | — / other side |
| Ear-distance, body-length outliers | > 3 robust SD (3 × 1.4826 MAD) from session median, or NaN | > 2 % / > 10 % |
| Nose-to-tail order | keypoints out of anatomical order along the body axis | > 5 % / > 15 % |
| Dark − light likelihood | median over keypoints | < −0.03 / < −0.08 |
| Brightness step at light switch | median over switches of (brightness 20 s after − 20 s before), lights-on minus lights-off sign, relative to the median brightness (from `analysis/…/illumination.h5`) | > 2 % / > 5 % either way |

Also on the page:

- **Frame viewer.** Per session about 15 still frames: the lowest-ranked frame
  in each of the eight low-confidence 10 s bins, the middle of the two longest
  ear-swap runs, the two largest jumps (for an out-and-back error, the frame
  where the keypoint is displaced) and three random reference frames, at least
  1 s apart (`hm2p.qc.frames.select_review_frames`). The frames are taken from
  the raw overhead video resampled with `ffmpeg -i … -r 30` (constant frame
  rate), which is how `scripts/run_dlc_retrain.py` made the DLC input; they are
  used only when the resampled frame count equals the pose file's (it did for
  both sessions checked), otherwise `labelled_30fps.mp4` is used and the page
  says so (that video already has the DLC points drawn in). The page draws the
  raw keypoints over the frame (filled at likelihood ≥ the pipeline cut-off,
  0.5 from `kinematics.h5` attrs, open below), the skeleton, and the stage 3
  head and body position and head direction mapped back into video pixels by
  undoing the stage 3 scale and orientation rotation (rotation centre = mean of
  all raw keypoint positions, as in `apply_orientation_rotation`). Clicking a
  row of the low-confidence table opens its frame. Frames are a lazily loaded
  asset, `results/qc/assets/tracking/frames-<sub>_<ses>.json` (grey JPEG,
  quality 70, full 832 × 608 resolution; ~1 MB per session). Collecting them
  downloads each raw video (~140 MB) and decodes it once (~3 min per session);
  `--no-tracking-frames` skips this.
- **Displacement trace.** Largest frame-to-frame displacement per keypoint (or
  over all keypoints) in 2 s bins, log scale, with the jump threshold.
- **Video brightness.** Mean frame brightness at 1 Hz with the DAQ light
  epochs.
- **DLC model.** `dlc-champion.json`, the promotion log
  (`dlc-champion-history/promotions.log`), the stored evaluation
  (`dlc-retrain/_eval_results.json`; flagged when it describes a different model
  than the current champion) and the promotion-gate verdict if present
  (`dlc-retrain/models/_compare_verdict.json`).

### Movement metrics

Frame timing (gaps > 2× median interval), head direction and position missing
after filtering (warning > 2 %, problem > 10 %), agreement of the separate
head-direction estimators stored in `kinematics.h5` with the ear-based estimate
(nose-neck median |Δ| warning > 20°, problem > 40°), implausible speed
(> 100 cm/s) and AHV (> 1500 °/s), head vs body speed, filtered vs raw maze
position, occupancy, and whether the session was computed with the current DLC
champion. A two-minute excerpt around a light switch shows the traces.
`bad_behav` intervals are drawn as red bars on the session timeline and the
excerpt.

Also on the page:

- **Trajectory map.** Every third frame (10 Hz) of the filtered body position
  (`x_maze`, `y_maze`) over the 23 maze cells, coloured by speed (capped at the
  95th percentile), head direction (cyclic scale) or light state, with the raw
  position (`x_maze_raw`) as an optional overlay and `bad_behav` frames marked.
  Lazily loaded asset `results/qc/assets/movement/traj-<sub>_<ses>.json`
  (~0.45 MB per session).
- **Bout durations.** Active and inactive bouts (runs of the frame-wise
  `active` state, outside `bad_behav`), lights on vs off, on a log time axis.
  Bouts that touch a light switch, a `bad_behav` interval or the session edge
  are censored and counted separately (`hm2p.qc.movement.bout_lengths`).

### Syllables

Identifies the fit from `syllables.provenance.json` and scores each session
against the criteria in `docs/kpms-improvement-report.md`: median bout
300–500 ms, 20–40 syllables cover 80 % of frames, usage entropy ratio 0.6–0.8,
< 5 % single-frame bouts. For power-law usage the two usage criteria are only
met together in a narrow range, so a small miss on one of them is not by itself
a reason to reject a fit. Also: usage across sessions, ethogram, per-syllable
median speed and |AHV|, light vs dark usage, and bout-to-bout (not frame-level)
transitions.

**What each syllable is** (`hm2p.qc.syllable_media`; pooled over the sessions
collected in the run). For the most-used syllables covering 95 % of frames (at
most 30), following the grid movies and typical-trajectory plots of
keypoint-MoSeq (Weinreb et al. 2024):

- *Grid movie*: up to 9 bouts, from different sessions where possible, picked
  with a fixed seed (1000 + syllable id) among bouts within 25 % of the
  syllable's median duration. A bout is skipped if its clip overlaps
  `bad_behav` frames or if neck, tail base or body centre have DLC likelihood
  < 0.5 in more than 10 % of its frames. Clips run from 10 frames before onset
  to 10 after offset (at most 75 frames), are cut from
  `pose/{sub}/{ses}/labelled_30fps.mp4`, cropped to a 180 mm square around
  `mouse_center` and rotated so tail base → neck at onset points up (fixed for
  the clip), and tiled 3 × 3 at 120 px. An orange border marks the bout.
  The labelled video is the cropped camera video resized to 416 × 304, not
  always by the same factor on both axes, so pose pixels are scaled with the
  crop size in `metadata/video_meta/` (sessions without it get no clips).
  Each grid is an H.264 mp4, base64 in
  `results/qc/assets/syllables/syl_<id>.json` (loaded lazily by the page;
  about 60–120 kB each).
- *Average trajectory*: champion DLC keypoints (likelihood < 0.5 removed, gaps
  ≤ 5 frames interpolated, 5-frame rolling median, via `movement`) in the
  egocentric frame of each bout onset, in mm, averaged over all usable bouts,
  frames −5 to +30. The egocentric frame uses camera-pixel keypoints, not the
  maze-frame keypoints in kinematics.h5 (reflected in some sessions).
- *Turn rate*: mean angular velocity of the neck → nose direction in the camera
  image, + = counter-clockwise seen from above (left turn). The left/right ear
  order in the camera image is checked per session; a session where it is
  reversed in > 80 % of frames would be flagged and its x axis and turn sign
  flipped.
- *Occupancy*: syllable frames per 0.25-cell bin of `x_maze`, `y_maze`, and the
  same divided by the occupancy of all good frames.

Per-session inputs are cached in `results/qc/cache/syllables_media/`
(`<exp_id>.json` + `.npz`); grid movies are re-rendered only when the bout
selection changes (or with `--refresh`). The video is read sequentially rather
than by seeking: seeking was very slow in some of the labelled videos.
Options: `--no-syllable-media`, `--no-syllable-clips`, `--tmp-dir` (where the
video and tile clips are written while rendering).

### Cell classification

Reference accuracy: leave-one-session-out predictions on the manually labelled
legacy sessions in `/data/s2p`, refitting with the champion's training recipe (hyper-parameters, balanced class weights, training-fold medians for
missing features) (`hm2p.qc.rois.cv_reference`); confusion matrix, per-class
precision/recall/F1, calibration, checked against the acceptance criteria in
`docs/soma-classifier.md` (macro F1 ≥ 0.85, each class ≥ 0.7, < 5 % artefacts
called soma). Per session: class counts, ambiguous ROIs (max probability
< 0.6), agreement between `roi_class.npy` and `ca.h5` `roi_types` (different
integer codes, recoded), whether re-applying the current model to recomputed
features reproduces the stored labels, ROI outlines on the mean image, and each
of the 26 features against the manual-label distributions (domain shift).

Suite2p inspection (`hm2p.qc.roi_assets`), loaded per session from
`docs/qc/assets/rois/<sub>_<ses>.js` when a session is selected:

- ROI map background: mean, enhanced mean (`meanImgE`), maximum projection or
  correlation image (`Vcorr`) at full resolution. `max_proj` and `Vcorr` are
  stored by Suite2p cropped to `ops["yrange"]` / `ops["xrange"]` and are placed
  back at that offset, so they line up with the ROI pixel coordinates.
- Selected ROI: a 41 × 41 px close-up with the Suite2p pixel weights (`lam`),
  the `stat.npy` fields (npix, radius, aspect_ratio, compact, skew, std,
  footprint), the `iscell.npy` probability (stage 1 overwrites `iscell.npy`
  with the classifier output, so this equals P(soma)) and the `ca.h5` `roi_qc` metrics
  with pass / fail against the thresholds in `hm2p.calcium.qc` (the ones the
  Streamlit ROI viewer uses).
- Traces of the soma and dendrite ROIs only (artefacts are not stored): Suite2p
  F and Fneu, FISSA `F_corr`, dF/F and CASCADE expected spikes per frame, with
  dark epochs shaded. Stored as int16; when the session would exceed
  `TRACE_BUDGET_BYTES` (2.5 MB) the traces are averaged over the smallest
  number of frames that fits, which the page states.
- `roi_qc` distributions by predicted class with the thresholds, and the
  registration offsets (`xoff`, `yoff`) and `corrXY` per frame.

Cached summaries from before the asset existed have no `asset` key; re-collect
with `python scripts/make_qc_reports.py --reports rois --refresh` (the page
then shows the downscaled inline mean image only).

### Spike extraction

Per ROI: robust noise SD (MAD of frame differences), SNR (99th percentile /
noise), V&H and SD-threshold event rates and their frame-level Jaccard overlap,
CASCADE mean rate (Hz: CASCADE outputs expected spikes per frame, multiplied
here by the frame rate; existing ca.h5 files carry a `spikes_units`
attribute "spikes/s", which is incorrect, and `scripts/run_cascade.py` now
writes "expected spikes per frame"), Spearman correlation of spikes
with dF/F, fraction of V&H
events containing CASCADE spikes, fraction of spike mass inside events, F0 drift
(last / first 5 %), fraction of frames with F below F0, plus the stored
`roi_qc` metrics. Per session: soma-mean dF/F and spike rate over time, pairwise
correlations, and the fraction of soma ROIs failing each threshold in
`hm2p.calcium.qc`. A 90 s trace viewer per ROI shows dF/F with both event masks
and the CASCADE rate, and the whole-session F with its F0 baseline.

## Findings, first full run (2026-10-06, all 26 sessions)

Code fixes for 1–3 are on branch `feat/qc-reports`; the stored data still
need re-running.

1. **Head direction: the ear estimate pointed backwards.**
   `_ear_perpendicular_angle` was ~180° from the nose-neck, nose-head and
   head-neck estimates and from travel direction while running (median
   |HD − travel| 157° vs ~20° for nose-neck). The fused `hd_deg` averaged
   opposite vectors: per session (median) 15 % of frames on the backward side
   and 17 % 45–135° off; while running it was further from travel than
   nose-neck alone (median 31° vs 20°). Fixed in `kinematics/compute.py`
   (also the head-body angle, which read +90° for a straight mouse). Stage 3,
   5 and 6 outputs predate the fix.
2. **ROI classifier and Suite2p ran at 29.97 Hz.** Stage 1 on EC2 had no
   timestamps.h5, so `fps_from_timestamps` fell back to 29.97 Hz (imaging is
   9.64–9.77 Hz) for Suite2p and for `classify_session`. Re-applying the
   model at 29.97 Hz reproduces every stored label; at the imaging rate 71 of
   4023 labels change (soma 463 → 467, dendrite 392 → 442). The fallback is
   removed and `classify_session` now requires `fps`.
3. **CASCADE units.** The stored `spikes` array is the expected number of
   spikes per frame, not spikes/s. Labels, rate conversions (× fps) and count
   conversions (plain sums) are fixed; results that use a fixed spike-count
   threshold (`sp_fraction_bins_active`, `sp_isi_cv_s`) change on re-run.
4. **Two dF/F scales.** The 9 sessions with the original FISSA run (all
   Penk+, up to 2021-12-03) have median F0 45–180; the 17 reprocessed through
   the EC2 FISSA bridge in 2026-06 have median F0 0.3–9, so their dF/F noise
   is ~6× higher (0.20–0.67 vs 0.05–0.14). CASCADE rate follows the noise
   (Spearman ρ = 0.83 across sessions); V&H event rate does not (ρ = 0.19).
   All Penk⁻CamKII+ sessions are in the reprocessed group, so amplitude- and
   noise-sensitive measures remain confounded with processing.
   Checked 2026-10-07 on one session per group (sub-1115465/ses-20211029T135008,
   sub-1117646/ses-20220804T112159): the two groups use different neuropil
   methods although both are labelled `neuropil_method=fissa`. In the original
   group `F_corr = F_raw − a·Fneu` exactly (per-cell a ≈ 0.3, intercept 0,
   R² = 1.000), i.e. coefficient subtraction, most likely the
   `subtract_estimated_coefficient` fallback taken when FISSA raised. The
   bridge group is FISSA output (baseline ≈ 0, a ≈ 1.06, R² = 0.86), so its
   dF/F relative to its own F0 is inflated 10–40×.
5. **Syllables** (κ = 10⁶, 4 PCs, AR-only 50 + 200 iterations): 3–10
   syllables cover 80 % of frames in every session (criterion 20–40); median
   bout 600–1100 ms in 20 of 26 (criterion 300–500 ms); 3 sessions do not
   match the kinematics length. `pca_variance.json` was never saved (fixed in
   `run_kpms.py`).
6. **Classifier reference** (leave-one-session-out, 3762 labelled ROIs,
   23 sessions): macro F1 0.824; soma F1 0.87; dendrite F1 0.64 (precision
   0.58: 116 of 3212 artefacts predicted as dendrite). Fails the dendrite
   criterion (≥ 0.7); artefacts labelled soma 1.5 % (passes).
7. **Tracking** is good in most sessions: ear side consistent in all
   sessions, no light/dark likelihood difference. Worst included session
   20220804_11_21_59 (13 % frames with a jump, 13 % out of nose-to-tail
   order, 11 % body-length outliers).
8. **Baseline drift:** soma F0 falls by > 40 % over the session (end/start
   < 0.6) in 13 of 26 sessions, down to 0.05.

## Re-run and second QC pass (2026-10-07)

Previous S3 outputs are kept under `s3://hm2p-derivatives/_backup_2026-10-07/`
(`calcium/`, `kinematics/`, `sync/`, `analysis/`).

What was re-run:

- **Stage 4, all 26 sessions, FISSA for all.** The 9 original sessions went
  through the EC2 FISSA bridge (`launch_stage4_fissa_ec2.py --sessions`,
  FISSA 1.0.0); two (sub-1115465 ses-20211028T112550, ses-20211102T151134)
  passed the registration alignment gate only at 0.85 (median Spearman 0.898
  and 0.888), the same threshold used for sub-1118317 ses-20221117T132031 in
  2026-06. The 17 bridge sessions were re-derived from their stored FISSA
  traces (`scripts/rerun_stage4_precomputed.py`). All 26: ROIs classified at
  the imaging frame rate; dF/F = (F_corr − F0(F_corr)) / F0(F_raw);
  `dff_denominator` attribute set.
- **CASCADE** on the new dF/F (stored array: expected spikes per frame).
- **Stage 3** (head-direction fix), **Stage 5** and **Stage 6**.

Second QC pass:

| Measure | Before | After |
| --- | --- | --- |
| Soma dF/F noise SD, Penk+ / Penk⁻CamKII+ (median over sessions) | 0.17 / 0.45 | 0.023 / 0.022 |
| Spearman, session noise vs CASCADE rate | 0.83 (p = 2×10⁻⁷) | 0.36 (p = 0.07) |
| Fused HD within 45° of the ear estimate | 4–13 % of frames | 99.8–99.9 % |
| Running: median \|HD − travel direction\| (2 sessions) | 30–32° | 21–22° |
| \|AHV\| > 1500 °/s (median over sessions) | 2.0 % | 0.07 % |
| ROI labels changed by re-classifying at the imaging rate | 71 / 4023 | 0 / 4023 |

Open points:

- **HD missing in 4–21 % of frames (median 12 %).** The per-keypoint
  confidence cut-off `quantile:0.25` discards each keypoint's lowest 25 % of
  frames by construction; for the current DLC model that cut-off is at
  likelihood 0.87–0.95 (median likelihood 0.96–1.00). A fixed threshold would
  keep more frames; not changed (affects all downstream results).
- **CASCADE rates are lower (0.01–0.24 Hz per soma).** FISSA dF/F is relative
  to the raw-trace baseline, which includes neuropil, so transient amplitudes
  are smaller than with coefficient subtraction; CASCADE is calibrated on dF/F
  amplitude, so absolute rates are likely underestimated, uniformly across
  sessions. In 3 sessions fewer than 60 % of V&H events contain ≥ 0.5 expected
  spikes (20210924, 20211216, 20220608_16_22).
- **Raw fluorescence falls by > 40 % over the session in 7 sessions**
  (bleaching or focus drift).
- **Syllables unchanged** (κ = 10⁶ fit still fails the bout-length and usage
  criteria); a κ sweep needs `launch_kpms_ec2.py` changes so trial fits do
  not overwrite `syllables.npz`.
- **Suite2p detection ran with fs = 29.97 Hz** (no timestamps.h5 on EC2);
  ROI sets were not re-detected.

## References

- Mathis A, Mamidanna P, Cury KM, et al. 2018. "DeepLabCut: markerless pose estimation of user-defined body parts with deep learning." Nature Neuroscience 21:1281–1289. doi:10.1038/s41593-018-0209-y. https://github.com/DeepLabCut/DeepLabCut
- Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics." Nature Methods 21:1329–1339. doi:10.1038/s41592-024-02318-2. https://github.com/dattalab/keypoint-moseq
- Chen T, Guestrin C. 2016. "XGBoost: a scalable tree boosting system." Proceedings of KDD 2016, 785–794. doi:10.1145/2939672.2939785. https://github.com/dmlc/xgboost
- Voigts J, Harnett MT. 2020. "Somatic and dendritic encoding of spatial variables in retrosplenial cortex differs during 2D navigation." Neuron 105:237–245. doi:10.1016/j.neuron.2019.10.016.
- Zong W, Obenhaus HA, Skytøen ER, et al. 2022. "Large-scale two-photon calcium imaging in freely moving mice." Cell 185:1240–1256. doi:10.1016/j.cell.2022.02.017.
- Rupprecht P, Carta S, Hoffmann A, et al. 2021. "A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging." Nature Neuroscience 24:1324–1337. doi:10.1038/s41593-021-00895-5. https://github.com/HelmchenLabSoftware/Cascade
- Mukamel EA, Nimmerjahn A, Schnitzer MJ. 2009. "Automated analysis of cellular signals from large-scale calcium imaging data." Neuron 63:747–760. doi:10.1016/j.neuron.2009.08.009.
