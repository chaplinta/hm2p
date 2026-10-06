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

### Movement metrics

Frame timing (gaps > 2× median interval), head direction and position missing
after filtering (warning > 2 %, problem > 10 %), agreement of the separate
head-direction estimators stored in `kinematics.h5` with the ear-based estimate
(nose-neck median |Δ| warning > 20°, problem > 40°), implausible speed
(> 100 cm/s) and AHV (> 1500 °/s), head vs body speed, filtered vs raw maze
position, occupancy, and whether the session was computed with the current DLC
champion. A two-minute excerpt around a light switch shows the traces.

### Syllables

Identifies the fit from `syllables.provenance.json` and scores each session
against the criteria in `docs/kpms-improvement-report.md`: median bout
300–500 ms, 20–40 syllables cover 80 % of frames, usage entropy ratio 0.6–0.8,
< 5 % single-frame bouts. For power-law usage the two usage criteria are only
met together in a narrow range, so a small miss on one of them is not by itself
a reason to reject a fit. Also: usage across sessions, ethogram, per-syllable
median speed and |AHV|, light vs dark usage, and bout-to-bout (not frame-level)
transitions.

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

## References

- Mathis A, Mamidanna P, Cury KM, et al. 2018. "DeepLabCut: markerless pose estimation of user-defined body parts with deep learning." Nature Neuroscience 21:1281–1289. doi:10.1038/s41593-018-0209-y. https://github.com/DeepLabCut/DeepLabCut
- Weinreb C, Pearl JE, Lin S, et al. 2024. "Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics." Nature Methods 21:1329–1339. doi:10.1038/s41592-024-02318-2. https://github.com/dattalab/keypoint-moseq
- Chen T, Guestrin C. 2016. "XGBoost: a scalable tree boosting system." Proceedings of KDD 2016, 785–794. doi:10.1145/2939672.2939785. https://github.com/dmlc/xgboost
- Voigts J, Harnett MT. 2020. "Somatic and dendritic encoding of spatial variables in retrosplenial cortex differs during 2D navigation." Neuron 105:237–245. doi:10.1016/j.neuron.2019.10.016.
- Zong W, Obenhaus HA, Skytøen ER, et al. 2022. "Large-scale two-photon calcium imaging in freely moving mice." Cell 185:1240–1256. doi:10.1016/j.cell.2022.02.017.
- Rupprecht P, Carta S, Hoffmann A, et al. 2021. "A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging." Nature Neuroscience 24:1324–1337. doi:10.1038/s41593-021-00895-5. https://github.com/HelmchenLabSoftware/Cascade
- Mukamel EA, Nimmerjahn A, Schnitzer MJ. 2009. "Automated analysis of cellular signals from large-scale calcium imaging data." Neuron 63:747–760. doi:10.1016/j.neuron.2009.08.009.
