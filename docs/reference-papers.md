# Reference Papers and Methods Comparison

This document summarises the three key reference papers for the hm2p pipeline, the methods they describe, and how they compare to both the legacy (`old-pipeline/`) and new (`src/hm2p/`) code.

---

## 1. Microscope — Zong et al. 2017 (Nature Methods)

**Paper:** "Fast high-resolution miniature two-photon microscopy for brain imaging in freely behaving mice"

**Relevance:** The FHIRM-TPM is the basis for the microscope used in hm2p, with modifications.

### Key specs (from paper)

| Parameter | Value |
|-----------|-------|
| Headpiece weight | 2.15 g |
| Excitation wavelength | 920 nm (custom HC-920 hollow-core PCF) |
| Lateral resolution | 0.64 um |
| Axial resolution | 3.35 um |
| Frame rate | 40 Hz at 256x256 (raster), 10 kHz (line scan) |
| FOV | 130 x 130 um^2 |
| Scanner | MEMS mirror |
| Detection | SFB (supple fiber bundle, 800 fused glass fibers) |
| Indicator | GCaMP6f |

### Ca2+ transient detection (Zong 2017 method)
- Template-matching algorithm considering amplitude and rise/decay time constants
- Acceptance criterion: peak amplitude > 3x SD of baseline
- This is a **different** method from what the hm2p pipeline uses (V&H percentile-based)

### Motion correction (Zong 2017)
- Running Z Projector (ImageJ) for 3-frame running projection to improve SNR
- Frame-to-frame correlation for drift correction
- Iterative image registration for residual motion artifacts

### Modifications in hm2p setup
- The exact modifications to the FHIRM-TPM should be documented by the user
- hm2p uses SciScan for acquisition (not the original LabVIEW FPGA control)
- Frame rates in hm2p data are ~9.6-9.8 Hz (not 40 Hz), consistent with larger FOV or different scan settings
- Single plane imaging per session

---

## 2. MINI2P analysis pipeline — Zong et al. 2022 (Cell)

**Paper:** "Large-scale two-photon calcium imaging in freely moving mice"

**Relevance:** The analysis methods from this paper are used/adapted in hm2p. The MINI2P is the next-generation microscope from the same group (Moser lab, NTNU).

### Data processing pipeline (NATEX)

#### Motion correction
- Suite2p rigid + non-rigid registration
- 256x256 frames, 6x6 blocks for non-rigid
- Quality metrics: 30 spatial principal components (SPCs), check for z-drift by comparing top/bottom 500 frames

**Comparison with hm2p:**
- hm2p uses Suite2p for motion correction identically (default settings)
- No additional z-drift checking is implemented in hm2p — could be added as a validation step

#### Neuropil subtraction
- Fixed coefficient: `Fcorr(t) = Fcell(t) - 0.7 * Fneu(t)`
- Coefficient 0.7 as suggested by Pachitariu et al. 2018

**Comparison with hm2p:**
- Old pipeline (`S2PData.__init__`): `self.FCorr = self.F - self.ops["neucoeff"] * self.Fneu` — uses Suite2p's `neucoeff` from ops (default 0.7)
- New pipeline (`calcium/neuropil.py`): `subtract_fixed_coefficient(F, Fneu, coefficient=0.7)` — same method, hardcoded 0.7
- **Match:** Both old and new code implement the same neuropil subtraction as Zong 2022

#### Baseline (F0) estimation
- Running baseline: `F0(t) = Fs(t) + m`
- `Fs(t)` = 8th percentile of `Fcorr(t)` within +/-15s moving window
- `m` = constant added so F0 is centered around zero during baseline periods
- Baseline points: local SD (within +/-15s) does not exceed `stdmin + 0.1*(stdmax - stdmin)`

**Comparison with hm2p:**
- Old pipeline (`S2PData`, calls `get_f0`): sliding window min of Gaussian-smoothed trace (Suite2p method — `win_baseline` and `sig_baseline` from ops)
- New pipeline (`calcium/dff.py`): sliding window minimum of Gaussian-smoothed trace (same Suite2p method)
- **Difference from Zong 2022:** hm2p uses Suite2p's sliding-window-minimum baseline (more standard), not the 8th-percentile + baseline-point method from Zong 2022. The Zong 2022 method is more sophisticated for handling photobleaching, but Suite2p's method is adequate for the hm2p data which shows minimal bleaching.

#### dF/F calculation
- `dF/F(t) = (Fcorr(t) - F0(t)) / F0(t)`

**Comparison:** Identical in old pipeline, new pipeline, and Zong 2022.

#### Significant transient detection
- Transients where dF/F > 2x local SD of baseline for > 0.75s
- Used to create "clean" traces: `dF/F(t)_clear` and `E(t)_clear` (signal outside events set to zero)

**Comparison with hm2p:**
- Old pipeline: uses V&H event detection (percentile-based noise model), NOT the 2x SD threshold
- New pipeline: uses V&H event detection (reimplemented in `calcium/events.py`)
- **Difference:** hm2p uses V&H which is more sensitive and does not require a fixed SD threshold. The Zong 2022 method is simpler but less adaptive.

#### Deconvolution
- Suite2p non-negative deconvolution with exponential kernel (tau = 1.5s for GCaMP6s)
- Deconvolved events used for all spatial tuning analyses
- Events = non-zero incidences of deconvolved calcium activity

**Comparison with hm2p:**
- Old pipeline: uses Suite2p deconvolution (`spks.npy`), normalized to max
- New pipeline: Suite2p deconvolution is enabled (`do_deconvolution = True`). The deconvolved
  trace (`deconv`) is stored in ca.h5 alongside dF/F and V&H events. The Stage 6 multi-signal
  analysis pipeline runs all metrics using all three signal types (dff, deconv, events) and
  saves results per signal type in analysis.h5, enabling direct comparison of whether
  conclusions hold across measures.
- **CASCADE** (calibrated spike inference) remains deferred — requires a separate conda
  environment (tensorflow==2.3, Python 3.8). When available, CASCADE spike rates would
  replace Suite2p deconv as the primary inferred-spike signal.
- **Note:** Zong 2022 uses deconvolved events for spatial tuning, not raw dF/F transients. The new pipeline follows this approach by including deconv as one of the three signal types in all analyses.

#### SNR calculation
- Signal = mean amplitude over all 90th percentiles of dF/F in significant transients
- Noise = mean of differences of dF/F outside significant transients
- Threshold: SNR > 3 required for spatial analysis

**Comparison with hm2p:**
- Old pipeline (`proc_ca.py`): `SNR = mean(event amplitudes) / std(dF/F during non-event periods)` — similar but not identical
- New pipeline (`calcium/events.py:compute_event_snr`): same as old pipeline
- **Difference:** Zong 2022 defines signal as 90th percentile within transients, hm2p uses peak amplitude. Zong 2022 defines noise as mean of dF/F differences (effectively derivative), hm2p uses std of dF/F.

#### Head direction computation
- HD = 90 degrees anticlockwise to the direction from left ear to right ear
- DLC with 4 body parts: left ear, right ear, body center, tail base
- Likelihood threshold: < 0.5 replaced with interpolated value
- Position smoothing: linear regression in 0.2s (3 frame) windows
- Speed: distance between adjacent frames / frame interval

**Comparison with hm2p:**
- Old pipeline: similar HD computation from ear positions
- New pipeline (`kinematics/compute.py`): HD from ear vector, with orientation correction from `experiments.csv`
- Body parts tracked in hm2p (SuperAnimal TopViewMouse names): `left_ear`, `right_ear`, `mid_back`, `mouse_center`, `tail_base` (5 vs 4 in Zong 2022 — `mouse_center` ≈ body-center; `mid_back` replaces back-upper)
- **Key difference:** Zong 2022 defines HD as 90deg anticlockwise from left-to-right ear vector. hm2p should verify this convention matches.

#### Spatial tuning analysis
- Place cells: spatial information (Skaggs 1996), shuffled controls (200-1000 permutations), intra-trial stability, place field size criteria
- HD cells: mean vector length (MVL), shuffled controls, intra-trial stability
- Grid cells: spatial autocorrelation, grid score
- Speed filter: data at < 2.5 cm/s excluded from all spatial analyses
- Bin sizes: 2.5x2.5 cm spatial, 3 deg HD, 1.5 cm/s speed
- Smoothing: Gaussian (3 cm spatial, 6 deg HD, 3 cm/s speed)

**Comparison with hm2p:**
- These tuning analyses are downstream of the current pipeline stages. They should be implemented following Zong 2022 conventions when the analysis stage is built.

---

## 3. Event detection — Voigts & Harnett 2020 (Neuron)

**Paper:** "Somatic and Dendritic Encoding of Spatial Variables in Retrosplenial Cortex Differs during 2D Navigation"

**Relevance:** The V&H calcium event detection algorithm is the primary method used in hm2p. The paper also images RSC (same brain region as hm2p) and studies HD tuning.

### Microscope setup
- Conventional 2P with resonant/galvo scanner (Neurolabware)
- Frame rate: 9-11 Hz (similar to hm2p's ~9.8 Hz)
- 980 nm excitation with 4x passive pulse splitter
- Two-plane imaging via ETL (electrically tunable lens): dendrites at 20-60 um, soma at 350-500 um
- Rotating headpost for free horizontal rotation (not freely moving)

### V&H calcium transient detection algorithm

This is the algorithm reimplemented in `src/hm2p/calcium/events.py`:

1. **Amplitude matching:** Dendritic and somatic traces are amplitude-matched by peak amplitudes
2. **Smoothing:** Gaussian smooth, sigma = 3 frames (~0.5s at 6 Hz per plane)
3. **Noise probability estimation:**
   - Fit Gaussian to noise using quantiles (not averages, to avoid bias from sparse transients)
   - Mean = 40th percentile
   - Std = range of 10th to 90th percentiles
   - Compute CDF-based noise probability per frame
4. **Joint noise probability:** Product of dendritic and somatic probabilities (implements OR function)
5. **Event detection:**
   - Onset: P(noise) drops below 0.2
   - Offset: P(noise) rises above 0.7 while increasing
   - Higher offset threshold captures more of GCaMP decay for better amplitude estimation

### Event classification (soma vs dendrite — V&H specific)
- Joint: contributions from both soma and dendrite
- Local dendritic: 25th percentile of (dendritic - somatic) > 0.01, somatic quantile < 0.05
- Local somatic: somatic > dendritic (diff < -0.02), dendritic quantile < 0.02

**Comparison with hm2p:**

| Aspect | V&H paper | Old pipeline | New pipeline |
|--------|-----------|-------------|-------------|
| Smoothing sigma | 3 frames | 3 frames | 3 frames |
| Noise mean percentile | 40th | 40th | 40th |
| Noise std percentiles | 10th-90th | 10th-90th | 10th-90th |
| Onset threshold | P(noise) < 0.2 | P(noise) < 0.2 | P(noise) < 0.2 |
| Offset threshold | P(noise) > 0.7, rising | P(noise) > 0.7, rising | P(noise) > 0.7, rising |
| Alpha (significance) | Not mentioned | 1.0 (disabled) | 1.0 (disabled) |
| Joint soma-dendrite | Product of probabilities | Product of probabilities | Single-trace only |
| Event classification | Joint/local dendritic/local somatic | Joint/local dendritic/local somatic | Not implemented |

**Key differences:**
1. The **joint event detection** (product of soma and dendrite noise probabilities) is in V&H and the old pipeline but NOT in the new pipeline. The new pipeline detects events on single traces only. This is correct for hm2p because the experiment images soma and dendrites in the **same plane** (not separate planes as in V&H), so the joint detection is not applicable in the same way. However, if soma-dendrite pairs are identified post-hoc by shape, the joint detection could be added later.
2. The **event classification** (local somatic vs local dendritic vs joint) from V&H is implemented in the old pipeline (`get_joint_ca_events` in `utils/ca.py`) but not in the new pipeline. This could be added when dendrite ROI classification is implemented.

### Neuropil correction (V&H)
- Estimates neuropil mixing coefficient per ROI by linear fit to lowest 10th percentile
- Subtracts neuropil with this per-ROI coefficient
- **Different from** the fixed 0.7 coefficient used in Suite2p/Zong 2022/hm2p

### HD tuning analysis (V&H)
- Event rates: event count / occupancy in 40 bins, circularly smoothed (sigma = 0.075 rad)
- Spatial: 8x8 grid
- Speed filter: exclude stationary periods
- Information content: KL divergence between rate distribution and occupancy distribution

**Comparison:** V&H uses KL divergence for information, Zong 2022 uses Skaggs information. Both are standard; Skaggs is more common in the place/grid cell literature.

### Angle-dependent brightness correction (V&H specific)
- The rotating headpost causes angle-dependent brightness changes
- F0 computed across angles (not time) by taking percentile at each angle
- Not relevant to hm2p (standard benchtop-style 2P, no rotation)

---

## Source Code and Data Repositories

| Paper | Repository | Notes |
|-------|-----------|-------|
| Zong 2017 | [Protocol Exchange](http://dx.doi.org/10.1038/protex.2017.048) | Assembly protocol only; no analysis code |
| Zong 2022 | [github.com/kavli-ntnu/MINI2P_toolbox](https://github.com/kavli-ntnu/MINI2P_toolbox) | MINI2P hardware + NATEX analysis (MATLAB) |
| Zong 2022 | [Zenodo: 10.5281/zenodo.6033997](https://doi.org/10.5281/zenodo.6033997) | NATEX pipeline code + sample data |
| V&H 2020 | [github.com/jvoigts/rotating-2p-image-correction](https://github.com/jvoigts/rotating-2p-image-correction) | Angle-dependent brightness correction (not used in hm2p) |

**Note:** The V&H repository contains the rotating headpost brightness correction code, not the calcium event detection algorithm. The event detection method is described in the paper's STAR Methods and reimplemented in `src/hm2p/calcium/events.py`. The NATEX repository (Zong 2022) contains the full MATLAB pipeline including motion correction, neuropil subtraction, baseline estimation, and spatial tuning analysis.

---

## Summary: Methods used in hm2p

| Processing step | Method source | Old pipeline | New pipeline | Notes |
|----------------|---------------|-------------|-------------|-------|
| Motion correction | Suite2p | Suite2p | Suite2p | Standard |
| Neuropil subtraction | Suite2p/Zong 2022 | ops["neucoeff"] (~0.7) | Fixed 0.7 | Could add FISSA later |
| Baseline (F0) | Suite2p | Sliding-window min of Gaussian-smoothed | Same | Zong 2022 uses 8th percentile method |
| dF/F | Standard | (F-F0)/F0 | (F-F0)/F0 | Identical |
| Event detection | V&H 2020 | Full V&H with joint | V&H single-trace | Joint detection not needed for single-plane |
| Deconvolution | Suite2p | spks.npy (normalized) | spks.npy in ca.h5 as `deconv`; used in multi-signal analysis | Zong 2022 uses deconvolved for spatial tuning |
| SNR | Custom | mean(amp)/std(non-event) | Same | Zong 2022 uses different definition |
| HD computation | Zong 2022 / V&H | Ear vector | Ear vector | Verify 90-deg convention |
| Spatial tuning | Zong 2022 | Skaggs SI, MVL, shuffled | `analysis/tuning.py`, `analysis/information.py` | Follows Zong 2022 methods |
| Speed filter | Zong 2022 | Exclude < threshold | `analysis/speed.py` | 2.5 cm/s threshold |
