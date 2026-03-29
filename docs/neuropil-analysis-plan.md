# Neuropil Signal Analysis Plan

## Background

The neuropil signal (Fneu in Suite2p) represents the aggregate fluorescence
from the ring of tissue surrounding each ROI — primarily axonal and dendritic
processes, not cell bodies. This signal reflects local network input activity
rather than the output (spiking) of individual neurons.

Key insight from the literature:

- **Kerr et al. 2005**: The neuropil calcium signal is tightly correlated with
  the electrocorticogram (ECoG) and represents bulk calcium signals in axonal
  structures — a measure of local input activity. They call this the "optical
  encephalogram" (OEG).

- **Dipoppa et al. 2018**: Locomotion modulates neural activity differently
  across cell types in V1. Locomotion increases baseline activity and changes
  effective synaptic connectivity. The neuropil signal captures these
  network-level state changes.

- **Ali & Kwan 2019**: Review of calcium signal interpretation. Neuropil
  contamination is a known confound in somatic imaging, but the neuropil
  signal itself contains information about local circuit input.

- **Vickers & McCormick 2024**: Pan-cortical mesoscale imaging reveals large
  populations that coordinate activity with spontaneous movement and arousal
  changes — consistent with widespread movement-related signals (Zagha et al.
  2022).

- **Nietz et al. 2022**: Wide-field calcium imaging captures mesoscale network
  dynamics. The neuropil-dominated signal at low magnification reflects
  population-level input patterns.

- **Helmchen & Denk 2005**: Technical basis for 2P imaging in deep tissue.
  Neuropil fluorescence originates from the dense mesh of axonal and dendritic
  processes in the neuropil layer.

## What the neuropil signal tells us in hm2p

In RSP during freely-moving behaviour, the mean neuropil signal should
reflect:

1. **Arousal / brain state** — correlated with pupil dilation, movement onset
2. **Movement-related input** — locomotion speed, AHV, vestibular signals
3. **Visual input** — light vs dark should modulate neuropil differently
   from somatic signals if visual input is a major driver
4. **Network synchrony** — neuropil correlation across ROIs reflects shared
   input vs independent processing

For the Penk+ vs Penk⁻CamKII+ comparison, differences in neuropil signal
could indicate differences in the input these populations receive, independent
of their somatic output (HD tuning).

## Analysis plan

### 1. Mean neuropil signal extraction

Suite2p stores Fneu.npy (n_rois × n_frames) — the neuropil ring fluorescence
for each ROI. The mean across all neuropil rings gives a population-level
input signal.

```
mean_fneu(t) = mean(Fneu[i, t] for all accepted ROIs i)
```

Also compute the neuropil-to-soma ratio per ROI:
```
neuropil_ratio(i) = mean(Fneu[i]) / mean(F[i])
```

### 2. Correlation with behaviour (per session)

For each session, correlate the mean neuropil signal with:

- **Speed** (cm/s) — Spearman correlation
- **|AHV|** (deg/s absolute) — Spearman correlation
- **Light state** — compare mean Fneu during light-on vs light-off
  (Mann-Whitney U on per-epoch means, not per-frame)
- **Active vs stationary** — compare Fneu above/below speed threshold

### 3. Neuropil modulation by condition (2×2 factorial)

Same as somatic analysis: moving×light, moving×dark, stationary×light,
stationary×dark. Compare mean neuropil signal across conditions.

### 4. Penk+ vs Penk⁻CamKII+ neuropil comparison

- Compare neuropil-to-soma ratios between cell types
- Compare neuropil modulation indices (movement, light) between cell types
- Test whether input-level signals differ between populations

### 5. Neuropil–soma decorrelation

For each ROI, compute the correlation between its neuropil signal and its
somatic dF/F. High correlation = the somatic signal is dominated by shared
input (neuropil contamination). Low correlation = the cell has independent
activity beyond the network input.

Compare this decorrelation between Penk+ and Penk⁻CamKII+.

### 6. Neuropil PCA

PCA on the neuropil matrix (n_rois × n_frames) — do the top PCs correlate
with movement/light more strongly than the somatic PCs? This tests whether
the neuropil captures brain-state signals more directly than somatic activity.

## Implementation

### Data source

Fneu.npy is already on S3 at:
`ca_extraction/{sub}/{ses}/suite2p/plane0/Fneu.npy`

Also need F.npy for neuropil-to-soma ratios, and kinematics from sync.h5
for behavioural correlations.

### New module: `src/hm2p/calcium/neuropil_analysis.py`

Functions:
- `compute_mean_neuropil(Fneu, cell_mask)` → (n_frames,)
- `compute_neuropil_ratio(F, Fneu)` → (n_rois,)
- `neuropil_behaviour_correlation(mean_fneu, speed, ahv, light_on)` → dict
- `neuropil_condition_rates(Fneu, speed, light_on, active_mask)` → dict
- `neuropil_soma_correlation(F_corr, Fneu, dff)` → (n_rois,)

### New frontend page: `frontend/pages/neuropil_analysis_page.py`

Tabs:
1. **Overview** — mean neuropil time course, neuropil-to-soma ratio distribution
2. **Behaviour** — correlations with speed, AHV, light
3. **Conditions** — 2×2 factorial bar plots
4. **Celltype comparison** — Penk+ vs Penk⁻CamKII+ neuropil properties
5. **Decorrelation** — neuropil–soma correlation per ROI
6. **PCA** — neuropil PCs vs behaviour

## References

- Kerr JND et al. 2005. PNAS 102(39):14063-14068.
- Dipoppa M et al. 2018. Neuron 98(3):602-615.
- Ali F & Kwan AC. 2019. Neurophotonics 7(1):011402.
- Vickers ED & McCormick DA. 2024. bioRxiv.
- Nietz AK et al. 2022. Biology 11(11):1601.
- Helmchen F & Denk W. 2005. Nature Methods 2(12):932-940.
