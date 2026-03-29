# Rastermap Analysis Plan

## Paper Summary

Stringer C et al. 2025. "Rastermap: a discovery method for neural population
recordings." Nature Neuroscience 28:201-212. doi:10.1038/s41593-024-01783-4

**What it does:** Rastermap sorts neurons along a 1D axis so that nearby
neurons have similar activity patterns, then displays a sorted raster plot.
Unlike t-SNE/UMAP (which embed in a separate space), Rastermap directly
shows the neural activity with neurons reordered for maximum interpretability.

**Algorithm:**
1. K-means clustering of neural activity into ~100 clusters
2. Compute asymmetric similarity between clusters using peak cross-correlation
   at non-negative time lags (captures sequential activity)
3. Optimize a permutation to match a target matrix combining global (power-law)
   and local (sequential) structure
4. Upsample to assign individual neurons positions in the sorted order

**Key properties:**
- Runs in <2 min on tens of thousands of neurons
- Handles multiplexed activity (sustained, sequential, tuned simultaneously)
- Outperforms t-SNE, UMAP, Isomap, Laplacian eigenmaps on neural data
- Works on: 2P calcium imaging, wide-field imaging, Neuropixels, artificial NNs
- Superneurons: average nearby neurons in the sorting for denoised visualization
- Does NOT require Suite2p — works on any (n_neurons × n_timepoints) matrix

**Limitation:** Ineffective when neural responses are intrinsically
high-dimensional (e.g., V1 responses to many natural images). Works best
when activity is low-dimensional (sequences, sustained states, tuning).

## Relevance to hm2p

RSP head direction cells during freely-moving behaviour should have clear
low-dimensional structure:
- HD-tuned cells sorted by preferred direction → sequential activation
- Movement-modulated cells → sustained activity during locomotion
- Light/dark responsive cells → state-dependent groups

Rastermap can reveal these patterns without knowing what the tuning is
beforehand. Comparing the sorted raster between Penk+ and Penk⁻CamKII+
sessions shows whether the two populations have similar or different
activity structures.

## Analysis Plan

### 1. Per-session Rastermap sorting

For each session, compute Rastermap on the dF/F matrix:

```python
from rastermap import Rastermap
model = Rastermap(n_clusters=100, n_PCs=200, time_lag_window=10).fit(dff)
isort = model.isort  # neuron sort order
```

This gives a sorting of all ROIs. Display as a sorted raster plot with
behavioural variables (speed, HD, light) alongside.

### 2. Superneuron time courses

Average groups of ~10 nearby neurons in the sorting to create denoised
"superneuron" traces. Correlate superneuron activity with:
- Head direction (circular correlation)
- Speed
- Light state
- AHV

### 3. Penk+ vs Penk⁻CamKII+ comparison

- Do the two populations produce similar Rastermap patterns?
- Is the dimensionality of activity different? (number of clusters with
  distinct patterns)
- Are the same behavioural variables represented?

### 4. Condition-specific Rastermap

Run Rastermap separately on:
- Light-on epochs only
- Light-off epochs only
- Moving epochs only
- Stationary epochs only

Compare: does the structure change between conditions? If HD sequences
disappear in the dark, that's evidence for visual anchoring.

### 5. All-ROIs Rastermap (including non-cells)

Since Rastermap works on any matrix, run it on ALL Suite2p ROIs (not
just accepted cells). The non-cell ROIs (neuropil) will cluster separately
from soma ROIs, giving a natural separation of cell vs neuropil activity.

## Implementation

### Module: `src/hm2p/analysis/rastermap_analysis.py`

```python
def compute_rastermap(dff, n_clusters=100, n_PCs=200, time_lag_window=10)
    → dict with isort, embedding, model

def compute_superneurons(dff, isort, bin_size=10)
    → (n_superneurons, n_frames) averaged traces

def rastermap_behaviour_correlation(superneurons, hd, speed, light_on)
    → per-superneuron correlations with behavioural variables

def compare_rastermap_conditions(dff, mask_a, mask_b, ...)
    → sorting similarity between two conditions
```

### Frontend: `frontend/pages/rastermap_page.py`

Tabs:
1. Sorted Raster — full session raster plot (dF/F heatmap, neurons sorted)
   with speed, HD, light traces alongside
2. Superneurons — denoised time courses + behavioural correlations
3. Condition Comparison — side-by-side rasters for light/dark, moving/stationary
4. Cross-Session — Penk+ vs Penk⁻CamKII+ structure comparison

## References

- Stringer C et al. 2025. Nat Neurosci 28:201-212. doi:10.1038/s41593-024-01783-4
- GitHub: https://github.com/MouseLand/rastermap
