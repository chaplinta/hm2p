# Penk-Patching Pipeline: MATLAB-to-Python Port Plan

## Overview

The `old-penk-patching/` directory contains a MATLAB pipeline for analysing
whole-cell patch-clamp electrophysiology and confocal morphology data from Penk+
and non-Penk RSP neurons. The pipeline processes WaveSurfer H5 ephys files and
SWC morphology tracings, extracts ~50 electrophysiological and morphological
metrics per cell, and runs population-level statistical comparisons (Mann-Whitney
U), correlations (Spearman), and PCA between cell types.

This document details the plan to reimplement this pipeline in Python under
`src/hm2p/patching/`, following the project conventions in CLAUDE.md.

---

## 1. Module Structure

```
src/hm2p/patching/
    __init__.py
    config.py           # paths, constants, sampling rate, filter params
    io.py               # WaveSurfer H5 loading, SWC loading via navis
    ephys.py            # trace processing: filtering, deconcat, spike detection
    protocols.py        # protocol-specific extraction: IV, Rheobase, Passive, Sag, Ramp
    spike_features.py   # spike waveform feature extraction via eFEL
    morphology.py       # SWC loading, soma subtraction, rotation, metrics via navis
    metrics.py          # assemble per-cell metrics table, compute derived metrics
    statistics.py       # Mann-Whitney U, Spearman correlations, summary stats
    pca.py              # PCA on metric subsets, variance explained, loadings
    plotting/
        __init__.py
        ephys_plots.py      # IV traces, rheobase, spike waveforms, steady-state
        morph_plots.py      # single-cell morphology, population overlays, density
        violin_box.py       # violin + box + swarm plots for metric comparisons
        correlation_plots.py # scatter + Spearman r/p annotation
        pca_plots.py        # PC scatter, scree, loading bar charts
        confocal_stack.py   # mean/max z-projections, GIF from TIFF stacks
    run.py              # orchestrator: load metadata, loop cells, save metrics.csv
```

---

## 2. Module Details and MATLAB Mapping

### 2.1 `config.py`

**Purpose:** Centralise all constants and path configuration.

**Functions:**
| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `load_config(config_path) -> PatchConfig` | `loadPCDirs.m` + `config.ini` | Pydantic `BaseSettings` with fields: `metadata_dir`, `morph_dir`, `ephys_dir`, `processed_dir`, `analysis_dir`. Reads from YAML (not INI). |
| Constants: `SAMPLE_RATE = 20_000`, `SAMPLE_RATE_KHZ = 20`, `FILTER_CUTOFF = 1000`, `FILTER_ORDER = 4` | Hard-coded in `ephys_intr.m`, `pc_lowpassfilt.m` | |

### 2.2 `io.py` — File I/O

**Purpose:** Read WaveSurfer H5 files and SWC morphology files.

**Functions:**
| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `load_wavesurfer(path) -> dict` | `loadDataFile_wavesurfer.m` | Use `h5py` to read WaveSurfer `.h5` files. Recursively walk HDF5 tree, extract sweep data, apply analog scaling coefficients. The MATLAB function applies a cubic polynomial scaling from raw ADC counts — replicate this exactly. |
| `get_sweep_traces(ws_data, sweep_idx) -> np.ndarray` | Inline in `ephys_intr.m` | Extract `analogScans[:,0]` (voltage channel) from specified sweep. |
| `load_swc_files(tracing_path) -> dict[str, navis.TreeNeuron]` | `getTracingFiles.m` | Find `Soma.swc`, `Apical_tree.swc`, `Basal*.swc`, `Surface.swc` (optional), `Axon.swc` (optional). Load each via `navis.read_swc()`. Return dict keyed by type. |
| `find_tracing_path(morph_dir, date_confocal, slice_folder, hemisphere, cell_slice_id) -> Path` | Lines 54-81 in `procPC.m` | Handle the hemisphere subfolder logic: if cells exist in both hemispheres or default path doesn't exist, add `{slice_folder}_{hemisphere}` subfolder. |

**Key mapping — WaveSurfer H5 reading:**
- MATLAB `loadDataFile_wavesurfer` uses `h5read` recursively and scales raw ADC
  counts via `ws.scaledDoubleAnalogDataFromRaw()` (cubic polynomial with channel
  scales and scaling coefficients).
- Python: use `h5py` to walk groups/datasets. Read `/header/AIScalingCoefficients`
  and `/header/AIChannelScales`. Apply: `scaled = sum(coeff[i] * raw**i for i in range(4)) / channel_scale`.

### 2.3 `ephys.py` — Trace Processing

**Purpose:** Filter traces, deconcat concatenated sweeps, detect spikes.

**Functions:**
| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `lowpass_filter(signal, order, cutoff, fs, filter_type) -> np.ndarray` | `pc_lowpassfilt.m` | Butterworth (default) or Bessel filter via `scipy.signal.butter` / `scipy.signal.bessel` + `scipy.signal.filtfilt`. Zero-phase bidirectional filtering. |
| `deconcat_traces(filtered, delay, delay_bp, pulse_dur, n_pulses, sr) -> np.ndarray` | Inline in `ephys_intr.m` (repeated for each protocol) | Slice concatenated trace into (samples_per_pulse, n_pulses) array. `startpoint = delay//2 : pulse_dur+delay_bp : trace_len`. |
| `build_stim_vector(first_amp, amp_change, n_pulses) -> np.ndarray` | Inline in `ephys_intr.m` | `np.arange(first_amp, first_amp + n_pulses * amp_change, amp_change)` |
| `detect_spikes(trace, threshold_factor) -> np.ndarray` | `spike_times.m` (Berg 2006) | Threshold crossings at `threshold_factor * max(trace)`. Group consecutive above-threshold samples; return index of peak within each group. |
| `count_spikes(traces, threshold) -> np.ndarray` | Inline in `ephys_intr.m` | Apply `detect_spikes` per column, return counts vector. |
| `compute_rmp(traces, baseline_samples) -> float` | Inline: `mean(traces(1:delay/2, :))` | Mean of pre-stimulus baseline across all sweeps. |

### 2.4 `protocols.py` — Protocol Extraction

**Purpose:** Extract structured data from each WaveSurfer protocol type.

**Functions:**
| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `extract_iv(ws_data) -> IVResult` | `ephys_intr.m` lines 35-89 | Parse stim params from `header.StimulusLibrary.Stimuli.element1.Delegate.*`. Deconcat, filter, count spikes, compute RMP. Return dataclass with `traces`, `stim_vec`, `spike_counts`, `rmp`. |
| `extract_rheobase(ws_data) -> RheobaseResult` | `ephys_intr.m` lines 92-151 | Parse from `element9`. Find first sweep with spike = rheobase current. Return `traces`, `stim_vec`, `spike_counts`, `rheo`, `rmp`. |
| `extract_passive(ws_data) -> PassiveResult` | `ephys_intr.m` lines 155-190 | Parse from `element6`. Compute input resistance (linear fit of dV vs I), membrane time constant (exponential fit). Return `traces`, `stim_vec`, `rin`, `tau`. |
| `extract_sag(ws_data) -> SagResult` | `ephys_intr.m` lines 193-215 | Parse from `element5`. Compute sag ratio: `100 * (Vss - Vmin) / (Vrest - Vmin)`. |
| `extract_ramp(ws_data) -> RampResult` | `ephys_intr.m` lines 217-223 | Parse from `element3`. Just store filtered traces. |
| `process_all_protocols(h5_files, ephys_folder) -> EphysData` | `ephys_intr.m` (outer loop) | Identify protocol type from filename (IV, Rheobase, Passive, Sag, Ramp). Call appropriate extractor. |

**Stim parameter extraction:** The MATLAB code reads stim parameters as strings
from HDF5 header fields and converts with `str2num`. In Python, `h5py` reads
these directly as numbers or byte strings — handle both cases.

### 2.5 `spike_features.py` — Spike Feature Extraction

**Purpose:** Replace PANDORA toolbox spike analysis with eFEL.

**MATLAB source:** `sp_parameters_pandora.m` uses the PANDORA toolbox
(`trace()`, `getProfileAllSpikes()`) to extract 18 parameters per spike.

**Python replacement:** Use [eFEL](https://github.com/BlueBrain/eFEL) (Blue
Brain Project, BSD-3 license).

**Function mapping — PANDORA parameters to eFEL features:**

| Index | PANDORA param | eFEL feature | Notes |
|-------|---------------|--------------|-------|
| 1 | `MinVm` | `minimum_voltage` | AHP minimum voltage |
| 2 | `PeakVm` | `peak_voltage` | Spike peak voltage |
| 3 | `InitVm` | `AP_begin_voltage` | Not used in final metrics |
| 4 | `InitVmBySlope` | `AP_begin_voltage` | Not used in final metrics |
| 5 | `MaxVmSlope` | `AP_rise_rate` | MATLAB assumes 10 kHz SR but actual is 20 kHz, so multiplies by 2. eFEL handles SR correctly — no correction needed. |
| 6 | `HalfVm` | Derive from `AP_begin_voltage` + `AP_amplitude` / 2 | Voltage at half-amplitude |
| 7 | `Amplitude` | `AP_amplitude` | Peak - threshold |
| 8 | `MaxAHP` | `AHP_depth_abs` or `min_AHP_values` | After-hyperpolarisation depth |
| 9 | `DAHPMag` | `AHP_depth` | Not used |
| 10 | `InitTime` | `AP_begin_time` | Not used |
| 11 | `RiseTime` | `AP_rise_time` | Not used |
| 12 | `FallTime` | `AP_fall_time` | Not used |
| 13 | `MinTime` | `min_AHP_indices` (convert to time) | Not used |
| 14 | `BaseWidth` | `AP_width` | Not used |
| 15 | `HalfWidth` | `AP_duration_half_width` | MATLAB divides by 2 (SR correction). eFEL handles SR correctly. |
| 16 | `FixVWidth` | — | Not used |
| 17 | `Index` | Spike index in trace | Internal bookkeeping |
| 18 | `Time` | Spike time | MATLAB multiplies by 10 (SR correction). eFEL returns correct times. |

**Functions:**
| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `extract_spike_features(trace, sr, spike_index) -> dict` | `sp_parameters_pandora.m` | Use eFEL: set `stim_start`, `stim_end`, `T`, `V`. Call `efel.getFeatureValues()` for the features above. Select the spike at `spike_index` (default: 2nd spike, matching MATLAB). |
| `extract_waveform(trace, spike_time_idx, sr, pre_ms, post_ms) -> tuple[np.ndarray, np.ndarray]` | `sumSpikeWaveforms.m` lines 40-50 | Extract a window around the spike. Return (time_ms, voltage). Pre=7ms, post=20ms. |

**Citation required:**
- eFEL: Van Geit et al. 2016. "BluePyOpt: Leveraging open source software and cloud computing for neuroscience optimization problems." Frontiers in Neuroinformatics. doi:10.3389/fninf.2016.00017. GitHub: https://github.com/BlueBrain/eFEL
- PANDORA (original): Gunay et al. 2009. "Channel density distributions explain spiking variability in the globus pallidus." J Neurosci. doi:10.1523/JNEUROSCI.2929-09.2009. GitHub: https://github.com/cengique/pandora-matlab

### 2.6 `morphology.py` — Morphology Processing

**Purpose:** Replace TREES toolbox operations with navis.

**MATLAB source:** `morphology_readout.m`, `stats_tree_sw.m`, `soma_subtract.m`,
`rotate_tree.m`, `dissect_tree_sw.m`, `cat_tree_sw.m`, `root_tree_sw.m`,
`getSurfDist.m`.

**Function mapping — TREES toolbox to navis:**

| TREES function | navis/scipy equivalent | Notes |
|----------------|----------------------|-------|
| `load_tree(file, 'swc')` | `navis.read_swc(file)` | Returns `TreeNeuron` |
| `cat_tree(tree1, tree2)` | `navis.stitch_neurons([tree1, tree2])` or manual node table concat | Concatenate basal trees into one combined tree |
| `len_tree(tree)` | `navis.cable_length(tree)` → total, or `tree.nodes` segment lengths | Segment lengths |
| `Pvec_tree(tree, len)` | `navis.geodesic_matrix(tree)` or `navis.dist_to_root(tree)` | Path length from root for each node |
| `eucl_tree(tree)` | `navis.dist_to_root(tree, weight='euclidean')` or manual Euclidean from root | Euclidean distance to root per node |
| `BO_tree(tree)` | `navis.strahler_index(tree)` (approximate) or manual BFS from root counting branch points | Branch order per node |
| `B_tree(tree)` | Nodes with >1 child in `tree.nodes` | Binary vector: 1 at branch points |
| `T_tree(tree)` | Leaf nodes (no children) | Binary vector: 1 at terminal points |
| `angleB_tree(tree)` | Manual: compute angle between child branches at each branch point | Branch angles in radians |
| `sholl_tree(tree, distances)` | `navis.sholl_analysis(tree, radii=distances)` | Intersection counts at concentric shells |
| `convhulln(pts)` | `scipy.spatial.ConvexHull(pts)` | Convex hull volume |
| `dissect_tree(tree)` | Manual: segment tree at branch/terminal points, compute branch lengths | Group nodes into branches |

**Functions:**
| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `load_morphology(tracing_path) -> MorphologyData` | `morphology_readout.m` lines 19-78 | Load SWC files via navis, concatenate basals, identify apical/soma/surface/axon. |
| `soma_subtract(neurons, soma_neuron) -> dict` | `morphology_readout.m` lines 82-111, `soma_subtract.m` | Subtract soma centroid from all trees. Flip Y axis (`*-1`). |
| `rotate_to_surface(neurons, surface_neuron) -> dict` | `morphology_readout.m` lines 117-153, `rotate_tree.m` | Fit line to surface near soma, compute rotation angle, apply 2D rotation matrix. |
| `compute_tree_stats(neuron) -> TreeStats` | `stats_tree_sw.m` | Compute: total length, max path length, branch points, mean path/eucl ratio, max branch order, mean branch angle, mean branch length, mean path length, mean branch order, width/height/depth, width/height ratio, width/depth ratio. |
| `compute_sholl(neuron, radii) -> np.ndarray` | `stats_tree_sw.m` (with extras) | Sholl intersection counts. Use `navis.sholl_analysis()`. |
| `compute_surface_distance(surface_pts, dendrite_pts) -> tuple[float, float]` | `getSurfDist.m` | Nearest-surface distance for most superficial and deepest dendrite points. Use `scipy.spatial.cKDTree`. |
| `combine_basal_trees(basal_neurons) -> navis.TreeNeuron` | `cat_tree_sw.m` (called in `morphology_readout.m`) | Concatenate multiple basal SWC files into one tree. |

**Citation required:**
- navis: Schlegel et al. 2021. "navis: neuron analysis and visualization in Python." GitHub: https://github.com/navis-org/navis
- TREES toolbox (original): Cuntz et al. 2010. "One rule to grow them all: a general theory of neuronal branching and its practical application." PLoS Comput Biol. doi:10.1371/journal.pcbi.1000877
- Sholl analysis: Sholl 1953. "Dendritic organization in the neurons of the visual and motor cortices of the cat." J Anat 87(4):387-406.

### 2.7 `metrics.py` — Metrics Assembly

**Purpose:** Combine ephys and morph metrics into a single DataFrame row per cell.

**MATLAB source:** `procPC.m` lines 100-385 (the big metrics struct assembly).

**Functions:**
| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `build_cell_metrics(cell_info, ephys_data, morph_data, spike_features) -> dict` | `procPC.m` lines 100-385 | Assemble all ~50 metrics into a flat dict. |
| `build_metrics_table(all_cell_metrics) -> pd.DataFrame` | `procPC.m` line 574: `struct2table` | Stack all cell dicts into a DataFrame. |
| `save_metrics(df, path)` | `procPC.m` line 577: `writetable` | Save to CSV. |

**Metric categories (from MATLAB metricsTable):**

Passive ephys (6): `RMP`, `rheobase`, `Rin`, `tau`, `input_capacitance` (tau/Rin), `sag_ratio`

Active ephys (7): `max_spike_rate`, `min_vm`, `peak_vm`, `max_vm_slope`, `half_vm`, `amplitude`, `max_ahp`, `half_width`

Apical morphology (15): `len`, `max_plen`, `bpoints`, `mpeucl`, `maxbo`, `mblen`, `mplen`, `mbo`, `width`, `height`, `depth`, `wh`, `wd`, `sholl_peak_crossings`, `sholl_peak_distance`, `ext_superficial`, `ext_deep`

Basal morphology (16): Same as apical + `n_basal_trees`

Metadata: `cell_index`, `animal_id`, `slice_id`, `cell_slice_id`, `hemisphere`, `cell_type`, `depth_slice`, `depth_pial`, `area`, `layer`

### 2.8 `statistics.py` — Population Statistics

**Purpose:** Summary statistics and group comparisons.

**MATLAB source:** `sumTableNum.m`, `sumCorrelations.m` / `plotPCCorr.m`.

**Functions:**
| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `compute_summary_stats(df, metric_cols, group_col) -> pd.DataFrame` | `sumTableNum.m` lines 39-77 | Per group: n, mean, median, std, sem, min, max. |
| `mann_whitney_comparison(df, metric_cols, group_col) -> pd.DataFrame` | `sumTableNum.m` line 74: `ranksum` | `scipy.stats.mannwhitneyu` for each metric. Add multiple-comparison correction (Benjamini-Hochberg FDR). |
| `spearman_correlation(df, x_col, y_col) -> tuple[float, float]` | `plotPCCorr.m` line 11: `corr(..., 'Spearman')` | `scipy.stats.spearmanr`. |
| `correlation_matrix(df, metric_cols) -> pd.DataFrame` | `sumCorrelations.m` (planned) | All pairwise Spearman correlations. |
| `save_stats_summary(stats_df, path)` | `sumTableNum.m` line 219: `writetable` | Save to CSV. |

### 2.9 `pca.py` — PCA

**Purpose:** Principal component analysis on metric subsets.

**MATLAB source:** `plotPCPCA.m` (called from `sumPCPCA.m` and `plotPCACellType.m`).

**Functions:**
| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `run_pca(df, metric_cols, n_components) -> PCAResult` | `plotPCPCA.m` lines 16-19 | `sklearn.preprocessing.StandardScaler` + `sklearn.decomposition.PCA`. Return scores, loadings, explained variance. |
| `get_metric_subsets(df) -> dict[str, list[str]]` | `plotPCACellType.m` | Define subsets: all_ephys, passive_ephys, active_ephys, all_morph, apical_morph, basal_morph, combined. |
| `filter_exclude_cols(cols, exclude) -> list[str]` | `plotPCACellType.m` lines 14-22 | Remove depth-dependent cols that are unreliable in z (confocal). |

**PCA subsets run (from `sumPCPCA.m`):**
1. All ephys metrics
2. Passive ephys only
3. Active ephys only
4. All morphology
5. Apical morphology only
6. Basal morphology only
7. Ephys + morphology combined
8. Each of the above with and without 3D-dependent columns excluded
9. With/without outlier cells removed

### 2.10 `plotting/` — Visualization

All plots use matplotlib + seaborn. No MATLAB-specific tooling (export_fig, Violinplot-Matlab).

#### `ephys_plots.py`

| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `plot_iv_traces(iv_data, path)` | `plot_intrinsic.m` plotType==1 | Sub-threshold + threshold + max traces, scale bar, RMP annotation. |
| `plot_rheobase_traces(rheo_data, path)` | `plot_intrinsic.m` plotType==2 | 1x rheobase trace + sub-threshold. |
| `plot_iv_spikes(stim_vec, spike_counts, path)` | `sumEphysPlots.m` lines 93-103 | F-I curve (current vs spike rate). |
| `plot_steady_state(stim_vec, vss, path)` | `plotSS.m` | V steady-state vs injected current. |
| `plot_population_iv(all_rates, cell_types, stim_vec, path)` | `sumEphysPlots.m` lines 167-198 | Mean +/- SEM per cell type, with individual traces. |
| `plot_population_iv_normalised(...)` | `sumEphysPlots.m` lines 200-233 | Normalised to max per cell. |
| `plot_population_subthresh(all_traces, cell_types, path)` | `sumEphysPlots.m` lines 236-282 | Mean sub-threshold traces per type. |
| `plot_population_ss(ss_data, cell_types, stim_vec, path)` | `plotSSCellType.m` | Steady-state mean +/- SEM per type. |

#### `morph_plots.py`

| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `plot_single_morphology(morph_data, path)` | `procPC.m` lines 398-430 | Apical (blue), basal (red), soma (black), surface, axon (magenta). Use `navis.plot2d()`. |
| `plot_population_morphology(all_morph, cell_type, path, components)` | `plotMorphsCombined.m` (trace mode) | Overlay all cells, optional flip for hemisphere. |
| `plot_density_morphology(all_morph, cell_type, path, components)` | `plotMorphsCombined.m` (density mode) | 2D histogram density with Gaussian smoothing. |

#### Spike waveform plots

| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `plot_spike_waveform(time, wave, path)` | `sumSpikeWaveforms.m` lines 53-72 | Single cell. |
| `plot_population_waveforms(all_waves, cell_types, time, path)` | `sumSpikeWaveforms.m` lines 82-121 | Raw overlay + mean per type. |
| `plot_population_waveforms_normalised(...)` | `sumSpikeWaveforms.m` lines 123-166 | Normalised to [0, 1]. |

#### `violin_box.py`

| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `plot_metric_comparison(df, metric, group_col, p_value, path)` | `sumTableNum.m` lines 94-200 | Violin + box + swarm plot with p-value annotation. Use `seaborn.violinplot` + `seaborn.swarmplot`. Replaces Violinplot-Matlab. |

#### `correlation_plots.py`

| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `plot_correlation(df, x_col, y_col, group_col, path)` | `plotPCCorr.m` | Scatter coloured by cell type, Spearman r/p in title. Cell index labels. |

#### `pca_plots.py`

| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `plot_pca_scatter(scores, cell_types, pc_x, pc_y, title, path)` | `plotPCPCA.m` lines 23-61 | PC score scatter, coloured by type, cell index labels. |
| `plot_scree(explained, title, path)` | `plotPCPCA.m` lines 64-74 | Variance explained bar chart. |
| `plot_loadings(loadings, feature_names, pc_idx, title, path)` | `plotPCPCA.m` lines 86-99 | PC loading bar chart per component. |

#### `confocal_stack.py`

| Python | MATLAB source | Notes |
|--------|---------------|-------|
| `make_projections(tiff_path, morph_data) -> tuple[np.ndarray, np.ndarray]` | `procPC.m` lines 433-548 | Mean and max z-projections. Read TIFF stack with `tifffile`, pad, rotate by surface angle. |
| `make_stack_gif(tiff_stack, path, delay)` | `procPC.m` lines 552-563 | Animated GIF from z-slices. Use `imageio`. |

---

## 3. Data Flow

```
Inputs:
  metadata/animals.csv + metadata/cells.csv
  ephys/*.h5 (WaveSurfer files, per protocol)
  confocal-raw/**/Tracing/Cell*/  (SWC files)
  confocal-raw/**/corrected.tif   (confocal z-stacks)

Processing:
  1. Load metadata, join animals + cells on animal_id
  2. For each cell:
     a. If has_ephys: load WaveSurfer H5 files → extract protocols
        → filter traces → count spikes → extract spike features (eFEL)
     b. If has_good_morph: load SWC files → soma subtract → rotate to
        surface → compute tree stats (navis) + Sholl → surface distances
     c. Assemble metrics dict
  3. Stack all metrics → DataFrame → metrics.csv
  4. Population analyses:
     a. Summary stats + Mann-Whitney U per metric → statsSum.csv
     b. Spearman correlations (morph-morph, ephys-ephys, morph-ephys)
     c. PCA on metric subsets
  5. Generate all plots

Outputs:
  analysis/metrics.csv          — per-cell metrics (ephys + morph + metadata)
  analysis/statsSum.csv         — population summary stats + p-values
  analysis/ephys-plots/         — per-cell ephys trace plots
  analysis/ephys-plots-pop/     — population ephys plots
  analysis/morph-plots/         — per-cell morphology plots
  analysis/morph-plots-pop/     — population morphology overlays/densities
  analysis/metric-violins/      — violin plots per metric
  analysis/metric-boxes-pts/    — box + swarm plots per metric
  analysis/corr-plots/          — correlation scatter plots
  analysis/pca-plots/           — PCA scatter, scree, loading plots
  analysis/zprojs/              — confocal mean/max projections + GIFs
```

---

## 4. navis/eFEL Function Mapping to MATLAB TREES/PANDORA

### 4.1 TREES Toolbox → navis

| Computation | TREES (MATLAB) | navis (Python) | Fallback |
|-------------|----------------|----------------|----------|
| Load SWC | `load_tree(f, 'swc')` | `navis.read_swc(f)` | — |
| Total cable length | `sum(len_tree(tree))` | `navis.cable_length(tree)` | Manual from node table |
| Path length to root | `Pvec_tree(tree, len)` | `navis.dist_to_root(tree)` | BFS on node table |
| Euclidean dist to root | `eucl_tree(tree)` | Manual: `np.linalg.norm(nodes[xyz] - root[xyz], axis=1)` | — |
| Branch order | `BO_tree(tree)` | Manual BFS counting BPs on path to root | `navis.strahler_index` (different definition) |
| Branch points | `B_tree(tree)` (binary) | `tree.nodes[tree.nodes.type == 'branch']` or count children > 1 | — |
| Terminal points | `T_tree(tree)` (binary) | `tree.nodes[tree.nodes.type == 'end']` | Leaf nodes |
| Branch angles | `angleB_tree(tree)` | Manual: vectors from parent to children at BPs, compute angle | — |
| Sholl analysis | `sholl_tree(tree, dsholl)` | `navis.sholl_analysis(tree, radii=dsholl)` | Manual: count intersections |
| Dissect into branches | `dissect_tree(tree)` | Manual: walk tree, cut at BP/terminal, compute branch lengths | — |
| Concatenate trees | `cat_tree(t1, t2)` | `navis.stitch_neurons([t1, t2])` | Manual node table merge |
| Convex hull | `convhulln([X,Y,Z])` | `scipy.spatial.ConvexHull(pts)` | — |
| Tree rotation | `rotate_tree(tree, angle, offset)` | Manual 2D rotation matrix on tree.nodes[x,y] | — |
| Soma subtraction | Custom (`soma_subtract.m`) | Manual: subtract centroid, flip Y | — |
| Width/height/depth | `max(X)-min(X)` etc. | `tree.nodes[col].max() - tree.nodes[col].min()` | — |

### 4.2 PANDORA Toolbox → eFEL

| Computation | PANDORA (MATLAB) | eFEL (Python) |
|-------------|------------------|---------------|
| Spike detection | `trace(data, dt, dy, props)` → `getProfileAllSpikes` | `efel.getFeatureValues(traces, features)` |
| Peak voltage | `spikes_db.data(:, 2)` (PeakVm) | `peak_voltage` |
| Min voltage (AHP) | `spikes_db.data(:, 1)` (MinVm) | `minimum_voltage` or `min_AHP_values` |
| Half-width | `spikes_db.data(:, 15)` (HalfWidth) | `AP_duration_half_width` |
| Max dV/dt | `spikes_db.data(:, 5)` (MaxVmSlope) | `AP_rise_rate` |
| Amplitude | `spikes_db.data(:, 7)` | `AP_amplitude` |
| AHP depth | `spikes_db.data(:, 8)` (MaxAHP) | `AHP_depth_abs` |
| Half-Vm | `spikes_db.data(:, 6)` | Derive: `AP_begin_voltage + AP_amplitude/2` |
| Spike time/index | `spikes_db.data(:, 17:18)` | `peak_time`, `peak_indices` |

**Important SR correction:** The MATLAB pipeline assumes 10 kHz sample rate in
PANDORA but the actual data is 20 kHz. This causes:
- `MaxVmSlope` to be multiplied by 2
- `HalfWidth` to be divided by 2
- Spike time to be multiplied by 10

eFEL accepts the true sample rate, so these corrections are unnecessary. However,
values must be validated against the MATLAB outputs to ensure equivalence.

---

## 5. Test Strategy

All tests in `tests/patching/` mirroring `src/hm2p/patching/`.

### Unit Tests (pytest + hypothesis)

| Module | Test file | Strategy |
|--------|-----------|----------|
| `io.py` | `test_io.py` | Create minimal WaveSurfer H5 files with `h5py` in fixtures. Test sweep extraction, scaling. Create minimal SWC strings. Test loading, type detection. |
| `ephys.py` | `test_ephys.py` | Synthetic sine wave → filter → check attenuation. Synthetic trace with known spikes → `detect_spikes` → check count/positions. `hypothesis` for filter edge cases (empty array, DC signal, Nyquist). |
| `protocols.py` | `test_protocols.py` | Construct mock `ws_data` dicts with known stim params. Verify deconcat dimensions, stim vectors, RMP. |
| `spike_features.py` | `test_spike_features.py` | Synthetic AP waveform (fast rise, slow decay). Verify eFEL extracts known half-width, amplitude, AHP. |
| `morphology.py` | `test_morphology.py` | Create synthetic SWC node tables (straight line, Y-branch, star). Verify: cable length, branch points, Sholl counts, rotation, soma subtraction. `hypothesis` for rotation angles. |
| `metrics.py` | `test_metrics.py` | Mock ephys/morph data → verify DataFrame columns, NaN handling. |
| `statistics.py` | `test_statistics.py` | Known distributions → verify Mann-Whitney p, Spearman r. |
| `pca.py` | `test_pca.py` | Synthetic correlated data → PCA → verify explained variance sums to ~100%, scores shape. |
| `plotting/` | `test_plotting.py` | Smoke tests: each plot function runs without error on synthetic data, produces a file. No pixel-level checks. |
| `config.py` | `test_config.py` | Verify YAML loading, default values, path resolution. |

### Validation Tests (against MATLAB outputs)

If MATLAB-generated `metrics.csv` is available, add a validation test that:
1. Loads the MATLAB metrics CSV.
2. Runs the Python pipeline on the same raw data.
3. Compares each metric within tolerance (e.g., 1% for ephys, 5% for morphology
   due to algorithmic differences between TREES and navis).

### Coverage Target

90% minimum as per CLAUDE.md. Plotting modules may have lower coverage (smoke
tests only), but all numerical modules must be near 100%.

---

## 6. Implementation Order

Dependencies flow downward — implement in this order:

```
Phase 1: Core I/O and processing
  1. config.py               — no deps
  2. io.py                   — depends on h5py, navis
  3. ephys.py                — depends on scipy.signal
  4. protocols.py            — depends on io.py, ephys.py
  5. spike_features.py       — depends on efel
  6. morphology.py           — depends on io.py, navis, scipy.spatial

Phase 2: Metrics and analysis
  7. metrics.py              — depends on protocols.py, spike_features.py, morphology.py
  8. statistics.py           — depends on scipy.stats
  9. pca.py                  — depends on sklearn

Phase 3: Orchestration
  10. run.py                 — depends on everything above

Phase 4: Plotting
  11. plotting/ephys_plots.py
  12. plotting/morph_plots.py
  13. plotting/violin_box.py
  14. plotting/correlation_plots.py
  15. plotting/pca_plots.py
  16. plotting/confocal_stack.py

Phase 5: Frontend
  17. frontend/pages/patching_page.py
  18. frontend/pages/patching_ephys_page.py
  19. frontend/pages/patching_morph_page.py
```

Each module gets its test file written **before or alongside** the implementation
(test-driven development).

---

## 7. Metadata CSVs

### `metadata/animals.csv`

Already exists. Required columns for patching:

| Column | Type | Description |
|--------|------|-------------|
| `animal_id` | str | Mouse ID (join key) |
| `sex` | str | M/F |
| `dob` | date | Date of birth |
| `date_slice` | date | Date of slicing/patching (YYMMDD) |
| `date_confocal` | date | Date of confocal imaging (YYMMDD) |

### `metadata/cells.csv`

Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `cell_index` | int | Unique cell ID across project |
| `animal_id` | str | Mouse ID (FK to animals.csv) |
| `slice_id` | str | Slice identifier (e.g., S1, S2) |
| `cell_slice_id` | int | Cell number within slice (e.g., 1, 2) |
| `ephys_id` | str | WaveSurfer recording folder name (e.g., SW0001). Empty if no ephys. |
| `cell_type` | str | `"penkpos"` or `"penkneg"` |
| `hemisphere` | str | `"L"` or `"R"` |
| `depth_slice` | float | Estimated depth from top of slice (um) |
| `depth_pial` | float | Estimated depth from pial surface (um). Can be NaN — computed from morphology. |
| `has_morph` | bool | Whether confocal morphology data exists |
| `good_morph` | bool | Whether morphology is traceable/usable |
| `area` | str | Brain area (e.g., `"RSP"`) |
| `layer` | str | Cortical layer (e.g., `"L5"`) |

### `analysis/metrics.csv` (output)

Generated by `metrics.py` / `run.py`. Contains all columns from cells.csv
metadata plus ~48 computed metrics (6 passive ephys + 7 active ephys + 15 apical
morph + 16 basal morph + derived).

### `analysis/statsSum.csv` (output)

Generated by `statistics.py`. One row per metric, columns:
`metric`, `penkpos_n`, `penkpos_mean`, `penkpos_median`, `penkpos_std`,
`penkpos_sem`, `penkpos_min`, `penkpos_max`, `penkneg_n`, `penkneg_mean`,
`penkneg_median`, `penkneg_std`, `penkneg_sem`, `penkneg_min`, `penkneg_max`,
`penkposneg_p` (Mann-Whitney U), `penkposneg_p_fdr` (FDR-corrected).

---

## 8. Frontend Pages

### 8.1 `patching_page.py` — Overview

- Summary table: N cells per type, N with ephys, N with morph
- Metrics table (from `metrics.csv` loaded from S3)
- Stats summary table (from `statsSum.csv`)
- Sidebar filters: cell type, area, layer, has_ephys, has_morph
- "No data" message if S3 files not yet available

### 8.2 `patching_ephys_page.py` — Electrophysiology

- Per-cell trace viewer: select cell → show IV, rheobase, passive traces
- F-I curves: individual + population mean +/- SEM by type
- Spike waveform comparison: overlay all + mean by type (raw and normalised)
- Steady-state V-I plots by type
- Violin/box plots for all passive and active ephys metrics
- Correlation scatter plots (ephys vs ephys)

### 8.3 `patching_morph_page.py` — Morphology

- Per-cell morphology viewer: 2D projection of SWC traces (navis plot)
- Population overlay: all cells of one type, with hemisphere flip
- Density heatmap: 2D histogram of dendrite point density
- Sholl analysis: individual + population mean +/- SEM
- Violin/box plots for all morph metrics (apical and basal)
- Correlation scatter plots (morph vs morph, morph vs ephys)
- Confocal z-projection viewer (mean/max images from S3)

### 8.4 `patching_pca_page.py` — PCA

- Dropdown to select metric subset (ephys, morph, combined, etc.)
- PC1 vs PC2 scatter, coloured by cell type
- Scree plot (variance explained)
- Loading bar charts for top PCs
- Option to exclude depth-dependent / 3D columns
- Option to exclude outlier cells

### Navigation Integration

Add a new **Patching** section to `frontend/app.py` navigation:

```python
"Patching": [
    patching_page,          # Overview
    patching_ephys_page,    # Electrophysiology
    patching_morph_page,    # Morphology
    patching_pca_page,      # PCA
]
```

All pages must:
- Load data from S3 (never synthetic/fake data)
- Show clear "No data available" message if files missing
- Include a "Methods & References" expander citing eFEL, navis, TREES, PANDORA,
  Sholl, and any relevant analysis papers

---

## 9. Dependencies to Add

```
# In pyproject.toml [project.optional-dependencies] or requirements
navis           # SWC morphology loading, metrics, Sholl, visualization
efel            # Spike feature extraction (Blue Brain Project)
h5py            # WaveSurfer H5 file reading
scipy           # Already present — signal processing, stats, spatial
scikit-learn    # Already present — PCA
matplotlib      # Already present
seaborn         # Already present
pandas          # Already present
tifffile        # TIFF stack reading (confocal)
imageio         # GIF writing
```

---

## 10. Known Quirks to Handle

1. **Sample rate mismatch:** PANDORA in the MATLAB code assumes 10 kHz but
   actual data is 20 kHz. The MATLAB code manually corrects `MaxVmSlope * 2`,
   `HalfWidth / 2`, `Time * 10`. eFEL handles the true SR correctly — no manual
   corrections needed. Validate outputs match.

2. **Y-axis flip:** Morphology readout flips Y coordinates: `Y = (Y - soma_y) * -1`.
   This is because confocal Y increases downward but the desired orientation has
   superficial (surface) upward.

3. **Hemisphere flip:** For left-hemisphere cells, X coordinates are negated to
   mirror all cells to the same orientation for population overlays.

4. **Surface rotation:** Trees are rotated so the pial surface is horizontal
   (aligned to X axis). This uses a linear fit to nearby surface points. The
   rotation angle is also used to rotate confocal z-stack images.

5. **WaveSurfer stim element naming:** Different protocols use different
   `element` indices (element1 for IV, element9 for Rheobase, element6 for
   Passive, element5 for Sag, element3 for Ramp). These are hard-coded in the
   MATLAB and must be replicated.

6. **Basal tree concatenation:** The MATLAB pipeline concatenates all basal
   dendrites into a single tree before computing stats. The `cat_tree_sw.m`
   script connects the root of tree2 to the closest node in tree1.

7. **Soma as trace root:** `root_tree_sw.m` adds a tiny segment at the root of
   each tree before dissection. This ensures the root node is properly handled
   in branch statistics.

8. **Confocal >4GB TIFF:** The MATLAB code skips single-frame TIFFs (which
   occur when the file exceeds 4 GB and fails to load properly). Use
   `tifffile` in Python which handles large TIFFs correctly.

9. **Missing data:** Cells may lack ephys (no `ephys_id`), morph
   (`has_morph=false`), or good morph (`good_morph=false`). Metrics are set to
   NaN for missing modalities. The pipeline must handle all combinations.

10. **IV stim count check:** Population averaging of F-I curves requires all
    cells to have the same number of stim steps (19 in MATLAB). Cells with
    different counts are excluded from population averages.
