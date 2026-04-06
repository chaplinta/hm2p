# NaviGraph Evaluation for the hm2p Pipeline

**Date:** 2026-04-02  
**Reviewer:** hm2p neuro-data-scientist agent  
**Subject:** NaviGraph v1.0.0, PBLab (Bezalel Bhanu Lab), Tel Aviv University  
**Repo:** https://github.com/PBLab/NaviGraph  
**Paper:** Preprint on biorxiv (2025.05.18.654725v1) — could not be fetched directly  
**Scope:** Evaluate feasibility and value of integrating NaviGraph into hm2p Stage 6 analysis

---

## 1. What NaviGraph Does

### Overview

NaviGraph is a graph-based multimodal behavioural analysis framework that maps
animal trajectories onto a spatial graph and then maps neural activity (or other
physiological signals) onto that graph for node-resolved analyses. It operates
across three analytical domains: temporal (time-series), spatial (continuous x/y
coordinates), and graph (discrete topology). It was designed for freely-moving
rodent maze experiments where the maze structure can be represented as a graph of
nodes and edges.

### Input Formats

| Data type | Accepted format |
|-----------|----------------|
| Pose / keypoints | DeepLabCut H5 files |
| Head direction | IMU/quaternion CSV files (qw, qx, qy, qz columns) |
| Neural activity | Minian zarr format (miniscope calcium imaging) |
| Graph structure | GraphML, GEXF, GML, or pickle |
| Spatial calibration | AVI video + transformation matrix (.npy) |
| Configuration | YAML via Hydra / OmegaConf |

The head direction pipeline converts quaternion IMU data to yaw using SciPy's
`Rotation` class with a session-specific yaw offset parameter. Video-based
keypoint pose is read from DLC H5 files. Neural data is read exclusively from
Minian zarr stores — this is the primary calcium imaging integration point and
is specific to miniscope / Minian, not Suite2p or CaImAn.

### Core Algorithms

**Graph construction:** The user defines spatial graphs interactively using a
PyQt5 GUI (the `navigraph setup graph` CLI command). Nodes are placed on an
image of the maze, and edges are drawn manually. The resulting graph is stored
as a NetworkX graph in GraphML/pickle format. No automated graph inference from
trajectories is implemented; the graph must be provided.

**Activity mapping:** Continuous animal position is mapped to the nearest graph
node at each time point. Neural activity (or any other physiological signal) is
then averaged over all time points assigned to each node. This is conceptually
identical to occupancy-normalised firing-rate maps, applied to discrete graph
nodes rather than 2D spatial bins.

**Decision-point analysis:** The framework tracks transitions between nodes and
records which outgoing edge the animal chose at each junction visit. Turn
direction (left, right, straight, back) is derived geometrically from the
incoming and outgoing edge vectors.

**Navigation metrics (built-in):**
- `time_a_to_b`: elapsed time for the animal to travel between two named nodes
- Speed at each node (distance / time)
- Rolling statistics over node-visit sequences

**Multi-session integration:** Spatial graphs are registered across sessions via
affine transformation matrices (`.npy`), enabling pooling of neural data mapped
to the same graph topology across sessions.

**Plugin system:** A `@register_data_source_plugin()` decorator allows custom
data sources to be loaded. A `@register_graph_builder()` decorator allows custom
graph construction. Both require writing Python classes extending
`NaviGraphPlugin`.

### Output Formats

- CSV files for per-node activity metrics
- Pickle files for serialized data structures
- MP4 video overlays (trajectory + neural activity)
- PNG/SVG images of graph-mapped activity

### Dependencies (from pyproject.toml)

```
python >=3.9, <3.13
opencv-python >=4.5.5
pandas >=2.0.0
numpy >=1.24.0
networkx >=3.0
pydantic >=2.0.0
click >=8.0.0
matplotlib >=3.7.0
seaborn >=0.12.0
PyQt5 >=5.15.0
h5py >=3.8.0
loguru >=0.7.0
omegaconf >=2.3.0
hydra-core >=1.3.0
tables >=3.8.0
xarray >=2023.0.0
dask >=2023.0.0
zarr >=2.14.0
scipy >=1.10.0
```

PyQt5 requires a display server (X11 or Wayland) for the interactive setup GUI.
This is a hard constraint in headless compute environments (Docker on EC2).

---

## 2. What We Already Have vs What NaviGraph Adds

### Functionality audit

| Capability | hm2p existing code | NaviGraph |
|------------|-------------------|-----------|
| Maze graph (nodes, edges, adjacency) | `maze/topology.py` — hardcoded 7×5 grid with internal walls, adjacency, BFS distances, junction classification | Interactive GUI-based graph definition |
| Continuous pos → discrete cell | `maze/discretize.py` — vectorised nearest-cell assignment | Node assignment by proximity |
| Node sequence (junctions + dead ends only) | `maze/discretize.py::node_sequence()` | Not explicit; uses full graph |
| Occupancy per cell | `maze/analysis.py::cell_occupancy()` | Node-level occupancy (same concept) |
| Turn bias at T-junctions | `maze/analysis.py::turn_bias()`, `per_junction_turn_bias()` | Decision-point turn tracking |
| Monotonic path detection | `maze/analysis.py::find_monotonic_paths()` | Not available |
| Path efficiency over time | `maze/analysis.py::path_efficiency_over_time()` | Not available |
| Exploration efficiency (new nodes/window) | `maze/analysis.py::exploration_efficiency()` | Not available |
| Markov transition models (1st + 2nd order) | `maze/analysis.py` (full implementation with AIC/BIC) | Not available |
| Sequence/conditional entropy | `maze/analysis.py::sequence_entropy()` | Not available |
| Behavioural mode segmentation | `maze/analysis.py::segment_modes()` | Not available |
| Simulate random walk null model | `maze/analysis.py::simulate_random_walk()` | Random walk statistics |
| All-pairs shortest path | `maze/topology.py::compute_distances()` (BFS) | Not available |
| Neural activity mapped to graph nodes | **Not implemented** | Core feature |
| Multi-session graph registration | **Not implemented** | GUI-based affine transforms |
| IMU / quaternion HD | Not used (we compute HD from ear vector via DLC) | Primary HD source |
| Graph-based speed / transit time | **Not implemented** | `time_a_to_b` + speed |
| Rastermap / population visualisation | `analysis/rastermap_analysis.py` | Not available |
| HD tuning curves per node | **Not implemented** | Not available |
| Light/dark comparison at node level | **Not implemented** | Not available |
| Cell-type-specific node activity | **Not implemented** | Not available |

### What NaviGraph adds that we lack

The one substantive capability NaviGraph offers beyond our existing code is
**associating neural activity with specific maze graph nodes** in a principled
way — computing a mean neural signal per node per cell, accounting for occupancy,
and visualising the result on the graph. This is the framework's core
contribution. Everything else in our maze module (Markov models, path efficiency,
entropy, turn bias, exploration efficiency) already matches or exceeds what
NaviGraph provides.

NaviGraph also provides multi-session graph registration via affine transforms,
which we would need to implement if we want to pool neural data across sessions
onto a shared graph. However, our maze is physically fixed and the camera is
overhead and roughly stationary across sessions, so inter-session alignment is
less critical than in a freely-chosen arena.

### What we have that NaviGraph does not

- Markov transition models with order selection (AIC/BIC)
- Monotonic path detection (goal-directed run identification)
- Path efficiency analysis
- Exploration efficiency (new nodes per window; Rosenberg NewNodes4 analogue)
- Conditional entropy / StringEntropy analogue
- Behavioural mode segmentation (directed vs exploratory)
- Full HD tuning analysis (tuning curves, MVL, Rayleigh test, PD, Skaggs information)
- Light/dark comparison
- Bayesian population decoder
- Cell-type-specific statistical comparisons
- CEBRA / NEMOS encoding models
- The entire calcium processing pipeline (dF/F, CASCADE, neuropil subtraction)

---

## 3. Integration Feasibility

### Neural signal format

**Critical incompatibility.** NaviGraph reads neural data from Minian zarr
stores, which are the native output of the Minian miniscope pipeline. Our neural
data lives in HDF5 files (ca.h5 and sync.h5) produced by Suite2p via
roiextractors. There is no built-in adapter for Suite2p HDF5 outputs.

A plugin could in principle bridge this gap: the `@register_data_source_plugin()`
decorator allows a custom class to load arbitrary data. However, this requires
writing and maintaining an adapter that loads our HDF5 schema and returns data
in whatever internal format NaviGraph expects. The Minian zarr loader is the only
documented neural data source; its internal representation is not publicly
documented.

### Pose / kinematics format

Partially compatible. NaviGraph reads DLC H5 files directly, and our pipeline
produces DLC H5 files at Stage 2. The relevant body part is `mid_back` or
`mouse_center` for position. However, NaviGraph's head direction pipeline is
quaternion-based (IMU data: qw, qx, qy, qz columns in CSV), whereas our HD is
computed from the left_ear / right_ear vector via DeepLabCut and the movement
library. There is no documented path for using DLC-derived HD in NaviGraph.

### Graph definition

NaviGraph requires manual interactive graph setup via its PyQt5 GUI. Our maze
topology is already fully defined programmatically in `maze/topology.py` with
all nodes, edges, junction classifications, shortest-path distances, and
internal wall definitions. The NaviGraph approach would require either (a)
re-entering the graph topology via the GUI on a per-session basis, or (b)
writing a tool to export our NetworkX-equivalent graph to GraphML and loading
it as NaviGraph's pre-defined graph. Option (b) is feasible but requires
implementation effort.

The PyQt5 GUI is also incompatible with headless Docker on EC2, which is where
Stage 6 analysis runs. The GUI requirement would need to be bypassed or the
graph setup run locally and the result uploaded.

### Python and dependency compatibility

NaviGraph requires Python >=3.9, <3.13. Our pipeline targets Python 3.11 and
3.12 (CI tests both). Version compatibility is satisfactory.

Core NaviGraph dependencies (networkx, xarray, scipy, numpy, pandas, h5py) are
already present in our environment. The unique additions are:
- PyQt5 >=5.15.0 — display-server dependency; problematic on EC2
- zarr >=2.14.0 — already present in the venv (used by MoSeq)
- hydra-core / omegaconf — new; could conflict with our pipeline.yaml config
- tables (PyTables) — new; HDF5 accessor we do not currently use
- loguru — new (we use structlog)

Dependency conflicts are unlikely but not zero; zarr version constraints could
interfere with keypoint-MoSeq's zarr pin. A full pip-compile check would be
needed.

### Cell-type-specific analyses

NaviGraph has no concept of cell populations, cell types, or the Penk+ /
Penk⁻CamKII+ distinction. All neural signals are treated as a flat array. Any
cell-type-resolved analysis — the primary scientific question of this project —
would need to be implemented outside NaviGraph and then mapped onto NaviGraph's
output format. This limits its value as an integrated framework.

### Light/dark condition handling

NaviGraph has no concept of time-segmented experimental conditions. It computes
session-level node averages without any mechanism for splitting by light_on /
light_off epochs. This is a substantial gap for our design, where the
light/dark comparison is the primary experimental manipulation.

---

## 4. Specific Analyses NaviGraph Could Enable

The following analyses would be valuable scientifically and are not currently
implemented. They draw on NaviGraph's node-activity mapping concept but would
likely need to be reimplemented natively rather than using NaviGraph directly.

### 4.1 Cell-type-specific activity mapped to maze graph nodes

**What:** Compute mean spike rate (or dF/F) per ROI per maze cell, split by
cell type (Penk+ vs Penk⁻CamKII+). Normalise by occupancy. Display as a
heatmap on the maze graph.

**Why:** This would reveal whether the two populations have spatially selective
activity patterns in the maze — i.e., whether either population preferentially
activates in corridors vs junctions vs dead ends, or in specific maze regions.

**NaviGraph approach:** NaviGraph would map a combined neural signal to nodes.
But it cannot split by cell type or by light condition. Node-activity mapping
would need to be written for hm2p regardless.

**Effort to implement natively:** Low. We already have `discretize_position_fast()`
to convert continuous (x, y) to cell indices. We have ROI-level signals in
sync.h5. The new computation is: for each ROI, compute `mean(signal[t])` for all
frames `t` where `cell_index[t] == c`, for each cell `c`, divided by occupancy.
This is a two-line numpy bincount operation.

### 4.2 Decision-point activity: does neural activity change at T-junctions?

**What:** Extract all T-junction visits from the node sequence. Align dF/F or
spike rate to junction entry time (say -2 s to +2 s). Compare peri-event
activity at junctions vs corridors vs dead ends, split by cell type and
light condition.

**Why:** If RSP neurons encode navigation-relevant information, their activity
should be modulated at choice points. Penk+ and Penk⁻CamKII+ could differ in
the timing or magnitude of junction-related activity.

**NaviGraph approach:** NaviGraph logs junction visits and could in principle
support event-triggered analysis, but no documented peri-event windowing function
exists. The node activity is epoch-averaged, not event-aligned.

**Effort to implement natively:** Medium. Requires extracting junction entry
times from the node sequence (already available from `node_sequence()`) and
aligning sync.h5 signals to those times using pynapple's event-triggered
averaging. This is a standard peri-event time histogram (PETH) using
`nap.compute_event_trigger_average()` or equivalent.

### 4.3 Path familiarity effects — does neural activity change as the animal repeats paths?

**What:** For each maze cell, compute mean activity in the first N visits vs
later visits within a session. Test whether activity in either population
increases or decreases as the animal becomes more familiar with a location.

**Why:** RSP is implicated in spatial memory consolidation. If Penk+ neurons
are more involved in initial exploration and Penk⁻CamKII+ in consolidated
navigation (or vice versa), this analysis would reveal it.

**NaviGraph approach:** Not available. NaviGraph aggregates over all visits.

**Effort to implement natively:** Medium. Requires tracking visit number per
cell and computing running averages per cell visit count.

### 4.4 Graph-based HD analysis — HD relative to corridor axis at each maze segment

**What:** For each corridor segment in the maze, define a "corridor axis" (the
direction of travel along that corridor). Compute HD relative to the corridor
axis at each moment the mouse is in that corridor. Compare tuning curves
expressed in corridor-relative coordinates between cell types.

**Why:** This tests whether RSP HD cells encode absolute allocentric HD or
corridor-relative egocentric HD. Laurent et al. (2025) showed RSP has room-
specific directional tuning in multi-room environments. The maze corridors
could function as analogous spatial contexts.

**NaviGraph approach:** NaviGraph does not implement corridor-relative HD.

**Effort to implement natively:** Medium. Requires: (a) defining corridor axes
from the maze topology (computable from the adjacency structure in
`maze/topology.py`), (b) computing HD_corridor = HD_absolute - corridor_axis
for all frames in each corridor, (c) computing tuning curves in this rotated
frame.

### 4.5 Light/dark comparison of maze-location-specific activity

**What:** For each maze cell and each cell type, compare mean activity in light
epochs vs dark epochs. Do certain maze locations show greater light-dependence
for one cell type?

**Why:** If visual cue availability differentially affects spatial coding in
specific maze regions (e.g., at junctions where visual landmarks are more
likely to be visible), this would suggest that visual and idiothetic anchoring
differ in their spatial specificity.

**NaviGraph approach:** Not available.

**Effort to implement natively:** Low. The same occupancy-normalised node
activity computation (section 4.1) applied separately to light_on and light_off
epochs from sync.h5.

---

## 5. Recommendation

### Verdict: Implement the useful analyses natively; do not integrate NaviGraph

The honest assessment is that NaviGraph does not offer a viable integration path
for the hm2p pipeline in its current form, and the one genuinely useful idea it
provides — mapping neural activity to maze graph nodes — is straightforward to
implement natively given what already exists.

The case against integration:

1. **Neural data format incompatibility.** NaviGraph reads Minian zarr (miniscope
   pipeline). Our data is in Suite2p HDF5. Bridging this gap requires writing and
   maintaining a custom plugin. The internal NaviGraph data format for neural
   signals is not documented. This is substantial reverse-engineering effort for
   uncertain outcome.

2. **HD source incompatibility.** NaviGraph's HD pipeline is IMU / quaternion-
   based. Our HD comes from DLC ear-vector tracking. There is no documented DLC-
   to-HD adapter in NaviGraph beyond its standard DLC H5 pose loader (which
   reads position, not angles).

3. **No cell-type handling.** The primary scientific question of this project is
   a Penk+ vs Penk⁻CamKII+ comparison. NaviGraph treats all ROIs as a flat pool
   with no mechanism for population-level splits. Every analysis relevant to our
   hypotheses would need to be layered on top outside the framework.

4. **No light/dark condition handling.** The primary experimental manipulation is
   a light/dark alternation. NaviGraph has no session-epoch concept.

5. **Interactive GUI for graph setup.** Our maze topology is already defined
   programmatically and precisely. Running the PyQt5 GUI to replicate it adds
   setup burden with no benefit, and the GUI cannot run in headless EC2 Docker
   containers.

6. **Dependency weight.** Adding PyQt5, hydra-core, omegaconf, and tables to the
   environment for a framework that covers only one of our needed analyses (node
   activity mapping) is disproportionate.

The case for adopting their approach:

The NaviGraph paper's conceptual contribution — that maze spatial structure
should be represented as a graph and neural activity mapped onto graph nodes
rather than treated as a 2D firing rate map — is sound and directly applicable
here. The node-activity mapping analyses described in Section 4 are worth
implementing.

### What to do

**Stage 6 extension — add a `maze/neural_mapping.py` module** with the following
functions, implemented natively:

1. `node_activity_map(sync_h5, cell_type_mask, condition_mask)` — mean spike rate
   per ROI per maze cell, occupancy-normalised, split by cell type and
   light condition. Returns a `(n_rois, n_cells)` array.

2. `population_node_profile(node_activity_map, roi_types)` — aggregate
   population-level activity per node, split by Penk+ vs Penk⁻CamKII+.

3. `junction_peth(sync_h5, node_seq_times, window_s)` — peri-event aligned
   activity around T-junction entries.

4. `corridor_relative_hd(hd, cell_indices, maze)` — HD in corridor-axis
   coordinates for each corridor segment.

These can all be implemented in a few hundred lines using existing infrastructure
(`maze/topology.py`, `maze/discretize.py`, `sync.h5` schema, `pynapple` for
event alignment). Unit tests follow the established pattern in `tests/maze/`.

The total implementation effort is 1–2 days for the core functions plus tests,
compared to an uncertain multi-day integration effort for NaviGraph with ongoing
maintenance risk.

**Pipeline placement:** All of the above belong in Stage 6 analysis
(`src/hm2p/analysis/` or `src/hm2p/maze/`). They do not require a new pipeline
stage. Output should go into `analysis.h5` under a `/maze_neural/` group.

**Citation:** Any implementation of node-activity mapping should cite NaviGraph
(preprint: biorxiv 2025.05.18.654725) as the conceptual source, in addition to
the Rosenberg et al. (2021) eLife paper that motivated the underlying graph
framework.

---

## Summary Table

| Question | NaviGraph | hm2p native |
|----------|-----------|-------------|
| Neural data format | Minian zarr only | Suite2p HDF5 |
| HD source | IMU quaternion CSV | DLC ear vector |
| Cell-type splits | Not supported | Native |
| Light/dark epochs | Not supported | Native |
| Graph definition | Interactive GUI | Programmatic (topology.py) |
| Node activity mapping | Core feature | Not implemented yet |
| Markov / entropy models | Not available | Fully implemented |
| Path efficiency | Not available | Fully implemented |
| Turn bias | Basic | Fully implemented |
| HD tuning | Not available | Fully implemented |
| Bayesian decoding | Not available | Fully implemented |
| Headless / EC2 compatibility | No (PyQt5 GUI) | Yes |

**Bottom line:** NaviGraph is a promising framework for miniscope / IMU
experiments in arbitrary arenas. For hm2p — two-photon Suite2p data, DLC
tracking, light/dark conditions, and cell-type comparisons — the integration
cost outweighs the benefit. The one missing analysis capability (node-activity
mapping) should be implemented natively in 1–2 days.
