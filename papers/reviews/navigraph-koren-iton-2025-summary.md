# Koren Iton et al. 2025 — Detailed Summary

## Citation

Koren Iton A, Iton E, Michaelson DM, Blinder P. 2025. "NaviGraph: A graph-based framework for multimodal analysis of spatial decision-making." *bioRxiv*. doi:10.1101/2025.05.18.654725

**Affiliations:** School of Neurobiology, Biochemistry and Biophysics, Tel Aviv University, Israel.

**Preprint posted:** 2025-05-21

**Note:** This summary is based on web search extracts (bioRxiv abstract/metadata, search snippets). The full PDF was not accessible from the devcontainer at time of writing. Some methodological details may be incomplete; update this summary when the full text is available.

---

## Overview

NaviGraph (Navigation on the Graph) is an open-source pipeline that represents maze environments as graphs and maps multimodal data streams — behavioural trajectories, head orientation, calcium imaging of neuronal activity — onto the graph structure. The framework was developed to address a gap in behavioural neuroscience: while graph-theoretic and topological methods are well established in network science and other domains, they remain underused for analysing spatial navigation data. NaviGraph provides a unified, modular framework that integrates diverse data types (behaviour, neural, physiological) into a single graph-based representation, enabling computation of spatial and topological metrics that are not accessible from standard trajectory-based analyses alone.

The framework was applied to a trial-based spatial memory task in a complex maze, with calcium imaging of retrosplenial cortex (RSP/RSC) neurons using head-mounted miniaturised microscopes (miniscopes) in freely moving mice. The study also examined an apolipoprotein E epsilon-4 (apoE4) knock-in mouse model, revealing sex- and genotype-specific navigation deficits detectable only through topological metrics.

---

## Key Methods and Algorithms

### 1. Graph representation of maze topology

The core idea is to model the maze as a graph where **decision points (junctions) are represented as nodes** and **corridors connecting them are edges**. This topological abstraction reduces continuous 2D trajectory data to a discrete sequence of node visits and edge traversals. By doing so, it enables graph-theoretic metrics (path length, visit frequency, efficiency, centrality) to be computed on the navigation behaviour.

The framework supports diverse maze configurations — the graph structure is defined by the maze layout rather than being hard-coded for a specific design.

### 2. Multimodal data integration onto graph nodes

NaviGraph populates each graph node (decision point) with multiple data layers:

- **Behavioural parameters:** visit frequency, dwell time, path choice at each junction, navigation efficiency metrics
- **Neuronal activity:** calcium fluorescence signals from miniscope recordings, mapped to the graph node where the animal was located at each timepoint
- **Physiological signals:** head orientation dynamics (head direction) aligned to graph position

This multi-layered annotation of graph nodes enables direct comparison of neural and behavioural states at specific topological locations (e.g., "what does RSP activity look like at this particular decision point?").

### 3. Calcium imaging of RSP

Neuronal activity was recorded from the retrosplenial cortex using head-mounted miniaturised one-photon (miniscope) calcium imaging in freely moving mice. RSP was chosen because of its extensive involvement in path integration and spatial navigation. The imaging enabled tracking of neuronal population activity at decision points throughout maze traversals.

### 4. Head orientation tracking

Head orientation (head direction) was tracked alongside position and mapped onto the graph structure, enabling analysis of how HD signals relate to maze topology and decision-making.

### 5. Topological and spatial metrics

The framework computes metrics including (inferred from described outputs):

- Navigation efficiency (path optimality relative to shortest graph path)
- Visit frequency per decision point
- Revisit patterns and perseverative behaviour
- Path familiarity dynamics (how neural/behavioural patterns change with repeated traversals)

---

## Key Findings

### 1. Decision-point-specific neuronal activity patterns

By mapping RSP calcium imaging data onto graph nodes, the framework revealed that neuronal activity patterns in RSP varied systematically across decision points. This is consistent with the known role of RSP in encoding spatial context (Mao et al. 2017, Nat Commun) and route planning (Alexander & Nitz 2015), but the graph-based representation provides a principled way to quantify and compare activity at topologically distinct locations.

### 2. Subpopulation dynamics linked to path familiarity

The framework detected changes in RSP neuronal subpopulation dynamics as a function of path familiarity — i.e., whether the animal was traversing a corridor/junction for the first time vs. on repeated visits. This suggests that RSP ensemble activity tracks not just current spatial position but also the history of navigation through the maze topology.

### 3. Sex differences in wildtype navigation

Wildtype female mice displayed more direct recall navigation compared to males. This sex difference in navigation strategy is consistent with broader literature on sex differences in spatial cognition.

### 4. apoE4 female-specific navigation deficits

In apoE4 knock-in mice (modelling the most prevalent genetic risk factor for Alzheimer's disease), **females** exhibited navigation deficits that were detectable only through the topological metrics enabled by NaviGraph. These deficits included:

- Inefficient navigation (longer graph-theoretic path lengths relative to optimal)
- Increased visits to decision points (more revisits/perseveration)

These deficits were not apparent from standard trajectory analyses, demonstrating the added value of the graph-based approach. The female-specific vulnerability aligns with epidemiological evidence that female apoE4 carriers have heightened risk for Alzheimer's disease.

### 5. Physiological-behavioural alignment

The framework enabled direct alignment of neural and behavioural data streams within the graph structure, allowing visualisation and quantification of how RSP activity relates to navigation behaviour at specific topological locations.

---

## Software and Code Availability

NaviGraph is described as an **open-source** pipeline with a **modular architecture** supporting diverse maze configurations and data types. The specific code repository URL was not identified in the search results; check the bioRxiv preprint for a GitHub link or data availability statement.

---

## Relevance to hm2p

NaviGraph is directly relevant to the hm2p project in several ways, though the match is not exact — there are important differences in maze design, imaging modality, and scientific questions.

### What aligns well

**1. Graph representation of the Rosenberg maze is natural and tractable.**
The hm2p Rosenberg maze is a binary-choice labyrinth with T-junctions at every decision point. This maps cleanly onto a graph where each T-junction is a node and each corridor segment is an edge. The Rosenberg maze (Rosenberg et al. 2021, eLife) has 63 T-junctions in the full tree, though the hm2p version may use a subset. The graph representation would enable:

- Quantifying exploration efficiency (how many unique nodes visited per unit time)
- Measuring path optimality (shortest graph path vs. actual path taken)
- Identifying perseverative behaviour at specific junctions
- Comparing exploration strategies between light and dark epochs at the topological level

**2. Decision-point-specific neural activity analysis.**
NaviGraph's approach of mapping neural activity to specific graph nodes could be applied to hm2p data to ask: does Penk+ vs Penk-CamKII+ RSP activity differ at decision points (T-junctions) compared to corridor traversals? This is related to the known RSP role in route planning and spatial context encoding (Alexander & Nitz 2015; Mao et al. 2017). Specifically:

- Do HD tuning properties change at decision points vs. straight corridors?
- Does one cell type show elevated activity or altered tuning at junctions where the animal must choose a direction?
- Is there a "deliberation signal" in either population at decision points, and does it differ between light and dark?

**3. Path familiarity analysis.**
NaviGraph's finding that RSP subpopulation dynamics change with path familiarity is relevant to hm2p. In the Rosenberg maze, mice explore freely and gradually learn the maze structure. Tracking how Penk+ and Penk-CamKII+ population activity evolves as specific corridors become familiar could reveal whether one population preferentially encodes novelty or familiarity — complementary to the primary HD anchoring question.

**4. Light/dark as an additional graph annotation layer.**
NaviGraph's multi-layered node annotation approach could be extended to include light condition as a layer. Each node visit could be tagged as light-on or light-off, enabling comparison of neural activity at the same topological location under different visual conditions. This is a natural extension not explored in the NaviGraph paper (which did not manipulate visual cues).

### What differs and requires adaptation

**1. Imaging modality.**
NaviGraph used one-photon miniscope imaging; hm2p uses two-photon head-mounted microscopy (~9.6 Hz). Two-photon provides better optical sectioning and less neuropil contamination, but the analytical framework for mapping activity to graph nodes is the same. The lower frame rate of 2P (9.6 Hz vs. typical miniscope 20-30 Hz) means fewer samples per decision-point dwell, which may limit the temporal resolution of junction-specific analyses.

**2. Cell-type specificity.**
NaviGraph did not image genetically defined subpopulations — they recorded from bulk RSP populations. The hm2p dataset's Penk+ vs Penk-CamKII+ distinction is a major advantage. Applying NaviGraph's framework separately to each cell type could reveal whether the decision-point and path-familiarity effects they observed are driven by one subpopulation or the other.

**3. Maze design.**
NaviGraph used an unspecified "complex maze" with a trial-based spatial memory task. The hm2p Rosenberg maze is a binary-tree labyrinth with free exploration (no explicit trial structure or goal). This difference affects what graph metrics are meaningful — path optimality requires a defined goal, which may not apply to free exploration. However, dead-end visits, exploration coverage, and turning bias at junctions are all computable without a goal.

**4. Scientific question.**
NaviGraph's primary application was detecting apoE4-related navigation deficits. The hm2p project focuses on visual vs. idiothetic HD anchoring in genetically defined RSP subpopulations. The framework is a tool, not a competing hypothesis — it provides a complementary analytical lens for the hm2p data.

### Concrete analysis ideas from NaviGraph for hm2p

1. **Build a graph representation of the Rosenberg maze.** Define T-junctions as nodes, corridor segments as edges. Map the animal's position at each frame to the nearest graph node or edge. This is Stage 3-level processing.

2. **Decision-point HD analysis.** At each T-junction, extract the HD tuning of Penk+ and Penk-CamKII+ neurons during the dwell period. Compare tuning curve properties (MVL, width, stability) at junctions vs. corridors. Compare light vs. dark at junctions.

3. **Topological exploration metrics.** Compute per-epoch (light/dark) metrics: fraction of unique nodes visited, mean path length between revisits to the same node, dead-end visit rate. Test whether exploration strategy changes between light and dark, and whether this correlates with population-level HD decoding accuracy.

4. **Path familiarity x cell type interaction.** For each corridor, count traversal number (1st, 2nd, 3rd, ...). Bin neural activity by traversal number and test whether Penk+ or Penk-CamKII+ populations show familiarity-dependent changes in activity or HD tuning stability.

5. **Junction choice prediction from neural state.** At each T-junction, use pre-junction population activity (100-200 ms before arrival) to predict left vs. right turn. Compare prediction accuracy between cell types and light conditions. This would test whether either population carries a prospective decision signal.

### Caveats for hm2p application

- **Dwell time at junctions may be short.** At 9.6 Hz, a 0.5-second pause at a junction gives ~5 frames — marginal for computing tuning curves. This analysis may need to be aggregated across all visits to a given junction class (e.g., all first-visited junctions pooled).
- **Graph size.** The full Rosenberg maze has 63 junctions, but mice in a single session may visit only a subset. The effective graph for analysis is the subgraph of visited nodes.
- **No explicit goal.** Unlike NaviGraph's trial-based task, hm2p free exploration has no defined optimal path. Efficiency metrics must be defined relative to exploration coverage rather than goal-directed performance.
- **This is exploratory, not primary.** Graph-based analyses would complement the core HD tuning and population decoding story. They are Tier 2 (supporting) or Tier 3 (exploratory) findings, unless a striking cell-type-specific effect at decision points emerges.

---

## Key References Cited or Related

| Reference | Relevance |
|---|---|
| Rosenberg et al. 2021 (eLife) | Binary-tree labyrinth design used in hm2p |
| Alexander & Nitz 2015 (Neuron) | RSP neurons encode routes and prospective spatial information |
| Mao et al. 2017 (Nat Commun) | Sparse orthogonal spatial context representations in RSP |
| Cho & Sharp 2001 (Behav Neurosci) | HD, place, and movement correlates in rat RSP |
| Jacob et al. 2017 (Nat Neurosci) | RSP HD cells anchored to visual landmarks |
| Fischer et al. 2020 (Curr Biol) | RSP integrates visual and self-motion cues |
