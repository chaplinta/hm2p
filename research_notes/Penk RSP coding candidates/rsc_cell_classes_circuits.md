# RSC functional cell classes, laminar organisation and input circuitry (focus: superficial L2/3 excitatory neurons)

Method note: Built from web searches (Oct 2026). Full-text fetches failed (network errors on PMC, bioRxiv, eLife), so most findings come from abstracts and search snippets. Percentages and layer details are given only where a source stated them. DOIs come from the source URL where possible. DOIs marked "(from memory, verify)" were not confirmed against a page in this session.

## Q1. Known RSC functional classes and their prevalence

### Takeaway
RSC has many overlapping, often conjunctive signals: HD (including landmark-anchored bidirectional HD in dysgranular RSC), AHV, speed, egocentric boundary vectors, turning/action, route progress, place-like sequences, landmark, reward/goal and time signals. Prevalence figures are scattered and depend on the method. The superficial layers carry a large share of the place/sequence, landmark, border, head-turning and speed cells reported in head-fixed mouse imaging.

### Cited Findings
**Head direction (HD), incl. bidirectional / landmark-anchored**
- Jacob, Casali, Spieser, Page, Overington & Jeffery 2017, "An independent, landmark-dominated head-direction signal in dysgranular retrosplenial cortex", *Nat Neurosci* 20:173, doi:10.1038/nn.4465 (rat). In a two-compartment environment with rotationally symmetric visual cues, some dysgranular RSC (RSPd/agranular) cells fired bidirectionally, i.e. with two tuning peaks 180 deg apart that follow the local visual landmark instead of the global HD signal. All 116 double-peaked cells had peak separations near 180 deg, compared with about 4% of random HD-cell pairs. Bidirectional firing persisted when visual information was removed, and the cells rotated with the apparatus — [Nature](https://www.nature.com/articles/nn.4465); [search summary](https://www.nature.com/articles/nn.4465)
- Sit & Goard 2023, "Coregistration of heading to visual cues in retrosplenial cortex", *Nat Commun*, doi:10.1038/s41467-023-37704-5 (mouse, head-fixed, 2P, rotating environment). RSC neurons are tuned to the animal's orientation relative to the environment even without head movement, and the paper proposes an RSC circuit that anchors heading to visual landmarks — [Nature Commun](https://www.nature.com/articles/s41467-023-37704-5)

**Angular head velocity (AHV)**
- Keshavarzi et al. 2022, "Multisensory coding of angular head velocity in the retrosplenial cortex", *Neuron* 110(3):532, doi:10.1016/j.neuron.2021.10.031 (from memory, verify) (mouse). RSC neurons track the direction and speed of head turns **in complete darkness from vestibular input**. Visual input increases the gain and SNR of AHV coding, and ensemble decoding of angular speed is best when vestibular and visual input are combined. The authors identified subsets coding HD, AHV, locomotion speed or combinations of these — [PubMed](https://pubmed.ncbi.nlm.nih.gov/34788632/)
- A cell-type-specific cholinergic modulation study of granular RSC (Ahmed lab, *Prog Neurobiol* 2025) frames its findings as having "implications for angular velocity coding across brain states" (title). Details were not retrieved — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12340933/); [PubMed](https://pubmed.ncbi.nlm.nih.gov/40639485/)

**Egocentric boundary vector cells (EBCs)**
- Alexander, Carstensen, Hinman, Raudies, Chapman & Hasselmo 2020, "Egocentric boundary vector tuning of the retrosplenial cortex", *Sci Adv* 6(8):eaaz2322, doi:10.1126/sciadv.aaz2322 (rat). **21.4% of RSC neurons (119/555)** were EBCs when referenced to movement direction — [Science Advances](https://www.science.org/doi/10.1126/sciadv.aaz2322)
- van Wijngaarden, Babl, Ito 2020, "Entorhinal-retrosplenial circuits for allocentric-egocentric transformation of boundary coding", *eLife* — [eLife](https://elifesciences.org/articles/59816) (details not retrieved)
- PNAS 2026, "Anterior and posterior retrosplenial cortex employ distinct strategies for egocentric–allocentric transformation in spatial coding", doi:10.1073/pnas.2600565123. Only the title was retrieved: anterior and posterior RSC handle the reference-frame transform differently — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2600565123)
- Oh, Yang, Shin & Kwag 2026 (bioRxiv), "Retrosplenial PV and SST interneurons shape egocentric spatial precision and stability", doi:10.64898/2026.05.10.724096 (mouse). PV interneurons are strongly modulated by self-motion and show bearing-aligned synchrony that precedes SST activation. SST cells are weakly self-motion-modulated but have robust boundary-anchored, globally coherent activity. Silencing PV degraded egocentric coding precision and impaired initial egocentric orientation. Silencing SST disrupted population organisation and impaired sustained updating — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.10.724096v1)

**Turning/action, route progress, position (conjunctive)**
- Alexander & Nitz 2015, "Retrosplenial cortex maps the conjunction of internal and external spaces", *Nat Neurosci* 18:1143, doi:10.1038/nn.4058 (rat, on track routes). RSC ensembles conjunctively encoded route progress, position in the larger environment, and left vs right turning — [Nature Neurosci](https://www.nature.com/articles/nn.4058)
- Alexander & Nitz 2017 (J Neurosci; bioRxiv 100537), "Spatially periodic activation patterns of retrosplenial cortex encode route sub-spaces and distance travelled" — [bioRxiv](https://www.biorxiv.org/content/10.1101/100537v1.full)

**Place-like / sequence cells**
- Mao, Kandler, McNaughton & Bonin 2017, "Sparse orthogonal population representation of spatial context in the retrosplenial cortex", *Nat Commun* 8:243, doi:10.1038/s41467-017-00180-9 (mouse, head-fixed treadmill, Ca2+ imaging). They found a population **located predominantly in superficial layers** whose activity resembles CA1 place cells: narrow fields, sequences during movement, and a sparse orthogonal code for location — [Nature Commun](https://www.nature.com/articles/s41467-017-00180-9)
- Mao et al. 2018, "Hippocampus-dependent emergence of spatial sequence coding in retrosplenial cortex", *PNAS* 115(31):8015 (mouse). The sequences depend on hippocampus — [PNAS](https://www.pnas.org/content/115/31/8015)

**Landmark / cue cells; laminar breakdown**
- Fischer, Mojica Soto-Albors, Buck & Harnett 2020, "Representation of visual landmarks in retrosplenial cortex", *eLife* 9:e51458, doi:10.7554/eLife.51458 (mouse, head-fixed VR, 2P). Both superficial and deep layers contained trial-onset, landmark and reward neurons, but **L5 contained substantially fewer landmark neurons**. A **large proportion of border cells, head-turning cells and locomotor speed cells were in the superficial layer** (search snippet; exact proportions not retrieved) — [eLife](https://elifesciences.org/articles/51458)
- A bioRxiv 2024 preprint, "Learning to use landmarks for navigation amplifies their representation in retrosplenial cortex": landmark representations grow with learning (chronic 2P) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.08.18.607457.full.pdf); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11370392/)
- Jeffery lab follow-up: "Uncoupling of dysgranular retrosplenial 'head direction' cells from the global head direction network" — [ResearchGate](https://www.researchgate.net/publication/320226134_Uncoupling_of_dysgranular_retrosplenial_head_direction_cells_from_the_global_head_direction_network)
- Comparison of RSC and postsubicular HD cells during visual landmark discrimination (Frontiers 2018) — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6124005/)

**Goal / reward / trajectory**
- Vedder, Miller, Harrison & Smith 2017, "Retrosplenial cortical neurons encode navigational cues, trajectories and reward locations during goal directed navigation", *Cereb Cortex* 27(7):3713 (rat) — [summary via search](https://blogs.cornell.edu/davidsmithlab/publications/)
- Miller, Mau & Smith 2019, "Retrosplenial cortical representations of space and future goal locations develop with learning", *Curr Biol* 29(12):2083 (rat). Spatial representations develop with learning, and population patterns simulate future goal locations, which suggests a role in navigational planning — [Curr Biol PDF](https://www.cell.com/current-biology/pdfExtended/S0960-9822(19)30603-7)
- RSC role in delayed spatial alternation (bioRxiv 2024) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.18.599656v1.full)
- "Dual-Factor Representation of the Environmental Context in the RSC" (Cereb Cortex 2021) — [OUP](https://academic.oup.com/cercor/article/31/5/2720/6056293)
- "Time cells in the retrosplenial cortex" (Hippocampus 2024) — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/hipo.23635)

**Anterior vs posterior**
- Wei et al. 2026, "Anterior and posterior retrosplenial cortex form distinct visuospatial circuits in the mouse", *Nat Commun* 17:4388, doi:10.1038/s41467-026-70762-z. Anterior RSC has **sharper position tuning** and prefers fast, low-spatial-frequency visual motion. Posterior RSC has **broader position selectivity** and responds more to slow, high-SF patterns. Anterior RSC receives denser motor, somatosensory and parietal input. Posterior RSC receives stronger V1 and posteromedial visual input. Code: github.com/ytsimon2004/rscvp — [Nature Commun](https://www.nature.com/articles/s41467-026-70762-z); [PubMed](https://pubmed.ncbi.nlm.nih.gov/41881980/)

**Review framing**
- Alexander, Place, Starrett, Chrastil & Nitz 2023, "Rethinking retrosplenial cortex: Perspectives and predictions", ***Neuron*** 111:150 (not Nat Rev Neurosci, as the task brief stated), doi:10.1016/j.neuron.2022.11.006 (from memory, verify). The authors propose that RSC activity relates spatial perspectives and generates predictions about interactions with the environment — [Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(22)01027-3)
- Vann, Aggleton & Maguire 2009, "What does the retrosplenial cortex do?", *Nat Rev Neurosci* 10:792 — [PubMed](https://ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=PubMed&list_uids=19812579)

### Inferences
- Superficial-layer imaging studies (Mao 2017; Fischer 2020) report place/sequence, border, head-turning, speed and landmark cells in L2/3. The clearest laminar bias found is that L5 has fewer landmark cells. "Superficial = less visual" therefore does **not** hold as a general rule. Landmark coding is, if anything, enriched superficially.
- Prevalences are not comparable across studies (rat tetrodes vs mouse 2P, freely moving vs VR). The one hard number found was EBCs at about 21% (rat, freely moving). In this project, a 10% HD yield for L2/3 calcium imaging is plausible but cannot be benchmarked against a superficial-specific HD prevalence in the literature (see Gaps).

### Gaps
- No source found with layer-resolved HD-cell prevalence in freely moving rodents. Most HD-in-RSC recordings are rat tetrode recordings without laminar assignment.
- Vale et al. 2020 (the brief lists it as a landmark/cue paper) could not be verified. It may refer to Vale, Campagner et al. 2020 *Nature* on a cortico-collicular RSC circuit for shelter-direction orientation during escape, but this was not confirmed in this session.
- Exact superficial-vs-deep percentages from Fischer 2020 were not retrieved (full text fetch failed).

## Q2. Laminar and cell-type biases (superficial vs deep, granular vs dysgranular, Cre lines / molecular types)

### Takeaway
The best-characterised superficial excitatory type is the **low-rheobase (LR) L2/3 pyramidal neuron of granular RSC (RSG / RSPv)**. It is Cxcl14+, hyperexcitable, non-adapting, and preferentially driven by anterior thalamus and dorsal subiculum. A neighbouring **regular-spiking (RS)** L2/3 population is preferentially driven by claustrum and ACC. Dysgranular RSC (RSPd/RSPagl) carries the landmark-dominated, bidirectional HD signal. No published work was found on Penk+ neurons in RSC specifically.

### Cited Findings
- Brennan, Sudhakar, Jedrasiak-Cape & Ahmed 2020, "Hyperexcitable neurons enable precise and persistent information encoding in the superficial retrosplenial cortex", *Cell Rep* 30(5) (mouse, slice + modelling). A uniquely excitable small pyramidal cell (LR) is the **most prominent cell type in L2/3 of granular RSC**. Biophysical models of LR (but not RS) cells precisely and continuously encode sustained input from postsubicular HD cells. The authors conclude LR properties support precision and persistence over multiple timescales. The HD-encoding claim comes from modelling, not in vivo recording — [Cell Rep](https://www.cell.com/cell-reports/fulltext/S2211-1247(19)31758-9); [bioRxiv](https://www.biorxiv.org/content/10.1101/673954v1.full)
- Sullivan et al. 2023, "Sharp cell-type-identity changes differentiate the retrosplenial cortex from the neocortex", *Cell Rep* 42(3):112206, doi:10.1016/j.celrep.2023.112206 (mouse, transcriptomics + physiology). LR neurons sit **exclusively in L2/3** of granular RSC and express **Cxcl14**. Their properties are low rheobase, high input resistance, no spike-frequency adaptation, and spike widths between FS interneurons and RS pyramids. The Cxcl14 cluster's location, size relative to all L2/3 excitatory cells, and prior physiology all point to LR = L2/3 Cxcl14 cluster — [Cell Rep](https://www.cell.com/cell-reports/fulltext/S2211-1247(23)00217-6); [ResearchGate](https://www.researchgate.net/publication/369045169_Sharp_cell-type-identity_changes_differentiate_the_retrosplenial_cortex_from_the_neocortex)
- Brooks, Jedrasiak-Cape, Rybicki-Kler, Ekins & Ahmed 2025 (bioRxiv 2024), "Unique transcriptomic cell types of the granular retrosplenial cortex are preserved across mice and rats despite dramatic changes in key marker genes", *J Neurosci* 45(48):e2246242025. RSG cell types are conserved between mouse and rat. L2/3 LR and L5a types are over-represented in rat. Marker genes differ strongly between species: *Scnn1a*, which tags mouse L5a RSG, is absent in rat. The authors warn that knock-in lines need species-specific markers — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12660178/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.09.17.613545v1.full)
- Kurotani et al. 2013 (rat). Superficial pyramidal neurons of RSC have a late-spiking (LS) firing property, likely the rat equivalent of the LR phenotype — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3535347/)
- Granular RSC L2/3 generates high-frequency oscillations coupled with hippocampal rhythms across brain states (*Cell Rep* 2024). ChR2 excitation of CaMKII+ cells in L2/3 or L5 induces HFOs, but **spontaneous HFOs occur only in L2/3** — [Cell Rep](https://www.sciencedirect.com/science/article/pii/S2211124724002389)
- Penk: proenkephalin (Penk) marks an excitatory subtype present in L2/3 and L6 in mouse **visual cortex**. The search returned no RSC-specific Penk literature — [search summary; Sci Rep 2021 V1 connectivity](https://www.nature.com/articles/s41598-021-82353-7)
- Dysgranular RSC hosts the landmark-dominated bidirectional HD cells (Jacob 2017, above) — [Nature Neurosci](https://www.nature.com/articles/nn.4465)
- Anterior–posterior gradient in tuning sharpness and visual input (Wei 2026, above) — [Nature Commun](https://www.nature.com/articles/s41467-026-70762-z)
- PV and SST interneurons have dissociable roles (self-motion/bearing vs boundary/stability) (Oh 2026, above) — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.10.724096v1)

### Inferences
- A sparse, highly excitable, L2/3 excitatory population in granular RSC fits the **LR/Cxcl14 phenotype**: hyperexcitable, high input resistance, non-adapting, thalamus-driven. Whether Penk+ RSC cells overlap with Cxcl14/LR cells is **unknown**, and this is the key thing to check. It can be checked directly in Allen Brain Cell Atlas / Yao et al. 2021/2023 MERFISH and scRNA-seq data for RSP L2/3 IT clusters, looking at Penk vs Cxcl14 co-expression. If Penk overlaps the LR cluster, the literature predicts anterior-thalamic/HD-like, persistent, integrator-type coding. If Penk instead labels RS-type cells, it predicts claustrum/ACC-driven, possibly more context/attention-related coding.
- The Ahmed lab line of work (Brennan 2020; Brennan 2021; Sullivan 2023; Brooks 2025; Prog Neurobiol 2025) predicts that LR cells relay HD and AHV with high fidelity and persistence. That fits a cell class whose HD representation would survive darkness, because ATN HD input is maintained in darkness by vestibular drive.

### Gaps
- No in vivo recordings of genetically identified LR/Cxcl14 or Penk RSC neurons during behaviour were found. The LR=HD-integrator idea rests on slice physiology plus modelling.
- No Cre-line functional studies (e.g. Rbp4-L5 vs L2/3 lines) in RSC were retrieved in this session.
- Granular vs dysgranular differences in superficial-layer function in freely moving mice are not resolved in what was retrieved.

## Q3. Inputs to superficial RSC and which would make cells less light-driven

### Takeaway
In granular RSC L2/3, **anterior thalamus (ATN) and dorsal subiculum preferentially drive LR cells**, through LR dendrites that converge with ATN axons in L1a. **Claustrum and ACC preferentially drive RS cells.** The newest work reports that ATN input reaches essentially every RSC pyramidal neuron in every subdivision and layer. Direct visual-cortex input is weighted to posterior RSC and dysgranular RSC. A superficial granular-RSC cell dominated by ATN/subicular input is the predicted "less visually driven" type.

### Cited Findings
- Brennan, Jedrasiak-Cape, Kailasa, Rice, Sudhakar & Ahmed 2021, "Thalamus and claustrum control parallel layer 1 circuits in retrosplenial cortex", *eLife* 10:e62207, doi:10.7554/eLife.62207 (mouse). Thalamic control of LR neurons is explained by precise convergence of LR dendrites and ATN axons in **L1a** of RSG. **Claustral inputs selectively drive RS, not LR, neurons**, consistent with greater overlap between RS dendrites and claustral axons. Precise sublaminar organisation in L1 supports parallel processing — [eLife](https://elifesciences.org/articles/62207); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8233040/)
- Yamawaki, Shepherd and colleagues (search summary of their RSG work and related papers). **ATN and dorsal subiculum preferentially recruit LR pyramidal cells in RSG L2/3. Neighbouring RS cells are preferentially controlled by claustral and ACC inputs** — [search results incl. PMC7669619, "Layer 2/3 pyramidal neurons of the mouse granular RSC and their innervation by cortico-cortical axons", Front Neural Circuits 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7669619/)
- Gao et al. 2021, "The subiculum sensitizes retrosplenial cortex layer 2/3 pyramidal neurons", *J Physiol*, doi:10.1113/JP281152. Subicular inputs to RSC L2/3 modulate later integration in L2/3 circuits. A related finding: CA1 apical-tuft-targeting long-range inhibitory neurons gate the RSC thalamocortical circuit — [Wiley](https://physoc.onlinelibrary.wiley.com/doi/10.1113/JP281152); [PubMed](https://pubmed.ncbi.nlm.nih.gov/33878801/); [ResearchGate (CA1 inhibitory)](https://www.researchgate.net/publication/327873950_Long-range_inhibitory_intersection_of_a_retrosplenial_thalamocortical_circuit_by_apical_tuft-targeting_CA1_neurons)
- Margetts-Smith, Andrianova, Kohli, Randall, Aggleton, Witton & Craig 2025 (bioRxiv), "Dissection of retrosplenial cortex inputs: ubiquitous drive from anterior thalamus", doi:10.1101/2025.02.06.636939 (mouse; tracing + optogenetics + patch clamp of ACC, dorsal subiculum and ATN inputs). **All recorded RSC pyramidal neurons received ATN input regardless of subdivision or layer.** NMDA receptor components of excitatory inputs were weaker than expected, which may explain why LTP is hard to induce ex vivo in RSC — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.06.636939v1)
- Same group, bioRxiv Dec 2025. ATN→RSC synaptic strength is unimpaired in granular and dysgranular RSC of hAPP-J20 mice at 3, 6 and 9 months — [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.01.691675v1)
- Wei et al. 2026 / companion bioRxiv 2025 ("Specific anterior-posterior brain-wide input patterns support specialized visuospatial processing in the mouse RSC"). **Posterior RSC gets stronger V1/PM visual input. Anterior RSC gets denser motor/somatosensory/parietal input.** Dorsal subiculum (geometric/spatial) targets anterior RSC preferentially, and ventral subiculum (contextual/affective) targets posterior RSC — [Nature Commun](https://www.nature.com/articles/s41467-026-70762-z); [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.24.661247v1.full)
- "Dissection of the long-range circuit of the mouse intermediate RSC" (*Commun Biol* 2025) — [Nature](https://www.nature.com/articles/s42003-025-07463-8) (details not retrieved)
- Projection-specific RSC circuits with differential contributions to spatial cognition (*Mol Psychiatry* 2024) — [Nature](https://www.nature.com/articles/s41380-024-02819-8) (details not retrieved)

### Inferences
- **Inputs predicted to reduce light-driven responses:** ATN (AD/AV carry HD maintained by vestibular/idiothetic signals in darkness) and dorsal subiculum/postsubiculum. These preferentially target LR cells in granular L2/3. **Inputs predicted to increase light-driven responses:** V1/PM visual cortex (posterior and dysgranular RSC). Claustrum/ACC (RS cells) carry attention/salience/context signals that are not specifically visual.
- A sparse, excitable, weakly light-driven L2/3 cell in granular, more anterior RSC would most likely be ATN/subiculum-dominated. Expected coding: HD and AHV maintained in darkness, persistent/integrated self-motion signals, possibly route/sequence position (hippocampus-dependent per Mao 2018). Landmark-anchored or visual-motion signals are less likely.
- Margetts-Smith 2025 (ATN input reaches all pyramids) means the difference between cell types is likely one of **weighting and dendritic location** (L1a convergence), not presence or absence of thalamic input.

### Gaps
- No direct data on which inputs target Penk+ RSC cells. Monosynaptic rabies from Penk-Cre RSC would answer this. None was found.
- LD thalamus (carries visual/landmark HD information) projections to RSC L2/3 subtypes were not retrieved.
- Contralateral RSC input to L2/3 subtypes was not retrieved.

## Q4. RSC in memory (contextual/episodic, consolidation, novelty, darkness/path integration, temporal association)

### Takeaway
RSC holds contextual fear-memory engrams that can be sufficient for recall and can drive systems consolidation. RSC inactivation selectively impairs navigation in darkness, which implies a role in path integration and idiothetic navigation. RSC also supports cue-conflict resolution, goal planning and temporal coding.

### Cited Findings
- Cowansage et al. 2014, "Direct reactivation of a coherent neocortical memory of context", *Neuron* 84:432–441, doi:10.1016/j.neuron.2014.09.022 (from memory, verify) (mouse). Optogenetic reactivation of RSC neurons tagged during fear conditioning elicited recall in a novel context — [summary via PNAS citing paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6486739/)
- de Sousa, Cowansage, Zutshi, Cardozo, Yadgarov, Kawashima & Mayford 2019, "Optogenetic reactivation of memory ensembles in the retrosplenial cortex induces systems consolidation", *PNAS* 116(17):8576, doi:10.1073/pnas.1818432116 (mouse). High-frequency reactivation of RSC engram ensembles 1 day after learning produced a remote-like memory: more neocortical engagement, contextual generalisation, and less hippocampal dependence. This happened **only if reactivation occurred during sleep or light anaesthesia** — [PNAS](https://www.pnas.org/doi/10.1073/pnas.1818432116); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6486739/)
- Cooper & Mizumori 1999, *NeuroReport*: **RSC inactivation selectively impairs navigation in darkness** (rat). Cooper, Manka & Mizumori 2001, *Behav Neurosci*: RSC is needed for spatial memory without visual cues. Cooper & Mizumori 2001, *J Neurosci*: RSC inactivation causes transient reorganisation of hippocampal place coding — [Mizumori lab publications](https://www.mizumorilab.com/publications); [Springer model paper citing these](https://link.springer.com/article/10.1007/s00422-020-00833-x)
- Pothuizen et al. 2008, "Do rats with retrosplenial cortex lesions lack direction?", *Eur J Neurosci* — [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1460-9568.2008.06550.x)
- Dysgranular RSC lesions disrupt cross-modal object recognition (*Learn Mem* 2014) — [CSHL](https://learnmem.cshlp.org/content/21/3/171.full.html)
- Anterior RSC is required for short-term object-in-place recognition memory retrieval (rat, 2024) — [PubMed](https://pubmed.ncbi.nlm.nih.gov/38411499)
- RSC 5-HT2A receptors contribute to recognition memory (2025) — [PubMed](https://pubmed.ncbi.nlm.nih.gov/41342013/)
- Spatial transcriptomic signature of RSC during memory consolidation (*Mol Psychiatry* 2025) — [Nature](https://www.nature.com/articles/s41380-025-03331-3)
- A model of path integration and spatial-context representation in RSC (*Biol Cybern* 2020) — [Springer](https://link.springer.com/article/10.1007/s00422-020-00833-x)
- Time cells in RSC (*Hippocampus* 2024) — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/hipo.23635)
- Goal/future-location coding develops with learning (Miller 2019, above) — [Curr Biol](https://www.cell.com/current-biology/pdfExtended/S0960-9822(19)30603-7)

### Inferences
- The darkness-specific behavioural deficit after RSC inactivation (Cooper & Mizumori) fits RSC holding an idiothetic or path-integration-based representation that is needed when vision is absent. That predicts preserved, not degraded, RSC spatial/HD coding in darkness. This matches the project's own null "dark vs light" results after occupancy matching.
- Engram work localises contextual memory to RSC but does not identify layer or cell type. Superficial L2/3 neurons are a plausible engram substrate, but this has not been shown.

### Gaps
- Temporal-association work (Todd, Bucci, Fournier) and novelty/familiarity coding were not retrieved in this session.
- No source tied engram allocation to a specific RSC layer or molecular type.

## Q5. RSC activity in darkness vs light and in mazes (route, junctions, cue removal)

### Takeaway
RSC HD and AHV signals persist in darkness (vestibular), with visual input adding gain and fast realignment. Dysgranular bidirectional HD cells keep firing without vision. In track/maze tasks RSC encodes route progress, turn direction and position conjunctively. No layer-specific light-vs-dark comparison was found.

### Cited Findings
- Keshavarzi 2022: AHV coding persists in complete darkness from vestibular input. Vision raises AHV gain and SNR — [PubMed](https://pubmed.ncbi.nlm.nih.gov/34788632/)
- HD-cell literature (search summary of RSC/landmark sources): turning on lights causes near-instant realignment of HD responses to visual cues, even when cues were moved in darkness. Bidirectional firing persists without vision — [Nature Neurosci, Jacob 2017](https://www.nature.com/articles/nn.4465); [Sit & Goard 2023](https://www.nature.com/articles/s41467-023-37704-5)
- Alexander & Nitz 2015: conjunctive route progress × environmental position × left/right turn coding on track routes (rat) — [Nature Neurosci](https://www.nature.com/articles/nn.4058)
- Vedder 2017: RSC encodes navigational cues, trajectories and reward locations during goal-directed navigation (rat, plus-maze-type task) — [via Smith lab](https://blogs.cornell.edu/davidsmithlab/publications/)
- Cooper & Mizumori 1999/2001: RSC inactivation impairs dark navigation — [Mizumori lab](https://www.mizumorilab.com/publications)

### Inferences
- In the q-rose (Rosenberg) maze, a weakly light-driven superficial population would be expected to carry signals that do not need vision: HD/AHV (vestibular/ATN), turn direction and route progress (self-motion-based), and possibly junction/turn conjunctions in the Alexander & Nitz sense. Landmark-bidirectional or visual-motion responses are less likely. Junction-by-turn conjunctive coding and route-phase coding are therefore reasonable candidates to test, alongside HD and AHV.

### Gaps
- No study was found comparing superficial vs deep RSC (or LR vs RS) neurons across light and dark in freely moving animals.
- No RSC junction/decision-point coding study was found that controls for visual cue removal.
