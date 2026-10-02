# Sparse, bursty, long-duration calcium events: what they imply about a population's code

Context being interpreted: one RSP excitatory population (Penk+) shows lower CASCADE rate (~0.13 vs 0.19 spikes/s), fewer active seconds, lower Fano factor, more skewed rate distributions, and rarer, LONGER (3.3 vs 2.6 s), slower-decaying, SMALLER calcium events than the comparison population (Penk-negative CamKII+). HD/AHV tuning weak; GLM (HD, AHV, speed, position, light) explains ~nothing. jGCaMP7f, ~9.6 Hz, freely moving.

Source-verification note: items marked [verified] were confirmed in a search result or page this session. Items marked [prior knowledge] are standard citations given from memory with DOIs; the report writer should treat the bibliographic details as likely-correct but not re-checked here.

---

## 1. Plateau potentials, complex spikes and BTSP: do long in-vivo calcium events reflect plateaus, and what do they encode?

### Takeaway
In CA1, dendritic plateau potentials drive complex-spike bursts and produce large, prolonged somatic calcium events that can create a place field in a single trial (BTSP), with plateaus most frequent during novelty and learning. The signature is large-amplitude events; the Penk+ events are longer but SMALLER, which fits the plateau/BTSP picture poorly. Direct evidence that RSP plateaus exist in behaving animals and do the same thing is thin.

### Cited Findings
- Bittner et al. 2015, "Conjunctive input processing drives feature selectivity in hippocampal CA1 neurons", Nat Neurosci 18:1133–1142, doi:10.1038/nn.4062. Dendritic plateau potentials arise from conjunctive, properly timed EC3 + CA3 input; plateaus positively modulate existing place fields and rapidly induce new place fields. Plateaus act as coincidence detectors that change both rate and mode (burst/complex-spike) of output. [verified] — [Nature Neuroscience](https://www.nature.com/articles/nn.4062)
- Bittner et al. 2017, "Behavioral time scale synaptic plasticity underlies CA1 place fields", Science 357:1033–1036, doi:10.1126/science.aan3846. A single in-vivo plateau/complex-spike event can produce a place field by potentiating inputs active seconds before and after the plateau (asymmetric, seconds-long window); the potentiated input need not coincide with spiking. Five slice pairings of subthreshold input with plateaus gave large potentiation with a seconds-long time course. Evidence strength: strong (intracellular in vivo + slice + model), but CA1-specific. [verified] — [Science](https://www.science.org/doi/10.1126/science.aan3846)
- Milstein et al. 2021, "Bidirectional synaptic plasticity rapidly modifies hippocampal representations", eLife 10:e73046. BTSP is bidirectional: plateaus can also depress inputs and shift/erase existing fields, so plateau events reshape representations rather than only adding fields. [verified title/venue in search; content from prior knowledge] — [eLife](https://elifesciences.org/articles/73046)
- Grienberger & Magee 2022, "Entorhinal cortex directs learning-related changes in CA1 representations", Nature 611:554–562, doi:10.1038/s41586-022-05378-6. EC input instructs plateau-driven BTSP so that CA1 over-represents behaviourally relevant locations (e.g. reward) during learning. [title verified in search; content prior knowledge] — [Nature](https://www.nature.com/articles/s41586-022-05378-6)
- Priestley, Bowler, Rolotti, Fusi & Losonczy 2022, "Signatures of rapid plasticity in hippocampal CA1 representations during novel experiences", Neuron 110:1978–1992, doi:10.1016/j.neuron.2022.03.026. In completely novel environments CA1 fields form rapidly with BTSP-like signatures (backward-shifting, single-lap onset), stabilising within minutes; plasticity strongly regulated by novelty. [verified] — [Neuron PDF](https://www.cell.com/neuron/pdf/S0896-6273(22)00262-8.pdf)
- A 2025 PubMed entry, "Diverse calcium dynamics underlie place field formation in hippocampal CA1 pyramidal cells" (PMID 41025505), indicates field formation is associated with heterogeneous calcium event types, not only one plateau-like event type. Not read in full. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/41025505/)
- Calcium indicator physics: bursts produce supralinearly larger fluorescence events than isolated spikes (Huang et al. 2021; Siegle et al. 2021, see section 4). So a plateau-driven complex-spike burst should appear as a LARGE event. [verified] — [Huang et al. 2021 eLife](https://elifesciences.org/articles/51675.pdf)

### Inferences
- If Penk+ long events were plateau/complex-spike bursts, they should be larger in amplitude than comparison-population events, not smaller. Longer + smaller + slower-decaying is more consistent with (a) low-rate sustained firing over 1–3 s (sparse spikes spread out in time), (b) slower indicator kinetics / different expression in that population, or (c) partial dendritic/neuropil contribution. This is inference, not tested.
- The BTSP framework predicts plateau-associated events cluster at novelty, at first visits to locations, and at reward/goal locations, and that a field appears from the next traversal onward. In a familiar maze without reward, BTSP-type events would be expected to be rare, which is consistent with low event rate but would not by itself predict longer duration.
- A cheap test: align Penk+ events to first entries into maze cells (per session) and to the first minutes of the session; check whether a cell's spatial preference after an event differs from before (single-event field formation). Positive result would support a plasticity/novelty interpretation.

### Gaps
- No source found showing plateau potentials or BTSP in RSP pyramidal cells in behaving animals. Whether RSP layer 2/3 or layer 5 cells generate CA1-like plateaus in vivo is unknown from this search.
- No source found that distinguishes plateau events from ordinary bursts using somatic GCaMP7f at ~10 Hz; at this frame rate the distinction is probably not resolvable without electrophysiology.

---

## 2. What do long, rare events in cortex correspond to (dendritic events, bursts, sustained episode firing)? RSP / parietal / hippocampal evidence

### Takeaway
RSP neurons imaged or recorded during navigation carry sparse place-like fields, landmark-anchored responses, route-progression, egocentric/allocentric conjunctions, and goal/reward signals. Apical dendrites in RSP carry signals partly independent of the soma. The variables that commonly drive sparse RSP activity (landmarks, route segments, goal locations, egocentric boundary/vertex relations) are NOT in the current GLM (HD, AHV, speed, position, light), which is a plausible reason it explains nothing.

### Cited Findings
- Mao, Kandler, McNaughton & Bonin 2017, "Sparse orthogonal population representation of spatial context in the retrosplenial cortex", Nat Commun 8:243, doi:10.1038/s41467-017-00180-9. Head-fixed treadmill, 2P imaging. Superficial-layer RSC neurons fire in sequences with narrow fields forming a sparse, orthogonal code for location, with partial remapping similar to CA1. [verified] — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5557927/)
- Mao et al. 2018, "Hippocampus-dependent emergence of spatial sequence coding in retrosplenial cortex", PNAS 115:8015–8018. RSC sequence/spatial coding depends on the hippocampus. [title/venue verified in search; content prior knowledge] — [PNAS](https://www.pnas.org/content/115/31/8015)
- Mao, Molina, Bonin & McNaughton 2020, "Vision and locomotion combine to drive path integration sequences in mouse retrosplenial cortex", Curr Biol 30:1680–1688, doi:10.1016/j.cub.2020.02.070. RSC sequences persist in darkness using locomotion signals but are shaped by visual input. [prior knowledge] — [doi](https://doi.org/10.1016/j.cub.2020.02.070)
- Alexander & Nitz 2015, "Retrosplenial cortex maps the conjunction of internal and external spaces", Nat Neurosci 18:1143–1151, doi:10.1038/nn.4058. Freely-moving rats: RSC ensembles conjunctively encode route progression, environmental position and the animal's actions (e.g. turns); individual neurons combine egocentric and allocentric frames. [verified] — [Nature Neuroscience](https://www.nature.com/articles/nn.4058)
- Vedder, Miller, Harrison & Smith 2017, "Retrosplenial cortical neurons encode navigational cues, trajectories and reward locations during goal directed navigation", Cereb Cortex 27:3713–3723, doi:10.1093/cercor/bhw192. [title verified in search; content prior knowledge] — [search listing via Fischer/Miller results](https://www.cell.com/neuron/fulltext/S0896-6273(22)01027-3)
- Miller, Mau & Smith 2019, "Retrosplenial cortical representations of space and future goal locations develop with learning", Curr Biol 29:2083–2090, doi:10.1016/j.cub.2019.05.034 (DOI from prior knowledge). RSC spatial firing and anticipatory goal-location signals emerge with learning. [title/venue verified] — [Rethinking RSC review listing](https://www.cell.com/neuron/fulltext/S0896-6273(22)01027-3)
- Fischer et al. 2020, "Representation of visual landmarks in retrosplenial cortex", eLife 9:e51458. Head-fixed mice learning landmark→hidden-reward relationships: landmarks were the dominant reference points for most task-active RSC neurons and anchored the spatial code. [verified] — [eLife](https://elifesciences.org/articles/51458)
- A follow-up (PMC11370392, "Learning to use landmarks for navigation amplifies their representation in retrosplenial cortex") reports landmark representation strengthens with learning. Not read in full. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11370392/)
- 2024 Nat Commun "Egocentric neural representation of geometric vertex in the retrosplenial cortex": RSC cells respond to the egocentric position of environment corners/vertices. Not read in full. — [Nature Communications](https://www.nature.com/articles/s41467-024-51391-w)
- Entorhinal–retrosplenial circuits for egocentric boundary coding (eLife 59816) — RSC egocentric boundary vector cells. Not read in full. — [eLife](https://elifesciences.org/articles/59816)
- Voigts & Harnett 2020, "Somatic and dendritic encoding of spatial variables in retrosplenial cortex differs during 2D navigation", Neuron 105:237–245, doi:10.1016/j.neuron.2019.10.016. Freely-rotating mice; somas and apical tuft dendrites imaged simultaneously. Both show global and local calcium transients; local tuft signals are tuned differently from the soma (HD and position), i.e. dendrites carry distinct navigational variables. [verified] — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/138266); [PDF](https://cenl.ucsd.edu/CompNeuro/Readings/week8/Voigts-Harnett+Somatic-dendritic-spatial-navigation-retrosplenial-cortex-differ-2D-navigation+Neuron+2020.pdf)
- Review: "Rethinking retrosplenial cortex: Perspectives and predictions", Neuron 2023 (S0896-6273(22)01027-3) — summarises RSC coding of landmarks, routes, egocentric/allocentric transformation, context. — [Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(22)01027-3)
- Hippocampal "time cells"/temporal context: Mau et al. 2018, "The same hippocampal CA1 population simultaneously codes temporal information over multiple timescales", Curr Biol 28:1499–1508 — the same population codes seconds, minutes and days; sequence membership changes gradually across days. [verified] — [PDF](https://www.bu.edu/hasselmo/MauSullivanKinskyHasselmoHowardEichenbaum2018.pdf)

### Inferences
- Candidate variables for sparse Penk+ events, ranked by how commonly RSC literature reports them and whether they are absent from the current GLM: (1) maze junctions / turn decisions / route segments (Alexander & Nitz 2015); (2) egocentric geometry — walls, corners, dead-end ends (egocentric boundary/vertex cells); (3) landmark/visual-cue onsets — here, light-transition moments (Fischer 2020); (4) episode/epoch identity — time-in-session or time-in-light-epoch (Mau 2018 analogue). Speculative.
- Long (~3 s) events match the duration of behavioural episodes such as traversing a maze arm, dwelling at a dead end, or a pause at a junction. That makes "sustained firing for the duration of a behavioural episode" a reasonable hypothesis to test with peri-event alignment. Speculative.
- Because the project's single plane mixes soma and dendrite ROIs and Voigts & Harnett show dendritic signals differ from somatic ones, a soma/dendrite misclassification difference between populations could contribute. Check that the event-statistics difference holds in soma-classified ROIs only.

### Gaps
- No source found specifically on Penk+ RSP neurons' in-vivo coding.
- No RSP study found reporting calcium event duration per se as a cell-type marker.
- Mao 2020 and Vedder 2017 content not re-verified this session.

---

## 3. Are sparse, low-Fano, bursty neurons more selective (conjunctive, few fields) or more contextual (epoch identity, slow drift)? Do sparse cells drift more?

### Takeaway
The best recent evidence (Climer et al. 2025, Nature) is that less excitable place cells drift more across days; excitability, not environment or behaviour, best predicted drift. Excitability fluctuations themselves can drive ensemble drift (Delamare et al.). So a low-rate population is, if anything, expected to be less stable and more "contextual" over long timescales. Within a single session, sparse coding in RSC/CA1 is usually read as high selectivity (narrow fields, orthogonal codes), but sparse cells give few events, so selectivity estimates from them are noisy.

### Cited Findings
- Climer, Davoudi, Oh & Dombeck 2025, "Hippocampal representations drift in stable multisensory environments", Nature 645:457–465, doi:10.1038/s41586-025-09245-y. Drift persists in highly reproducible VR; sensory environment and behaviour differences did not detectably change drift rate; excitability of individual place cells was the best predictor of subsequent drift, with more excitable cells drifting less. [verified via search summary] — [Nature](https://www.nature.com/articles/s41586-025-09245-y)
- Delamare et al. 2024, "Drift of neural ensembles driven by slow fluctuations of intrinsic excitability", eLife RP88053. Model + data: gradual excitability changes drive drift in cell rates; ensemble rate correlations decline over time even during spontaneous activity. [verified via search summary] — [eLife reviewed preprint](https://elifesciences.org/reviewed-preprints/88053)
- Ziv et al. 2013, "Long-term dynamics of CA1 hippocampal place codes", Nat Neurosci 16:264–266, doi:10.1038/nn.3329. Place-cell ensemble membership turns over across days while place-field locations of persisting cells are stable. [prior knowledge] — [doi](https://doi.org/10.1038/nn.3329)
- Rubin et al. 2015, "Hippocampal ensemble dynamics timestamp events in long-term memory", eLife 4:e12247. Ensembles evolve independently of environment, giving each episode a unique timestamp. [verified] — [eLife](https://elifesciences.org/articles/12247)
- Driscoll et al. 2017, "Dynamic reorganization of neuronal activity patterns in parietal cortex", Cell 170:986–999, doi:10.1016/j.cell.2017.07.021. PPC task-related sequences reorganise across days despite stable behaviour. [prior knowledge] — [doi](https://doi.org/10.1016/j.cell.2017.07.021)
- Rule, O'Leary & Harvey 2019, "Causes and consequences of representational drift", Curr Opin Neurobiol 58:141–147, doi:10.1016/j.conb.2019.08.005; and Rule et al. 2020 "Stable task information from an unstable neural population" (eLife) — population-level readout can stay stable while single cells drift. [Rule 2020 verified as listed; Rule 2019 prior knowledge] — [Rule 2020 PDF](https://harveylab.hms.harvard.edu/pdf/rule2020.pdf)
- Mao et al. 2017 (above): sparse RSC code is narrow-field and orthogonal — the sparse-selective interpretation. [verified] — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5557927/)
- Contrast for the HD system: "Months-long stability of the head-direction system", Nature 2025 (s41586-025-10096-w) — appeared in results; not read. If confirmed, HD cells are a low-drift reference against which non-HD sparse cells could be compared. — [Nature](https://www.nature.com/articles/s41586-025-10096-w)

### Inferences
- Low Fano factor plus skewed/bursty rate distributions is an unusual combination: low Fano (computed on binned counts) can arise simply from very low counts (Fano approaches 1 or below for sparse Poisson-like or refractory processes at low means), so it should not be read as "reliable" firing without a rate-matched comparison. Speculative / methodological.
- Within one 1-h session, slow drift is measurable as time-in-session decodability. If Penk+ cells are "contextual/drifting", epoch identity (early vs late session, light-epoch index) should be decodable from them better than from the comparison population, after controlling for bleaching. Speculative.
- Since only ~1 session per animal per condition seems available, across-day drift cannot be tested directly; within-session drift is the accessible proxy.

### Gaps
- No study found that directly compares drift rates between molecularly defined cortical excitatory subtypes.
- No source found on Fano factor interpretation for calcium-inferred rates at <0.2 spikes/s.

---

## 4. Indicator kinetics (GCaMP7f) and CASCADE: how much of "long, small, slow-decaying events" could be artefact vs sustained firing?

### Takeaway
jGCaMP7f's single-spike half-decay in vivo is ~0.27 s and 10-AP half-decay ~0.5 s, so 2.6–3.3 s events must reflect multiple spikes spread over time, slow indicator clearance, or both. Calcium imaging systematically sparsifies responses and supralinearly amplifies bursts, and spike-inference results depend on the algorithm and on assumptions about kinetics. A between-population difference in event duration and amplitude is a known way that expression level / cell-type calcium handling shows up; it should be ruled out before interpreting it as a coding difference.

### Cited Findings
- Dana et al. 2019, "High-performance calcium sensors for imaging activity in neuronal populations and microcompartments", Nat Methods 16:649–657, doi:10.1038/s41592-019-0435-6. jGCaMP7f: half-rise 75 ms and half-decay 520 ms for 10 APs; 1-AP half-decay ~265 ms in culture and ~270 ms in vivo. [verified via search snippet] — [ResearchGate record](https://www.researchgate.net/publication/333836480_High-performance_calcium_sensors_for_imaging_activity_in_neuronal_populations_and_microcompartments)
- Huang et al. 2021, "Relationship between simultaneously recorded spiking activity and fluorescence signal in GCaMP6 transgenic mice", eLife 10:e51675. Spike-to-fluorescence transform is nonlinear and low-pass; bursts yield events many times larger than isolated spikes; GCaMP6f has lower single-spike detectability than 6s. [verified] — [eLife PDF](https://elifesciences.org/articles/51675.pdf)
- Siegle et al. 2021, "Reconciling functional differences in populations of neurons recorded with two-photon imaging and electrophysiology", eLife 10:e69068. Ephys shows more responsive neurons; imaging shows responsive neurons as more selective. Calcium-indicator dynamics sparsify responses and supralinearly amplify bursts; a spikes-to-calcium forward model reconciled the modalities only for neurons above a minimum event rate. [verified] — [eLife](https://elifesciences.org/articles/69068)
- Wei et al. 2020, "A comparison of neuronal population dynamics measured with calcium imaging and electrophysiology", PLoS Comput Biol 16:e1008198. Choice of spike-inference algorithm changed the inferred fractions of neuron response categories (e.g. monophasic vs multiphasic), i.e. inference alters scientific conclusions about population dynamics. [verified] — [PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008198)
- Rupprecht et al. 2021, "A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging", Nat Neurosci 24:1324–1337, doi:10.1038/s41593-021-00895-5. CASCADE: supervised deep nets trained on ground truth (298 neurons, >35 h, zebrafish + mouse), resampled to the user's frame rate and noise level; outputs absolute spike rates; outperforms model-based methods on unseen data. [verified] — [Nature Neuroscience](https://www.nature.com/articles/s41593-021-00895-5)
- A 2026 Nat Methods paper, "Spike inference from calcium imaging data acquired with GCaMP8 indicators", extends calibrated spike inference to GCaMP8 (indicates indicator-specific training matters). Not read. — [Nature Methods](https://www.nature.com/articles/s41592-026-03183-x)

### Inferences
- CASCADE is trained to map fluorescence shape to spikes. If Penk+ cells have slower decay because of expression/handling (not firing), CASCADE trained on a different kinetic regime may misestimate rates — possibly spreading spikes across the long tail (inflating) or missing low-amplitude events (deflating). The direction is not predictable without ground truth. Inference.
- The two populations are labelled with different viral strategies (Cre-ON ADD3 in Penk-Cre vs Cre-OFF virus 344). Different constructs/expression levels can change baseline F, nuclear filling and effective decay. Recommended checks (not from a source): compare baseline F, fraction of nuclear-filled ROIs, and the decay time constant of the smallest isolated events per population. If isolated-event decay differs, the duration difference is at least partly indicator/expression, not firing.
- Smaller + longer events are the opposite of the burst signature (bursts → larger events per Huang/Siegle). This argues against "Penk+ cells fire in bursts" from event amplitude alone. The "bursty" description from rate distributions (skewness) may reflect clustering of rare events in time rather than intra-event spike bursts.
- Because imaging over-represents bursts and under-represents isolated spikes (Siegle 2021), a lower-rate, smaller-event population may have more undetected single spikes; the true rate difference may be smaller than CASCADE suggests.

### Gaps
- No ground-truth (simultaneous ephys + imaging) data found for jGCaMP7f in RSP or for Penk+ cortical neurons.
- No source found quantifying expression-level effects on GCaMP decay in vivo by cell type; this needs local checks.

---

## 5. Analysis approaches for sparse, low-rate populations (~20 cells/session)

### Takeaway
With few events per cell, marginal tuning curves and GLMs are underpowered; event-centred methods (event-triggered behaviour averages, peri-event alignment to discrete maze/light events, information per event with shuffle controls) and population/sequence discovery (rastermap, assembly detection, epoch decoding) are better matched to the data.

### Cited Findings
- Rastermap (Stringer et al. 2024, "Rastermap: a discovery method for neural population recordings", Nat Neurosci, doi:10.1038/s41593-024-01783-4) sorts neurons along one axis by activity similarity, finds sequences and clusters in single-trial data without trial averaging or behavioural labels. Designed for hundreds to hundreds of thousands of neurons. [verified] — [Nature Neuroscience](https://www.nature.com/articles/s41593-024-01783-4); [GitHub](https://github.com/MouseLand/rastermap)
- Mau et al. 2018 used Bayesian decoding of time at seconds, minutes and days scales from CA1 calcium data — a template for epoch-identity decoding. [verified] — [PDF](https://www.bu.edu/hasselmo/MauSullivanKinskyHasselmoHowardEichenbaum2018.pdf)
- Rubin et al. 2015 used ensemble-similarity over time to show unique per-episode "timestamps" — template for within-session slow-drift analysis. [verified] — [eLife](https://elifesciences.org/articles/12247)
- Fischer et al. 2020 aligned RSC activity to landmark and reward positions to identify landmark-anchored cells — template for peri-event alignment to discrete cues. [verified] — [eLife](https://elifesciences.org/articles/51458)
- Siegle et al. 2021: imaging-based selectivity is inflated for sparse cells; comparisons should use forward models or rate-matched controls. [verified] — [eLife](https://elifesciences.org/articles/69068)

### Inferences (methods recommendations; not sourced as a package)
- Event-triggered averages (ETA) of behaviour: for each Penk+ event onset, average HD, AHV, speed, maze-graph node, distance-to-junction, time-since-light-transition in a ±5 s window; compare to circularly shifted event times (keeps event count and ISI structure). Works with few events and makes no linear-encoding assumption.
- Peri-event alignment the other way round: align dF/F / event probability to discrete behavioural events — junction entry, turn choice, dead-end arrival, reversal, light on/off, stop/start of locomotion. Test per cell with circular-shift shuffles; summarise per population with Mann-Whitney / Wilcoxon (non-parametric per project rules).
- Information per event (Skaggs bits/event) with shuffle debiasing; report the shuffle-corrected value, since raw bits/event is upward-biased for low event counts.
- Rate-matching: subsample events in the comparison population to the Penk+ event count before comparing selectivity or information, to remove count-driven bias.
- Epoch-identity decoding: decode light-epoch index, time-in-session tertile, or light vs dark from population vectors (cross-validated, chance by label shuffle). A "contextual" population should decode epoch/time better than position/HD.
- Assembly detection: co-activation of events within ~1–2 s windows across cells versus shuffles (e.g. PCA/ICA-based assemblies). With ~20 cells, report count of significant pairs/assemblies rather than fitting complex models.
- Rastermap on each session to see whether Penk+ events form sequences tied to maze traversals.
- Before any of the above, run the indicator checks from section 4 (isolated-event decay, baseline F, nuclear filling by population) and restrict to soma-classified ROIs.

### Gaps
- No published benchmark found for the minimum event count needed for reliable shuffle-corrected selectivity at ~10 Hz imaging; thresholds would need to be set empirically (e.g. by subsampling the comparison population).
- No source found applying these methods to a molecularly defined RSP subtype.
