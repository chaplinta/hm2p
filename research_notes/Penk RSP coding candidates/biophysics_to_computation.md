# Biophysics to computation: excitatory neurons with high Rin, small dendrites, narrow spikes, high max firing, low rheobase, hyperpolarised rest, prominent sag, sparse in vivo firing

Method note: WebSearch worked throughout; WebFetch failed on every attempt (connection refused / no output), so full texts could not be read. Findings below come from search-result abstracts/snippets of primary sources. DOIs marked "(DOI from prior knowledge, not re-verified this session)" should be checked before citing in a manuscript. Species are noted where the source made them clear.

## Q1. How do high input resistance + small dendritic tree + hyperpolarised rest combine, and what firing regime and selectivity does this produce in vivo?

### Takeaway
The closest match to the queried profile is the low-rheobase (LR) / late-spiking L2/3 pyramidal cell of granular retrosplenial cortex (RSG): small, high-Rin, low-capacitance, narrow-ish spikes, non-adapting, strongly driven by anterior thalamic (HD) input but largely not by distant cortico-cortical input. Across brain areas, cells with "small and excitable per input, but few/selective inputs" (piriform semilunar cells, dentate granule cells, superficial CA1) tend to be dominated by a single feedforward afferent stream and fire sparsely in vivo; the more broadly connected, more integrative neighbours (piriform pyramidal cells, deep CA1) fire more and are more often spatially tuned. High per-input excitability does not imply high in vivo activity: sparse firing arises from few/selective inputs, hyperpolarised rest and strong inhibition.

### Cited Findings
**RSC LR cells (mouse) – the reference case**
- RSG layer 3 (L2/3) contains two principal pyramidal classes, low-rheobase (LR) and regular-spiking (RS). LR cells are defined by low rheobase, high input resistance, lack of spike-frequency adaptation, and narrower spikes than RS cells — [Brennan et al. 2020, Cell Reports, "Hyperexcitable Neurons Enable Precise and Persistent Information Encoding in the Superficial Retrosplenial Cortex", doi:10.1016/j.celrep.2019.12.093](https://www.cell.com/cell-reports/fulltext/S2211-1247(19)31758-9); preprint [bioRxiv 673954](https://www.biorxiv.org/content/10.1101/673954v1.full)
- Quantitative values (mouse, slice): LR input resistance 402.69 ± 16.75 MΩ, input capacitance 38.42 ± 1.32 pF, rheobase 91.79 ± 12.89 pA; spike width LR 0.55 ± 0.02 ms vs RS 0.86 ± 0.05 ms vs FS interneurons 0.22 ± 0.05 ms — [Brennan et al. bioRxiv 673954](https://www.biorxiv.org/content/10.1101/673954v1.full)
- Anterior thalamus (a head-direction source) strongly recruits these small LR pyramidal cells in RSG L3 — [Brennan et al. 2020 / bioRxiv](https://www.biorxiv.org/content/10.1101/673954v1.full)
- The authors describe LR cells as able to fire at high rates for extended periods ("persistent and fast") — [bioRxiv 673954](https://www.biorxiv.org/content/10.1101/673954v1.full)
- Transcriptomic identity: in RSG, L2/3 excitatory cells split into Cxcl14+ and Calb1+ clusters; L2/3 Cxcl14 cells correspond to physiological LR cells, L2/3 Calb1 cells to RS cells — [Sullivan et al. 2023, Cell Reports 42(3), "Sharp cell-type-identity changes differentiate the retrosplenial cortex from the neocortex"](https://www.researchgate.net/publication/369045169_Sharp_cell-type-identity_changes_differentiate_the_retrosplenial_cortex_from_the_neocortex). These types are reported to be preserved between mouse and rat despite marker-gene changes — [Unique transcriptomic cell types of the granular RSC preserved across mice and rats (PMC12660178)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12660178/)

**RSC late-spiking cells (rat, mouse)**
- Rat RSG L2 and adjoining L3 small pyramidal neurons are late-spiking; they have higher input resistance with similar membrane time constant, implying lower capacitance / smaller size; the late-spiking delay is due to delayed-rectifier and A-type K+ channels (Kv1.1, Kv1.4, Kv4.3) — [Kurotani et al. 2013, "Pyramidal neurons in the superficial layers of rat retrosplenial cortex exhibit a late-spiking firing property" (PMC3535347)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3535347/)
- In ventral RSG, L2/3 is almost exclusively late-spiking (94% in rat, Kurotani et al. 2013, as cited by) — [Robles et al. 2020, Front Neural Circuits, doi:10.3389/fncir.2020.576504](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7669619/)
- Mouse RSG late-spiking L2/3 neurons do not receive direct excitatory contacts from contralateral homotopic RSC or ipsilateral dysgranular RSC; they receive only weak disynaptic responses of local origin — [Robles et al. 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7669619/). This is direct evidence for "few, selective inputs" in these cells.
- Thalamus and claustrum control parallel L1 circuits in RSC; LR/late-spiking cells are part of this layer-specific input segregation — [Brennan et al. 2021, eLife 62207, "Thalamus and claustrum control parallel layer 1 circuits in retrosplenial cortex"](https://elifesciences.org/articles/62207)
- Late-spiking RSC neurons are reported not to be synchronised with neocortical slow waves in anaesthetised mice — [ResearchGate listing, "Late-spiking retrosplenial cortical neurons are not synchronized with neocortical slow waves in anesthetized mice"](https://www.researchgate.net/publication/377388916_Late-spiking_retrosplenial_cortical_neurons_are_not_synchronized_with_neocortical_slow_waves_in_anesthetized_mice) (abstract only seen; authors/journal not verified)

**Piriform semilunar vs pyramidal (rodent)**
- Semilunar cells (L2a) have higher input resistance and shorter membrane time constants than L2b superficial pyramidal cells, receive stronger olfactory bulb (afferent) excitation but weaker associational input; they lack basal dendrites and their dendrites thin rapidly (<1 µm beyond ~150 µm vs pyramidal >1 µm to ~400 µm) — [J Neurosci, "Properties of Piriform Cortex Pyramidal Cell Dendrites" 29(40):12641](https://www.jneurosci.org/content/29/40/12641); [Nagappan & Franks 2021-era eLife 73668, "Parallel processing by distinct classes of principal neurons in the olfactory cortex"](https://elifesciences.org/articles/73668)
- Proposed division of labour: semilunar cells integrate afferent bulb input; pyramidal cells receive, transform and relay semilunar input downstream — "parallel channels" — [eLife 73668](https://elifesciences.org/articles/73668); [Duke thesis, "Odor coding by distinct classes of principal neurons in piriform cortex"](https://dukespace.lib.duke.edu/items/d38b3929-7636-41a7-b388-eadc64229cac)

**MEC layer 2 stellate vs pyramidal (rodent)**
- Stellate cells (reelin+, calbindin−) have large sag and superficially branching dendrites; pyramidal cells (calbindin+, Wfs1+) have thick apical dendrites and strong septal cholinergic input — [eLife 36664, "Functional properties of stellate cells in MEC layer II"](https://elifesciences.org/articles/36664)
- Conflicting reports on which class carries grid signals: Tang et al. 2014 (Neuron) found pyramidal cells more often grid cells (only 3/94 putative stellates were grid cells); other groups found grid cells in both classes — [J Neurosci 36(7):2283, "Cell-type specific phase precession in layer II of the MEC"](https://www.jneurosci.org/content/36/7/2283); [Tang et al. 2014 Neuron (ResearchGate)](https://www.researchgate.net/publication/269281645_Pyramidal_and_Stellate_Cell_Specificity_of_Grid_and_Border_Representations_in_Layer_2_of_Medial_Entorhinal_Cortex)
- Calbindin pyramidal cells form hexagonally arranged patches; stellates fill inter-patch space — [J Neurophysiol, "Structural modularity and grid activity in the MEC"](https://journals.physiology.org/doi/full/10.1152/jn.00574.2017)

**Dentate granule cells (rodent)**
- Granule cells fire ultra-sparsely in both electrophysiology and calcium imaging, consistent with pattern separation; strong feedforward/feedback/lateral inhibition promotes sparse, competitive activation — [Hippocampus 2024, "Granule cells perform frequency-dependent pattern separation..."](https://onlinelibrary.wiley.com/doi/full/10.1002/hipo.23585); [Hippocampus, "Dendrites of dentate gyrus granule cells contribute to pattern separation by controlling sparsity"](https://onlinelibrary.wiley.com/doi/abs/10.1002/hipo.22675)
- Mature granule cell Rin ~100–150 MΩ; young adult-born cells higher (e.g. 171 ± 16 MΩ in mice) — [same Hippocampus 2024 model paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hipo.23585). Note: mature GCs are therefore NOT especially high-Rin; their sparseness is mainly due to hyperpolarised rest and inhibition. Kv4.1 is reported as a key channel for low-frequency GC firing and necessary for pattern separation — [bioRxiv 670018](https://www.biorxiv.org/content/10.1101/670018.full.pdf)

**Superficial vs deep CA1 (rat)**
- Deep CA1 pyramidal cells fire at higher rates, burst more, are more likely to have place fields, and are more modulated by sleep slow oscillations than superficial cells; place-cell propensity increases superficial → deep; deep cells shift preferred theta phase in REM — [Mizuseki et al. 2011, Nat Neurosci, "Hippocampal CA1 pyramidal cells form functionally distinct sublayers", doi:10.1038/nn.2894](https://www.nature.com/articles/nn.2894)
- Deep CA1 place cells are more strongly tied to landmarks than superficial ones — [Geiller et al. 2017, Nat Commun, ncomms14531](https://www.nature.com/articles/ncomms14531); deep and superficial subcircuits support different coding regimes across environments — [Sharif et al. 2021, Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(20)30858-8). Calbindin+ (superficial) CA1 cells are reported to encode spatial information more efficiently — [eNeuro 2023, ENEURO.0411-22.2023](https://www.eneuro.org/content/10/3/ENEURO.0411-22.2023)

**Sparse L2/3 neocortex**
- Review of experimental evidence for sparse firing in neocortex, with L2/3 as the clearest case — Barth & Poulet 2012, Trends Neurosci 35:345–355, "Experimental evidence for sparse firing in the neocortex" (bibliographic details confirmed only via citing papers in search results; no direct URL retrieved; DOI 10.1016/j.tins.2012.03.008 from prior knowledge, not re-verified)
- In awake mouse vibrissal cortex, sparse L2/3 activity is explained largely by diverse/narrow stimulus selectivity plus brain state, not merely by low excitability — [Ranjbar-Slamloo & Arabzadeh 2019, J Physiol, "Diverse tuning underlies sparse activity in layer 2/3 vibrissal cortex of awake mice"](https://physoc.onlinelibrary.wiley.com/doi/abs/10.1113/JP277506); [PubMed 31243764](https://pubmed.ncbi.nlm.nih.gov/31243764/)
- Differential wiring of L2/3 neurons drives sparse and reliable firing during development — [Cereb Cortex 23(11):2690](https://academic.oup.com/cercor/article/23/11/2690/302895)

### Inferences
- The biophysical profile (high Rin, low C, low rheobase) means each input produces a large voltage change, so a cell needs relatively few coincident inputs to fire. If anatomy restricts it to one dominant afferent stream (thalamic HD input for RSG LR cells; bulb input for semilunar cells; perforant path for GCs), the expected in vivo regime is low background rate with strong, reliable, input-locked responses: a sparse, selective "relay/feature-copy" code rather than a broadly integrating mixed code. A hyperpolarised rest adds a threshold-like gap that suppresses weak/background input — sharpening selectivity.
- Pattern across systems: the smaller, higher-Rin, afferent-dominated class (semilunar, superficial CA1, possibly MEC stellate/GC) tends to be sparser and less often "classically tuned" than the larger integrating neighbour (piriform pyramidal, deep CA1, MEC calbindin pyramidal). If Penk+ RSP cells sit in the small/high-Rin class, a lower fraction of classically HD-tuned cells and sparser events would be consistent with this pattern, not evidence of weaker coding per se. This is a speculative mapping; Penk identity relative to Cxcl14/LR vs Calb1/RS is not established in the sources found.
- Caution: the RSC LR cell is described as persistently and rapidly firing in vitro, while the query profile includes sparse in vivo firing. These are compatible (high capability, low drive), but there are no in vivo identified LR-cell recordings in the sources found.

### Gaps
- No source found giving in vivo firing rates or HD tuning of identified RSG LR/Cxcl14 cells. Brennan's AHV role is model-based.
- No source found linking Penk expression to LR/Cxcl14 or Calb1/RS RSG types (worth checking Allen/Yao et al. transcriptomic atlases directly).
- Could not retrieve sag/Vrest values for LR vs RS cells (full text inaccessible this session).
- Layer 2 vs layer 3 differences specifically in agranular/dysgranular RSC not found.

## Q2. What does HCN/Ih (sag) do for synaptic integration and temporal coding? Somatic vs dendritic Ih.

### Takeaway
Ih shortens EPSP time courses and narrows the temporal summation window, normalising summation across dendritic locations (Magee 1999). It thereby penalises asynchronous inputs more than synchronous ones (coincidence detection), produces theta-band resonance (with M-current), and sets Rin and resting potential. In small, compact neurons somatic Ih will dominate; dendritic gradients of Ih (CA1, L5) are a feature of large dendritic trees. Mouse L2/3 has relatively low HCN1 compared with human L2/3, so prominent sag in a mouse superficial excitatory cell is itself a distinguishing feature.

### Cited Findings
- Dendritic Ih in CA1 pyramidal neurons normalises temporal summation: deactivation of the non-uniform Ih counterbalances dendritic filtering and removes location dependence of temporal integration, which enhances synchronisation of populations — [Magee 1999, Nat Neurosci 2:508–514, "Dendritic Ih normalizes temporal summation in hippocampal CA1 neurons", doi:10.1038/9158 (DOI from prior knowledge)](https://www.nature.com/articles/nn0699_508)
- Modelling: dendritic Ih selectively blocks temporal summation of unsynchronised distal inputs while sparing synchronised inputs — [J Comput Neurosci, "Dendritic Ih selectively blocks temporal summation of unsynchronized distal inputs in CA1 pyramidal neurons"](https://link.springer.com/article/10.1023/B:JCNS.0000004837.81595.b0)
- HCN channels inhibit EPSPs partly by interacting with M-type K+ channels (the depolarised rest set by Ih increases M-current activation) — [George, Abbott & Siegelbaum 2009, Nat Neurosci, nn.2307](https://www.nature.com/articles/nn.2307)
- Two forms of theta-frequency (2–7 Hz) electrical resonance in rat CA1 pyramidal cells: M-resonance at depolarised and h-resonance at hyperpolarised potentials, amplified by persistent Na+ current — [Hu, Vervaeke & Storm 2002, J Physiol, doi:10.1113/jphysiol.2002.029249](https://physoc.onlinelibrary.wiley.com/doi/10.1113/jphysiol.2002.029249)
- Blocking HCN channels at hyperpolarised voltages turns the somatodendritic arbor into a simple low-pass filter, abolishing resonance and phase lead at all dendritic locations — [PMC4289900, "Dendritic atrophy constricts functional maps in resonance and impedance properties of hippocampal model neurons"](https://pmc.ncbi.nlm.nih.gov/articles/PMC4289900/) (building on Narayanan & Johnston 2007 Neuron)
- Ih contributes to spike-time precision and intrinsic resonance in cortical neurons in vitro — [PMC3171884, "The role of hyperpolarization-activated cationic current in spike-time precision and intrinsic resonance in cortical neurons in vitro"](https://pmc.ncbi.nlm.nih.gov/articles/PMC3171884/)
- Review: HCN1/HCN2 predominate in cortex, mostly in pyramidal dendrites (trafficked via TRIP8b); somatodendritic HCN shapes firing and synaptic integration through effects on membrane resistance and resting potential; also present in some axons/terminals — [Shah 2014, J Physiol 592:2711–2719, "Cortical HCN channels: function, trafficking and plasticity", doi:10.1113/jphysiol.2013.270058](https://physoc.onlinelibrary.wiley.com/doi/10.1113/jphysiol.2013.270058)
- Species difference: HCN1 is ubiquitous in human but not mouse L2/3; h-channels shape human supragranular pyramidal properties more than mouse; Ih speeds somatic EPSP kinetics and reduces temporal summation, making summation independent of dendritic input site — [Kalmbach et al. 2018, Neuron, "h-Channels contribute to divergent intrinsic membrane properties of supragranular pyramidal neurons in human versus mouse cerebral cortex"](https://providence.elsevierpure.com/en/publications/h-channels-contribute-to-divergent-intrinsic-membrane-properties-/); [bioRxiv 312298](https://www.biorxiv.org/content/10.1101/312298v1.full) (DOI 10.1016/j.neuron.2018.10.012 from prior knowledge)
- "Hidden" HCN channels in mouse L2/3 pyramidal neurons permit pathway-specific synaptic amplification — i.e. Ih effects can be input-pathway specific rather than global — [eLife reviewed preprint 96002](https://elifesciences.org/reviewed-preprints/96002v1)
- Ih is modulated by synaptic potentiation differently in excitatory vs inhibitory neurons (intrinsic plasticity of Ih) — [PMC11574699](https://pmc.ncbi.nlm.nih.gov/articles/PMC11574699)
- MEC L2 stellate cells — the canonical high-sag excitatory cell — show cell-type-specific phase precession relative to pyramidal cells — [J Neurosci 36(7):2283](https://www.jneurosci.org/content/36/7/2283)

### Inferences
- Combination of prominent sag with a hyperpolarised rest is mechanistically coherent: at hyperpolarised potentials more HCN is open, so Ih is maximally engaged at rest, shortening the membrane time constant and the EPSP integration window. Prediction: such a cell behaves more as a coincidence detector (responds to synchronous bursts of input) than an integrator, and may show h-resonance in the theta band.
- Ih also opposes hyperpolarisation (rebound): after inhibition or input withdrawal, rebound depolarisation could favour transient/onset responses — relevant to event/change detection. Speculative for RSC.
- Because high-Rin + Ih favour synchronous input, these cells are well placed to read out synchronous thalamic HD volleys (consistent with Brennan's thalamus → LR circuit), but this specific link was not shown in the sources found.

### Gaps
- No direct measurement of Ih/sag in identified RSG LR or Penk+ cells found.
- No in vivo test of Ih-dependent coincidence detection in RSC.

## Q3. Narrow spikes in excitatory cells: implications for Kv3/Kv1 channels, high-frequency firing, temporal precision, Ca2+ entry per spike

### Takeaway
Narrow spikes in excitatory cells usually indicate fast-activating high-threshold K+ channels (Kv3) and/or Kv1 contribution, which allow rapid repolarisation, Na+ channel recovery and high sustained firing rates, and improve spike-timing precision. RSG LR cells have spikes ~0.55 ms (intermediate between RS ~0.86 ms and FS 0.22 ms) and their late-spiking/low-adaptation profile depends on Kv1 and Kv4 channels. Shorter spikes generally reduce Ca2+ entry per spike, which would make calcium indicators less sensitive per spike in these cells.

### Cited Findings
- Kv3 channels have high-voltage, fast activation and deactivation, narrow the AP and allow Na+ channel recovery during brief ISIs, supporting firing >100 Hz; mis-tuned Kv3 levels alter temporal accuracy across a neural ensemble — [Kaczmarek & Zhang 2017, Physiol Rev, "Kv3 channels: enablers of rapid firing, neurotransmitter release, and neuronal endurance", doi:10.1152/physrev.00002.2017](https://journals.physiology.org/doi/full/10.1152/physrev.00002.2017)
- Gradients and modulation of K+ channels (Kv3) optimise temporal accuracy in auditory neuron networks — [PMC3305353](https://pmc.ncbi.nlm.nih.gov/articles/PMC3305353/); Kv3.1b modulation adjusts firing-pattern fidelity — [J Neurosci 23(4):1133](https://www.jneurosci.org/content/23/4/1133)
- Excitatory projection neurons can use Kv3 to generate very narrow spikes (songbird motor cortex analogue, homologous to mammalian L5 PT cells with high Kv3.1) — [PMC10241522](https://pmc.ncbi.nlm.nih.gov/articles/PMC10241522)
- Kv3.3 controls presynaptic AP waveform and transmitter release at an excitatory synapse — [eLife 75219](https://elifesciences.org/articles/75219)
- RSG late-spiking pyramidal neurons: late spiking from delayed-rectifier and A-type K+ (Kv1.1, Kv1.4, Kv4.3) — [Kurotani et al. 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3535347/)
- RSG LR spike width 0.55 ± 0.02 ms vs RS 0.86 ± 0.05 ms — [Brennan et al. bioRxiv](https://www.biorxiv.org/content/10.1101/673954v1.full)

### Inferences
- Narrow spikes + non-adapting firing + high max rate (LR cells) supports faithful rate tracking of fast-changing input (e.g. HD changing during rapid head turns) — this is the basis of Brennan's AHV model (Q4).
- Calcium imaging consequence (project-relevant): narrower spikes → less Ca2+ influx per spike, plus small soma volume and possibly different buffering. For GCaMP data, a cell class with narrow spikes may show smaller/briefer per-spike transients; observed "sparser, smaller events" could partly reflect lower detectability rather than lower firing. CASCADE models are calibrated mostly on broad-spiking pyramidal ground truth. This is an inference; no source directly measuring Ca2+ per spike vs spike width in RSC was found.

### Gaps
- No direct evidence on which Kv3 subunits (if any) RSG LR/Penk+ cells express; the RSC source attributes firing phenotype to Kv1/Kv4, not Kv3.
- No source found quantifying Ca2+ entry per spike vs AP width in cortical pyramids this session (Bean 2007 Nat Rev Neurosci and related work would be the place to look).

## Q4. Retrosplenial superficial cell types specifically: proposed computations

### Takeaway
The RSC literature proposes that small, high-Rin, low-rheobase, non-adapting L2/3 cells (LR / late-spiking / Cxcl14+) receive dense anterior thalamic HD input and, via depressing thalamic synapses plus lack of adaptation, compute angular head velocity (the time derivative of HD) — i.e. a HD-to-AHV converter ("compass to gyroscope"). Unlike other midline cortical principal cells, LR cells do not show cholinergic persistent firing, which Jedrasiak-Cape et al. 2025 argue is consistent with an AHV (transient) rather than a memory/persistent role. They are also anatomically isolated from long-range cortico-cortical input.

### Cited Findings
- Thalamic HD input + short-term synaptic depression + absent adaptation let LR neuron models rapidly and faithfully compute the derivative of HD = AHV; described as converting compass signals into a gyroscope-like signal — [Brennan et al. 2020, Cell Reports, doi:10.1016/j.celrep.2019.12.093](https://www.cell.com/cell-reports/fulltext/S2211-1247(19)31758-9); [Technology Networks summary](https://www.technologynetworks.com/neuroscience/news/unique-neuron-study-identifies-the-compass-of-the-brain-330322); [Futurity summary](https://www.futurity.org/retrosplenial-cortex-excitatory-neurons-direction-2275192-2/) (lay summaries; the modelling claim is from the paper)
- Original preprint framed LR cells as enabling "precise and persistent information transmission" — [bioRxiv 673954](https://www.biorxiv.org/content/10.1101/673954v1.full)
- LR neurons do NOT fire persistently in response to cholinergic agonists, in contrast to all other principal subtypes examined in RSG and midline cortex; modelling tested whether this absence of persistent activity supports AHV coding; published in Progress in Neurobiology 251:102804 (2025), authors include Jedrasiak-Cape, Rybicki-Kler, Brooks, Ghosh, Brennan, Ahmed (mouse) — [Jedrasiak-Cape et al. 2025, PubMed 38895393](https://pubmed.ncbi.nlm.nih.gov/38895393); [PMC12340933](https://pmc.ncbi.nlm.nih.gov/articles/PMC12340933/); [bioRxiv 2024.06.04.597341](https://www.biorxiv.org/content/10.1101/2024.06.04.597341v1)
- RSC receives cholinergic input in both superficial and deep layers — [Jedrasiak-Cape et al. 2025](https://pubmed.ncbi.nlm.nih.gov/38895393)
- Separate L1 circuits: thalamus and claustrum target parallel L1 circuits in RSC — [Brennan et al. 2021, eLife 62207](https://elifesciences.org/articles/62207)
- Late-spiking L2/3 RSG cells lack direct excitatory input from contralateral RSC and dysgranular RSC — [Robles et al. 2020, Front Neural Circuits](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7669619/)
- L2/3 Cxcl14 = LR, L2/3 Calb1 = RS — [Sullivan et al. 2023, Cell Reports](https://www.researchgate.net/publication/369045169_Sharp_cell-type-identity_changes_differentiate_the_retrosplenial_cortex_from_the_neocortex)
- Broader reviews of RSC computation — [Alexander et al. 2023, Neuron, "Rethinking retrosplenial cortex: perspectives and predictions"](https://www.cell.com/neuron/fulltext/S0896-6273(22)01027-3); [Trends Neurosci 2022, "Mechanistic flexibility of the retrosplenial cortex enables its contribution to spatial cognition"](https://www.sciencedirect.com/science/article/pii/S0166223622000194)
- Rat RSC heterogeneity of firing type and morphology — [Heterogeneity of neuronal firing type and morphology in RSC of male F344 rats (ResearchGate)](https://www.researchgate.net/publication/340522924_Heterogeneity_of_Neuronal_Firing_Type_and_Morphology_in_Retrosplenial_Cortex_of_Male_F344_Rats)

### Inferences
- Testable predictions for a cell class matching this profile in the project's 2P data: (1) more AHV-tuned (or HD×AHV conjunctive, or turn-onset) than pure HD-tuned cells; (2) transient responses at head-turn onsets rather than sustained HD-locked activity; (3) little light/dark difference if driven by thalamic (vestibular-derived) input; (4) at ~9.6 Hz imaging, derivative coding will be partly smeared by GCaMP kinetics, so AHV tuning may be underestimated.
- The "no cholinergic persistent firing" result argues against these cells supporting working-memory-like persistent HD representation; the RS/Calb1 neighbours are the more likely persistent-activity substrate.
- Whether Penk+ RSP neurons overlap with LR/Cxcl14 is unknown; Penk is not named in any source found here. Strong caveat.

### Gaps
- No in vivo recording of identified LR cells confirming AHV tuning — the AHV claim is from slice physiology + modelling.
- No source on Penk expression in RSG L2/3 types.
- Could not retrieve exact sag ratios or Vrest for LR cells.

## Q5. Computational/modelling work linking excitability heterogeneity to coding (sparse coding theory, heterogeneity, population codes)

### Takeaway
Sparse coding theory holds that representing inputs with strong activation of few neurons improves energy efficiency, storage capacity in associative memories and downstream readability. Intrinsic biophysical heterogeneity decorrelates neurons and increases population information (about twofold in mitral cells), with an intermediate level of diversity being optimal. A mixed population of low-threshold, sparse/selective cells and more broadly tuned cells is therefore expected to improve population coding rather than being noise.

### Cited Findings
- Sparse coding: sensory events represented by strong activation of relatively small neuron groups; sparse codes increase energy efficiency, make structure explicit, ease downstream readout and increase associative-memory capacity; lateral inhibition maintains sparseness — [Olshausen & Field 2004, Curr Opin Neurobiol 14:481–487, "Sparse coding of sensory inputs", doi:10.1016/j.conb.2004.07.007 (DOI from prior knowledge)](https://www.researchgate.net/publication/8391099_Sparse_coding_of_sensory_inputs); [Scholarpedia: Sparse coding](http://www.scholarpedia.org/article/Sparse_coding)
- Theoretical treatment of sparse and silent coding in neural circuits — [Spanne & Jörntell, arXiv 1010.4138](https://arxiv.org/pdf/1010.4138)
- Intrinsic diversity in mouse olfactory bulb mitral cells decorrelates firing and lets diverse populations encode about twofold more information than homogeneous ones — [Padmanabhan & Urban 2010, Nat Neurosci 13:1276–1282, doi:10.1038/nn.2630](https://www.nature.com/articles/nn.2630)
- Intermediate (not maximal) intrinsic diversity maximises population coding — [Tripathy et al. 2013, PNAS 110:8248, "Intermediate intrinsic diversity enhances neural population coding"](https://pnas.org/content/110/20/8248)
- Heterogeneity of intrinsic properties among cochlear nucleus neurons improves population coding of temporal information — [J Neurophysiol, jn.00836.2013](https://journals.physiology.org/doi/full/10.1152/jn.00836.2013)
- Dentate GC dendrites contribute to pattern separation by controlling sparsity (model) — [Hippocampus, hipo.22675](https://onlinelibrary.wiley.com/doi/abs/10.1002/hipo.22675)
- Energy-efficient coding dynamics in visual cortex — [J Neurophysiol 2025, jn.00078.2025](https://journals.physiology.org/doi/full/10.1152/jn.00078.2025)

### Inferences
- For a two-population comparison (Penk+ vs Penk⁻CamKII+), the theory predicts that a sparse, high-gain class could contribute disproportionately to decorrelated, information-rich codes per spike even if fewer cells pass single-cell tuning thresholds; single-cell MVL/Skaggs thresholds may systematically under-rate such cells. Population decoding with per-class subsampling would be a fairer comparison.

### Gaps
- No modelling work found specifically on small-dendrite/high-Rin excitatory classes within a cortical microcircuit and their contribution to HD population codes.

## Q6. Are more excitable neurons preferentially recruited into engrams, novelty or context representations?

### Takeaway
Yes, strong causal evidence (mainly lateral amygdala and hippocampus, mouse): neurons with higher relative excitability at the time of learning are preferentially allocated to the memory trace; raising excitability (CREB, other manipulations) biases recruitment and enhances memory, and blocking the excitability increase (Kir2.1) prevents biased allocation. RSC itself holds a context engram that can be reactivated independently of hippocampus. Whether intrinsically high-excitability cell types (vs transiently excitable individual neurons) are preferentially allocated is less established.

### Cited Findings
- Neurons are recruited to a memory trace based on relative excitability immediately before training; multiple ways of increasing excitability bias recruitment and enhance memory; co-expressing Kir2.1 blocks CREB-driven allocation; activating the allocated neurons alone acts as a retrieval cue (mouse lateral amygdala, fear memory) — [Yiu et al. 2014, Neuron 83:722–735, doi:10.1016/j.neuron.2014.07.017 (DOI from prior knowledge)](https://pubmed.ncbi.nlm.nih.gov/25102562/)
- Reviews of memory allocation and CREB/excitability mechanisms — [Rogerson et al. 2014, Neuropsychopharmacology, "Memory allocation" (npp2014234)](https://www.nature.com/articles/npp2014234); [Josselyn & Frankland 2018, Annu Rev Neurosci, "Memory allocation: mechanisms and function"](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-080317-061956); [Neuropsychopharmacology npp201673, "Neuronal allocation to a hippocampal engram"](https://www.nature.com/articles/npp201673); [Josselyn & Tonegawa 2020, Science, "Memory engrams: recalling the past and imagining the future"](https://www.science.org/doi/10.1126/science.aaw4325)
- Time-dependent role for CREB in allocation shown with an optogenetic CREB tool — [Neuropsychopharmacology 2019, s41386-019-0588-0](https://www.nature.com/articles/s41386-019-0588-0)
- Excitability-based allocation and memory linking contribute to false-memory generation — [Neurobiol Learn Mem 2020](https://www.sciencedirect.com/science/article/abs/pii/S1074742720301283)
- RSC context engram: optogenetic reactivation of c-fos-tagged RSC neurons from contextual fear conditioning elicits context-specific fear and can bypass hippocampal inactivation — [Cowansage et al. 2014, Neuron 84:432–441, "Direct reactivation of a coherent neocortical memory of context"](https://www.researchgate.net/publication/266950362_Direct_Reactivation_of_a_Coherent_Neocortical_Memory_of_Context); repeated RSC ensemble reactivation induces systems consolidation — [de Sousa et al. 2019, PNAS, doi:10.1073/pnas.1818432116](https://www.pnas.org/doi/10.1073/pnas.1818432116)
- Spatial transcriptomic signature of RSC during memory consolidation at single-cell resolution — [Mol Psychiatry 2025, s41380-025-03331-3](https://www.nature.com/articles/s41380-025-03331-3)
- Deep CA1 (higher rate, more bursting) are more likely to be place cells; deep cells more landmark-bound — [Mizuseki et al. 2011](https://www.nature.com/articles/nn.2894); [Geiller et al. 2017](https://www.nature.com/articles/ncomms14531). Note this is the opposite direction from "small high-Rin = more recruited": the more active, more integrative class is more often tuned.

### Inferences
- Engram allocation work concerns relative, often transient, excitability among neurons of the same type; extrapolating to a stable intrinsically excitable cell class (e.g. LR/Penk+) is speculative. If it applies, a low-rheobase class would be expected to be over-represented in newly formed context/novelty ensembles — testable as greater novelty/first-epoch responses or faster representational change across sessions.
- The CA1 deep/superficial contrast warns that intrinsic excitability alone does not predict tuning: input connectivity matters at least as much.

### Gaps
- No study found testing whether RSC LR/late-spiking cells are preferentially c-fos/engram-tagged.
- No evidence found tying Penk+ cortical neurons to engram allocation or novelty responses.
- Burst coding: not covered by the sources retrieved (LR cells are described as non-adapting, regular high-rate firers rather than bursters; deep CA1 bursts more — Mizuseki 2011).
- Gain control via Ih/shunting in this cell class not directly addressed in retrieved sources.
