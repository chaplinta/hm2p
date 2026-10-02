# Penk-expressing excitatory neurons in cortex and hippocampus, and enkephalin/opioid circuit effects

Research notes (2026-10-01). Scope: mouse/rat primary literature. WebFetch failed for every full-text page this session (connection refused / no output), so findings below come from search-result abstracts and snippets of the cited primary pages. Claims that could not be checked against a primary source are in Gaps, not Findings. DOIs marked "(from citation record)" were taken from the journal URL or volume/e-locator rather than read off the paper.

---

## Q1. Transcriptomics: which excitatory types express Penk across cortex and hippocampus, and is Penk a marker of a specific L2/3 IT subtype in RSC?

### Takeaway
Penk is expressed in subsets of L2/3 IT glutamatergic neurons (and L6 IT) across isocortex, and in the whole-cortex taxonomy it names a supertype within the L2/3 IT entorhinal subclass ("Penk Ndst4"). Penk is also expressed in VIP interneurons. I found no source showing that Penk marks the RSG-specific L2/3 low-rheobase (Cxcl14+) type, so it is still open whether the Penk+ RSP population is that type or a more generic L2/3 IT population.

### Cited Findings
- Tasic et al. 2018. "Shared and distinct transcriptomic cell types across neocortical areas." Nature. doi:10.1038/s41586-018-0654-5. This is the VISp + ALM scRNA-seq taxonomy (~23,000 neurons) that later neuropeptide analyses build on. — [Tasic 2018, Nature](https://www.nature.com/articles/s41586-018-0654-5)
- Yao et al. 2021. "A taxonomy of transcriptomic cell types across the isocortex and hippocampal formation." Cell. doi:10.1016/j.cell.2021.04.021 (from citation record). The search summary of this taxonomy says a "Penk Ndst4" supertype is specific to L2/3 IT neurons of entorhinal cortex, and that the L2/3 IT entorhinal subclass contains the Fign and Ndst4 supertypes. This needs checking against the paper's supplementary tables before it is relied on. — [Yao 2021, Cell](https://www.sciencedirect.com/science/article/pii/S0092867421005018); [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2020.03.30.015214v1.full)
- Smith et al. 2019. "Single-cell transcriptomic evidence for dense intracortical neuropeptide networks." eLife 8:e47889. doi:10.7554/eLife.47889. Uses the Tasic 2018 data (22,439 neurons, VISp + ALM) to show that 18 neuropeptide-precursor and receptor genes are highly expressed in most cortical neurons, most neurons express several of them, and their expression differs between cell types. This supports the idea of cell-type-specific local peptide signalling. Penk and the opioid receptors (Oprd1, Oprm1) are among the NPP/NP-GPCR genes analysed. — [Smith 2019, eLife](https://elifesciences.org/articles/47889); [PubMed](https://pubmed.ncbi.nlm.nih.gov/31710287/)
- RSG-specific cell types: mouse granular RSC (RSG) contains hyperexcitable low-rheobase (LR) pyramidal neurons that are found only in L2/3 and express Cxcl14. Together, L2/3 LR and L5a RSG (Scnn1a+ in mouse) neurons make up more than 50% of RSG neurons in both mouse and rat. Scnn1a expression is completely absent in rat, so the cell types are conserved even though some marker genes are not. — [RSG cross-species types, J Neurosci 2025 45(48):e2246242025, doi:10.1523/JNEUROSCI.2246-24.2025 (from citation record)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12660178/); [bioRxiv 2024.09.17.613545](https://www.biorxiv.org/content/10.1101/2024.09.17.613545v1)
- L2/3 LR neurons are described as the dominant cell type in RSG L2/3, and they transmit information precisely and persistently. — [Brennan et al., bioRxiv 673954, "Uniquely excitable neurons enable precise and persistent information transmission through the retrosplenial cortex"](https://www.biorxiv.org/content/10.1101/673954.full.pdf)
- RSG L2/3 cell types also show cell-type-specific cholinergic control, with implications for angular-velocity coding. — [PubMed 38895393](https://pubmed.ncbi.nlm.nih.gov/38895393)
- Search snippets of the RSG cross-species paper did not mention Penk among its marker genes. Without full text, this is absence of evidence, not evidence of absence. — [J Neurosci 2025](https://www.jneurosci.org/content/45/48/e2246242025)

### Inferences
- Penk looks like a marker of some L2/3 IT neurons spread across cortical areas, with an especially clear named supertype in entorhinal L2/3. RSC sits between visual/parietal isocortex and the parahippocampal (entorhinal) territory, so Penk+ L2/3 cells in RSP could plausibly be either (i) a subset of generic isocortical L2/3 IT or (ii) related to the RSG-specific L2/3 LR type. Which one is not established.
- If Penk+ RSP cells are mainly the RSG L2/3 LR type, they would be very excitable, low-rheobase neurons. That fits a population that could reach the high firing rates needed to release peptide (see Q4). This is speculation.

### Gaps
- Could not confirm Penk expression levels per type in the Yao 2021 / Yao 2023 (Nature, whole-brain atlas) / Allen ABC atlas tables for RSP specifically. Full text and atlas pages were unreachable. Recommended direct check: query the ABC Atlas (knowledge.brain-map.org/abcatlas) for Penk in "L2/3 IT RSP" and "L2/3 IT CTX" supertypes, and check whether Penk co-expresses with Cxcl14 in RSP.
- Sullivan et al. 2023 Cell Rep (RSC types) was not found in my searches. Not verified.
- Co-markers of Penk+ L2/3 IT (other than Ndst4/Fign in entorhinal cortex) were not established.

---

## Q2. Penk-Cre / Penk-IRES2-Cre: what does it label in cortex and hippocampus, and is there functional imaging or ephys of Penk-Cre cortical neurons?

### Takeaway
Penk-IRES2-Cre (JAX 025112) labels L2/3 and L6 IT glutamatergic neurons plus several VIP interneuron types in cortex, and restricted populations in the hippocampal formation. Its main functional use in cortex so far is as an L2/3 excitatory driver for connectivity mapping (Hage et al. 2022). I found no published in vivo functional imaging of Penk-Cre RSC neurons.

### Cited Findings
- Penk-IRES2-Cre-neo: reporter expression is enriched in cortical layers 2 and 6, and in restricted populations in olfactory areas, hippocampal formation, striatum, pallidum and hypothalamus. — [Allen Mouse Connectivity transgenic characterization](https://connectivity.brain-map.org/transgenic); [JAX 025112](https://www.jax.org/strain/25112)
- Transcriptomic profiling of reporter-labelled dissociated cells shows that Penk-Cre labels L2/3 IT neurons, L6 IT neurons and several types of VIP interneurons. — [Hage et al. 2022, eLife, "Synaptic connectivity to L2/3 of primary visual cortex measured by two-photon optogenetic stimulation," doi:10.7554/eLife.71103](https://elifesciences.org/articles/71103)
- Hage et al. 2022 used Penk-Cre with AAV-ChrimsonR to target L2/3 excitatory neurons in V1 for two-photon optogenetic connectivity mapping. Snippets did not give Penk-specific connectivity numbers. — [Hage 2022, eLife](https://elifesciences.org/articles/71103)
- The Allen local-circuit connectivity work ties connection properties to transcriptomic types, and Penk-Cre is one of the drivers used to reach L2/3 IT. — [Transcriptomic cell-type specificity of local cortical circuits, Neuron 2024](https://www.sciencedirect.com/science/article/pii/S0896627324006512)

### Inferences
- **This matters for the hm2p design.** Penk-Cre also labels VIP interneurons. A Cre-ON virus (ADD3) in Penk-Cre mice would therefore label any VIP+/Penk+ interneurons in the field unless the virus has an excitatory-specific promoter. The "Penk+" imaged population may then not be purely excitatory. Check the ADD3 construct's promoter, and/or look for VIP-like (small, non-pyramidal) ROIs in the Penk+ fields.
- The Penk⁻CamKII+ (Cre-OFF) population is defined by Cre absence. Any Penk+ neurons whose Cre expression is too low to block the Cre-OFF construct could leak into the "nonpenk" group.

### Gaps
- No in vivo calcium-imaging or ephys study of Penk-Cre neurons in RSC, ACC or V1 tuning was found.
- The fraction of Penk-Cre-labelled L2/3 cells that are excitatory vs VIP in RSP specifically is unknown.

---

## Q3. Hippocampus and entorhinal cortex: Penk+ excitatory populations and enkephalin function

### Takeaway
The best-established hippocampal case of excitatory enkephalin is co-release from glutamatergic pathways: the lateral perforant path and the dentate mossy fibres. There, endogenous opioids are needed for LTP induction, and they act by suppressing GABAergic inhibition. In CA2/3a, enkephalin from VIP interneurons drives delta-receptor-mediated iLTD at PV synapses and is required for social memory. Penk expression is dense in the CA1/CA2 pyramidal layer, but I found no primary source on a functionally defined Penk+ CA1 pyramidal subpopulation.

### Cited Findings
- Opioid peptides are co-stored with glutamate in three hippocampal inputs: the lateral perforant path (LPP) to dentate granule cells, the LPP to CA3, and the mossy fibre projection to CA3. — [Bramham & Sarvey 1996, J Neurosci 16(24):8123, "Endogenous activation of mu and delta-1 opioid receptors is required for long-term potentiation induction in the lateral perforant path: dependence on GABAergic inhibition"](https://www.jneurosci.org/content/16/24/8123)
- LTP in the LPP in vivo requires delta opioid receptor activation. Opioid facilitation of LTP depends on GABAergic inhibition: endogenous opioids lower GABA_A inhibition and so allow LTP. — [Bramham et al. 1991, Brain Res, "δ opioid receptor activation is required to induce LTP of synaptic transmission in the lateral perforant path in vivo"](https://www.sciencedirect.com/science/article/abs/pii/0006899391914332); [Bramham & Sarvey 1996](https://www.jneurosci.org/content/16/24/8123); [naloxone depression of DG LTP reversed by GABA_A blockade](https://www.sciencedirect.com/science/article/abs/pii/000689939500510W)
- Leu-enkephalin in mossy fibres is altered by acute and chronic stress in female rats. — [PMC4225781](https://pmc.ncbi.nlm.nih.gov/articles/PMC4225781)
- Leroy et al. 2022 (online 2021). "Enkephalin release from VIP interneurons in the hippocampal CA2/3a region mediates heterosynaptic plasticity and social memory." Mol Psychiatry 27:2879–2900. doi:10.1038/s41380-021-01124-y. Main points:
  - VIP interneurons increase CA3→CA2 transmission by releasing enkephalin.
  - The enkephalin causes long-term depression of feedforward inhibition (DOR-mediated iLTD at PV basket-cell → pyramidal synapses).
  - CA2 VIP activity rises selectively while the mouse explores a novel conspecific.
  - The mechanism is required for social memory.
  - The paper reports the highest density of Penk-expressing cells in the CA1 and CA2 pyramidal layer.
  - [Leroy 2021, Mol Psychiatry](https://www.nature.com/articles/s41380-021-01124-y); [PubMed 33990774](https://pubmed.ncbi.nlm.nih.gov/33990774/)
- DORs also regulate temporoammonic (entorhinal → CA1) feedforward inhibition in mouse CA1. — [Rezai et al. (PMC3829835)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3829835/)
- Cembrowski et al. 2016. "Spatial gene-expression gradients underlie prominent heterogeneity of CA1 pyramidal neurons." Neuron. doi:10.1016/j.neuron.2015.12.013 (from citation record). The paper reports 33, 71 and 265 differentially expressed transcripts along the proximal-distal, superficial-deep and dorsal-ventral axes, and describes CA1 as a continuum of graded expression rather than discrete types. The snippets did not say whether Penk is among the superficial/deep markers. — [Cembrowski 2016 PDF (Janelia)](https://www.janelia.org/sites/default/files/Labs/Spruston%20Lab/Cembrowski%20et%20al.pdf)
- Ventral CA1 pyramidal cells (projecting to nucleus accumbens shell) store social memory. These cells were not defined by Penk. — [Okuyama et al. 2016, Science, doi:10.1126/science.aaf7003](https://www.science.org/doi/10.1126/science.aaf7003)

### Inferences
- The hippocampal pattern is consistent across sites: glutamate/enkephalin co-release from excitatory pathways (LPP, mossy fibres), or enkephalin release from VIP cells, acts mainly by suppressing local GABAergic inhibition (PV/feedforward) and so gates LTP and plasticity at co-active synapses. Applied by analogy to RSC (not shown there), a Penk+ excitatory population could, when strongly active, lower local inhibition and facilitate plasticity in its targets.
- The entorhinal "Penk Ndst4" L2/3 IT supertype (Q1) and the LPP enkephalin system could be the same cells. LEC L2 projection neurons forming the LPP would be the obvious candidate. This identity is not confirmed by any source I found.

### Gaps
- No primary source found for a Penk+ CA1 pyramidal subpopulation in social or contextual memory. The "Penk+ CA1 social memory" premise in the brief could not be verified.
- Cembrowski 2018 (follow-up) was not retrieved.
- No direct source for Penk expression in LEC L2 projection neurons specifically. A search on enkephalin/LEC/novelty returned only non-Penk LEC papers.

---

## Q4. Opioid circuit effects in cortex: MOR/DOR on PV, SST and VIP interneurons; disinhibition; plasticity; activity dependence of release

### Takeaway
In frontal cortex, DORs sit on most PV interneurons and suppress their GABA release, which disinhibits pyramidal cells. DORs also suppress SST→pyramidal release through a separate signalling route, and MOR and DOR act on dissociable interneuron targets. Peptide release from dense-core vesicles needs high-frequency or burst firing, so enkephalin from a Penk+ population should come mainly during episodes of strong activity, not from single spikes.

### Cited Findings
- Birdsong et al. 2019. "Synapse-specific opioid modulation of thalamo-cortico-striatal circuits." eLife 8:e45146. doi:10.7554/eLife.45146. Key results:
  - 90–95% of PV interneurons in ACC express Oprd1.
  - The DOR agonist DPDPE reduces feedforward inhibition onto L5 pyramidal cells, i.e. DORs disinhibit thalamocortical circuits.
  - The brief cited this as Neuron 2019; it is eLife.
  - Source: search summary of the Birdsong 2019 paper, alongside [PMC9242007 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9242007/) and [Opioid modulation of prefrontal cortex cells and circuits, Neuropharmacology 2024](https://www.sciencedirect.com/science/article/pii/S0028390824000583).
- Alexander & Bender 2025. "Delta opioid receptors engage multiple signaling cascades to differentially modulate prefrontal GABA release with input and target specificity." Cell Reports. Key results:
  - DORs suppress GABA release from both PV (perisomatic) and SST (dendritic) interneurons onto mouse PFC L5 pyramidal cells.
  - At PV boutons, the canonical Gβγ → presynaptic Ca²⁺ channel shift lowers release probability and increases short-term plasticity.
  - At SST boutons, several DOR cascades act in parallel in the same bouton and lower release probability without changing short-term plasticity.
  - Result: inhibition is temporally filtered in a way that depends on presynaptic cell identity.
  - [Cell Rep 2025](https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00064-6); [PubMed 39149233](https://pubmed.ncbi.nlm.nih.gov/39149233/)
- "Opioid Receptors Modulate Inhibition within the Prefrontal Cortex through Dissociable Cellular and Molecular Mechanisms." J Neurosci 2025 45(27):e1963242025. doi:10.1523/JNEUROSCI.1963-24.2025 (from citation record). Shows that MOR and DOR modulate PFC inhibition through dissociable cellular and molecular mechanisms. Snippets did not give the per-interneuron-class breakdown. — [J Neurosci 2025](https://www.jneurosci.org/content/45/27/e1963242025); [PMC12225597](https://pmc.ncbi.nlm.nih.gov/articles/PMC12225597/)
- A search summary reports that DORs can increase GABA transmission from SST to PV interneurons, which would disinhibit pyramidal cells. This claim came from an aggregated summary and could not be tied to a specific primary paper. Treat as unverified. — (search summary; see the [PMC9242007 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9242007/))
- DORs on PV cells are needed for the convulsant and anxiolytic effects of the DOR agonist SNC80. This shows that DOR action on PV cells has network-level (seizure/oscillatory) and behavioural consequences. — [PMC12642499](https://pmc.ncbi.nlm.nih.gov/articles/PMC12642499/)
- DOR agonists activate PI3K–mTORC1 signalling in infralimbic PV interneurons and produce acute antidepressant-like effects. — [Mol Psychiatry 2024](https://www.nature.com/articles/s41380-024-02814-z)
- van den Pol 2012. "Neuropeptide transmission in brain circuits." Neuron 76(1):98–115. doi:10.1016/j.neuron.2012.09.014 (from citation record). Neuropeptides are released from dense-core vesicles. High-frequency firing or bursts drive DCV fusion through the resulting calcium influx, which separates peptide release from small-synaptic-vesicle release. — [van den Pol 2012, Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(12)00847-1)
- In mammalian CNS neurons, DCV fusion competence depends on CAPS-1, and DCV pools are small. These points support the view that peptide release is limited and activity-dependent. — [PMC4341531](https://pmc.ncbi.nlm.nih.gov/articles/PMC4341531/); [Persoon et al., EMBO J](https://link.springer.com/article/10.15252/embj.201899672)
- In a defined sensory neuron (C. elegans), neuropeptide release was dissected separately at high and low activity, which supports activity-level dependence (invertebrate, for context only). — [PNAS 2018](https://www.pnas.org/content/115/29/E6890)

### Inferences
- Applied to RSC by analogy (not demonstrated there):
  - If Penk+ L2/3 neurons release enkephalin locally, the most likely targets are DOR on PV cells (perisomatic) and DOR/MOR on SST cells (dendritic).
  - The net effect would be presynaptic suppression of inhibition onto nearby pyramidal cells, i.e. disinhibition with a time-filtering effect that differs by interneuron type.
  - Hippocampal results (LPP, CA2) suggest this disinhibition could gate LTP or heterosynaptic plasticity.
  - DOR-on-PV action could also alter gamma-band activity. This is unverified for RSC.
- Because DCV release needs bursts or high-frequency firing, a Penk+ population that fires sparsely but with long, large bursts would release peptide only during those episodes. Its peptidergic output would be event-locked and slow-acting (GPCR timescale: seconds), not a continuous code.
  - The project's own result (Penk+ cells show sparser, longer events; memory: "Penk+ sparser/longer/smaller events, FDR 0.06") fits this pattern loosely.
  - However, 2P calcium event duration cannot be read as burst firing without ephys calibration.
- A slow, burst-gated disinhibitory signal would suit a modulatory role: flagging salient or novel moments for local plasticity, as in CA2 social novelty. It would suit a moment-to-moment HD signal much less. This would be consistent with weak HD tuning in Penk+ cells, but it is speculative.

### Gaps
- No RSC-specific data on Oprd1/Oprm1 expression by interneuron class, or on opioid effects on RSC circuits.
- No primary source on minimum firing frequency or burst length for enkephalin release from cortical pyramidal neurons. Ludwig & Leng (Nat Rev Neurosci 2006, dendritic peptide release) was not retrieved.
- No data on effects of endogenous (as opposed to agonist-applied) cortical enkephalin on oscillations.

---

## Q5. Behavioural roles of Penk+ / enkephalin-releasing neurons, and any link to spatial navigation

### Takeaway
The best-supported behavioural roles of enkephalin outside the striatum and hypothalamus are hippocampal social memory and novelty (CA2 VIP enkephalin), LTP-dependent hippocampal plasticity, stress/anxiety, and affective pain. I found no study linking Penk+ cortical excitatory neurons, or enkephalin, to spatial navigation or head-direction coding.

### Cited Findings
- Social novelty and memory: CA2 VIP enkephalin release is increased during exploration of a novel conspecific and is required for social memory. — [Leroy 2021, Mol Psychiatry](https://www.nature.com/articles/s41380-021-01124-y)
- Penk is linked to fear conditioning, anxiety and stress responses. Lateral hypothalamic Penk neurons drive threat-induced overeating linked to a negative emotional state. — [Nat Commun 2023, doi:10.1038/s41467-023-42623-6 (from citation record)](https://www.nature.com/articles/s41467-023-42623-6)
- Pain: Penk+ spinal and brainstem enkephalinergic circuits modulate mechanical pain. — [Neuron 2017, brainstem–spinal GABA/enkephalin circuit](https://www.sciencedirect.com/science/article/pii/S0896627317300107)
- ACC: excitation of ACC pyramidal neurons is necessary and sufficient for pain-related negative emotion (not Penk-specific). — [PMC4340873](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4340873/); [PMC7431632](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7431632/)
- Stress alters mossy-fibre leu-enkephalin. — [PMC4225781](https://pmc.ncbi.nlm.nih.gov/articles/PMC4225781)
- Antidepressant- and anxiolytic-like effects of DOR agonists act through PV interneurons. — [PMC12642499](https://pmc.ncbi.nlm.nih.gov/articles/PMC12642499/); [Mol Psychiatry 2024](https://www.nature.com/articles/s41380-024-02814-z)
- RSC L2/3 and L5 represent task features (landmark, trial onset, reward) differently. This is not Penk-specific, but is relevant context for which L2/3 signals a subpopulation might carry. — [search summary of RSC landmark studies, e.g. Fischer et al. 2020, eLife "Representation of visual landmarks in retrosplenial cortex"](https://elifesciences.org/articles/51458)

### Inferences
- Taken together, enkephalin systems are associated with **salience, novelty, social or affective state, and plasticity gating**, not with continuous spatial variables. A plausible but untested hypothesis for Penk+ RSP L2/3: rather than encoding HD or position strongly, these cells may signal salient or novel events (new context, landmark change, light transitions, reward or aversive moments) and, through burst-gated enkephalin release, transiently disinhibit local RSC circuits to allow plasticity, e.g. updating landmark–HD associations. Testable predictions in the hm2p data:
  - Penk+ activity locked to light-on/off transitions or first exposure to maze regions.
  - Larger Penk+ responses early in a session (novelty) than late.
  - Bursts (long/large events) concentrated at those moments.
- **Established fact** (from cited work): enkephalin suppresses PV/SST inhibition and gates hippocampal LTP and social memory. **Speculation** (no source): any of these mechanisms operates in RSC, or Penk+ RSP neurons release enkephalin locally in vivo.

### Gaps
- No source links Penk, enkephalin, DOR or MOR to spatial navigation, HD cells or RSC function.
- No source on enkephalin and contextual (non-social) memory in cortex was retrieved.
- Novelty-specific roles of Penk+ excitatory (as opposed to VIP) neurons were not found.
