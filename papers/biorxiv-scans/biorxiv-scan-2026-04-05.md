# bioRxiv Scan — 5 April 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk-CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-04-05. Searches covered: retrosplenial cortex, head direction +
two-photon, head direction + darkness/landmarks/drift, retrosplenial + navigation,
Penk/enkephalin + cortex, calcium imaging + maze navigation, neuropil + two-photon,
head-mounted two-photon, head direction + cell type specific, spatial navigation +
two-photon, visual landmark + navigation.

---

## Summary

A quiet week for our core topics. No new preprints on RSP head direction cells,
RSP visual landmark anchoring, or Penk-expressing cortical neurons appeared in the
March 29 -- April 5 window. The most relevant new paper is a subiculum projectome
study (April 1) that maps cell-type-specific projection patterns — relevant for
understanding the RSP input landscape. One RSP paper from March 27 (just before the
previous scan) on hypoglycemia vulnerability was not covered previously and is
included below for completeness, though it is tangential to our HD/navigation focus.

---

## Moderately relevant papers

### 1. Mapping projectome heterogeneity of subiculum neuron cell types

Authors not fully specified. 2026.
"Mapping Projectome Heterogeneity of Subiculum Neuron Cell Types." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.04.01.716004v1

Posted: April 1 or 4, 2026 (within scan window).

**Findings:** Classified 689 subiculum projection neurons into 12 cell-type groups
based on laminar and columnar distributions using the Hippocampus Gene Expression
Atlas (HGEA). Identified distinct connectivity motifs and axonal collateralisation
patterns for each cell type, with gene expression boundaries corresponding to
brain-wide networks involved in spatial navigation, social behaviour, and
neuroendocrine regulation.

**Relevance to hm2p:** The subiculum is a major input to RSP and a key node in the
HD circuit (PoS/subiculum → RSP). Understanding subicular cell-type diversity and
their distinct projection targets constrains how we interpret RSP inputs. If specific
subicular cell types preferentially project to RSP, the HD signals arriving in RSP
may already be cell-type-filtered. However, this is an anatomical tracing study, not
a functional one — no direct implications for our calcium imaging analyses. Useful
background for the discussion of RSP input pathways.

---

## Tangentially relevant papers

### 2. Retrosplenial cortex vulnerability links severe hypoglycemia to cognitive impairment

Joo JY, Lee S, Shin MK, Kim S, Park S, Heo JH, Kim M, Lee H, Park K, Koo D,
Lee HY, Kim JI, Kwon O. 2026.
"Retrosplenial cortex vulnerability links severe hypoglycemia to cognitive impairment
through neuron-microglia crosstalk." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.03.27.714654v1

Posted: March 27, 2026 (just before the previous scan window; not covered in the
April 2 scan).

**Findings:** Identified the retrosplenial cortex as particularly vulnerable to
hypoglycemia-induced neuronal damage in mouse models. The injury is driven by
neuron-specific Drp1-dependent mitochondrial fission and microglial IL-1 signalling.
Targeting either pathway rescued neuronal damage and reversed cognitive impairment.

**Relevance to hm2p:** Not directly relevant to HD tuning or spatial navigation. The
paper is notable because it highlights RSP vulnerability to metabolic stress, which
could be relevant if any of our animals experienced health issues. More broadly, it
contributes to the picture of RSP as a region with distinctive cellular properties,
but does not inform our cell-type-specific HD analyses.

---

### 3. Extracting large-scale neural activity with Suite2p (updated pipeline paper)

Pachitariu M, Stringer C, et al. 2026.
"Extracting large-scale neural activity with Suite2p." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.02.04.703741v1

Posted: February 4, 2026 (outside scan window; included because it was not covered
in the April 2 scan and is methodologically important).

**Findings:** Updated Suite2p paper describing GPU-accelerated motion correction,
improved cell detection (outperforming CaImAn/Fiola), neuropil correction, and spike
deconvolution. Demonstrated recordings of >100,000 neurons from mouse cortex.

**Relevance to hm2p:** We use Suite2p as our primary extractor. This updated paper
should be cited in our methods. The neuropil correction approach described (fixed
coefficient subtraction) is the same method we use. Worth checking whether the
updated version includes any changes to default neuropil correction parameters or
cell detection algorithms that might affect our ROI extraction.

---

## Searches with no relevant results

**Penk/enkephalin + cortex:** No new preprints on Penk-expressing neurons in cortical
circuits in a spatial navigation or HD context. The only 2026 Penk paper found was on
striatal D2-MSN enkephalin in cocaine abstinence (March 11, 2026) — not relevant.
The gap in Penk+ cortical neuron characterisation remains open.

**Head direction + two-photon:** No new papers beyond the Tian et al. 2026 MEC paper
already covered in the April 2 scan.

**Head direction + darkness/landmarks/drift:** No new papers within the scan window.

**Head direction + cell type specific:** Two papers from January 2026 (already covered
or outside window): "Divergent excitatory and inhibitory signaling in a head direction
circuit" (Jan 18) and "Active locomotion predictively rescues head direction attractor
dynamics in head-fixed mice" (Jan 12). Neither is new this week.

**Calcium imaging + maze navigation:** No new relevant rodent papers within the scan
window.

**Neuropil contamination + two-photon:** No new papers beyond the Suite2p update
noted above.

**Head-mounted two-photon:** No new miniature 2P technology papers within the scan
window.

**Retrosplenial + navigation:** No new papers beyond those already covered in the
April 2 scan.

---

## Implications for hm2p

**No changes to project strategy.** This was a quiet week for our core research
areas. The literature gap identified in the April 2 scan remains: no papers
characterise HD properties of genetically-defined RSP subpopulations. Our study
continues to address a genuine gap.

**Action items:**
- Update Suite2p citation to include the new 2026 paper (Pachitariu & Stringer 2026)
  in our methods.
- Note the subiculum projectome paper for the discussion of RSP input diversity.
