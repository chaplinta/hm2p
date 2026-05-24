# bioRxiv Scan — 18 May 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁺CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-05-18. Primary window: May 17–18, 2026. This scan also recovers one
paper posted May 15 that was not captured in the May 16 scan (coppaFISH 3D; see
Moderately relevant below). Searches covered: retrosplenial cortex (RSP/RSC),
Penk/enkephalin + cortex, head direction + two-photon imaging, head direction +
darkness/landmarks/drift, spatial navigation + maze (rodents), visual processing in RSP,
spatial navigation in RSP, head-mounted two-photon microscopy, calcium imaging + maze
navigation, neuropil contamination + two-photon, calcium imaging + cell type identification
+ transcriptomics.

Papers listed in any prior scan (2026-04-02 through 2026-05-16) are not repeated here.

---

## Highly relevant papers

No new highly relevant preprints were found in the May 17–18 window.

---

## Moderately relevant papers

### 1. Spatially resolved transcriptomic identification of thousands of neurons recorded in vivo (coppaFISH 3D + CASTalign)

Prankerd I, Shinn M, Shuker PC, Zhou Z, Tilbury R, Duffield JAM, Maat CA,
Nicoloutsopoulos D, Ritoux A, Maglio Cauhy PV, Orme D, Bourdenx M, Duff KE,
Bugeron S, Isogai Y, Harris KD. 2026.
"Spatially resolved transcriptomic identification of thousands of neurons recorded
in vivo." bioRxiv.
https://www.biorxiv.org/content/10.64898/2026.05.15.725413v1

**Findings:** Introduces coppaFISH 3D, a modified in situ hybridisation chemistry
optimised for thick free-floating brain sections (reagents access from both faces),
and CASTalign, a computational framework that registers the post-hoc transcriptomic
volume back to the in vivo two-photon calcium imaging coordinate frame. Together the
pipeline assigns transcriptomic cell-type identity to thousands of simultaneously
recorded neurons in mouse cortex. The workflow is: mice undergo in vivo two-photon
imaging → acquisition of a reference z-stack → perfusion → sectioning → coppaFISH 3D
→ gene calling → cell typing → alignment back to in vivo imaging volume or to post hoc
antibody staining. Demonstrated in layer 2/3 of mouse cortex with a panel of marker
genes sufficient to resolve principal cell subtypes. The method is compatible with
standard two-photon calcium imaging rigs without hardware modification.

**Relevance to hm2p:** We already resolve Penk+ vs Penk⁺CamKII+ populations via viral
intersectional labelling, so coppaFISH 3D is not needed for our primary cell-type
classification. However, three uses are plausible. (1) Validation: post hoc coppaFISH
against Penk mRNA in a subset of animals could confirm that viral labelling faithfully
captures the intended population and identify any off-target expression. (2) Additional
subtypes: within our Penk⁺CamKII+ population, transcriptomic subtyping could reveal
whether HD tuning differences correlate with molecular identity beyond the CamKII
promoter level. (3) Interneuron context: applying coppaFISH to the SST/PV targets
identified in Oh et al. 2026 (May 14 scan) in the same tissue would allow direct
test of whether PV/SST interneuron density correlates with cell-type-specific HD
tuning precision in our RSP plane. This paper is a future-facing methods note; no
immediate citation planned but relevant to mention in the methods discussion if we
perform any post hoc histological validation.

---

## Tangentially relevant / methods papers

No new papers in this category were found in the May 17–18 window.

---

## Searches with no new relevant results (May 17–18)

**Retrosplenial cortex (RSP/RSC):** No new RSC/RSP preprints identified in the May 17–18
window on spatial coding, HD tuning, visual processing, or cell-type-specific function.

**Penk/enkephalin + cortex:** No new preprints. Subcortical Penk papers (striatum,
brainstem, hypothalamus) continue to dominate. This is the twenty-third consecutive
scan cycle without a cortical Penk+ navigational or spatial paper. The absence of any
literature on Penk+ RSP neuron function in a navigational context remains total across
all 25 scans to date.

**Head direction + darkness / visual landmarks / drift:** No new experimental papers
this window. The landmark-drift and cue-conflict landscape is unchanged since the
April 2026 scans.

**Head direction + two-photon imaging (freely moving):** No new preprints.

**Spatial navigation + maze + calcium imaging (freely moving mouse):** No new papers
matching all criteria.

**Visual processing in RSP / spatial navigation in RSP:** No new preprints.

**Head-mounted / miniature two-photon microscopy:** No new technology papers.

**Neuropil contamination + two-photon:** No new correction methods or benchmarking papers.

---

## Summary

**Total new relevant preprints this scan:** 1 (0 highly relevant, 1 moderately relevant,
0 tangential). The coppaFISH 3D / CASTalign paper (May 15, missed in the May 16 scan)
is the sole addition.

The May 17–18 window is quiet across all monitored topic areas, extending the pattern
of low output in RSP/HD neuroscience relative to hippocampus and MEC. No paper posted
this week directly addresses RSP HD cell-type comparisons, visual cue dependence, or
light/dark alternation.

**Cumulative picture (April 2–May 18, 47 days, 25 scans):**

No preprint has characterised HD tuning of genetically-defined RSP excitatory neuron
populations (Penk+ or Penk⁺CamKII+) across any scan window. The RSP Penk+ gap is
unchanged. The most recent substantive RSP-specific preprints remain:

- Oh et al. 2026 (May 10): RSC PV/SST interneurons and egocentric spatial coding
  precision/stability — first RSC interneuron-subtype dissociation in freely moving mice.
- Coordinated representational drift paper (May 5, first captured May 16 scan):
  RSC confirmed as highest-density spatially tuned region in dorsal cortex; population
  manifold mirrors maze geometry.

**Persistent action items (unchanged):**

1. Separate moving vs. stationary epochs in all light/dark HD tuning comparisons.
2. Document imaging position along the RSC anterior-posterior axis; cite Wei et al.
   Nature Communications (doi:10.1038/s41467-026-70762-z).
3. CEBRA orthogonal subspace test: does light/dark context separate from the HD
   coding subspace in population activity?
4. Update Jayakumar et al. citation to Current Biology DOI (S0960-9822(26)00222-8).
5. In the Discussion, frame cell-type HD tuning differences in terms of the
   PV-precision / SST-stability dissociation from Oh et al. 2026.
