# bioRxiv Scan — 7 June 2026

Literature scan for recent preprints relevant to the hm2p project: two-photon calcium
imaging of Penk+ and Penk⁻CamKII+ RSP head-direction cells in freely moving mice,
light/dark alternation in a rose maze.

Search date: 2026-06-07. Primary window: June 6–7, 2026 (since the June 6 scan).
The June 6 scan covered June 3–6 exhaustively; only new papers not captured in any
prior scan (2026-04-02 through 2026-06-06) are included here.
Searches covered: retrosplenial cortex (RSP/RSC), Penk/enkephalin + cortex, head
direction + two-photon imaging, head direction + darkness/landmarks/drift, spatial
navigation + maze (rodents), visual processing in RSP, spatial navigation in RSP,
head-mounted two-photon microscopy, calcium imaging + maze navigation, neuropil
contamination + two-photon, spike inference / deconvolution + calcium imaging.

Note: direct bioRxiv API access (api.biorxiv.org) returned HTTP 403 errors throughout
this session. All searches performed via web search; newly posted papers indexed with a
lag of days to weeks may not be captured.

---

## Highly relevant papers

No new highly relevant preprints found for June 6–7, 2026.

---

## Moderately relevant papers

### 1. Precise calcium-to-spike inference using biophysical generative models (v3)

Authors not specified in search results (preprint originally posted December 2024,
v3 update reported as June 4, 2026). bioRxiv.
https://www.biorxiv.org/content/10.1101/2024.12.31.630967v3

**Findings:** Characterises the calcium response kinetics of GCaMP6f, jGCaMP7f, and
jGCaMP8f indicators to build biophysically-inspired Bayesian Sequential Monte Carlo and
machine learning inference models. The approach outperforms existing deconvolution
methods (including CASCADE) on spike timing accuracy and correlation benchmarks, with
particular gains in low-firing-rate regimes. Tested on in vivo data.

**Relevance to hm2p:** Directly relevant to Stage 4 of the pipeline. CASCADE is our
current primary spike-inference tool. This v3 update (if confirmed as June 4, 2026)
represents a competing method worth benchmarking against our GCaMP6s data. Note: the
paper tests GCaMP6f not GCaMP6s; kinetic differences mean the biophysical model
parameters would need re-fitting before direct application. The benchmark comparison
against CASCADE is the most actionable finding — if CASCADE performance is competitive
with GCaMP6f data, it is likely adequate for our GCaMP6s recordings where the slower
indicator kinetics are less demanding for spike timing. No immediate pipeline change
recommended; file for Stage 4 methods decision log.

---

## Tangentially relevant / methods papers

No new tangentially relevant or methods papers found for June 6–7, 2026 beyond the
spike inference paper noted above.

---

## Searches with no new results (June 6–7)

**Retrosplenial cortex (RSP/RSC):** No new RSC/RSP preprints on HD tuning, visual
processing, or cell-type-specific function. The most recent RSC preprints in the
monitoring window remain unchanged: Oh et al. 2026 (May 10, RSC PV/SST egocentric
coding), the coordinated drift paper (May 5), and Prankerd et al. 2026 (May 15,
coppaFISH for transcriptomic cell-type assignment).

**Penk/enkephalin + cortex:** No new preprints. This is the thirty-fifth consecutive
scan cycle without a cortical Penk+ spatial or navigational paper. All Penk hits across
the full monitoring period remain subcortical (striatum, hypothalamus, MPOA, brainstem,
dorsal raphe). The absence of any literature characterising Penk+ RSP neurons in a
spatial or HD context is complete and now well-documented.

**Head direction + two-photon imaging (freely moving):** No new preprints.

**Head direction + darkness / visual landmarks / drift:** No new experimental papers.
The most recent relevant paper in the broader literature remains the Jayakumar et al.
2025 path-integration recalibration paper (Current Biology doi:S0960-9822(26)00222-8).

**Spatial navigation + maze + calcium imaging (freely moving mouse):** No new papers.

**Visual processing in RSP / spatial navigation in RSP:** No new preprints.

**Head-mounted / miniature two-photon microscopy:** No new technology papers.

**Neuropil contamination + two-photon:** No new correction methods or benchmarking.

---

## Summary

**Total new relevant preprints this scan:** 1 (moderately relevant, methods).

The June 6–7 window is quiet across all core topic areas, consistent with the pattern
since the May 14–18 cluster. The spike inference v3 is the only new find; it does not
require immediate action.

**Cumulative picture (April 2–June 7, 66 days, 35 scans):**

No preprint has characterised HD tuning of genetically-defined RSP excitatory neuron
populations (Penk+ or Penk⁻CamKII+) across any scan window. The three most recent
substantive preprints within the active monitoring window remain unchanged from
the June 6 scan:

- **Oh et al. 2026** (May 10): RSC PV/SST interneurons and egocentric spatial coding
  precision/stability. PV cells govern egocentric coding precision via motion-linked
  bearing-aligned synchrony; SST cells govern long-term global stability via
  boundary-anchored activity. Optogenetic silencing confirms independent roles.
  https://www.biorxiv.org/content/10.64898/2026.05.10.724096v1

- **Coordinated representational drift across mouse cortex** (May 5): RSC confirmed as
  highest-density spatially tuned region in dorsal cortex alongside V1; coordinated
  multi-region drift over a 47-day longitudinal dataset; within-session stability
  substantially higher than cross-session.
  https://www.biorxiv.org/content/10.64898/2026.05.05.723038v1

- **Prankerd et al. 2026** (May 15): coppaFISH 3D volumetric in situ sequencing +
  CASTalign pipeline for post-hoc transcriptomic cell-type assignment to in vivo
  2P-imaged neurons.
  https://www.biorxiv.org/content/10.64898/2026.05.15.725413v1

**Persistent action items (unchanged from June 6 scan):**

1. Separate moving vs. stationary epochs in all light/dark HD tuning comparisons
   (Jayakumar et al. locomotion-dependent recalibration result).
2. Document imaging position along the RSC anterior-posterior axis; cite Wei et al.
   Nature Communications (doi:10.1038/s41467-026-70762-z).
3. CEBRA orthogonal subspace test: does light/dark context separate from the HD
   coding subspace in population activity?
4. Update Jayakumar et al. citation to Current Biology DOI (S0960-9822(26)00222-8).
5. In the Discussion, frame cell-type HD tuning differences in terms of the
   PV-precision / SST-stability dissociation from Oh et al. 2026.
6. Note coordinated representational drift result when reporting session-wise HD
   tuning stability — within-session estimates are more reliable than cross-session.
7. (New) File the biophysical spike inference v3 (10.1101/2024.12.31.630967v3) for
   Stage 4 methods decision log; benchmark against CASCADE if GCaMP6f data become
   available, but no immediate pipeline change needed for our GCaMP6s recordings.
