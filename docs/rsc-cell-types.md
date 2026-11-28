# Functional Cell Types in Retrosplenial Cortex

Reference list of functional cell types reported in retrosplenial cortex (RSC), and
how each would be expected to appear in the q-rose maze used in this project.

**Status of this document**

- Bibliographic details (authors, year, journal, DOI) were checked by web search on
  2026-11-28 against abstracts and index records. Full texts were not read. Check
  the primary source before relying on a specific number or claim.
- Section 4 ("Expected appearance in the q-rose maze") contains predictions derived
  from the literature and the maze geometry. None of them has been tested on the
  hm2p data.

---

## 1. Cell types with published evidence in RSC

| Type | Description | Reference |
| --- | --- | --- |
| Head direction (HD) cell | Fires at one allocentric heading. About 8.5% of recorded cells in both granular and dysgranular RSC in rat. | Chen et al. 1994; Cho & Sharp 2001 |
| Bidirectional cell | Fires at two headings 180° apart in a two-compartment environment with opposed cues. Dysgranular RSC. Tuning follows local landmarks, not the global HD signal. | Jacob et al. 2017 |
| Multidirectional (symmetry) cell | HD tuning symmetry matches the rotational symmetry of the environment (two-fold or four-fold). Experience-dependent. Persisted in darkness and was reported not to be explained by egocentric boundary coding. | Zhang et al. 2022 |
| Local structure cell | Tuned to local walls and corners; gives four-fold HD tuning in a square arena. Compared with postrhinal cortex. | LaChance & Hasselmo 2024 |
| Angular head velocity (AHV) cell | Firing varies with turning speed, often direction-selective. Vestibular input dominates; visual input improves coding accuracy. About 10% of layer 2/3 neurons in head-fixed rotation. | Keshavarzi et al. 2022; Hennestad et al. 2021 |
| Egocentric boundary cell (EBC) | Fires when a boundary is at a given distance and bearing relative to the animal. Mostly dysgranular RSC. Independent of self-motion; context-invariant. Does not require visual or whisker input; depends on medial entorhinal input. | Alexander et al. 2020; van Wijngaarden et al. 2020 |
| Vertex cell | Fires when a corner of the environment is at a given egocentric distance and bearing. Granular RSC. Abolished when the physical boundaries are removed. | Park et al. 2024 |
| Place-like / spatial sequence cell | Sparse place fields tiling a linear treadmill track; develops with experience and depends on the hippocampus. | Mao et al. 2017; Mao et al. 2018 |
| Path-integration sequence cell | Sequential activity driven by locomotion-gated optic flow; follows the visual environment when optic flow and locomotion are decoupled. | Mao et al. 2020 |
| Route / turn cell | Encodes conjunctions of progress along a route, allocentric position, and left versus right turns. Some cells repeat their firing pattern across route segments of the same shape. | Alexander & Nitz 2015; Alexander & Nitz 2017 |
| Landmark cell | Responds to visual landmarks that predict reward location; stronger after learning and during active navigation. Head-fixed virtual reality. | Fischer et al. 2020 |
| Visual-heading cell | Combines visual input with anterodorsal thalamic input to register heading to visual cues. Head-fixed, rotating chamber. | Sit & Goard 2023 |
| Visual cell | Retinotopic, position- and orientation-tuned responses, stable over weeks; strong locomotion modulation. | Powell et al. 2020 |
| Goal / reward location cell | Encodes reward location and, after learning, the upcoming goal before the choice point. | Vedder et al. 2017; Miller et al. 2019 |
| Speed-modulated cell | Firing varies with running speed, usually in conjunction with other variables. | Cho & Sharp 2001 |

Grid cells have not been established in RSC.

### Conjunctive coding

Single-variable cells are a minority. Cho & Sharp (2001) reported cells responding to
combinations of location, direction, and movement. Voigts & Harnett (2020) imaged
freely moving mice with two-photon microscopy and reported that apical tuft dendrites
and the soma of the same neuron can be tuned to different navigational variables.

### Intrinsic cell types

Brennan et al. (2020) described small, low-rheobase pyramidal cells as the most
common cell type in layer 2/3 of mouse granular RSC. These cells fire persistently at
high rates with little adaptation. This is an in vitro and modelling result; it is
the closest published precedent for relating an intrinsically or molecularly defined
RSC population to HD-related coding, and is therefore a point of comparison for the
Penk+ versus Penk⁻CamKII+ analysis in this project.

### Recent items not yet checked

- A 2026 PNAS paper (PMID 42520117), "Anterior and posterior retrosplenial cortex
  employ distinct strategies for egocentric-allocentric transformation in spatial
  coding", reports two-photon imaging in freely navigating mice, with anterior RSC
  dominated by egocentric boundary vector coding and posterior RSC showing more
  allocentric boundary coding. Only the search summary was seen. Relevant to the
  anteroposterior position of the imaging field of view.
- A 2026 bioRxiv preprint, "Active pursuit gates egocentric coding in the
  retrosplenial cortex" (doi:10.64898/2026.02.14.705560), reports egocentric target
  cells during prey pursuit in rats. Not applicable to this experiment.

---

## 2. HD coordination with thalamus in darkness

van der Goes et al. (2024) recorded RSC and the anterodorsal thalamic nucleus
simultaneously in freely moving mice and reported that RSC HD signals follow
thalamic HD signals during cue rotation, and that the coordination is maintained in
darkness.

---

## 3. Maze geometry

The maze polygon is defined in [data-guide.md](data-guide.md) ("Maze Coordinate
System") on a 7 × 5 unit grid. Open squares:

```text
■ ■ ■│■│■ ■ ■
  ■   ■   ■
  ■ ■ ■ ■ ■
  ■       ■
■ ■ ■   ■ ■ ■        │ = wall between adjacent squares
```

The orientation of this diagram is arbitrary (image coordinates; see the
`orientation` column in `experiments.csv`).

- Two columns, each five units long, each capped at both ends by a three-unit bar.
- A crossbar joining the two columns at their midpoints.
- A two-unit dead-end stub off the middle of the crossbar.
- 23 open squares, 9 dead ends, 7 junctions.
- Mirror-symmetric about the line through the stub. The stub breaks the second
  mirror axis and the two-fold rotational symmetry.
- One unit is about 70 mm (maze ROI 608 × 437 px at 0.8113 mm/px in the example
  `meta.txt` for session `20210823_16_59_50_1114353`), so the maze is about
  49 × 35 cm and every corridor is about 7 cm wide.

Consequences for tuning analyses:

1. The animal is always within about 3.5 cm of a wall on both sides. There is no
   open space.
2. In a 7 cm corridor the body is aligned with the corridor, so heading is
   concentrated on four directions and is correlated with position. Oblique
   headings are sampled mainly at junctions and during turns.
3. Turning (high AHV, low speed) occurs at junctions and dead ends, so AHV and
   speed are also correlated with position.
4. Each heading is sampled in several spatially separate corridors: five segments
   along the long axis (four caps and the crossbar) and three along the short axis
   (two columns and the stub).
5. The four caps are locally identical, and the eight cap ends face in two opposite
   directions (four each).

---

## 4. Expected appearance in the q-rose maze

Predictions only. "Light versus dark" refers to the 1 min on / 1 min off protocol.

| Type | Expected appearance | Light versus dark |
| --- | --- | --- |
| HD cell | Same preferred heading in every corridor of the same orientation, regardless of location. | Preferred direction drifts during the dark period and resets at light onset. |
| Place-like / sequence cell | Field in one corridor segment, possibly for one running direction only. This produces a narrow HD tuning curve that does not replicate in other corridors of the same orientation. | Not established for this preparation; visually driven fields would weaken in darkness. |
| EBC, lateral preferred bearing | A side wall is within range almost continuously, so firing is nearly constant and the cell appears untuned. Preferred distances longer than the corridor half-width cannot be sampled laterally. | No change expected (van Wijngaarden et al. 2020). |
| EBC, wall ahead or behind | Fires at dead ends and when facing the wall at T-junctions. Because the cap ends face two opposite directions, the HD tuning curve has two lobes about 180° apart, with smaller lobes from the column ends and the stub. | No change expected. |
| Vertex cell | Fields at junctions and dead ends, where the corners are. Difficult to separate from an EBC in this geometry. | Not reported. |
| Multidirectional / local structure cell | Two-fold or four-fold HD tuning. If experience-dependent as in Zhang et al. (2022), the effect should increase with `maze_session_num`. | Persists in darkness (Zhang et al. 2022). |
| AHV cell | Activity concentrated at junctions and dead ends. Direction-selective cells depend on the turn taken. | Coding accuracy expected to be higher with lights on (Keshavarzi et al. 2022). |
| Speed-modulated cell | Activity concentrated mid-corridor; approximately the inverse of the AHV map. | No specific prediction. |
| Route / turn cell | Left versus right selectivity at T-junctions, or a field at the equivalent position in each of the four caps. Reported for trained routes; may be weaker in free exploration. | No specific prediction. |
| Visual / visual-heading cell | Transient at each light transition, or activity only during light periods. | Present in light only. |
| Goal / reward cell | Not applicable; the protocol as documented has no reward. | Not applicable. |

---

## 5. Implications for analysis

1. **Bimodal HD tuning has several possible sources.** Wall-ahead EBCs, symmetry
   cells, and place-like cells with fields on both passes through a corridor all
   produce two lobes 180° apart. A unimodal mean-vector-length criterion has low
   sensitivity to two-fold and four-fold tuning; the vector length computed on 2θ
   and 4θ is one way to detect it.
2. **Stable tuning in darkness does not by itself indicate path integration.** EBC
   and symmetry-driven tuning are expected to be unchanged in darkness. Progressive
   drift of the preferred direction during the dark period, with reset at light
   onset, is the pattern expected of an HD cell that is anchored to visual cues.
3. **Replication across corridors as an HD criterion.** A cell classified as an HD
   cell should keep its preferred direction across spatially separate corridors of
   the same orientation.
4. **Compare event rates between light and dark before comparing tuning.** Visually
   driven cells change rate between conditions, and tuning and information metrics
   depend on rate.
5. **Fit the variables jointly.** HD, position, AHV, speed, and egocentric boundary
   bearing are mutually correlated in this maze. A joint encoding model (for example a
   GLM fitted with NEMOS) separates their contributions more reliably than
   single-variable tuning curves.

---

## References

Alexander et al. 2020. "Egocentric boundary vector tuning of the retrosplenial cortex." Science Advances 6:eaaz2322. doi:10.1126/sciadv.aaz2322

Alexander et al. 2023. "Rethinking retrosplenial cortex: Perspectives and predictions." Neuron 111:150–175. PMID 36460006 (review; DOI not confirmed)

Alexander & Nitz 2015. "Retrosplenial cortex maps the conjunction of internal and external spaces." Nature Neuroscience 18:1143–1151. doi:10.1038/nn.4058

Alexander & Nitz 2017. "Spatially periodic activation patterns of retrosplenial cortex encode route sub-spaces and distance traveled." Current Biology 27(11). Preprint doi:10.1101/100537 (journal DOI not confirmed)

Brennan et al. 2020. "Hyperexcitable neurons enable precise and persistent information encoding in the superficial retrosplenial cortex." Cell Reports 30:1598–1612. doi:10.1016/j.celrep.2019.12.093

Chen et al. 1994. "Head-direction cells in the rat posterior cortex. I. Anatomical distribution and behavioral modulation." Experimental Brain Research 101:8–23. doi:10.1007/BF00243212

Cho & Sharp 2001. "Head direction, place, and movement correlates for cells in the rat retrosplenial cortex." Behavioral Neuroscience 115:3–25. doi:10.1037/0735-7044.115.1.3

Fischer et al. 2020. "Representation of visual landmarks in retrosplenial cortex." eLife 9:e51458. doi:10.7554/eLife.51458

Hennestad et al. 2021. "Mapping vestibular and visual contributions to angular head velocity tuning in the cortex." Cell Reports 37:110134. doi:10.1016/j.celrep.2021.110134

Jacob et al. 2017. "An independent, landmark-dominated head-direction signal in dysgranular retrosplenial cortex." Nature Neuroscience 20:173–175. doi:10.1038/nn.4465

Keshavarzi et al. 2022. "Multisensory coding of angular head velocity in the retrosplenial cortex." Neuron 110:532–543. doi:10.1016/j.neuron.2021.10.031

LaChance & Hasselmo 2024. "Distinct codes for environment structure and symmetry in postrhinal and retrosplenial cortices." Nature Communications 15. doi:10.1038/s41467-024-52315-4

Mao et al. 2017. "Sparse orthogonal population representation of spatial context in the retrosplenial cortex." Nature Communications 8:243. doi:10.1038/s41467-017-00180-9

Mao et al. 2018. "Hippocampus-dependent emergence of spatial sequence coding in retrosplenial cortex." PNAS 115:8015–8018. doi:10.1073/pnas.1803224115

Mao et al. 2020. "Vision and locomotion combine to drive path integration sequences in mouse retrosplenial cortex." Current Biology 30:1680–1688. doi:10.1016/j.cub.2020.02.070

Miller et al. 2019. "Retrosplenial cortical representations of space and future goal locations develop with learning." Current Biology 29:2083–2090. doi:10.1016/j.cub.2019.05.034

Park et al. 2024. "Egocentric neural representation of geometric vertex in the retrosplenial cortex." Nature Communications 15. doi:10.1038/s41467-024-51391-w

Powell et al. 2020. "Stable encoding of visual cues in the mouse retrosplenial cortex." Cerebral Cortex 30:4424–4437. doi:10.1093/cercor/bhaa030

Sit & Goard 2023. "Coregistration of heading to visual cues in retrosplenial cortex." Nature Communications 14. doi:10.1038/s41467-023-37704-5

van der Goes et al. 2024. "Coordinated head direction representations in mouse anterodorsal thalamic nucleus and retrosplenial cortex." eLife. doi:10.7554/eLife.82952

van Wijngaarden et al. 2020. "Entorhinal-retrosplenial circuits for allocentric-egocentric transformation of boundary coding." eLife 9:e59816. doi:10.7554/eLife.59816

Vedder et al. 2017. "Retrosplenial cortical neurons encode navigational cues, trajectories and reward locations during goal directed navigation." Cerebral Cortex 27:3713–3723. doi:10.1093/cercor/bhw192

Voigts & Harnett 2020. "Somatic and dendritic encoding of spatial variables in retrosplenial cortex differs during 2D navigation." Neuron 105:237–245. doi:10.1016/j.neuron.2019.10.016

Zhang et al. 2022. "Environment symmetry drives a multidirectional code in rat retrosplenial cortex." Journal of Neuroscience 42:9227–9241. doi:10.1523/JNEUROSCI.0619-22.2022
