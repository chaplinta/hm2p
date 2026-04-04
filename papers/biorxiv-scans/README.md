# bioRxiv Literature Scans

Daily automated literature scans by the RSP science advisor agent.

## Standing Instructions

The RSP science advisor runs daily to scan bioRxiv for recent preprints relevant to
the hm2p project. Each scan produces a markdown file in this directory named
`biorxiv-scan-YYYY-MM-DD.md`.

### Search Topics

Each scan must search for recent preprints (last 7 days) across these topics:

1. **Retrosplenial cortex (RSP/RSC)** — any new RSP papers (anatomy, function, imaging)
2. **Penk / enkephalin + cortex** — Penk-expressing neurons in cortical circuits
3. **Head direction cells + two-photon** — HD cell imaging with 2P (miniscope or head-fixed)
4. **Head direction + darkness / landmarks / drift** — visual anchoring of HD signals
5. **Spatial navigation + maze** — maze-based navigation studies in rodents
6. **Visual processing in RSP** — visual landmark coding, scene processing in RSP
7. **Spatial navigation in RSP** — path integration, spatial coding in RSP
8. **Head-mounted two-photon** — freely-moving 2P microscopy technology
9. **Calcium imaging + maze navigation** — any 2P/miniscope maze navigation studies
10. **Neuropil contamination + two-photon** — neuropil correction methods

### Output Format

Each scan file must follow the format established in `biorxiv-scan-2026-04-02.md`:

- Title: `# bioRxiv Scan — {date}`
- Header paragraph listing search date and terms covered
- Sections: `## Highly relevant papers`, `## Moderately relevant papers`,
  `## Tangentially relevant / methods papers`
- Each paper entry includes:
  - Full citation (authors, year, title, journal/preprint, URL)
  - **Findings:** summary of key results
  - **Relevance to hm2p:** how it connects to our project specifically
- If no new papers found for a topic, note "No new preprints found" for that search
- End with a brief summary of total papers found and any notable trends

### Display

Scan files are automatically displayed on the Literature page of the hm2p frontend
(Biorxiv Scans tab), sorted newest first.
