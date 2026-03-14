# Notepad — hm2p-v2

Working notes and decisions log. Not a design doc — use PLAN.md / ARCHITECTURE.md for that.

---

## Current Status (2026-03-14)

### Pipeline Completion

| Stage | Status | Sessions |
|-------|--------|----------|
| 0 — Ingest (DAQ) | COMPLETE | 26/26 timestamps.h5 on S3 |
| 1 — Suite2p | COMPLETE | 26/26 ca_extraction on S3 |
| 2 — DLC Pose | COMPLETE | 26/26 pose on S3 |
| 3 — Kinematics | COMPLETE | 21/21 kinematics.h5 on S3 |
| 4 — Calcium | COMPLETE | 26/26 ca.h5 on S3 (391 ROIs) |
| 5 — Sync | COMPLETE | 21/21 sync.h5 on S3 |
| 6 — Analysis | COMPLETE | 21/21 analysis.h5 on S3 |

Note: 21/21 refers to non-excluded sessions (5 sessions excluded via experiments.csv).

### Test Coverage

- 1119+ total tests, 91%+ coverage
- 227 patching-specific tests (config, io, ephys, protocols, spike_features, morphology, metrics, statistics, pca, run)

### Frontend

- 43+ pages in 5 navigation sections
- All analysis pages implemented and loading real data from S3

### Remaining Work

- CASCADE spike inference (needs separate conda env)
- FISSA neuropil subtraction (optional)
- neuroconv NWB export
- Patching pipeline: plotting modules + frontend pages
- Credential rotation (hm2p-agent S3 keys)

---

## Decisions Made

### 2026-02-28

- **Repo location**: `/Users/tristan/Neuro/hm2p-v2` -> GitHub `chaplinta/hm2p` (private)
- **File rules**: read-only on `hm2p-analysis/` and Dropbox data; can copy INTO `hm2p-v2`
- **Calcium abstraction**: `roiextractors` as unified API (Suite2p default, CaImAn alt)
- **Pose abstraction**: `movement` as unified API (DLC default, SLEAP/LP alts)
- **Primary behavioural outputs**: HD, position, speed (AHV also computed)
- **Local processing**: all CPU stages can run locally; GPU stages need GPU (cloud or local)
- **Snakemake profiles**: `local`, `local-gpu`, `aws-batch`
- **Cloud provider**: AWS (S3 + EC2 Spot), region ap-southeast-2 (Sydney)
- **Design philosophy**: ground-up redesign, fully unit-tested, not a copy of old code
- **Versions**: always latest stable

### MD filename convention
- Root-level: ALL CAPS (PLAN, ARCHITECTURE, AGENTS, README, NOTEPAD)
- `docs/`: lowercase-hyphenated
