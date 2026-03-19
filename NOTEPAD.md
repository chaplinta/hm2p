# Notepad — hm2p-v2

Working notes and decisions log. Not a design doc — use PLAN.md / ARCHITECTURE.md for that.

---

## Current Status (2026-03-19)

### Pipeline Completion

| Stage | Status | Sessions |
|-------|--------|----------|
| 0 — Ingest (DAQ) | COMPLETE | 26/26 timestamps.h5 on S3 |
| 1 — Suite2p | COMPLETE | 26/26 ca_extraction on S3 |
| 2 — DLC Pose | RE-RUNNING | max_individuals=1 + video_adapt=True (i-023165e775ad63b3d) |
| 3 — Kinematics | PENDING RE-RUN | Blocked on DLC; perspective correction implemented |
| 3b — MoSeq | PENDING RE-RUN | Code ready; blocked on DLC |
| 4 — Calcium | COMPLETE | 26/26 ca.h5 on S3 (391 ROIs) — not affected by DLC re-run |
| 5 — Sync | PENDING RE-RUN | Blocked on kinematics |
| 6 — Analysis | PENDING RE-RUN | Blocked on sync |

Dependency chain: DLC → Kinematics → MoSeq → Sync → Analysis.

### Test Coverage

- 1500+ total tests, 91%+ coverage
- 227 patching-specific tests
- 31 perspective correction tests (new)

### Frontend

- 53 pages in 5 navigation sections
- All analysis pages implemented and loading real data from S3

### Remaining Work

- DLC re-run completion → re-run downstream stages (3, 3b, 5, 6)
- CASCADE spike inference (needs separate conda env)
- FISSA neuropil subtraction (optional)
- neuroconv NWB export
- Credential rotation (hm2p-agent S3 keys)
- Terminate stopped EC2 instances (7 instances accruing EBS costs)

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
